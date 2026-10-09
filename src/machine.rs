//! Where an agent's commands and file operations happen: an ErisSandbox [`Sandbox`], which
//! isolates them, or [`Direct`], which runs them on this machine as this user. Tools see only
//! the [`Machine`] trait, so the model cannot tell them apart. Sandboxes need Linux; direct
//! machines run on any Unix.

#[cfg(target_os = "linux")]
pub use erissandbox::SandboxSpec;
pub use erissandbox::{DRAIN_GRACE, ExitStatus, Killer, OpenMode, Output, Process};

use anyhow::{Result, bail};
use erissandbox::open_path;
#[cfg(target_os = "linux")]
use erissandbox::{Host, OutsideCommand, Sandbox};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::io;
use std::os::fd::OwnedFd;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

pub trait Machine: Send + Sync {
    /// Working directory for commands and relative paths.
    fn cwd(&self) -> &str;
    /// What `~` means.
    fn home(&self) -> &str;
    /// Starts `argv` in `cwd` (the machine's own when `None`). Its combined output must be
    /// read, or it blocks once the pipe fills.
    fn spawn<'a>(&'a self, argv: &'a [String], cwd: Option<&'a str>) -> BoxFuture<'a, Result<Process>>;
    /// Opens an absolute path as the machine sees it.
    fn open<'a>(&'a self, path: &'a str, mode: OpenMode) -> BoxFuture<'a, io::Result<OwnedFd>>;
}

/// The machine an agent runs on, persisted with the agent.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MachineSpec {
    #[cfg(target_os = "linux")]
    Sandbox(SandboxSpec),
    Direct(DirectSpec),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DirectSpec {
    pub cwd: String,
    /// The complete environment for commands, or `None` to inherit the harness's.
    pub env: Option<Vec<(String, String)>>,
}

/// Runs `argv` to completion and collects its output.
pub async fn run(machine: &dyn Machine, argv: &[String]) -> Result<Output> {
    let mut bytes = Vec::new();
    let status = machine.spawn(argv, None).await?.drain(|chunk| bytes.extend_from_slice(chunk)).await;
    Ok(Output { bytes, status })
}

#[cfg(target_os = "linux")]
impl Machine for Sandbox {
    fn cwd(&self) -> &str {
        &self.spec().cwd
    }

    fn home(&self) -> &str {
        self.spec().home()
    }

    fn spawn<'a>(&'a self, argv: &'a [String], cwd: Option<&'a str>) -> BoxFuture<'a, Result<Process>> {
        Box::pin(Sandbox::spawn(self, argv, cwd))
    }

    fn open<'a>(&'a self, path: &'a str, mode: OpenMode) -> BoxFuture<'a, io::Result<OwnedFd>> {
        Box::pin(Sandbox::open(self, path, mode))
    }
}

/// Runs commands on this machine as this user. In a program that bootstrapped ErisSandbox,
/// commands start outside its namespace through its host (see [`Direct::outside`]), so they see
/// the machine exactly as the user does.
pub struct Direct {
    spec: DirectSpec,
    home: String,
    #[cfg(target_os = "linux")]
    host: Option<Host>,
}

impl Direct {
    pub fn new(spec: DirectSpec) -> Self {
        let from_spec = spec.env.as_ref().and_then(|env| env.iter().find(|(k, _)| k == "HOME").map(|(_, v)| v.clone()));
        let home = from_spec.or_else(|| std::env::var("HOME").ok()).unwrap_or_else(|| "/".into());
        Self {
            spec,
            home,
            #[cfg(target_os = "linux")]
            host: None,
        }
    }

    /// A direct machine for a program running inside ErisSandbox's namespace: its commands
    /// start outside it through `host`.
    #[cfg(target_os = "linux")]
    pub fn outside(spec: DirectSpec, host: Host) -> Self {
        Self { host: Some(host), ..Self::new(spec) }
    }

    async fn start(&self, argv: &[String], cwd: &str) -> Result<Process> {
        if !Path::new(cwd).is_dir() {
            bail!("Working directory does not exist: {cwd}");
        }
        #[cfg(target_os = "linux")]
        if let Some(host) = &self.host {
            let env = self.spec.env.clone().unwrap_or_else(|| std::env::vars().collect());
            return host
                .spawn_outside(OutsideCommand { argv: argv.to_vec(), cwd: cwd.into(), env, terminal: None })
                .await;
        }
        self.start_here(argv, cwd)
    }

    fn start_here(&self, argv: &[String], cwd: &str) -> Result<Process> {
        let Some((program, args)) = argv.split_first() else { bail!("empty command") };
        let (output, input) = io::pipe()?;
        let mut command = std::process::Command::new(program);
        command.args(args).current_dir(cwd).stdin(std::process::Stdio::null());
        command.stdout(input.try_clone()?).stderr(input);
        if let Some(env) = &self.spec.env {
            command.env_clear().envs(env.iter().map(|(k, v)| (k, v)));
        }
        // SAFETY: setsid is async-signal-safe. Each command leads its own process group, so a
        // kill reaches its pipelines and children.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        let child = command.spawn().map_err(|e| anyhow::anyhow!("{program}: {e}"))?;
        drop(command);
        let pid = child.id() as libc::pid_t;
        // Once the leader is reaped its pid can be reused; the group is never signalled after.
        let reaped = Arc::new(Mutex::new(false));
        let exit = reap_unless_killing(pid, reaped.clone())?;
        let killer = Killer::new(move || {
            let reaped = reaped.clone();
            Box::pin(async move {
                let reaped = reaped.lock().unwrap();
                if !*reaped {
                    // SAFETY: signalling the process group of an unreaped leader.
                    unsafe { libc::kill(-pid, libc::SIGKILL) };
                }
            })
        });
        Ok(Process::new(OwnedFd::from(output), exit, killer, None)?)
    }
}

/// Reaps the child once it exits, under the lock a kill takes, so a kill never races the pid
/// being freed. A pidfd says when it has exited, with no thread per child.
#[cfg(target_os = "linux")]
fn reap_unless_killing(pid: libc::pid_t, reaped: Arc<Mutex<bool>>) -> Result<oneshot::Receiver<ExitStatus>> {
    use std::os::fd::{AsRawFd, FromRawFd};
    // SAFETY: pidfd_open on our own unreaped child.
    let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
    if pidfd < 0 {
        return Err(io::Error::last_os_error().into());
    }
    // SAFETY: pidfd_open returned a descriptor we now own.
    let pidfd = tokio::io::unix::AsyncFd::new(unsafe { OwnedFd::from_raw_fd(pidfd) })?;
    let (exited, exit) = oneshot::channel();
    tokio::spawn(async move {
        let _ = pidfd.readable().await;
        let status = {
            let mut reaped = reaped.lock().unwrap();
            let status = erissandbox::wait_exited(pidfd.as_raw_fd());
            *reaped = true;
            status
        };
        let _ = exited.send(status);
    });
    Ok(exit)
}

/// Reaps the child once it exits, under the lock a kill takes, so a kill never races the pid
/// being freed. Without pidfds, a thread waits for the exit while leaving the child waitable,
/// then reaps it under the lock.
#[cfg(not(target_os = "linux"))]
fn reap_unless_killing(pid: libc::pid_t, reaped: Arc<Mutex<bool>>) -> Result<oneshot::Receiver<ExitStatus>> {
    let (exited, exit) = oneshot::channel();
    std::thread::Builder::new().name(format!("reap-{pid}")).spawn(move || {
        let interrupted =
            |result: libc::c_int| result < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted;
        // SAFETY: waiting on our own unreaped child into a zeroed siginfo; WNOWAIT leaves it
        // waitable, so its pid stays ours until the reap below.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        while interrupted(unsafe {
            libc::waitid(libc::P_PID, pid as libc::id_t, &mut info, libc::WEXITED | libc::WNOWAIT)
        }) {}
        let status = {
            let mut reaped = reaped.lock().unwrap();
            let mut status = 0;
            let mut result;
            loop {
                // SAFETY: reaping our own exited child.
                result = unsafe { libc::waitpid(pid, &mut status, 0) };
                if !interrupted(result) {
                    break;
                }
            }
            *reaped = true;
            match result {
                ..0 => erissandbox::KILLED,
                _ if libc::WIFEXITED(status) => ExitStatus { code: Some(libc::WEXITSTATUS(status)), signal: None },
                _ => ExitStatus { code: None, signal: Some(libc::WTERMSIG(status)) },
            }
        };
        let _ = exited.send(status);
    })?;
    Ok(exit)
}

impl Machine for Direct {
    fn cwd(&self) -> &str {
        &self.spec.cwd
    }

    fn home(&self) -> &str {
        &self.home
    }

    fn spawn<'a>(&'a self, argv: &'a [String], cwd: Option<&'a str>) -> BoxFuture<'a, Result<Process>> {
        Box::pin(self.start(argv, cwd.unwrap_or(&self.spec.cwd)))
    }

    fn open<'a>(&'a self, path: &'a str, mode: OpenMode) -> BoxFuture<'a, io::Result<OwnedFd>> {
        Box::pin(async move { open_path(path, mode).map(OwnedFd::from) })
    }
}
