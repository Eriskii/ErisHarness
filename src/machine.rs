//! Where an agent's commands and file operations happen. An ErisSandbox [`Sandbox`] isolates
//! them; [`Direct`] runs them on this machine as this user, in a working directory. Tools see
//! only the [`Machine`] trait, so both behave identically to the model.

pub use erissandbox::{DRAIN_GRACE, ExitStatus, Killer, OpenMode, Output, Process, SandboxSpec};

use anyhow::{Result, bail};
use erissandbox::{Host, OutsideCommand, Sandbox, open_path, wait_exited};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::io::unix::AsyncFd;
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
    Sandbox(SandboxSpec),
    Direct(DirectSpec),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DirectSpec {
    pub cwd: String,
    /// The complete environment for commands, or `None` to inherit the harness's.
    pub env: Option<Vec<(String, String)>>,
}

/// How [`Machine::open`] opens a path.
/// Runs `argv` to completion and collects its output.
pub async fn run(machine: &dyn Machine, argv: &[String]) -> Result<Output> {
    let mut bytes = Vec::new();
    let status = machine.spawn(argv, None).await?.drain(|chunk| bytes.extend_from_slice(chunk)).await;
    Ok(Output { bytes, status })
}

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

/// Reaps under the lock a killer takes, so a kill never races a pid being freed.
async fn reap_unless_killing(pidfd: &AsyncFd<OwnedFd>, reaped: &Mutex<bool>) -> ExitStatus {
    let _ = pidfd.readable().await;
    let mut reaped = reaped.lock().unwrap();
    let status = wait_exited(pidfd.as_raw_fd());
    *reaped = true;
    status
}

/// Runs commands on this machine as this user. In a program that bootstrapped ErisSandbox,
/// commands start outside its namespace through `host`, so they see the machine exactly as
/// the user does.
pub struct Direct {
    spec: DirectSpec,
    home: String,
    host: Option<Host>,
}

impl Direct {
    pub fn new(spec: DirectSpec, host: Option<Host>) -> Self {
        let from_spec = spec.env.as_ref().and_then(|env| env.iter().find(|(k, _)| k == "HOME").map(|(_, v)| v.clone()));
        let home = from_spec.or_else(|| std::env::var("HOME").ok()).unwrap_or_else(|| "/".into());
        Self { spec, home, host }
    }

    async fn start(&self, argv: &[String], cwd: &str) -> Result<Process> {
        if argv.is_empty() {
            bail!("empty command");
        }
        if !Path::new(cwd).is_dir() {
            bail!("Working directory does not exist: {cwd}");
        }
        match &self.host {
            Some(host) => {
                let env = self.spec.env.clone().unwrap_or_else(|| std::env::vars().collect());
                host.spawn_outside(OutsideCommand { argv: argv.to_vec(), cwd: cwd.into(), env, terminal: None }).await
            }
            None => self.start_here(argv, cwd),
        }
    }

    fn start_here(&self, argv: &[String], cwd: &str) -> Result<Process> {
        let Some((program, args)) = argv.split_first() else { bail!("empty command") };
        let (output, input) = nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC)?;
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
        // SAFETY: pidfd_open on our own unreaped child.
        let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        if pidfd < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: pidfd_open returned a descriptor we now own.
        let pidfd = AsyncFd::new(unsafe { OwnedFd::from_raw_fd(pidfd) })?;
        let (exited, exit) = oneshot::channel();
        // Once the leader is reaped its pid can be reused; the group is never signalled after.
        let reaped = Arc::new(Mutex::new(false));
        let reaper = reaped.clone();
        tokio::spawn(async move {
            let status = reap_unless_killing(&pidfd, &reaper).await;
            let _ = exited.send(status);
        });
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
        Ok(Process::new(output, exit, killer, None)?)
    }
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
