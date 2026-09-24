//! Where an agent's commands and file operations happen. A [`Sandbox`](crate::sandbox::Sandbox)
//! isolates them; [`Direct`] runs them on this machine as this user, in a working directory.
//! Tools see only the [`Machine`] trait, so both behave identically to the model.

use crate::sandbox::SandboxSpec;
use anyhow::{Result, bail};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenMode {
    Read,
    /// Create or truncate, optionally creating missing parent directories.
    Write {
        create_parents: bool,
    },
    /// Read and write an existing file without truncating it.
    Update,
}

/// Opens a path in the calling process's view of the filesystem. A sandbox init calls this
/// from inside the sandbox; [`Direct`] calls it on the host.
pub fn open_path(path: &str, mode: OpenMode) -> io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    // Non-blocking so opening a FIFO cannot hang; the flag is cleared before returning.
    options.custom_flags(libc::O_NOCTTY | libc::O_NONBLOCK);
    match mode {
        OpenMode::Read => {
            options.read(true);
        }
        OpenMode::Write { create_parents } => {
            if create_parents && let Some(parent) = Path::new(path).parent() {
                fs::create_dir_all(parent)?;
            }
            options.write(true).create(true).truncate(true).mode(0o644);
        }
        OpenMode::Update => {
            options.read(true).write(true);
        }
    }
    let file = options.open(path)?;
    // SAFETY: F_GETFL/F_SETFL on a descriptor we own.
    unsafe {
        let flags = libc::fcntl(file.as_raw_fd(), libc::F_GETFL);
        libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags & !libc::O_NONBLOCK);
    }
    Ok(file)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExitStatus {
    pub code: Option<i32>,
    pub signal: Option<i32>,
}

impl ExitStatus {
    /// The shell convention: the exit code, or 128 plus the terminating signal.
    pub fn code(&self) -> i32 {
        self.code.unwrap_or_else(|| 128 + self.signal.unwrap_or(0))
    }

    pub fn success(&self) -> bool {
        self.code == Some(0)
    }
}

pub(crate) const KILLED: ExitStatus = ExitStatus { code: None, signal: Some(libc::SIGKILL) };

/// How long output is still collected after a command exits.
pub const DRAIN_GRACE: Duration = Duration::from_millis(100);

/// Combined stdout and stderr of a finished command.
#[derive(Clone, Debug)]
pub struct Output {
    pub bytes: Vec<u8>,
    pub status: ExitStatus,
}

impl Output {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

type KillFn = dyn Fn() -> BoxFuture<'static, ()> + Send + Sync;

/// Kills a command's whole process group.
#[derive(Clone)]
pub struct Killer(Arc<KillFn>);

impl Killer {
    pub(crate) fn new(kill: impl Fn() -> BoxFuture<'static, ()> + Send + Sync + 'static) -> Self {
        Self(Arc::new(kill))
    }

    pub async fn kill(&self) {
        (self.0)().await
    }
}

/// A running command. Dropping it leaves the command running.
pub struct Process {
    output: Option<tokio::net::unix::pipe::Receiver>,
    exit: oneshot::Receiver<ExitStatus>,
    killer: Killer,
    /// Whatever must live as long as the command, such as a sandbox's activity count.
    _keep: Option<Box<dyn Send + Sync>>,
}

impl Process {
    pub(crate) fn new(
        output: OwnedFd,
        exit: oneshot::Receiver<ExitStatus>,
        killer: Killer,
        keep: Option<Box<dyn Send + Sync>>,
    ) -> io::Result<Self> {
        set_nonblocking(&output)?;
        let output = tokio::net::unix::pipe::Receiver::from_owned_fd(output)?;
        Ok(Self { output: Some(output), exit, killer, _keep: keep })
    }

    /// Combined stdout and stderr. Reaches end of file once every process holding it exits.
    pub fn take_output(&mut self) -> tokio::net::unix::pipe::Receiver {
        self.output.take().expect("output already taken")
    }

    pub async fn kill(&self) {
        self.killer.kill().await;
    }

    /// A handle that kills this command, usable while [`Process::wait`] owns the process.
    pub fn killer(&self) -> Killer {
        self.killer.clone()
    }

    pub async fn wait(self) -> ExitStatus {
        self.exit.await.unwrap_or(KILLED)
    }

    /// Feeds output to `sink` until the command exits, then for at most [`DRAIN_GRACE`]
    /// longer. Background processes it started may keep the pipe open indefinitely; they
    /// are not waited for.
    pub async fn drain(mut self, mut sink: impl FnMut(&[u8])) -> ExitStatus {
        let mut output = self.take_output();
        let mut buffer = vec![0u8; 16 * 1024];
        let mut open = true;
        let status = loop {
            tokio::select! {
                exit = &mut self.exit => break exit.unwrap_or(KILLED),
                read = output.read(&mut buffer), if open => match read {
                    Ok(n) if n > 0 => sink(&buffer[..n]),
                    _ => open = false,
                },
            }
        };
        let grace = tokio::time::Instant::now() + DRAIN_GRACE;
        while open {
            match tokio::time::timeout_at(grace, output.read(&mut buffer)).await {
                Ok(Ok(n)) if n > 0 => sink(&buffer[..n]),
                _ => open = false,
            }
        }
        status
    }
}

/// Runs `argv` to completion and collects its output.
pub async fn run(machine: &dyn Machine, argv: &[String]) -> Result<Output> {
    let mut bytes = Vec::new();
    let status = machine.spawn(argv, None).await?.drain(|chunk| bytes.extend_from_slice(chunk)).await;
    Ok(Output { bytes, status })
}

/// Tokio drives these descriptors; a blocking read would stall a runtime worker.
pub(crate) fn set_nonblocking(fd: &impl AsRawFd) -> io::Result<()> {
    // SAFETY: F_GETFL/F_SETFL on a valid descriptor.
    unsafe {
        let flags = libc::fcntl(fd.as_raw_fd(), libc::F_GETFL);
        if flags < 0 || libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Waits for the process behind a pidfd to exit and reaps it. Reaping through the pidfd
/// needs no thread per child. Repeating it is harmless: a reaped pidfd stays readable.
pub(crate) async fn reap(pidfd: &AsyncFd<OwnedFd>) -> ExitStatus {
    let _ = pidfd.readable().await;
    wait_exited(pidfd.as_raw_fd())
}

/// Reaps an exited child by pidfd.
fn wait_exited(pidfd: i32) -> ExitStatus {
    // SAFETY: waitid on a pidfd we own, into a zeroed siginfo.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::waitid(libc::P_PIDFD, pidfd as libc::id_t, &mut info, libc::WEXITED) };
    if result < 0 {
        return KILLED;
    }
    // SAFETY: waitid filled a SIGCHLD siginfo.
    let status = unsafe { info.si_status() };
    if info.si_code == libc::CLD_EXITED {
        ExitStatus { code: Some(status), signal: None }
    } else {
        ExitStatus { code: None, signal: Some(status) }
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

/// Runs commands on this machine as this user.
pub struct Direct {
    spec: DirectSpec,
    home: String,
}

impl Direct {
    pub fn new(spec: DirectSpec) -> Self {
        let from_spec = spec.env.as_ref().and_then(|env| env.iter().find(|(k, _)| k == "HOME").map(|(_, v)| v.clone()));
        let home = from_spec.or_else(|| std::env::var("HOME").ok()).unwrap_or_else(|| "/".into());
        Self { spec, home }
    }

    fn start(&self, argv: &[String], cwd: &str) -> Result<Process> {
        let Some((program, args)) = argv.split_first() else { bail!("empty command") };
        if !Path::new(cwd).is_dir() {
            bail!("Working directory does not exist: {cwd}");
        }
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
        Box::pin(async move { self.start(argv, cwd.unwrap_or(&self.spec.cwd)) })
    }

    fn open<'a>(&'a self, path: &'a str, mode: OpenMode) -> BoxFuture<'a, io::Result<OwnedFd>> {
        Box::pin(async move { open_path(path, mode).map(OwnedFd::from) })
    }
}
