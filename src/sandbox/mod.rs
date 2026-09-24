//! Agent sandboxes. A [`Sandbox`] is mostly directories on disk: an overlay upper layer for
//! the image and one per [`Layer`]. It becomes live only while something runs in it: the
//! first request starts an init process in fresh namespaces, and the sandbox hibernates again
//! once it has been idle for [`Sandboxes::idle_grace`] with no processes left.

pub(crate) mod init;
mod launch;
mod proto;
mod seccomp;

pub use crate::cgroup::Limits;
pub use proto::OpenMode;

use crate::Host;
use anyhow::{Context, Result, bail};
use proto::{BindMount, LayerMount, Reply, Request, Setup};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::io::unix::AsyncFd;
use tokio::sync::oneshot;

/// What an agent's machine looks like.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxSpec {
    /// Read-only image root, shared by every sandbox that uses it.
    pub rootfs: PathBuf,
    /// Copy-on-write directories mounted over the image, such as a project at `/workspace`.
    pub layers: Vec<Layer>,
    /// Live host directories, such as `.ae` records other agents write.
    pub binds: Vec<Bind>,
    pub limits: Limits,
    pub hostname: String,
    /// Working directory for commands.
    pub cwd: String,
    pub env: Vec<(String, String)>,
}

impl SandboxSpec {
    pub fn default_env() -> Vec<(String, String)> {
        [
            ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
            ("HOME", "/root"),
            ("LANG", "C.UTF-8"),
            ("TERM", "dumb"),
        ]
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .to_vec()
    }

    pub fn home(&self) -> &str {
        self.env.iter().find(|(k, _)| k == "HOME").map_or("/root", |(_, v)| v)
    }
}

/// `lower` must not change while a sandbox using it is live; overlayfs requires it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Layer {
    pub lower: PathBuf,
    pub target: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Bind {
    pub source: PathBuf,
    pub target: String,
    pub writable: bool,
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

/// Owns every sandbox under one state directory.
pub struct Sandboxes {
    host: Host,
    dir: PathBuf,
    idle_grace: Duration,
    open: Mutex<HashMap<String, Weak<Sandbox>>>,
    live: Registry,
}

/// Live sandboxes, held strongly so a running init outlives every other handle.
type Registry = Arc<Mutex<HashMap<String, Arc<Sandbox>>>>;

impl Sandboxes {
    pub fn new(host: &Host, dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            host: host.clone(),
            dir,
            idle_grace: Duration::from_secs(10),
            open: Mutex::default(),
            live: Registry::default(),
        })
    }

    /// How long a sandbox with no running commands stays live before hibernating.
    pub fn idle_grace(mut self, grace: Duration) -> Self {
        self.idle_grace = grace;
        self
    }

    /// The sandbox with this id. Its filesystem persists across calls and restarts. A live
    /// sandbox keeps its spec until it hibernates.
    pub fn sandbox(&self, id: &str, spec: SandboxSpec) -> Result<Arc<Sandbox>> {
        if !valid_id(id) {
            bail!("sandbox ids are 1-64 of [A-Za-z0-9_-]: {id:?}");
        }
        let mut open = self.open.lock().unwrap();
        if let Some(existing) = open.get(id).and_then(Weak::upgrade) {
            if existing.spec == spec {
                return Ok(existing);
            }
            if existing.is_live() {
                bail!("sandbox {id} is live with a different spec");
            }
        }
        let sandbox = Arc::new(Sandbox {
            id: id.to_owned(),
            dir: self.dir.join(id),
            cgroup: self.host.cgroups.agents().join(id),
            host: self.host.clone(),
            idle_grace: self.idle_grace,
            spec,
            live: tokio::sync::Mutex::new(None),
            live_flag: AtomicBool::new(false),
            registry: self.live.clone(),
        });
        open.insert(id.to_owned(), Arc::downgrade(&sandbox));
        Ok(sandbox)
    }

    /// Stops every live sandbox, killing whatever runs in them.
    pub async fn shutdown_all(&self) {
        let live: Vec<Arc<Sandbox>> = self.live.lock().unwrap().values().cloned().collect();
        futures_util::future::join_all(live.iter().map(|s| s.shutdown(true))).await;
    }

    /// Sandboxes with a running init.
    pub fn live_count(&self) -> usize {
        self.live.lock().unwrap().len()
    }

    /// Deletes a sandbox's filesystem. It must not be live.
    pub fn remove(&self, id: &str) -> Result<()> {
        if !valid_id(id) {
            bail!("invalid sandbox id {id:?}");
        }
        self.open.lock().unwrap().remove(id);
        match std::fs::remove_dir_all(self.dir.join(id)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }
}

fn valid_id(id: &str) -> bool {
    (1..=64).contains(&id.len()) && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

pub struct Sandbox {
    id: String,
    dir: PathBuf,
    cgroup: PathBuf,
    host: Host,
    idle_grace: Duration,
    spec: SandboxSpec,
    live: tokio::sync::Mutex<Option<Arc<Live>>>,
    live_flag: AtomicBool,
    registry: Registry,
}

impl Sandbox {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn spec(&self) -> &SandboxSpec {
        &self.spec
    }

    /// Where writes to `spec.layers[index]` accumulate.
    pub fn layer_upper(&self, index: usize) -> PathBuf {
        self.dir.join("layers").join(index.to_string()).join("upper")
    }

    pub fn is_live(&self) -> bool {
        self.live_flag.load(Ordering::Acquire)
    }

    /// Starts `argv` in the working directory (the spec's unless given). Its combined output
    /// must be read, or the command blocks once the pipe fills.
    pub async fn spawn(self: &Arc<Self>, argv: &[String], cwd: Option<&str>) -> Result<Process> {
        let (live, activity) = self.live().await?;
        let (output, input) = nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC)?;
        let id = live.next_id();
        let exit = live.expect_exit(id);
        let request = Request::Spawn {
            id,
            argv: argv.to_vec(),
            cwd: cwd.unwrap_or(&self.spec.cwd).to_owned(),
            env: self.spec.env.clone(),
        };
        match live.call(id, &request, &[input.as_raw_fd()]).await? {
            (Reply::Spawned { .. }, _) => {}
            (Reply::SpawnFailed { error, .. }, _) => bail!(error),
            (other, _) => bail!("unexpected reply {other:?}"),
        }
        drop(input);
        launch::set_nonblocking(&output)?;
        let output = tokio::net::unix::pipe::Receiver::from_owned_fd(output)?;
        Ok(Process { id, live, output: Some(output), exit, _activity: activity })
    }

    /// Runs `argv` to completion and collects its output.
    pub async fn run(self: &Arc<Self>, argv: &[String]) -> Result<Output> {
        let mut bytes = Vec::new();
        let status = self.spawn(argv, None).await?.drain(|chunk| bytes.extend_from_slice(chunk)).await;
        Ok(Output { bytes, status })
    }

    /// Opens an absolute path as the sandbox sees it, with the sandbox's permissions.
    pub async fn open(self: &Arc<Self>, path: &str, mode: OpenMode) -> io::Result<OwnedFd> {
        let (live, _activity) = self.live().await.map_err(io::Error::other)?;
        let id = live.next_id();
        let request = Request::Open { id, path: path.to_owned(), mode };
        match live.call(id, &request, &[]).await.map_err(io::Error::other)? {
            (Reply::Opened { .. }, mut fds) if !fds.is_empty() => Ok(fds.remove(0)),
            (Reply::OpenFailed { errno, .. }, _) => Err(io::Error::from_raw_os_error(errno)),
            (other, _) => Err(io::Error::other(format!("unexpected reply {other:?}"))),
        }
    }

    /// Stops the sandbox. Without `force`, it stays up while any process runs in it or a
    /// request is in flight, and this returns false. The filesystem is kept either way.
    pub async fn shutdown(&self, force: bool) -> bool {
        let mut guard = self.live.lock().await;
        let Some(live) = guard.clone() else { return true };
        if !force && live.active.load(Ordering::Acquire) > 0 {
            return false;
        }
        if !live.dead() {
            let id = live.next_id();
            if let Ok((Reply::ShutdownRefused { .. }, _)) = live.call(id, &Request::Shutdown { id, force }, &[]).await {
                return false;
            }
        }
        live.exited().await;
        self.tear_down(&mut guard);
        true
    }

    /// Forgets an exited init. Runs under the `live` lock, by whichever of shutdown, restart
    /// or the reply reader gets there first, so it can never undo a newer init.
    fn tear_down(&self, guard: &mut Option<Arc<Live>>) {
        *guard = None;
        self.live_flag.store(false, Ordering::Release);
        let mut registry = self.registry.lock().unwrap();
        if registry.get(&self.id).is_some_and(|s| std::ptr::eq(Arc::as_ptr(s), self)) {
            registry.remove(&self.id);
        }
        drop(registry);
        let _ = std::fs::remove_dir(&self.cgroup);
    }

    /// The running init, started if needed. The activity is taken under the lock so an idle
    /// shutdown cannot slip in between.
    async fn live(self: &Arc<Self>) -> Result<(Arc<Live>, Activity)> {
        let mut guard = self.live.lock().await;
        if let Some(live) = guard.as_ref().filter(|live| !live.dead()) {
            return Ok((live.clone(), live.activity()));
        }
        if let Some(stale) = guard.clone() {
            stale.exited().await;
            self.tear_down(&mut guard);
        }
        let live = self.start().await.with_context(|| format!("starting sandbox {}", self.id))?;
        *guard = Some(live.clone());
        self.live_flag.store(true, Ordering::Release);
        self.registry.lock().unwrap().insert(self.id.clone(), self.clone());
        Ok((live.clone(), live.activity()))
    }

    async fn start(self: &Arc<Self>) -> Result<Arc<Live>> {
        let setup = self.prepare()?;
        let cgroup_dir = std::fs::File::open(&self.cgroup)?;
        let (socket, pidfd) = launch::init(self.host.exe.as_fd(), cgroup_dir.as_fd())?;
        let live = Live::new(socket, pidfd, Arc::downgrade(self))?;
        live.send(&Request::Setup(setup), &[]).await?;
        let ready = live.setup.lock().unwrap().take().expect("setup receiver");
        match ready.await {
            Ok(Reply::Ready) => Ok(live),
            Ok(Reply::SetupFailed { error }) => bail!(error),
            other => bail!("sandbox init failed: {other:?}"),
        }
    }

    fn prepare(&self) -> Result<Setup> {
        let make = |path: PathBuf| -> Result<PathBuf> {
            std::fs::create_dir_all(&path).with_context(|| format!("creating {}", path.display()))?;
            Ok(path)
        };
        std::fs::create_dir_all(&self.cgroup).context("creating the sandbox cgroup")?;
        self.spec.limits.apply(&self.cgroup).context("applying limits")?;
        let layers = self
            .spec
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let dir = self.dir.join("layers").join(index.to_string());
                Ok(LayerMount {
                    lower: layer.lower.clone(),
                    upper: make(dir.join("upper"))?,
                    work: make(dir.join("work"))?,
                    target: layer.target.clone(),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Setup {
            rootfs: self.spec.rootfs.clone(),
            root_upper: make(self.dir.join("root/upper"))?,
            root_work: make(self.dir.join("root/work"))?,
            mount_point: make(self.dir.join("mnt"))?,
            layers,
            binds: self
                .spec
                .binds
                .iter()
                .map(|b| BindMount { source: b.source.clone(), target: b.target.clone(), writable: b.writable })
                .collect(),
            hostname: if self.spec.hostname.is_empty() { self.id.clone() } else { self.spec.hostname.clone() },
        })
    }

    fn became_idle(self: &Arc<Self>, live: &Arc<Live>) {
        let sandbox = Arc::downgrade(self);
        let grace = self.idle_grace;
        let runtime = live.runtime.clone();
        let live = Arc::downgrade(live);
        runtime.spawn(async move {
            tokio::time::sleep(grace).await;
            if let (Some(sandbox), Some(_)) = (sandbox.upgrade(), live.upgrade()) {
                sandbox.shutdown(false).await;
            }
        });
    }
}

/// A running command. Dropping it leaves the command running.
pub struct Process {
    id: u64,
    live: Arc<Live>,
    output: Option<tokio::net::unix::pipe::Receiver>,
    exit: oneshot::Receiver<ExitStatus>,
    _activity: Activity,
}

impl Process {
    /// Combined stdout and stderr. Reaches end of file once every process holding it exits.
    pub fn take_output(&mut self) -> tokio::net::unix::pipe::Receiver {
        self.output.take().expect("output already taken")
    }

    /// Kills the command's process group.
    pub async fn kill(&self) {
        self.killer().kill().await;
    }

    /// A handle that kills this command, usable while [`Process::wait`] owns the process.
    pub fn killer(&self) -> Killer {
        Killer { id: self.id, live: self.live.clone() }
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
        let mut status = None;
        let mut deadline = None;
        loop {
            let read = async {
                match deadline {
                    Some(at) => tokio::time::timeout_at(at, output.read(&mut buffer)).await.ok(),
                    None => Some(output.read(&mut buffer).await),
                }
            };
            tokio::select! {
                exit = &mut self.exit, if status.is_none() => {
                    status = Some(exit.unwrap_or(KILLED));
                    deadline = Some(tokio::time::Instant::now() + DRAIN_GRACE);
                }
                result = read => match result {
                    Some(Ok(n)) if n > 0 => sink(&buffer[..n]),
                    Some(Ok(_)) | Some(Err(_)) if status.is_none() => {
                        return (&mut self.exit).await.unwrap_or(KILLED);
                    }
                    _ => return status.unwrap_or(KILLED),
                },
            }
        }
    }
}

#[derive(Clone)]
pub struct Killer {
    id: u64,
    live: Arc<Live>,
}

impl Killer {
    pub async fn kill(&self) {
        let _ = self.live.send(&Request::Kill { id: self.id }, &[]).await;
    }
}

const KILLED: ExitStatus = ExitStatus { code: None, signal: Some(libc::SIGKILL) };

/// How long output is still collected after a command exits.
pub const DRAIN_GRACE: Duration = Duration::from_millis(100);

type Pending = oneshot::Sender<(Reply, Vec<OwnedFd>)>;

/// The harness end of a running init.
struct Live {
    socket: AsyncFd<OwnedFd>,
    pidfd: AsyncFd<OwnedFd>,
    ids: AtomicU64,
    active: AtomicUsize,
    dead: AtomicBool,
    setup: Mutex<Option<oneshot::Receiver<Reply>>>,
    pending: Mutex<HashMap<u64, Pending>>,
    exits: Mutex<HashMap<u64, oneshot::Sender<ExitStatus>>>,
    sandbox: Weak<Sandbox>,
    /// Handles may be dropped outside the runtime; idle timers still need one.
    runtime: tokio::runtime::Handle,
}

/// Counts toward keeping a sandbox live; the last one dropped starts the idle timer.
struct Activity(Arc<Live>);

impl Drop for Activity {
    fn drop(&mut self) {
        if self.0.active.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(sandbox) = self.0.sandbox.upgrade()
        {
            sandbox.became_idle(&self.0);
        }
    }
}

impl Live {
    fn new(socket: OwnedFd, pidfd: OwnedFd, sandbox: Weak<Sandbox>) -> Result<Arc<Self>> {
        let (setup_tx, setup_rx) = oneshot::channel();
        let live = Arc::new(Self {
            socket: AsyncFd::new(socket)?,
            pidfd: AsyncFd::new(pidfd)?,
            ids: AtomicU64::new(1),
            active: AtomicUsize::new(0),
            dead: AtomicBool::new(false),
            setup: Mutex::new(Some(setup_rx)),
            pending: Mutex::default(),
            exits: Mutex::default(),
            sandbox,
            runtime: tokio::runtime::Handle::current(),
        });
        tokio::spawn(Self::read_replies(live.clone(), setup_tx));
        Ok(live)
    }

    fn dead(&self) -> bool {
        self.dead.load(Ordering::Acquire)
    }

    fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::Relaxed)
    }

    fn activity(self: &Arc<Self>) -> Activity {
        self.active.fetch_add(1, Ordering::AcqRel);
        Activity(self.clone())
    }

    fn expect_exit(&self, id: u64) -> oneshot::Receiver<ExitStatus> {
        let (tx, rx) = oneshot::channel();
        self.exits.lock().unwrap().insert(id, tx);
        rx
    }

    async fn send(&self, request: &Request, fds: &[i32]) -> Result<()> {
        loop {
            let mut ready = self.socket.writable().await?;
            match ready.try_io(|socket| proto::send(socket.get_ref().as_fd(), request, fds)) {
                Ok(result) => return Ok(result?),
                Err(_would_block) => continue,
            }
        }
    }

    async fn call(&self, id: u64, request: &Request, fds: &[i32]) -> Result<(Reply, Vec<OwnedFd>)> {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if self.dead() {
            bail!("sandbox stopped");
        }
        self.send(request, fds).await?;
        rx.await.context("sandbox stopped")
    }

    async fn read_replies(self: Arc<Self>, setup: oneshot::Sender<Reply>) {
        let mut setup = Some(setup);
        loop {
            let Ok(mut ready) = self.socket.readable().await else { break };
            let received = match ready.try_io(|socket| proto::recv::<Reply>(socket.get_ref().as_fd())) {
                Ok(Ok(Some(message))) => message,
                Ok(_) => break,
                Err(_would_block) => continue,
            };
            match received {
                (Reply::Exited { id, code, signal }, _) => {
                    if let Some(tx) = self.exits.lock().unwrap().remove(&id) {
                        let _ = tx.send(ExitStatus { code, signal });
                    }
                }
                (reply @ (Reply::Ready | Reply::SetupFailed { .. }), _) => {
                    if let Some(tx) = setup.take() {
                        let _ = tx.send(reply);
                    }
                }
                (reply, fds) => {
                    let waiter = reply.id().and_then(|id| self.pending.lock().unwrap().remove(&id));
                    if let Some(tx) = waiter {
                        let _ = tx.send((reply, fds));
                    }
                }
            }
        }
        self.dead.store(true, Ordering::Release);
        self.pending.lock().unwrap().clear();
        self.exits.lock().unwrap().clear();
        self.exited().await;
        if let Some(sandbox) = self.sandbox.upgrade() {
            let mut guard = sandbox.live.lock().await;
            if guard.as_ref().is_some_and(|current| std::ptr::eq(Arc::as_ptr(current), Arc::as_ptr(&self))) {
                sandbox.tear_down(&mut guard);
            }
        }
    }

    /// Waits for init to exit and reaps it. Safe to repeat: a reaped pidfd stays readable.
    async fn exited(&self) {
        let _ = self.pidfd.readable().await;
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: reaping the init this pidfd refers to; it is readable once init has exited.
        unsafe { libc::waitid(libc::P_PIDFD, self.pidfd.as_raw_fd() as libc::id_t, &mut info, libc::WEXITED) };
    }
}
