//! Process entry. [`bootstrap`] must run first in `main`, before any thread exists.
//!
//! On the host it claims a delegated cgroup subtree, then forks. The child becomes the harness
//! inside a new user and mount namespace: the invoking user is root there, and the user's
//! subordinate id range backs ids 1..=65536, so agent images keep their file ownership. The
//! parent stays behind only to forward signals, clean up the cgroups and exit with the
//! harness's status. The same binary re-executed with [`INIT_ARG`] becomes a sandbox init.

use crate::cgroup::Cgroups;
use anyhow::{Context, Result, bail};
use std::fs;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

pub(crate) const INIT_ARG: &str = "__erisharness-sandbox-init";

/// Ids available inside the harness namespace: root plus one subordinate range.
pub(crate) const ID_COUNT: u32 = 65537;

/// The harness's view of its environment, produced by [`bootstrap`].
#[derive(Clone)]
pub struct Host {
    pub(crate) cgroups: Arc<Cgroups>,
    /// This executable, opened before any sandbox exists so inits can be exec'd from it.
    pub(crate) exe: Arc<OwnedFd>,
}

pub fn bootstrap() -> Result<Host> {
    if std::env::args_os().nth(1).is_some_and(|arg| arg == INIT_ARG) {
        crate::sandbox::init::main();
    }
    let cgroups = Cgroups::claim().context("claiming a delegated cgroup subtree")?;
    let ids = SubordinateIds::current()?;
    enter_namespace(&ids, &cgroups)?;
    // Each live sandbox holds a socket and a pidfd; thousands of them outgrow the usual 1024.
    let (_, hard) = nix::sys::resource::getrlimit(nix::sys::resource::Resource::RLIMIT_NOFILE)?;
    nix::sys::resource::setrlimit(nix::sys::resource::Resource::RLIMIT_NOFILE, hard, hard)?;
    let exe = fs::File::open("/proc/self/exe").context("opening own executable")?;
    // Keep the descriptor clear of the low numbers a sandbox init receives.
    // SAFETY: F_DUPFD_CLOEXEC returns a new descriptor we own.
    let exe = unsafe { OwnedFd::from_raw_fd(libc::fcntl(exe.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 10)) };
    Ok(Host { cgroups: Arc::new(cgroups), exe: Arc::new(exe) })
}

struct SubordinateIds {
    uid: u32,
    gid: u32,
    subuid: u32,
    subgid: u32,
}

impl SubordinateIds {
    fn current() -> Result<Self> {
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        let user = nix::unistd::User::from_uid(uid.into())?.map(|u| u.name).unwrap_or_default();
        let find = |file: &str| -> Result<u32> {
            let text = fs::read_to_string(file).with_context(|| format!("reading {file}"))?;
            text.lines()
                .filter_map(|line| {
                    let mut parts = line.split(':');
                    let (owner, start, count) = (parts.next()?, parts.next()?, parts.next()?);
                    let owned = owner == user || owner == uid.to_string();
                    (owned && count.parse::<u32>().ok()? >= ID_COUNT - 1).then(|| start.parse().ok())?
                })
                .next()
                .with_context(|| format!("{file} has no range of {} ids for {user}", ID_COUNT - 1))
        };
        Ok(Self { uid, gid, subuid: find("/etc/subuid")?, subgid: find("/etc/subgid")? })
    }
}

static CHILD: AtomicI32 = AtomicI32::new(0);

extern "C" fn forward(signal: libc::c_int) {
    let child = CHILD.load(Ordering::Relaxed);
    if child > 0 {
        // SAFETY: kill is async-signal-safe.
        unsafe { libc::kill(child, signal) };
    }
}

/// Returns in the child, inside the new namespaces. The parent never returns.
fn enter_namespace(ids: &SubordinateIds, cgroups: &Cgroups) -> Result<()> {
    let (ready_read, ready_write) = nix::unistd::pipe()?;
    let (go_read, go_write) = nix::unistd::pipe()?;
    let parent = std::process::id() as libc::pid_t;
    // SAFETY: bootstrap runs before any other thread exists.
    match unsafe { nix::unistd::fork() }? {
        nix::unistd::ForkResult::Child => {
            drop((ready_read, go_write));
            // SAFETY: plain syscalls in a single-threaded child.
            unsafe {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                if libc::getppid() != parent {
                    libc::_exit(1);
                }
                if libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNS) != 0 {
                    bail!("unshare: {}", std::io::Error::last_os_error());
                }
            }
            nix::unistd::write(&ready_write, b"u")?;
            let mut byte = [0u8];
            if nix::unistd::read(&go_read, &mut byte)? != 1 {
                bail!("id mapping failed");
            }
            nix::mount::mount(
                None::<&str>,
                "/",
                None::<&str>,
                nix::mount::MsFlags::MS_REC | nix::mount::MsFlags::MS_PRIVATE,
                None::<&str>,
            )?;
            Ok(())
        }
        nix::unistd::ForkResult::Parent { child } => {
            drop((ready_write, go_read));
            CHILD.store(child.as_raw(), Ordering::Relaxed);
            for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT] {
                // SAFETY: installing an async-signal-safe forwarding handler.
                unsafe { libc::signal(signal, forward as *const () as libc::sighandler_t) };
            }
            let mut byte = [0u8];
            let mapped = nix::unistd::read(&ready_read, &mut byte).map(|n| n == 1).unwrap_or(false)
                && map_ids(child.as_raw(), ids).is_ok();
            if mapped {
                let _ = nix::unistd::write(&go_write, b"g");
            }
            drop(go_write);
            let status = loop {
                match nix::sys::wait::waitpid(child, None) {
                    Err(nix::errno::Errno::EINTR) => continue,
                    other => break other,
                }
            };
            cgroups.release();
            let code = match status {
                Ok(nix::sys::wait::WaitStatus::Exited(_, code)) => code,
                Ok(nix::sys::wait::WaitStatus::Signaled(_, signal, _)) => 128 + signal as i32,
                _ => 1,
            };
            std::process::exit(if mapped { code } else { 1 })
        }
    }
}

fn map_ids(pid: libc::pid_t, ids: &SubordinateIds) -> Result<()> {
    let range = (ID_COUNT - 1).to_string();
    for (tool, own, sub) in [("newuidmap", ids.uid, ids.subuid), ("newgidmap", ids.gid, ids.subgid)] {
        let status = Command::new(tool)
            .args([
                pid.to_string(),
                "0".into(),
                own.to_string(),
                "1".into(),
                "1".into(),
                sub.to_string(),
                range.clone(),
            ])
            .status()
            .with_context(|| format!("running {tool}"))?;
        if !status.success() {
            bail!("{tool} failed: {status}");
        }
    }
    Ok(())
}
