//! PID 1 of an agent sandbox. The harness execs its own binary into fresh user, mount, PID,
//! network, IPC, UTS and cgroup namespaces; this code then builds the root filesystem, drops
//! to Docker's default capabilities under a seccomp filter, and serves requests until the
//! harness closes the socket or asks it to stop. It is single-threaded and blocking.

use super::proto::{self, BindMount, OpenMode, Reply, Request, Setup};
use super::seccomp;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::ffi::{CString, OsStr};
use std::fs;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, symlink};
use std::path::{Path, PathBuf};

/// Descriptor the harness installs for the control socket before exec.
pub const SOCKET_FD: RawFd = 3;

/// CHOWN, DAC_OVERRIDE, FOWNER, FSETID, KILL, SETGID, SETUID, SETPCAP, NET_BIND_SERVICE,
/// NET_RAW, SYS_CHROOT, MKNOD, AUDIT_WRITE and SETFCAP: what Docker grants by default.
pub const KEPT_CAPABILITIES: u64 = 0xa804_25fb;

pub fn main() -> ! {
    // SAFETY: the harness installs the control socket at this descriptor before exec.
    let socket = unsafe { OwnedFd::from_raw_fd(SOCKET_FD) };
    let code = match serve(socket.as_fd()) {
        Ok(()) => 0,
        Err(error) => {
            let _ = proto::send(socket.as_fd(), &Reply::SetupFailed { error: format!("{error:#}") }, &[]);
            1
        }
    };
    std::process::exit(code)
}

fn serve(socket: BorrowedFd) -> Result<()> {
    // The descriptor survived exec for init alone; agent commands must not inherit it.
    check(unsafe { libc::fcntl(SOCKET_FD, libc::F_SETFD, libc::FD_CLOEXEC) })?;
    quiet_stdio()?;
    let Some((Request::Setup(setup), _)) = proto::recv::<Request>(socket)? else {
        bail!("expected setup");
    };
    build_root(&setup).context("building the sandbox root")?;
    lock_down().context("dropping privileges")?;
    proto::send(socket, &Reply::Ready, &[])?;
    Init::new(socket)?.run()
}

/// Init must not hold the harness's terminal or inherited output.
fn quiet_stdio() -> io::Result<()> {
    let null = fs::OpenOptions::new().read(true).write(true).open("/dev/null")?;
    for fd in 0..3 {
        check(unsafe { libc::dup2(null.as_raw_fd(), fd) })?;
    }
    Ok(())
}

fn check(result: libc::c_int) -> io::Result<libc::c_int> {
    if result < 0 { Err(io::Error::last_os_error()) } else { Ok(result) }
}

fn cstr(path: impl AsRef<OsStr>) -> CString {
    CString::new(path.as_ref().as_bytes()).expect("path contains NUL")
}

fn mount(source: &str, target: &Path, fstype: Option<&str>, flags: libc::c_ulong, data: Option<&str>) -> Result<()> {
    let (source_c, target_c) = (cstr(source), cstr(target));
    let fstype = fstype.map(cstr);
    let data = data.map(cstr);
    // SAFETY: every pointer is a valid NUL-terminated string or null.
    let result = unsafe {
        libc::mount(
            source_c.as_ptr(),
            target_c.as_ptr(),
            fstype.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            flags,
            data.as_ref().map_or(std::ptr::null(), |s| s.as_ptr().cast()),
        )
    };
    check(result).with_context(|| format!("mount {source} on {}", target.display()))?;
    Ok(())
}

/// Overlay option values separate entries with `,` and layers with `:`.
fn overlay_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\").replace(',', "\\,").replace(':', "\\:")
}

fn overlay(lower: &Path, upper: &Path, work: &Path, target: &Path) -> Result<()> {
    let options =
        format!("lowerdir={},upperdir={},workdir={}", overlay_path(lower), overlay_path(upper), overlay_path(work));
    mount("overlay", target, Some("overlay"), 0, Some(&options))
}

fn inside(root: &Path, target: &str) -> PathBuf {
    root.join(target.trim_start_matches('/'))
}

fn make_dir(path: &Path) -> Result<()> {
    fs::DirBuilder::new().recursive(true).mode(0o755).create(path).with_context(|| format!("mkdir {}", path.display()))
}

fn build_root(setup: &Setup) -> Result<()> {
    nix::unistd::setgroups(&[]).context("setgroups")?;
    mount("none", Path::new("/"), None, libc::MS_REC | libc::MS_PRIVATE, None)?;
    let root = &setup.mount_point;
    overlay(&setup.rootfs, &setup.root_upper, &setup.root_work, root)?;
    for layer in &setup.layers {
        let target = inside(root, &layer.target);
        make_dir(&target)?;
        overlay(&layer.lower, &layer.upper, &layer.work, &target)?;
    }
    for bind in &setup.binds {
        bind_mount(root, bind)?;
    }
    system_mounts(root)?;
    pivot(root)?;
    nix::unistd::sethostname(&setup.hostname).context("sethostname")?;
    loopback_up().context("bringing up loopback")?;
    Ok(())
}

fn bind_mount(root: &Path, bind: &BindMount) -> Result<()> {
    let target = inside(root, &bind.target);
    if bind.source.is_dir() {
        make_dir(&target)?;
    } else {
        make_dir(target.parent().unwrap_or(root))?;
        fs::OpenOptions::new().create(true).append(true).open(&target)?;
    }
    let source = bind.source.to_string_lossy();
    mount(&source, &target, None, libc::MS_BIND | libc::MS_REC, None)?;
    if !bind.writable {
        set_read_only(&target)?;
    }
    Ok(())
}

/// `mount_setattr` makes a whole bind tree read-only without having to restate the flags the
/// kernel locks on mounts inherited from a more privileged namespace.
fn set_read_only(target: &Path) -> Result<()> {
    #[repr(C)]
    struct MountAttr {
        attr_set: u64,
        attr_clr: u64,
        propagation: u64,
        userns_fd: u64,
    }
    const MOUNT_ATTR_RDONLY: u64 = 0x1;
    let attr = MountAttr { attr_set: MOUNT_ATTR_RDONLY, attr_clr: 0, propagation: 0, userns_fd: 0 };
    let path = cstr(target);
    // SAFETY: valid path and a correctly sized mount_attr.
    let result = unsafe {
        libc::syscall(
            libc::SYS_mount_setattr,
            libc::AT_FDCWD,
            path.as_ptr(),
            libc::AT_RECURSIVE,
            &attr as *const MountAttr,
            std::mem::size_of::<MountAttr>(),
        )
    };
    check(result as libc::c_int).with_context(|| format!("read-only {}", target.display()))?;
    Ok(())
}

fn system_mounts(root: &Path) -> Result<()> {
    let hardened = libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC;
    let proc_dir = root.join("proc");
    make_dir(&proc_dir)?;
    mount("proc", &proc_dir, Some("proc"), hardened, None)?;
    let sys = root.join("sys");
    make_dir(&sys)?;
    mount("sysfs", &sys, Some("sysfs"), hardened | libc::MS_RDONLY, None)?;
    // The agent sees only its own cgroup, read-only, so tools can read their limits.
    mount("cgroup2", &sys.join("fs/cgroup"), Some("cgroup2"), hardened | libc::MS_RDONLY, None)?;

    let dev = root.join("dev");
    make_dir(&dev)?;
    mount("tmpfs", &dev, Some("tmpfs"), libc::MS_NOSUID | libc::MS_NOEXEC, Some("mode=755,size=64k"))?;
    for node in ["null", "zero", "full", "random", "urandom", "tty"] {
        let target = dev.join(node);
        fs::File::create(&target)?;
        mount(&format!("/dev/{node}"), &target, None, libc::MS_BIND, None)?;
    }
    let pts = dev.join("pts");
    make_dir(&pts)?;
    mount(
        "devpts",
        &pts,
        Some("devpts"),
        libc::MS_NOSUID | libc::MS_NOEXEC,
        Some("newinstance,ptmxmode=0666,mode=0620,gid=5"),
    )?;
    let shm = dev.join("shm");
    make_dir(&shm)?;
    mount("tmpfs", &shm, Some("tmpfs"), libc::MS_NOSUID | libc::MS_NODEV, Some("mode=1777"))?;
    for (link, target) in [
        ("ptmx", "pts/ptmx"),
        ("fd", "/proc/self/fd"),
        ("stdin", "/proc/self/fd/0"),
        ("stdout", "/proc/self/fd/1"),
        ("stderr", "/proc/self/fd/2"),
    ] {
        symlink(target, dev.join(link))?;
    }
    Ok(())
}

fn pivot(root: &Path) -> Result<()> {
    std::env::set_current_dir(root)?;
    let dot = cstr(".");
    // SAFETY: pivot_root(".", ".") stacks the old root beneath the new one; it is detached next.
    check(unsafe { libc::syscall(libc::SYS_pivot_root, dot.as_ptr(), dot.as_ptr()) } as libc::c_int)
        .context("pivot_root")?;
    check(unsafe { libc::umount2(dot.as_ptr(), libc::MNT_DETACH) }).context("detaching the old root")?;
    std::env::set_current_dir("/")?;
    Ok(())
}

fn loopback_up() -> io::Result<()> {
    // SAFETY: plain socket and ioctl calls on a zeroed ifreq naming "lo".
    unsafe {
        let socket =
            OwnedFd::from_raw_fd(check(libc::socket(libc::AF_INET, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0))?);
        let mut request: libc::ifreq = std::mem::zeroed();
        for (slot, byte) in request.ifr_name.iter_mut().zip(b"lo\0") {
            *slot = *byte as libc::c_char;
        }
        check(libc::ioctl(socket.as_raw_fd(), libc::SIOCGIFFLAGS, &mut request))?;
        request.ifr_ifru.ifru_flags |= libc::IFF_UP as libc::c_short;
        check(libc::ioctl(socket.as_raw_fd(), libc::SIOCSIFFLAGS, &request))?;
    }
    Ok(())
}

/// After this, init and everything it runs are root in name only: Docker's capability set,
/// no new privileges, a seccomp filter that denies namespace and mount escapes, and a
/// non-dumpable init that agent processes cannot ptrace or inspect.
fn lock_down() -> Result<()> {
    for capability in 0..64 {
        if KEPT_CAPABILITIES & (1 << capability) == 0 {
            // SAFETY: PR_CAPBSET_DROP takes a capability number; unknown numbers return EINVAL.
            unsafe { libc::prctl(libc::PR_CAPBSET_DROP, capability, 0, 0, 0) };
        }
    }
    #[repr(C)]
    struct Header {
        version: u32,
        pid: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Data {
        effective: u32,
        permitted: u32,
        inheritable: u32,
    }
    let low = KEPT_CAPABILITIES as u32;
    let high = (KEPT_CAPABILITIES >> 32) as u32;
    let header = Header { version: 0x2008_0522, pid: 0 };
    let data = [
        Data { effective: low, permitted: low, inheritable: 0 },
        Data { effective: high, permitted: high, inheritable: 0 },
    ];
    // SAFETY: version 3 capset takes a header and two data words.
    check(unsafe { libc::syscall(libc::SYS_capset, &header as *const Header, data.as_ptr()) } as libc::c_int)
        .context("capset")?;
    check(unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) }).context("non-dumpable")?;
    check(unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) }).context("no_new_privs")?;
    seccomp::install().context("seccomp")?;
    Ok(())
}

struct Init<'a> {
    socket: BorrowedFd<'a>,
    signals: OwnedFd,
    commands: HashMap<libc::pid_t, u64>,
}

impl<'a> Init<'a> {
    fn new(socket: BorrowedFd<'a>) -> Result<Self> {
        // SAFETY: SIGCHLD is blocked and read through a signalfd so reaping stays in the loop.
        let signals = unsafe {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGCHLD);
            check(libc::sigprocmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()))?;
            OwnedFd::from_raw_fd(check(libc::signalfd(-1, &set, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK))?)
        };
        Ok(Self { socket, signals, commands: HashMap::new() })
    }

    fn run(&mut self) -> Result<()> {
        loop {
            let mut fds = [
                libc::pollfd { fd: self.socket.as_raw_fd(), events: libc::POLLIN, revents: 0 },
                libc::pollfd { fd: self.signals.as_raw_fd(), events: libc::POLLIN, revents: 0 },
            ];
            // SAFETY: two valid pollfd entries.
            if unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) } < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error.into());
            }
            if fds[1].revents != 0 {
                self.reap()?;
            }
            if fds[0].revents != 0 {
                let Some((request, received)) = proto::recv::<Request>(self.socket)? else {
                    return Ok(());
                };
                if !self.handle(request, received)? {
                    return Ok(());
                }
            }
        }
    }

    fn reply(&self, reply: &Reply, fds: &[RawFd]) -> Result<()> {
        proto::send(self.socket, reply, fds)?;
        Ok(())
    }

    /// Returns false when init should exit.
    fn handle(&mut self, request: Request, received: Vec<OwnedFd>) -> Result<bool> {
        match request {
            Request::Setup(_) => bail!("sandbox is already set up"),
            Request::Spawn { id, argv, cwd, env } => {
                let Some(output) = received.into_iter().next() else { bail!("spawn without output") };
                match spawn(&argv, &cwd, &env, output.as_fd()) {
                    Ok(pid) => {
                        self.commands.insert(pid, id);
                        self.reply(&Reply::Spawned { id }, &[])?;
                    }
                    Err(error) => self.reply(&Reply::SpawnFailed { id, error }, &[])?,
                }
            }
            Request::Kill { id } => {
                for (&pid, &command) in &self.commands {
                    if command == id {
                        // SAFETY: each command leads its own process group.
                        unsafe { libc::kill(-pid, libc::SIGKILL) };
                    }
                }
            }
            Request::Open { id, path, mode } => match open(&path, mode) {
                Ok(file) => self.reply(&Reply::Opened { id }, &[file.as_raw_fd()])?,
                Err(error) => {
                    self.reply(&Reply::OpenFailed { id, errno: error.raw_os_error().unwrap_or(libc::EIO) }, &[])?
                }
            },
            Request::Shutdown { id, force } => {
                if force || only_init_remains() {
                    return Ok(false);
                }
                self.reply(&Reply::ShutdownRefused { id }, &[])?;
            }
        }
        Ok(true)
    }

    fn reap(&mut self) -> Result<()> {
        let mut info = [0u8; std::mem::size_of::<libc::signalfd_siginfo>()];
        // SAFETY: draining the non-blocking signalfd into a correctly sized buffer.
        while unsafe { libc::read(self.signals.as_raw_fd(), info.as_mut_ptr().cast(), info.len()) } > 0 {}
        loop {
            let mut status = 0;
            // SAFETY: non-blocking wait for any child; orphans are reparented to init.
            let pid = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
            if pid <= 0 {
                return Ok(());
            }
            if let Some(id) = self.commands.remove(&pid) {
                let (code, signal) = if libc::WIFEXITED(status) {
                    (Some(libc::WEXITSTATUS(status)), None)
                } else {
                    (None, Some(libc::WTERMSIG(status)))
                };
                self.reply(&Reply::Exited { id, code, signal }, &[])?;
            }
        }
    }
}

fn only_init_remains() -> bool {
    fs::read_dir("/proc").map_or(true, |entries| {
        !entries.flatten().any(|entry| {
            entry.file_name().to_str().and_then(|name| name.parse::<u32>().ok()).is_some_and(|pid| pid != 1)
        })
    })
}

fn open(path: &str, mode: OpenMode) -> io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    // Non-blocking so a FIFO cannot stall init; the flag is cleared before handing it over.
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

fn find_program(program: &str, env: &[(String, String)]) -> Option<CString> {
    if program.contains('/') {
        return Some(cstr(program));
    }
    let path = env.iter().find(|(key, _)| key == "PATH").map_or("/usr/bin:/bin", |(_, value)| value.as_str());
    path.split(':')
        .map(|dir| Path::new(dir).join(program))
        .find(|candidate| {
            let c = cstr(candidate);
            // SAFETY: access() on a valid path.
            unsafe { libc::access(c.as_ptr(), libc::X_OK) == 0 }
        })
        .map(cstr)
}

/// Forks the command into its own session and process group. A close-on-exec pipe reports
/// whether `chdir` or `exec` failed in the child before the harness is told it started.
fn spawn(argv: &[String], cwd: &str, env: &[(String, String)], output: BorrowedFd) -> Result<libc::pid_t, String> {
    let Some(first) = argv.first() else { return Err("empty command".into()) };
    let program = find_program(first, env).ok_or_else(|| format!("{first}: command not found"))?;
    let args: Vec<CString> = argv.iter().map(cstr).collect();
    let vars: Vec<CString> = env.iter().map(|(key, value)| cstr(format!("{key}={value}"))).collect();
    let mut arg_ptrs: Vec<*const libc::c_char> = args.iter().map(|a| a.as_ptr()).collect();
    arg_ptrs.push(std::ptr::null());
    let mut env_ptrs: Vec<*const libc::c_char> = vars.iter().map(|v| v.as_ptr()).collect();
    env_ptrs.push(std::ptr::null());
    let dir = cstr(cwd);
    let null = cstr("/dev/null");
    let oom_adj = cstr("/proc/self/oom_score_adj");
    let mut pipe = [0; 2];
    // SAFETY: creating a close-on-exec pipe.
    if unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error().to_string());
    }
    // SAFETY: init is single-threaded; the child only makes async-signal-safe calls.
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        unsafe {
            let fail = |stage: u8| -> ! {
                let report = [stage, *libc::__errno_location() as u8];
                libc::write(pipe[1], report.as_ptr().cast(), 2);
                libc::_exit(127)
            };
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigprocmask(libc::SIG_SETMASK, &set, std::ptr::null_mut());
            libc::setsid();
            // Init cannot lower its own OOM score without host privileges, so commands raise
            // theirs: the OOM killer then always prefers them, and the sandbox survives.
            let adj = libc::open(oom_adj.as_ptr(), libc::O_WRONLY);
            libc::write(adj, c"500".as_ptr().cast(), 3);
            libc::close(adj);
            let stdin = libc::open(null.as_ptr(), libc::O_RDONLY);
            libc::dup2(stdin, 0);
            libc::dup2(output.as_raw_fd(), 1);
            libc::dup2(output.as_raw_fd(), 2);
            if libc::chdir(dir.as_ptr()) < 0 {
                fail(b'c');
            }
            libc::execve(program.as_ptr(), arg_ptrs.as_ptr(), env_ptrs.as_ptr());
            fail(b'e');
        }
    }
    // SAFETY: closing our copy of the write end and reading the child's report.
    unsafe { libc::close(pipe[1]) };
    let mut report = [0u8; 2];
    let read = unsafe { libc::read(pipe[0], report.as_mut_ptr().cast(), 2) };
    unsafe { libc::close(pipe[0]) };
    if pid < 0 {
        return Err(io::Error::last_os_error().to_string());
    }
    if read <= 0 {
        return Ok(pid);
    }
    let mut status = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };
    let errno = io::Error::from_raw_os_error(report[1] as i32);
    Err(match report[0] {
        b'c' => format!("Working directory does not exist: {cwd}"),
        _ => format!("{first}: {errno}"),
    })
}
