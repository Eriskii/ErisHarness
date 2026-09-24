//! Starting a sandbox init. `clone3` creates the process directly in its namespaces and its
//! cgroup, so no moment exists where it runs unconfined. The harness is multi-threaded, so
//! the child only makes async-signal-safe calls: it waits for its id mapping, then execs this
//! binary as init.

use crate::bootstrap::{ID_COUNT, INIT_ARG};
use anyhow::{Context, Result};
use std::ffi::CString;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

#[repr(C)]
#[derive(Default)]
struct CloneArgs {
    flags: u64,
    pidfd: u64,
    child_tid: u64,
    parent_tid: u64,
    exit_signal: u64,
    stack: u64,
    stack_size: u64,
    tls: u64,
    set_tid: u64,
    set_tid_size: u64,
    cgroup: u64,
}

const CLONE_INTO_CGROUP: u64 = 0x2_0000_0000;

/// Returns the harness end of the control socket and a pidfd for the init.
pub fn init(exe: BorrowedFd, cgroup: BorrowedFd) -> Result<(OwnedFd, OwnedFd)> {
    let (ours, theirs) = nix::sys::socket::socketpair(
        nix::sys::socket::AddressFamily::Unix,
        nix::sys::socket::SockType::SeqPacket,
        None,
        nix::sys::socket::SockFlag::SOCK_CLOEXEC,
    )?;
    let argv = [c"erisharness-init".to_owned(), CString::new(INIT_ARG).unwrap()];
    let argv_ptrs = [argv[0].as_ptr(), argv[1].as_ptr(), std::ptr::null()];
    let envp: [*const libc::c_char; 1] = [std::ptr::null()];
    let socket = theirs.as_raw_fd();
    let exe = exe.as_raw_fd();
    let mut pidfd: libc::c_int = -1;
    let args = CloneArgs {
        flags: (libc::CLONE_NEWUSER
            | libc::CLONE_NEWNS
            | libc::CLONE_NEWPID
            | libc::CLONE_NEWNET
            | libc::CLONE_NEWIPC
            | libc::CLONE_NEWUTS
            | libc::CLONE_NEWCGROUP
            | libc::CLONE_PIDFD) as u64
            | CLONE_INTO_CGROUP,
        pidfd: &mut pidfd as *mut libc::c_int as u64,
        exit_signal: libc::SIGCHLD as u64,
        cgroup: cgroup.as_raw_fd() as u64,
        ..CloneArgs::default()
    };
    // SAFETY: clone3 without CLONE_VM behaves like fork. The child branch below only calls
    // async-signal-safe functions on memory prepared before the call.
    let pid = unsafe { libc::syscall(libc::SYS_clone3, &args as *const CloneArgs, std::mem::size_of::<CloneArgs>()) };
    if pid == 0 {
        unsafe { child(socket, exe, &argv_ptrs, &envp) }
    }
    if pid < 0 {
        return Err(io::Error::last_os_error()).context("clone3");
    }
    // SAFETY: CLONE_PIDFD stored a descriptor we now own.
    let pidfd = unsafe { OwnedFd::from_raw_fd(pidfd) };
    drop(theirs);
    crate::machine::set_nonblocking(&ours)?;
    // One extent per extent of the harness map (root, then the subordinate range): the kernel
    // rejects an extent that spans two parent extents.
    let map = format!("0 0 1\n1 1 {}\n", ID_COUNT - 1);
    let written = std::fs::write(format!("/proc/{pid}/uid_map"), &map)
        .map_err(|e| io::Error::new(e.kind(), format!("uid_map: {e}")))
        .and_then(|()| {
            std::fs::write(format!("/proc/{pid}/gid_map"), &map)
                .map_err(|e| io::Error::new(e.kind(), format!("gid_map: {e}")))
        })
        .and_then(|()| nix::unistd::write(&ours, b"m").map(drop).map_err(io::Error::from));
    if let Err(error) = written {
        // SAFETY: the child is ours and has not started; killing and reaping it is safe.
        unsafe {
            libc::syscall(libc::SYS_pidfd_send_signal, pidfd.as_raw_fd(), libc::SIGKILL, 0, 0);
            libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), 0);
        }
        return Err(error).context("mapping sandbox ids");
    }
    Ok((ours, pidfd))
}

/// Runs in the cloned child. Never returns.
unsafe fn child(socket: i32, exe: i32, argv: &[*const libc::c_char; 3], envp: &[*const libc::c_char; 1]) -> ! {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigprocmask(libc::SIG_SETMASK, &set, std::ptr::null_mut());
        let socket = if socket == super::init::SOCKET_FD {
            libc::fcntl(socket, libc::F_SETFD, 0);
            socket
        } else {
            libc::dup2(socket, super::init::SOCKET_FD)
        };
        // Everything else closes at exec, whatever other threads opened.
        libc::syscall(libc::SYS_close_range, super::init::SOCKET_FD + 1, u32::MAX, libc::CLOSE_RANGE_CLOEXEC);
        // Wait until the harness has written the id maps.
        let mut byte = 0u8;
        if socket < 0 || libc::read(socket, (&mut byte as *mut u8).cast(), 1) != 1 {
            libc::_exit(126);
        }
        libc::syscall(libc::SYS_execveat, exe, c"".as_ptr(), argv.as_ptr(), envp.as_ptr(), libc::AT_EMPTY_PATH);
        libc::_exit(127)
    }
}
