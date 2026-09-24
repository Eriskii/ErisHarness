//! Filesystem errors worded as Node words them (`ENOENT: no such file or directory, open
//! '/x'`). Pi's tools surface Node's messages, and models are used to reacting to them.

use std::io;

pub fn code(error: &io::Error) -> String {
    match error.raw_os_error() {
        Some(errno) => nix::errno::Errno::from_raw(errno).to_string().split(':').next().unwrap_or("EIO").to_owned(),
        None => "EIO".to_owned(),
    }
}

fn description(error: &io::Error) -> String {
    let known = match error.raw_os_error() {
        Some(libc::ENOENT) => "no such file or directory",
        Some(libc::EACCES) => "permission denied",
        Some(libc::EISDIR) => "illegal operation on a directory",
        Some(libc::ENOTDIR) => "not a directory",
        Some(libc::ELOOP) => "too many symbolic links encountered",
        Some(libc::ENAMETOOLONG) => "name too long",
        Some(libc::EROFS) => "read-only file system",
        Some(libc::ENOSPC) => "no space left on device",
        Some(libc::EPERM) => "operation not permitted",
        Some(libc::EEXIST) => "file already exists",
        Some(libc::EBUSY) => "resource busy or locked",
        Some(libc::ETXTBSY) => "text file is busy",
        Some(libc::EFBIG) => "file too large",
        Some(libc::EMFILE) => "too many open files",
        Some(libc::EIO) => "i/o error",
        _ => "",
    };
    if known.is_empty() {
        error.to_string().split(" (os error").next().unwrap_or_default().to_lowercase()
    } else {
        known.to_owned()
    }
}

/// `CODE: description, syscall 'path'`, or without the path when it is empty.
pub fn node(error: &io::Error, syscall: &str, path: &str) -> String {
    let base = format!("{}: {}, {syscall}", code(error), description(error));
    if path.is_empty() { base } else { format!("{base} '{path}'") }
}
