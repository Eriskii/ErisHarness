//! Messages between the harness and a sandbox's init over a `SOCK_SEQPACKET` pair. Each
//! message is one JSON packet; file descriptors ride along as `SCM_RIGHTS`. Packets larger
//! than [`INLINE_LIMIT`] travel in a sealed memfd so a huge command never hits socket limits.

use nix::sys::socket::{ControlMessage, ControlMessageOwned, MsgFlags, recvmsg, sendmsg};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, IoSlice, IoSliceMut, Read, Seek, Write};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::path::PathBuf;

const INLINE_LIMIT: usize = 48 * 1024;
const MAX_FDS: usize = 4;
const MEMFD_MARKER: &[u8] = b"\0memfd";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Setup {
    pub rootfs: PathBuf,
    pub root_upper: PathBuf,
    pub root_work: PathBuf,
    pub mount_point: PathBuf,
    pub layers: Vec<LayerMount>,
    pub binds: Vec<BindMount>,
    pub hostname: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMount {
    pub lower: PathBuf,
    pub upper: PathBuf,
    pub work: PathBuf,
    pub target: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindMount {
    pub source: PathBuf,
    pub target: String,
    pub writable: bool,
}

/// How [`Request::Open`] opens a path inside the sandbox.
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Request {
    Setup(Setup),
    /// Runs `argv` with the passed descriptor as stdout and stderr, in a new session.
    Spawn {
        id: u64,
        argv: Vec<String>,
        cwd: String,
        env: Vec<(String, String)>,
    },
    /// Kills the spawned command's whole process group.
    Kill {
        id: u64,
    },
    /// Opens a path as the sandbox sees it; the reply carries the descriptor.
    Open {
        id: u64,
        path: String,
        mode: OpenMode,
    },
    /// Ends the sandbox. Without `force`, init refuses while other processes are alive.
    Shutdown {
        id: u64,
        force: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Reply {
    Ready,
    SetupFailed { error: String },
    Spawned { id: u64 },
    SpawnFailed { id: u64, error: String },
    Exited { id: u64, code: Option<i32>, signal: Option<i32> },
    Opened { id: u64 },
    OpenFailed { id: u64, errno: i32 },
    ShutdownRefused { id: u64 },
}

impl Reply {
    pub fn id(&self) -> Option<u64> {
        match self {
            Reply::Ready | Reply::SetupFailed { .. } => None,
            Reply::Spawned { id }
            | Reply::SpawnFailed { id, .. }
            | Reply::Exited { id, .. }
            | Reply::Opened { id }
            | Reply::OpenFailed { id, .. }
            | Reply::ShutdownRefused { id } => Some(*id),
        }
    }
}

/// Sends one message. Returns `WouldBlock` on a full non-blocking socket.
pub fn send<T: Serialize>(socket: BorrowedFd, message: &T, fds: &[RawFd]) -> io::Result<()> {
    let body = serde_json::to_vec(message).map_err(io::Error::other)?;
    if body.len() <= INLINE_LIMIT {
        return send_packet(socket, &body, fds);
    }
    let memfd = nix::sys::memfd::memfd_create(c"erisharness-message", nix::sys::memfd::MFdFlags::MFD_CLOEXEC)?;
    let mut file = std::fs::File::from(memfd);
    file.write_all(&body)?;
    let mut all = fds.to_vec();
    all.push(file.as_raw_fd());
    send_packet(socket, MEMFD_MARKER, &all)
}

fn send_packet(socket: BorrowedFd, bytes: &[u8], fds: &[RawFd]) -> io::Result<()> {
    let control = [ControlMessage::ScmRights(fds)];
    let control: &[ControlMessage] = if fds.is_empty() { &[] } else { &control };
    sendmsg::<()>(socket.as_raw_fd(), &[IoSlice::new(bytes)], control, MsgFlags::MSG_NOSIGNAL, None)?;
    Ok(())
}

/// Receives one message and its descriptors. `Ok(None)` means the peer closed the socket.
pub fn recv<T: DeserializeOwned>(socket: BorrowedFd) -> io::Result<Option<(T, Vec<OwnedFd>)>> {
    let mut buffer = vec![0u8; INLINE_LIMIT + 1];
    let mut control = nix::cmsg_space!([RawFd; MAX_FDS + 1]);
    let (length, mut fds) = {
        let mut iov = [IoSliceMut::new(&mut buffer)];
        let message = recvmsg::<()>(socket.as_raw_fd(), &mut iov, Some(&mut control), MsgFlags::MSG_CMSG_CLOEXEC)?;
        let mut fds = Vec::new();
        for cmsg in message.cmsgs()? {
            if let ControlMessageOwned::ScmRights(received) = cmsg {
                // SAFETY: SCM_RIGHTS installs fresh descriptors that this process now owns.
                fds.extend(received.into_iter().map(|fd| unsafe { OwnedFd::from_raw_fd(fd) }));
            }
        }
        (message.bytes, fds)
    };
    if length == 0 {
        return Ok(None);
    }
    let body = &buffer[..length];
    let message = if body == MEMFD_MARKER {
        let memfd = fds.pop().ok_or_else(|| io::Error::other("memfd message without descriptor"))?;
        let mut file = std::fs::File::from(memfd);
        file.rewind()?;
        let mut body = Vec::new();
        file.read_to_end(&mut body)?;
        serde_json::from_slice(&body)
    } else {
        serde_json::from_slice(body)
    };
    Ok(Some((message.map_err(io::Error::other)?, fds)))
}
