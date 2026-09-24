//! Agent images are plain directory trees. [`import`] unpacks an image tarball, such as the
//! output of `docker export`, with its ownership intact. It must run inside the harness
//! namespace so ids beyond the invoking user map onto the subordinate range.

use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

pub fn import(mut tar: impl Read, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // Device nodes cannot be created in a user namespace; sandboxes mount their own /dev.
    let mut child = Command::new("tar")
        .args(["-x", "-p", "--numeric-owner", "--exclude=dev/*", "--exclude=.dockerenv", "-C"])
        .arg(dir)
        .stdin(Stdio::piped())
        .spawn()
        .context("running tar")?;
    std::io::copy(&mut tar, child.stdin.as_mut().unwrap())?;
    drop(child.stdin.take());
    let status = child.wait()?;
    if !status.success() {
        bail!("tar exited with {status}");
    }
    Ok(())
}
