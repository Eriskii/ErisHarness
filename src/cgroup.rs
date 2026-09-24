//! cgroup v2 placement. The harness claims a subtree it owns:
//!
//! ```text
//! <delegated>/erisharness-<pid>/     controllers enabled for children
//!     harness/                       the harness process itself
//!     agents/<sandbox id>/           one leaf per live sandbox, with its limits
//! ```
//!
//! `<delegated>` is the process's own cgroup when the harness is alone in it (a systemd unit
//! with `Delegate=yes`), otherwise the nearest ancestor the user owns with the memory, pids
//! and cpu controllers available, such as `user@<uid>.service`.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

const MOUNT: &str = "/sys/fs/cgroup";
const CONTROLLERS: [&str; 3] = ["memory", "pids", "cpu"];

#[derive(Debug)]
pub struct Cgroups {
    root: PathBuf,
    created_root: bool,
}

impl Cgroups {
    pub fn claim() -> Result<Self> {
        let own = own_cgroup()?;
        let pid = std::process::id();
        let alone = fs::read_to_string(own.join("cgroup.procs"))?.split_whitespace().eq([pid.to_string().as_str()]);
        let (root, created_root) = if alone && writable(&own) {
            (own.clone(), false)
        } else {
            let parent = own
                .ancestors()
                .skip(1)
                .take_while(|dir| dir.starts_with(MOUNT) && *dir != Path::new(MOUNT))
                .find(|dir| writable(dir) && has_controllers(dir))
                .with_context(|| format!("no writable cgroup above {} delegates {CONTROLLERS:?}", own.display()))?;
            sweep_stale(parent);
            let root = parent.join(format!("erisharness-{pid}"));
            fs::create_dir(&root).with_context(|| format!("creating {}", root.display()))?;
            (root, true)
        };
        let cgroups = Self { root, created_root };
        let harness = cgroups.root.join("harness");
        fs::create_dir_all(&harness)?;
        fs::write(harness.join("cgroup.procs"), pid.to_string()).context("moving the harness into its cgroup")?;
        enable_controllers(&cgroups.root)?;
        let agents = cgroups.agents();
        fs::create_dir_all(&agents)?;
        enable_controllers(&agents)?;
        Ok(cgroups)
    }

    pub fn agents(&self) -> PathBuf {
        self.root.join("agents")
    }

    /// Kills whatever is left in the subtree and removes the directories this process made.
    pub fn release(&self) {
        let agents = self.agents();
        let _ = fs::write(agents.join("cgroup.kill"), "1");
        for _ in 0..50 {
            let leaves: Vec<PathBuf> = fs::read_dir(&agents)
                .map(|entries| entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
                .unwrap_or_default();
            if leaves.iter().all(|leaf| fs::remove_dir(leaf).is_ok()) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let _ = fs::remove_dir(&agents);
        if self.created_root {
            let _ = fs::remove_dir(self.root.join("harness"));
            let _ = fs::remove_dir(&self.root);
        }
    }
}

/// Removes subtrees left by harnesses that were killed before they could clean up.
fn sweep_stale(parent: &Path) {
    let Ok(entries) = fs::read_dir(parent) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|n| n.strip_prefix("erisharness-")).and_then(|p| p.parse::<u32>().ok())
        else {
            continue;
        };
        if !Path::new(&format!("/proc/{pid}")).exists() {
            Cgroups { root: entry.path(), created_root: true }.release();
        }
    }
}

fn own_cgroup() -> Result<PathBuf> {
    let text = fs::read_to_string("/proc/self/cgroup")?;
    let Some(path) = text.lines().find_map(|line| line.strip_prefix("0::")) else {
        bail!("cgroup v2 is required");
    };
    Ok(Path::new(MOUNT).join(path.trim_start_matches('/')))
}

fn writable(dir: &Path) -> bool {
    nix::unistd::access(dir, nix::unistd::AccessFlags::W_OK).is_ok()
        && nix::unistd::access(&dir.join("cgroup.procs"), nix::unistd::AccessFlags::W_OK).is_ok()
}

fn has_controllers(dir: &Path) -> bool {
    let enabled = fs::read_to_string(dir.join("cgroup.subtree_control")).unwrap_or_default();
    CONTROLLERS.iter().all(|c| enabled.split_whitespace().any(|e| e == *c))
}

fn enable_controllers(dir: &Path) -> Result<()> {
    let request = CONTROLLERS.map(|c| format!("+{c}")).join(" ");
    fs::write(dir.join("cgroup.subtree_control"), request)
        .with_context(|| format!("enabling controllers in {}", dir.display()))
}

/// Resource limits for one sandbox. `None` leaves a resource unlimited.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Limits {
    pub memory_bytes: Option<u64>,
    pub pids: Option<u64>,
    /// CPU bandwidth in cores, such as `1.5`.
    pub cpus: Option<f64>,
}

impl Limits {
    pub(crate) fn apply(&self, cgroup: &Path) -> Result<()> {
        let max = |value: Option<u64>| value.map_or("max".to_owned(), |v| v.to_string());
        fs::write(cgroup.join("memory.max"), max(self.memory_bytes))?;
        // Without this, a limited sandbox swaps instead of meeting its limit.
        if self.memory_bytes.is_some() && cgroup.join("memory.swap.max").exists() {
            fs::write(cgroup.join("memory.swap.max"), "0")?;
        }
        fs::write(cgroup.join("pids.max"), max(self.pids))?;
        let period = 100_000u64;
        let cpu = self.cpus.map_or("max".to_owned(), |cores| ((cores * period as f64) as u64).max(1000).to_string());
        fs::write(cgroup.join("cpu.max"), format!("{cpu} {period}"))?;
        Ok(())
    }
}
