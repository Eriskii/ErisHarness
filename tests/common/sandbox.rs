use super::block_on;
use erissandbox::{Host, Limits, Output, Sandbox, SandboxSpec, Sandboxes};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

pub type Test = fn(&Fixture);

/// Bootstraps into the harness namespace, then runs each test against one shared fixture.
/// Test binaries use this as `main` because bootstrap must precede every thread.
pub fn run(tests: &[(&'static str, Test)]) -> ! {
    let host = erissandbox::bootstrap().expect("bootstrap");
    let fixture = Arc::new(Fixture::new(host));
    let trials = tests
        .iter()
        .map(|&(name, test)| {
            let fixture = fixture.clone();
            libtest_mimic::Trial::test(name, move || {
                test(&fixture);
                Ok(())
            })
        })
        .collect();
    libtest_mimic::run(&libtest_mimic::Arguments::from_args(), trials).exit()
}

pub struct Fixture {
    pub host: Host,
    pub sandboxes: Arc<Sandboxes>,
    rootfs: PathBuf,
    temp: tempfile::TempDir,
}

impl Fixture {
    pub fn new(host: Host) -> Self {
        let rootfs = test_rootfs();
        let temp = tempfile::Builder::new().prefix("erisharness-test").tempdir().unwrap();
        let sandboxes = Arc::new(Sandboxes::new(&host, temp.path().join("sandboxes")).unwrap());
        Self { host, sandboxes, rootfs, temp }
    }

    pub fn rootfs(&self) -> &Path {
        &self.rootfs
    }

    pub fn spec(&self) -> SandboxSpec {
        sandbox_spec(self.rootfs.clone())
    }

    pub fn sandbox(&self, name: &str, mut spec: SandboxSpec) -> Arc<Sandbox> {
        spec.hostname = format!("agent-{name}");
        self.sandboxes.sandbox(name, spec).unwrap()
    }

    pub fn host_dir(&self, name: &str) -> PathBuf {
        let dir = self.temp.path().join("host").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub async fn run(&self, sandbox: &Arc<Sandbox>, script: &str) -> Output {
        sandbox.run(&["/bin/bash".into(), "-c".into(), script.into()]).await.expect("run")
    }
}

pub fn sandbox_spec(rootfs: PathBuf) -> SandboxSpec {
    SandboxSpec {
        rootfs,
        layers: Vec::new(),
        binds: Vec::new(),
        limits: Limits { memory_bytes: Some(512 << 20), pids: Some(256), cpus: Some(2.0) },
        hostname: String::new(),
        cwd: "/root".into(),
        env: SandboxSpec::default_env(),
        devices: Vec::new(),
        forwards: Vec::new(),
    }
}

/// Debian bookworm-slim, exported from Docker once into `target/` and reused.
fn test_rootfs() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("rootfs-bookworm-slim");
    if dir.join("bin/bash").exists() {
        return dir;
    }
    let container = Command::new("docker").args(["create", "debian:bookworm-slim"]).output().expect("docker create");
    assert!(container.status.success(), "docker create: {container:?}");
    let id = String::from_utf8(container.stdout).unwrap().trim().to_owned();
    let mut export = Command::new("docker").args(["export", &id]).stdout(Stdio::piped()).spawn().unwrap();
    let staging = dir.with_extension("partial");
    let _ = std::fs::remove_dir_all(&staging);
    erissandbox::rootfs::import(export.stdout.take().unwrap(), &staging).expect("import rootfs");
    assert!(export.wait().unwrap().success());
    let _ = Command::new("docker").args(["rm", &id]).output();
    std::fs::rename(&staging, &dir).unwrap();
    dir
}
