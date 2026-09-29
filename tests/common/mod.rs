#![allow(dead_code, unused_macros)]

pub mod model;

/// `(name, function)` for each test function, for [`run`].
macro_rules! named {
    ($($test:ident),* $(,)?) => { &[$((stringify!($test), $test)),*] };
}

use erisharness::agent::{AgentSpec, AgentState, Item};
use erisharness::machine::{DirectSpec, MachineSpec};
use erisharness::tools::{self, Content};
use erisharness::{Harness, HarnessBuilder};
use erissandbox::{Host, Limits, Output, Sandbox, SandboxSpec, Sandboxes};
use model::Model;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

pub fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap())
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    runtime().block_on(future)
}

/// Opens `builder` with the built-in tools and one provider per named model.
pub fn open(builder: HarnessBuilder, models: &[(&str, &Model)]) -> Arc<Harness> {
    let builder = models.iter().fold(builder, |b, (name, model)| b.provider(name, model.provider()));
    block_on(builder.tools(tools::builtin()).open()).expect("open harness")
}

/// A harness in `dir` whose only provider, `test`, is `model`.
pub fn harness(dir: &Path, model: &Model) -> Arc<Harness> {
    open(Harness::builder(dir), &[("test", model)])
}

/// An agent working directly in `cwd` with every built-in tool and the `test` provider.
pub fn direct(cwd: &Path) -> AgentSpec {
    AgentSpec {
        system_prompt: "You are a test agent.".into(),
        tools: tools::builtin().iter().map(|t| t.name().to_owned()).collect(),
        provider: "test".into(),
        model: "test-model".into(),
        reasoning_effort: None,
        context_window: None,
        metadata: serde_json::Value::Null,
        machine: MachineSpec::Direct(DirectSpec { cwd: cwd.to_str().unwrap().into(), env: None }),
    }
}

/// Waits until `agent` is in `state`, panicking after 20 seconds.
pub fn settle(h: &Harness, agent: &str, state: AgentState) {
    block_on(async { tokio::time::timeout(Duration::from_secs(20), h.wait_for(agent, state)).await })
        .unwrap_or_else(|_| panic!("agent never became {state:?}: {:?}", h.agent(agent).map(|a| a.state)));
}

pub fn items(h: &Harness, agent: &str) -> Vec<Item> {
    h.transcript(agent).unwrap().into_iter().map(|e| e.item).collect()
}

pub fn text(content: &[Content]) -> String {
    content.iter().map(|c| if let Content::Text(t) = c { t.as_str() } else { "" }).collect()
}

/// Every tool result: whether it is an error, and its text.
pub fn results(h: &Harness, agent: &str) -> Vec<(bool, String)> {
    items(h, agent)
        .into_iter()
        .filter_map(|item| match item {
            Item::ToolResult { output, .. } => Some((output.is_error, text(&output.content))),
            _ => None,
        })
        .collect()
}

/// Every message delivered: its sender and text.
pub fn inputs(h: &Harness, agent: &str) -> Vec<(String, String)> {
    items(h, agent)
        .into_iter()
        .filter_map(|item| match item {
            Item::Input { from, text } => Some((from, text)),
            _ => None,
        })
        .collect()
}

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
