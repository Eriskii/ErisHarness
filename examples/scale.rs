//! Measures memory per agent at scale.
//!
//! ```sh
//! cargo run --release --example scale -- <rootfs> [idle agents] [live sandboxes] [concurrent turns]
//! ```
//!
//! Idle agents are records and transcripts only. Live sandboxes each keep an init and a
//! background `sleep`. Concurrent turns each call a model (an in-process stub) that runs one
//! bash command, so every turn starts a sandbox, runs a real process and answers.

use erisharness::agent::{AgentSpec, AgentState, Item, Usage};
use erisharness::machine::MachineSpec;
use erisharness::provider::{Completion, Provider, Request};
use erisharness::sandbox::{Limits, SandboxSpec, Sandboxes};
use erisharness::{Harness, tools};
use futures_util::future::BoxFuture;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Calls bash once, then answers.
struct Stub;

impl Provider for Stub {
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        _: &'a (dyn Fn(&str) + Send + Sync),
    ) -> BoxFuture<'a, anyhow::Result<Completion>> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let answered = matches!(request.items.last(), Some(Item::ToolResult { .. }));
            let item = if answered {
                Item::Assistant { text: "done".into() }
            } else {
                Item::ToolCall {
                    call_id: "c".into(),
                    name: "bash".into(),
                    arguments: r#"{"command":"echo hi; sleep 2"}"#.into(),
                }
            };
            Ok(Completion { items: vec![item], usage: Usage::default() })
        })
    }
}

fn kib(field: &str, text: &str) -> u64 {
    text.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

fn self_rss() -> u64 {
    kib("VmRSS:", &std::fs::read_to_string("/proc/self/status").unwrap_or_default())
}

fn agents_cgroup() -> PathBuf {
    let own = std::fs::read_to_string("/proc/self/cgroup").unwrap();
    let own = own.trim().trim_start_matches("0::");
    Path::new("/sys/fs/cgroup").join(own.trim_start_matches('/')).parent().unwrap().join("agents")
}

/// Proportional set size of every process in the agents' cgroups, and their total charge.
fn sandbox_memory() -> (u64, u64, usize) {
    let root = agents_cgroup();
    let mut pss = 0;
    let mut processes = 0;
    for leaf in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        for pid in std::fs::read_to_string(leaf.path().join("cgroup.procs")).unwrap_or_default().lines() {
            pss += kib("Pss:", &std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).unwrap_or_default());
            processes += 1;
        }
    }
    let charged: u64 =
        std::fs::read_to_string(root.join("memory.current")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    (pss, charged / 1024, processes)
}

fn main() -> anyhow::Result<()> {
    let host = erisharness::bootstrap()?;
    let args: Vec<String> = std::env::args().collect();
    let rootfs = PathBuf::from(args.get(1).expect("usage: scale <rootfs> [idle] [live] [turns]"));
    let count = |i: usize, default: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(default);
    let (idle, live, turns) = (count(2, 10_000), count(3, 1000), count(4, 500));
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move {
        let dir = tempfile::Builder::new().prefix("erisharness-scale").tempdir()?;
        let spec = SandboxSpec {
            rootfs,
            layers: Vec::new(),
            binds: Vec::new(),
            limits: Limits { memory_bytes: Some(256 << 20), pids: Some(64), cpus: Some(1.0) },
            hostname: String::new(),
            cwd: "/root".into(),
            env: SandboxSpec::default_env(),
        };
        let harness = Harness::builder(dir.path())
            .sandboxes(&host)
            .provider("stub", Arc::new(Stub))
            .tools(tools::builtin())
            .idle_grace(Duration::from_secs(600))
            .open()
            .await?;
        let agent_spec = AgentSpec { system_prompt: "bench".into(), tools: vec!["bash".into()], provider: "stub".into(), machine: MachineSpec::Sandbox(spec.clone()) };

        let before = self_rss();
        let started = Instant::now();
        for _ in 0..idle {
            harness.create_agent(agent_spec.clone())?;
        }
        let after = self_rss();
        println!(
            "idle agents: {idle} created in {:.1}s; harness RSS {} -> {} KiB ({:.2} KiB per agent)",
            started.elapsed().as_secs_f64(),
            before,
            after,
            (after.saturating_sub(before)) as f64 / idle as f64
        );

        let sandboxes = Arc::new(Sandboxes::new(&host, dir.path().join("live"))?.idle_grace(Duration::from_secs(600)));
        let before = self_rss();
        let started = Instant::now();
        let mut handles = Vec::new();
        for chunk in (0..live).collect::<Vec<_>>().chunks(64) {
            let batch = chunk.iter().map(|i| {
                let sandbox = sandboxes.sandbox(&format!("live-{i}"), spec.clone()).unwrap();
                async move {
                    sandbox.run(&["/bin/bash".into(), "-c".into(), "sleep 3600 >/dev/null 2>&1 &".into()]).await.map(|_| sandbox)
                }
            });
            for result in futures_util::future::join_all(batch).await {
                handles.push(result?);
            }
        }
        let (pss, charged, processes) = sandbox_memory();
        let after = self_rss();
        println!(
            "live sandboxes: {live} started in {:.1}s; {processes} processes; PSS {:.0} KiB/sandbox; cgroup charge {:.0} KiB/sandbox; harness RSS +{:.1} KiB/sandbox",
            started.elapsed().as_secs_f64(),
            pss as f64 / live as f64,
            charged as f64 / live as f64,
            after.saturating_sub(before) as f64 / live as f64
        );
        sandboxes.shutdown_all().await;
        drop(handles);

        let ids: Vec<String> = (0..turns).map(|_| harness.create_agent(agent_spec.clone())).collect::<anyhow::Result<_>>()?;
        let before = self_rss();
        let started = Instant::now();
        for id in &ids {
            harness.send(id, "user", "go")?;
        }
        let mut peak = (0, 0, 0, 0);
        loop {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let (pss, charged, processes) = sandbox_memory();
            let rss = self_rss();
            if rss + pss > peak.0 + peak.1 {
                peak = (rss, pss, charged, processes);
            }
            let done = ids.iter().filter(|id| harness.agent(id).is_ok_and(|a| a.state == AgentState::Idle && !a.held)).count();
            let answered = ids
                .iter()
                .filter(|id| harness.transcript(id).is_ok_and(|t| matches!(t.last().map(|e| &e.item), Some(Item::Assistant { .. }))))
                .count();
            if answered == turns && done == turns {
                break;
            }
        }
        println!(
            "concurrent turns: {turns} finished in {:.1}s; at peak {} processes, harness RSS +{:.1} KiB/turn, sandbox PSS {:.0} KiB/turn, cgroup charge {:.0} KiB/turn",
            started.elapsed().as_secs_f64(),
            peak.3,
            peak.0.saturating_sub(before) as f64 / turns as f64,
            peak.1 as f64 / turns as f64,
            peak.2 as f64 / turns as f64
        );
        harness.shutdown().await;
        anyhow::Ok(())
    })
}
