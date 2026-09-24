//! Direct mode: agents run on this machine as this user. No bootstrap, namespaces, cgroups
//! or images are needed, so this binary uses the standard test harness.

mod common;

use common::block_on;
use common::model::{Model, calls, says};
use erisharness::agent::{AgentSpec, AgentState, Item};
use erisharness::machine::{DirectSpec, MachineSpec};
use erisharness::{Harness, tools};
use erissandbox::{Limits, SandboxSpec};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn harness(dir: &Path, model: &Model) -> Arc<Harness> {
    block_on(Harness::builder(dir).provider("test", model.provider()).tools(tools::builtin()).open()).unwrap()
}

fn direct(cwd: &Path) -> AgentSpec {
    AgentSpec {
        system_prompt: "Direct agent.".into(),
        tools: vec!["read".into(), "bash".into(), "edit".into(), "write".into(), "send_message".into()],
        provider: "test".into(),
        model: "test-model".into(),
        reasoning_effort: Some("low".into()),
        context_window: None,
        machine: MachineSpec::Direct(DirectSpec { cwd: cwd.to_str().unwrap().into(), env: None }),
    }
}

fn settle(h: &Harness, agent: &str) {
    block_on(async { tokio::time::timeout(Duration::from_secs(20), h.wait_for(agent, AgentState::Idle)).await })
        .expect("agent settled");
}

fn results(h: &Harness, agent: &str) -> Vec<String> {
    h.transcript(agent)
        .unwrap()
        .into_iter()
        .filter_map(|e| match e.item {
            Item::ToolResult { output, .. } => Some(
                output
                    .content
                    .iter()
                    .map(|c| match c {
                        tools::Content::Text(t) => t.clone(),
                        tools::Content::Image { .. } => String::new(),
                    })
                    .collect(),
            ),
            _ => None,
        })
        .collect()
}

#[test]
fn agents_work_on_the_host_filesystem_in_their_directory() {
    let temp = tempfile::tempdir().unwrap();
    let project = temp.path().join("project");
    std::fs::create_dir(&project).unwrap();
    let outside = temp.path().join("outside.txt");
    std::fs::write(&outside, "host file\n").unwrap();
    let model = Model::start(vec![
        calls("c1", "bash", json!({"command": format!("pwd; cat {}; id -u", outside.display())})),
        calls("c2", "write", json!({"path": "made.txt", "content": "by agent"})),
        says("done"),
    ]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(direct(&project)).unwrap();
    h.send(&agent, "user", "look around").unwrap();
    settle(&h, &agent);
    let uid = nix::unistd::getuid();
    assert_eq!(
        results(&h, &agent),
        [format!("{}\nhost file\n{uid}\n", project.display()), "Successfully wrote to made.txt".into()]
    );
    assert_eq!(std::fs::read_to_string(project.join("made.txt")).unwrap(), "by agent");
}

#[test]
fn direct_agents_inherit_the_harness_environment_unless_given_one() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "echo \"$HOME|$ERIS_MARKER\""})), says("ok")]);
    let h = harness(&temp.path().join("state"), &model);
    let mut spec = direct(temp.path());
    let MachineSpec::Direct(machine) = &mut spec.machine else { unreachable!() };
    machine.env = Some(vec![("HOME".into(), "/elsewhere".into()), ("ERIS_MARKER".into(), "set".into())]);
    let agent = h.create_agent(spec).unwrap();
    h.send(&agent, "user", "env").unwrap();
    settle(&h, &agent);
    assert_eq!(results(&h, &agent), ["/elsewhere|set\n"]);

    let inherited = Model::start(vec![calls("c1", "bash", json!({"command": "echo \"$HOME\""})), says("ok")]);
    let h = harness(&temp.path().join("state2"), &inherited);
    let agent = h.create_agent(direct(temp.path())).unwrap();
    h.send(&agent, "user", "env").unwrap();
    settle(&h, &agent);
    assert_eq!(results(&h, &agent), [format!("{}\n", std::env::var("HOME").unwrap())]);
}

#[test]
fn interrupting_kills_the_commands_process_group() {
    let temp = tempfile::tempdir().unwrap();
    let marker = format!("{}.{}", 30, std::process::id());
    let model = Model::start(vec![calls("c1", "bash", json!({"command": format!("sleep {marker} | cat")}))]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(direct(temp.path())).unwrap();
    h.send(&agent, "user", "sleep").unwrap();
    std::thread::sleep(Duration::from_millis(800));
    h.interrupt(&agent);
    settle(&h, &agent);
    assert_eq!(results(&h, &agent), ["Command aborted"]);
    std::thread::sleep(Duration::from_millis(200));
    let survivors = std::fs::read_dir("/proc")
        .unwrap()
        .flatten()
        .filter_map(|e| std::fs::read(e.path().join("cmdline")).ok())
        .filter(|cmdline| String::from_utf8_lossy(cmdline).contains(&marker))
        .count();
    assert_eq!(survivors, 0);
}

#[test]
fn sandboxed_agents_need_sandboxes_enabled() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![]);
    let h = harness(&temp.path().join("state"), &model);
    let spec = AgentSpec {
        machine: MachineSpec::Sandbox(SandboxSpec {
            rootfs: "/nonexistent".into(),
            layers: Vec::new(),
            binds: Vec::new(),
            limits: Limits::default(),
            hostname: String::new(),
            cwd: "/root".into(),
            env: SandboxSpec::default_env(),
            devices: Vec::new(),
            forwards: Vec::new(),
        }),
        ..direct(temp.path())
    };
    let error = h.create_agent(spec).unwrap_err().to_string();
    assert!(error.contains("sandboxes are not enabled"), "{error}");
}
