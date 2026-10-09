//! Direct agents run on this machine as this user. They need no bootstrap, namespaces,
//! cgroups or images, so this binary uses the standard test harness.

mod common;

use common::model::{Model, calls, says};
use common::{direct, harness, results, settle};
use erisharness::agent::AgentState;
use erisharness::machine::MachineSpec;
use serde_json::json;
use std::time::Duration;

#[test]
fn agents_work_on_the_host_filesystem_in_their_directory() {
    let temp = tempfile::tempdir().unwrap();
    // `pwd` prints the resolved path, and macOS's temporary directory is behind a symlink.
    let project = temp.path().canonicalize().unwrap().join("project");
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
    settle(&h, &agent, AgentState::Idle);
    let uid = nix::unistd::getuid();
    assert_eq!(
        results(&h, &agent),
        [
            (false, format!("{}\nhost file\n{uid}\n", project.display())),
            (false, "Successfully wrote to made.txt".into())
        ]
    );
    assert_eq!(std::fs::read_to_string(project.join("made.txt")).unwrap(), "by agent");
}

#[test]
#[allow(irrefutable_let_patterns)]
fn direct_agents_inherit_the_harness_environment_unless_given_one() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "echo \"$HOME|$ERIS_MARKER\""})), says("ok")]);
    let h = harness(&temp.path().join("state"), &model);
    let mut spec = direct(temp.path());
    let MachineSpec::Direct(machine) = &mut spec.machine else { unreachable!() };
    machine.env = Some(vec![("HOME".into(), "/elsewhere".into()), ("ERIS_MARKER".into(), "set".into())]);
    let agent = h.create_agent(spec).unwrap();
    h.send(&agent, "user", "env").unwrap();
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(results(&h, &agent), [(false, "/elsewhere|set\n".into())]);

    let inherited = Model::start(vec![calls("c1", "bash", json!({"command": "echo \"$HOME\""})), says("ok")]);
    let h = harness(&temp.path().join("state2"), &inherited);
    let agent = h.create_agent(direct(temp.path())).unwrap();
    h.send(&agent, "user", "env").unwrap();
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(results(&h, &agent), [(false, format!("{}\n", std::env::var("HOME").unwrap()))]);
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
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(results(&h, &agent), [(true, "Command aborted".into())]);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(command_lines().iter().filter(|line| line.contains(&marker)).count(), 0);
}

/// Every running process's command line.
#[cfg(target_os = "linux")]
fn command_lines() -> Vec<String> {
    std::fs::read_dir("/proc")
        .unwrap()
        .flatten()
        .filter_map(|e| std::fs::read(e.path().join("cmdline")).ok())
        .map(|cmdline| String::from_utf8_lossy(&cmdline).into_owned())
        .collect()
}

/// Every running process's command line.
#[cfg(not(target_os = "linux"))]
fn command_lines() -> Vec<String> {
    let ps = std::process::Command::new("ps").args(["-A", "-o", "command="]).output().unwrap();
    String::from_utf8_lossy(&ps.stdout).lines().map(str::to_owned).collect()
}

#[test]
#[cfg(target_os = "linux")]
fn sandboxed_agents_need_sandboxes_enabled() {
    let temp = tempfile::tempdir().unwrap();
    let h = harness(&temp.path().join("state"), &Model::start(vec![]));
    let spec = erisharness::agent::AgentSpec {
        machine: MachineSpec::Sandbox(common::sandbox_spec("/nonexistent".into())),
        ..direct(temp.path())
    };
    let error = h.create_agent(spec).unwrap_err().to_string();
    assert!(error.contains("sandboxes are not enabled"), "{error}");
}
