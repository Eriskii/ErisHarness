//! Isolation contract for agent sandboxes. Runs inside the harness user namespace, so it uses
//! its own `main` that bootstraps before any thread starts.

mod common;

use common::{Fixture, block_on};
use erisharness::sandbox::{Bind, Layer, Limits, Output};
use libtest_mimic::Failed;
use std::fs;

fn main() {
    common::run(&[
        ("agent_is_root_in_a_private_hostname", identity),
        ("host_files_are_unreachable", host_files_are_unreachable),
        ("writes_stay_in_each_agents_layer", writes_stay_in_each_agents_layer),
        ("workspace_layer_is_copy_on_write", workspace_layer_is_copy_on_write),
        ("read_only_bind_is_live_and_immutable", read_only_bind_is_live_and_immutable),
        ("agents_cannot_see_other_processes", agents_cannot_see_other_processes),
        ("each_agent_has_its_own_loopback_network", each_agent_has_its_own_loopback_network),
        ("memory_limit_kills_the_runaway_process", memory_limit_kills_the_runaway_process),
        ("pid_limit_bounds_process_creation", pid_limit_bounds_process_creation),
        ("namespace_and_mount_escapes_are_denied", namespace_and_mount_escapes_are_denied),
        ("background_processes_keep_the_sandbox_alive", background_processes_keep_the_sandbox_alive),
        ("files_open_inside_the_sandbox_view", files_open_inside_the_sandbox_view),
        ("filesystem_persists_across_hibernation", filesystem_persists_across_hibernation),
        ("missing_working_directory_is_reported", missing_working_directory_is_reported),
        ("kill_stops_the_whole_command", kill_stops_the_whole_command),
    ]);
}

/// Counts `sleep` processes visible in the sandbox; Debian slim has no procps.
const SLEEPERS: &str = "grep -l '^sleep' /proc/[0-9]*/cmdline 2>/dev/null | wc -l";

fn sh(fixture: &Fixture, sandbox: &std::sync::Arc<erisharness::sandbox::Sandbox>, script: &str) -> Output {
    block_on(fixture.run(sandbox, script))
}

fn check(condition: bool, message: impl Into<String>) -> Result<(), Failed> {
    if condition { Ok(()) } else { Err(message.into().into()) }
}

fn identity(f: &Fixture) -> Result<(), Failed> {
    let sandbox = f.sandbox("identity", f.spec());
    let out = sh(f, &sandbox, "id -u; hostname; echo $HOME; pwd");
    check(out.text() == "0\nagent-identity\n/root\n/root\n", format!("{out:?}"))
}

fn host_files_are_unreachable(f: &Fixture) -> Result<(), Failed> {
    let secret = f.host_dir("secret").join("key");
    fs::write(&secret, "host secret").unwrap();
    let sandbox = f.sandbox("host-files", f.spec());
    let script = format!(
        "cat {0} 2>/dev/null; test -e {0} && echo visible; ls /home; test -e /nix && echo nix; test -e /run/current-system && echo system; true",
        secret.display()
    );
    let out = sh(f, &sandbox, &script);
    check(out.text().is_empty(), format!("host content leaked: {out:?}"))
}

fn writes_stay_in_each_agents_layer(f: &Fixture) -> Result<(), Failed> {
    let a = f.sandbox("layer-a", f.spec());
    let b = f.sandbox("layer-b", f.spec());
    sh(f, &a, "echo a > /etc/agent && mkdir -p /usr/local/x && touch /usr/local/x/y");
    sh(f, &b, "echo b > /etc/agent");
    check(sh(f, &a, "cat /etc/agent").text() == "a\n", "agent a lost its file")?;
    check(
        sh(f, &b, "cat /etc/agent; test -e /usr/local/x/y || echo absent").text() == "b\nabsent\n",
        "agent b saw agent a",
    )?;
    check(!f.rootfs().join("etc/agent").exists(), "image was modified")
}

fn workspace_layer_is_copy_on_write(f: &Fixture) -> Result<(), Failed> {
    let project = f.host_dir("project");
    fs::write(project.join("main.rs"), "fn main() {}\n").unwrap();
    let mut spec = f.spec();
    spec.layers.push(Layer { lower: project.clone(), target: "/workspace".into() });
    spec.cwd = "/workspace".into();
    let sandbox = f.sandbox("workspace", spec);
    let out = sh(f, &sandbox, "cat main.rs; echo changed > main.rs; echo new > added; cat main.rs");
    check(out.text() == "fn main() {}\nchanged\n", format!("{out:?}"))?;
    check(fs::read_to_string(project.join("main.rs")).unwrap() == "fn main() {}\n", "project changed")?;
    check(!project.join("added").exists(), "project gained a file")?;
    let upper = sandbox.layer_upper(0);
    check(fs::read_to_string(upper.join("added")).unwrap() == "new\n", "change missing from the layer")
}

fn read_only_bind_is_live_and_immutable(f: &Fixture) -> Result<(), Failed> {
    let records = f.host_dir("records");
    fs::write(records.join("transcript.jsonl"), "one\n").unwrap();
    let mut spec = f.spec();
    spec.binds.push(Bind { source: records.clone(), target: "/workspace/.ae".into(), writable: false });
    let sandbox = f.sandbox("bind", spec);
    check(sh(f, &sandbox, "cat /workspace/.ae/transcript.jsonl").text() == "one\n", "bind not visible")?;
    fs::write(records.join("transcript.jsonl"), "one\ntwo\n").unwrap();
    let out = sh(
        f,
        &sandbox,
        "cat /workspace/.ae/transcript.jsonl; (echo x > /workspace/.ae/new) 2>/dev/null || echo denied",
    );
    check(out.text() == "one\ntwo\ndenied\n", format!("{out:?}"))?;
    check(!records.join("new").exists(), "bind was writable")
}

fn agents_cannot_see_other_processes(f: &Fixture) -> Result<(), Failed> {
    let a = f.sandbox("pids-a", f.spec());
    let b = f.sandbox("pids-b", f.spec());
    sh(f, &a, "sleep 300 >/dev/null 2>&1 &");
    let out = sh(f, &b, &format!("{SLEEPERS}; ls /proc | grep -cE '^[0-9]+$'"));
    let text = out.text();
    let lines: Vec<&str> = text.lines().collect();
    let visible: usize = lines.get(1).and_then(|n| n.parse().ok()).unwrap_or(usize::MAX);
    let result = check(lines.first() == Some(&"0") && visible < 8, format!("visible processes: {out:?}"));
    block_on(a.shutdown(true));
    result
}

fn each_agent_has_its_own_loopback_network(f: &Fixture) -> Result<(), Failed> {
    let a = f.sandbox("net-a", f.spec());
    let b = f.sandbox("net-b", f.spec());
    let interfaces = sh(f, &a, "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' '");
    check(interfaces.text() == "lo\n", format!("interfaces: {interfaces:?}"))?;
    let up = sh(
        f,
        &a,
        "cat /sys/class/net/lo/operstate; (exec 3<>/dev/tcp/1.1.1.1/53) 2>/dev/null && echo online || echo offline",
    );
    check(up.text() == "unknown\noffline\n", format!("{up:?}"))?;
    let ns = |s| sh(f, s, "readlink /proc/self/ns/net").text();
    check(ns(&a) != ns(&b), "agents share a network namespace")
}

fn memory_limit_kills_the_runaway_process(f: &Fixture) -> Result<(), Failed> {
    let mut spec = f.spec();
    spec.limits = Limits { memory_bytes: Some(64 << 20), ..spec.limits };
    let sandbox = f.sandbox("memory", spec);
    // The shell running the pipeline may be an OOM victim too, so judge the command itself.
    let out = sh(f, &sandbox, "head -c 512M /dev/zero | tail > /dev/null && echo completed");
    check(!out.text().contains("completed") && out.status.code() == 137, format!("{out:?}"))?;
    check(sh(f, &sandbox, "echo alive").text() == "alive\n", "sandbox died with the process")
}

fn pid_limit_bounds_process_creation(f: &Fixture) -> Result<(), Failed> {
    let mut spec = f.spec();
    spec.limits = Limits { pids: Some(16), ..spec.limits };
    let sandbox = f.sandbox("pids-limit", spec);
    // The kernel counts every fork the limit refuses in pids.events. timeout kills the
    // whole process group, so bash's fork retries end quickly.
    let out = sh(
        f,
        &sandbox,
        "cat /sys/fs/cgroup/pids.max; { timeout -s KILL 1 bash -c 'for i in $(seq 40); do sleep 30 & done'; } 2>/dev/null; \
         grep -c '^max [1-9]' /sys/fs/cgroup/pids.events",
    );
    block_on(sandbox.shutdown(true));
    check(out.text() == "16\n1\n", format!("{out:?}"))
}

fn namespace_and_mount_escapes_are_denied(f: &Fixture) -> Result<(), Failed> {
    let sandbox = f.sandbox("escapes", f.spec());
    let out = sh(
        f,
        &sandbox,
        "unshare -U true 2>/dev/null && echo userns; unshare -n true 2>/dev/null && echo netns; \
         mount -t tmpfs t /mnt 2>/dev/null && echo mount; \
         (echo 1 > /sys/fs/cgroup/memory.max) 2>/dev/null && echo cgroup; grep CapEff /proc/self/status",
    );
    // Docker's default capability set: no SYS_ADMIN, NET_ADMIN, SYS_PTRACE or SYS_MODULE.
    check(out.text() == "CapEff:\t00000000a80425fb\n", format!("escape succeeded: {out:?}"))
}

fn background_processes_keep_the_sandbox_alive(f: &Fixture) -> Result<(), Failed> {
    let sandbox = f.sandbox("background", f.spec());
    sh(f, &sandbox, "(sleep 300 &) ; echo started");
    check(!block_on(sandbox.shutdown(false)), "idle shutdown stopped a background process")?;
    check(sh(f, &sandbox, SLEEPERS).text() == "1\n", "background process vanished")?;
    check(block_on(sandbox.shutdown(true)), "forced shutdown failed")?;
    check(sh(f, &sandbox, SLEEPERS).text() == "0\n", "forced shutdown left processes")
}

fn files_open_inside_the_sandbox_view(f: &Fixture) -> Result<(), Failed> {
    use erisharness::sandbox::OpenMode;
    use std::io::{Read, Write};
    let sandbox = f.sandbox("open", f.spec());
    let mut file: fs::File = block_on(sandbox.open("/root/deep/note.txt", OpenMode::Write { create_parents: true }))
        .map_err(|e| e.to_string())?
        .into();
    file.write_all(b"from harness").unwrap();
    drop(file);
    check(sh(f, &sandbox, "cat /root/deep/note.txt").text() == "from harness", "write not visible")?;
    let mut text = String::new();
    fs::File::from(block_on(sandbox.open("/etc/debian_version", OpenMode::Read)).map_err(|e| e.to_string())?)
        .read_to_string(&mut text)
        .unwrap();
    check(!text.is_empty(), "read failed")?;
    let err = block_on(sandbox.open("/nope/missing", OpenMode::Read)).unwrap_err();
    check(err.raw_os_error() == Some(libc::ENOENT), format!("{err:?}"))?;
    sh(f, &sandbox, "ln -s /../../../../etc/shadow /root/escape");
    let escaped = block_on(sandbox.open("/root/escape", OpenMode::Read)).map(fs::File::from);
    let mut contents = String::new();
    if let Ok(mut file) = escaped {
        file.read_to_string(&mut contents).unwrap();
    }
    check(!contents.contains("eriskii"), "symlink escaped to the host")
}

fn filesystem_persists_across_hibernation(f: &Fixture) -> Result<(), Failed> {
    let sandbox = f.sandbox("hibernate", f.spec());
    sh(f, &sandbox, "echo kept > /root/state");
    check(block_on(sandbox.shutdown(false)), "idle sandbox refused shutdown")?;
    check(!sandbox.is_live(), "sandbox still live")?;
    check(sh(f, &sandbox, "cat /root/state").text() == "kept\n", "state lost")
}

fn missing_working_directory_is_reported(f: &Fixture) -> Result<(), Failed> {
    let mut spec = f.spec();
    spec.cwd = "/does/not/exist".into();
    let sandbox = f.sandbox("cwd", spec);
    let err = block_on(sandbox.spawn(&["true".into()], None)).err().map(|e| e.to_string()).unwrap_or_default();
    check(err.contains("Working directory does not exist: /does/not/exist"), err)
}

fn kill_stops_the_whole_command(f: &Fixture) -> Result<(), Failed> {
    let sandbox = f.sandbox("kill", f.spec());
    let started = std::time::Instant::now();
    let process =
        block_on(sandbox.spawn(&["/bin/bash".into(), "-c".into(), "sleep 100 | cat; sleep 100".into()], None)).unwrap();
    block_on(async {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        process.kill().await;
    });
    let status = block_on(process.wait());
    check(status.signal == Some(libc::SIGKILL), format!("{status:?}"))?;
    check(started.elapsed().as_secs() < 5, "kill did not stop the command")?;
    check(sh(f, &sandbox, SLEEPERS).text() == "0\n", "pipeline members survived")
}
