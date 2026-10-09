//! The built-in tools on both machines, a real sandbox and this one directly: what the model
//! sees must not depend on isolation. Model-facing text matches Pi 0.87 exactly.

#[macro_use]
mod common;

use common::{Fixture, block_on};
use erisharness::machine::{Direct, DirectSpec, Machine};
use erisharness::tools::{Content, Mailbox, Reply, Tool, ToolContext, ToolOutput, builtin};
use futures_util::future::BoxFuture;
use libtest_mimic::{Arguments, Trial};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn main() {
    let host = erissandbox::bootstrap().expect("bootstrap");
    let fixture = Arc::new(Fixture::new(host));
    let tests: &[(&str, Test)] = named![
        read_returns_the_file,
        read_pages_with_offset_and_limit,
        read_truncates_long_files,
        read_reports_a_huge_first_line,
        read_errors_like_node,
        read_resolves_paths_like_pi,
        read_returns_images_as_attachments,
        write_creates_parents,
        edit_replaces_blocks_preserving_endings,
        edit_accepts_legacy_argument_shapes,
        edit_reports_missing_files,
        bash_returns_output,
        bash_reports_failures,
        bash_times_out,
        bash_is_cancellable,
        bash_keeps_the_tail_and_saves_full_output,
        tools_describe_themselves,
    ];
    let trials = [Mode::Sandbox, Mode::Direct]
        .into_iter()
        .flat_map(|mode| {
            let fixture = fixture.clone();
            tests.iter().map(move |&(name, test)| {
                let fixture = fixture.clone();
                Trial::test(format!("{mode:?}::{name}"), move || {
                    test(&Agent::new(&fixture, mode, name));
                    Ok(())
                })
            })
        })
        .collect();
    libtest_mimic::run(&Arguments::from_args(), trials).exit();
}

type Test = fn(&Agent);

#[derive(Clone, Copy, Debug)]
enum Mode {
    Sandbox,
    Direct,
}

struct NoMail;

impl Mailbox for NoMail {
    fn send(&self, _: &str, _: &str, _: &str) -> anyhow::Result<i64> {
        anyhow::bail!("no mail in tool tests")
    }

    fn reply<'a>(&'a self, _: &'a str, _: &'a str, _: i64) -> BoxFuture<'a, anyhow::Result<Reply>> {
        Box::pin(async { anyhow::bail!("no mail in tool tests") })
    }
}

struct Agent {
    context: ToolContext,
    tools: Vec<Arc<dyn Tool>>,
}

impl Agent {
    fn new(f: &Fixture, mode: Mode, name: &str) -> Self {
        let machine: Arc<dyn Machine> = match mode {
            Mode::Sandbox => f.sandbox(name, f.spec()),
            Mode::Direct => {
                let home = f.host_dir(&format!("direct-{name}"));
                let home = home.to_str().unwrap().to_owned();
                let env = vec![("HOME".to_owned(), home.clone()), ("PATH".to_owned(), std::env::var("PATH").unwrap())];
                Arc::new(Direct::outside(DirectSpec { cwd: home, env: Some(env) }, f.host.clone()))
            }
        };
        let context =
            ToolContext { agent: name.into(), machine, cancel: CancellationToken::new(), mailbox: Arc::new(NoMail) };
        Self { context, tools: builtin() }
    }

    /// An absolute path under the machine's home directory.
    fn path(&self, relative: &str) -> String {
        format!("{}/{relative}", self.context.machine.home())
    }

    fn call(&self, tool: &str, args: Value) -> ToolOutput {
        let tool = self.tools.iter().find(|t| t.name() == tool).expect("tool");
        block_on(tool.call(&self.context, args))
    }

    fn sh(&self, script: &str) -> String {
        text(&self.call("bash", json!({ "command": script })))
    }
}

fn text(out: &ToolOutput) -> String {
    common::text(&out.content)
}

#[track_caller]
fn expect(out: &ToolOutput, error: bool, expected: &str) {
    assert_eq!((out.is_error, text(out).as_str()), (error, expected));
}

fn read_returns_the_file(a: &Agent) {
    a.sh("printf 'one\\ntwo\\n' > ~/f.txt");
    expect(&a.call("read", json!({"path": a.path("f.txt")})), false, "one\ntwo\n");
}

fn read_pages_with_offset_and_limit(a: &Agent) {
    a.sh("seq 1 10 > ~/n.txt");
    expect(
        &a.call("read", json!({"path": a.path("n.txt"), "offset": 3, "limit": 2})),
        false,
        "3\n4\n\n[7 more lines in file. Use offset=5 to continue.]",
    );
    expect(&a.call("read", json!({"path": a.path("n.txt"), "offset": 10})), false, "10\n");
    expect(
        &a.call("read", json!({"path": a.path("n.txt"), "offset": 20})),
        true,
        "Offset 20 is beyond end of file (11 lines total)",
    );
}

fn read_truncates_long_files(a: &Agent) {
    a.sh("seq 1 2500 > ~/long.txt; for i in $(seq 1 30); do head -c 2000 /dev/zero | tr '\\0' x; echo; done > ~/wide.txt");
    let long = text(&a.call("read", json!({"path": a.path("long.txt")})));
    assert!(long.ends_with("2000\n\n[Showing lines 1-2000 of 2501. Use offset=2001 to continue.]"), "{long}");
    let wide = text(&a.call("read", json!({"path": a.path("wide.txt")})));
    assert!(wide.ends_with("\n\n[Showing lines 1-25 of 31 (50.0KB limit). Use offset=26 to continue.]"), "{wide}");
}

fn read_reports_a_huge_first_line(a: &Agent) {
    a.sh("head -c 61440 /dev/zero | tr '\\0' x > ~/big");
    expect(
        &a.call("read", json!({"path": "big"})),
        false,
        "[Line 1 is 60.0KB, exceeds 50.0KB limit. Use bash: sed -n '1p' big | head -c 51200]",
    );
}

fn read_errors_like_node(a: &Agent) {
    expect(
        &a.call("read", json!({"path": "missing.txt"})),
        true,
        &format!("ENOENT: no such file or directory, access '{}'", a.path("missing.txt")),
    );
    expect(&a.call("read", json!({"path": "/etc"})), true, "EISDIR: illegal operation on a directory, read");
}

fn read_resolves_paths_like_pi(a: &Agent) {
    a.sh("mkdir -p ~/sub && echo hi > ~/sub/x");
    for path in ["sub/x".to_owned(), "~/sub/x".into(), "@sub/x".into(), a.path("sub/../sub/x")] {
        expect(&a.call("read", json!({"path": path})), false, "hi\n");
    }
}

fn read_returns_images_as_attachments(a: &Agent) {
    a.sh("printf '\\x89PNG\\r\\n\\x1a\\n\\0\\0\\0\\rIHDR' > ~/i.png");
    let out = a.call("read", json!({"path": "i.png"}));
    expect(&out, false, "Read image file [image/png]");
    assert!(out.content.iter().any(|c| matches!(c, Content::Image { mime, .. } if mime == "image/png")), "{out:?}");
}

fn write_creates_parents(a: &Agent) {
    expect(
        &a.call("write", json!({"path": "deep/er/new.txt", "content": "made\n"})),
        false,
        "Successfully wrote to deep/er/new.txt",
    );
    assert_eq!(a.sh("cat ~/deep/er/new.txt"), "made\n");
}

fn edit_replaces_blocks_preserving_endings(a: &Agent) {
    a.sh("printf '\\xef\\xbb\\xbfalpha\\r\\nbeta\\r\\ngamma\\r\\n' > ~/e.txt");
    let out = a.call(
        "edit",
        json!({"path": "e.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}, {"oldText": "gamma\n", "newText": "GAMMA\nDELTA\n"}]}),
    );
    expect(&out, false, "Successfully replaced 2 block(s) in e.txt.");
    let patch = out.details["patch"].as_str().unwrap_or_default();
    assert!(patch.contains("+GAMMA"), "{patch}");
    assert_eq!(
        a.sh("od -An -c ~/e.txt | tr -s ' '"),
        " 357 273 277 A L P H A \\r \\n b e t a \\r \\n\n G A M M A \\r \\n D E L T A \\r \\n\n"
    );
}

fn edit_accepts_legacy_argument_shapes(a: &Agent) {
    a.sh("echo 'one two three' > ~/l.txt");
    for args in [
        json!({"path": "l.txt", "oldText": "one", "newText": "1"}),
        json!({"path": "l.txt", "edits": "[{\"oldText\":\"two\",\"newText\":\"2\"}]"}),
        json!({"path": "l.txt", "edits": {"oldText": "three", "newText": "3"}}),
    ] {
        expect(&a.call("edit", args), false, "Successfully replaced 1 block(s) in l.txt.");
    }
    assert_eq!(a.sh("cat ~/l.txt"), "1 2 3\n");
}

fn edit_reports_missing_files(a: &Agent) {
    expect(
        &a.call("edit", json!({"path": "nope.txt", "edits": [{"oldText": "a", "newText": "b"}]})),
        true,
        "Could not edit file: nope.txt. Error code: ENOENT.",
    );
    expect(
        &a.call("edit", json!({"path": "nope.txt", "edits": []})),
        true,
        "Edit tool input is invalid. edits must contain at least one replacement.",
    );
}

fn bash_returns_output(a: &Agent) {
    expect(&a.call("bash", json!({"command": "echo out; echo err >&2"})), false, "out\nerr\n");
    expect(&a.call("bash", json!({"command": "true"})), false, "(no output)");
}

fn bash_reports_failures(a: &Agent) {
    expect(&a.call("bash", json!({"command": "echo bad; exit 3"})), true, "bad\n\n\nCommand exited with code 3");
    expect(&a.call("bash", json!({"command": "exit 4"})), true, "(no output)\n\nCommand exited with code 4");
}

fn bash_times_out(a: &Agent) {
    let started = std::time::Instant::now();
    let out = a.call("bash", json!({"command": "echo start; sleep 30", "timeout": 1}));
    expect(&out, true, "start\n\n\nCommand timed out after 1 seconds");
    assert!(started.elapsed() < Duration::from_secs(5), "timeout ignored");
}

fn bash_is_cancellable(a: &Agent) {
    let cancel = a.context.cancel.clone();
    common::runtime().spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
    });
    expect(&a.call("bash", json!({"command": "sleep 30"})), true, "Command aborted");
}

fn bash_keeps_the_tail_and_saves_full_output(a: &Agent) {
    let body = text(&a.call("bash", json!({"command": "seq 1 3000"})));
    let (kept, notice) = body.split_once("\n\n[").expect("a notice");
    assert!(kept.starts_with("1001\n") && kept.ends_with("\n3000"), "{kept}");
    let path = notice
        .strip_prefix("Showing lines 1001-3000 of 3000. Full output: ")
        .and_then(|p| p.strip_suffix(']'))
        .unwrap_or_else(|| panic!("{notice}"));
    assert_eq!(a.sh(&format!("wc -l < {path}; head -1 {path}")), "3000\n1\n");
}

fn tools_describe_themselves(_: &Agent) {
    let names: Vec<String> = builtin().iter().map(|t| t.name().to_owned()).collect();
    assert_eq!(names, ["read", "bash", "edit", "write", "send_message"]);
    for tool in builtin() {
        let schema = tool.parameters();
        assert!(schema["type"] == "object" && !tool.description().is_empty(), "{} schema {schema}", tool.name());
    }
}
