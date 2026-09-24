//! The built-in tools inside a real sandbox. Model-facing text matches Pi 0.87 exactly.

mod common;

use common::{Fixture, block_on};
use erisharness::machine::{Direct, DirectSpec, Machine};
use erisharness::tools::{Content, Tool, ToolContext, ToolOutput, builtin};
use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Every tool test runs on both machines: model-facing behavior must not depend on isolation.
fn main() {
    let host = erissandbox::bootstrap().expect("bootstrap");
    let fixture = Arc::new(Fixture::new(host));
    let tests: &[(&str, Test)] = &[
        ("read_returns_the_file", read_returns_the_file),
        ("read_pages_with_offset_and_limit", read_pages_with_offset_and_limit),
        ("read_truncates_long_files", read_truncates_long_files),
        ("read_reports_a_huge_first_line", read_reports_a_huge_first_line),
        ("read_errors_like_node", read_errors_like_node),
        ("read_resolves_paths_like_pi", read_resolves_paths_like_pi),
        ("read_returns_images_as_attachments", read_returns_images_as_attachments),
        ("write_creates_parents", write_creates_parents),
        ("edit_replaces_blocks_preserving_endings", edit_replaces_blocks_preserving_endings),
        ("edit_accepts_legacy_argument_shapes", edit_accepts_legacy_argument_shapes),
        ("edit_reports_missing_files", edit_reports_missing_files),
        ("bash_returns_output", bash_returns_output),
        ("bash_reports_failures", bash_reports_failures),
        ("bash_times_out", bash_times_out),
        ("bash_is_cancellable", bash_is_cancellable),
        ("bash_keeps_the_tail_and_saves_full_output", bash_keeps_the_tail_and_saves_full_output),
        ("tools_describe_themselves", tools_describe_themselves),
    ];
    let trials = [Mode::Sandbox, Mode::Direct]
        .into_iter()
        .flat_map(|mode| {
            let fixture = fixture.clone();
            tests.iter().map(move |&(name, test)| {
                let fixture = fixture.clone();
                Trial::test(format!("{mode:?}::{name}"), move || test(&Agent::new(&fixture, mode, name)))
            })
        })
        .collect();
    libtest_mimic::run(&Arguments::from_args(), trials).exit();
}

type Test = fn(&Agent) -> Result<(), Failed>;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Sandbox,
    Direct,
}

struct NoMail;

impl erisharness::tools::Mailbox for NoMail {
    fn send(&self, _: &str, _: &str, _: &str) -> anyhow::Result<()> {
        anyhow::bail!("no mail in tool tests")
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
                Arc::new(Direct::new(DirectSpec { cwd: home, env: Some(env) }))
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
        let out = self.call("bash", json!({ "command": script }));
        text(&out)
    }
}

fn text(out: &ToolOutput) -> String {
    out.content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            Content::Image { .. } => None,
        })
        .collect()
}

fn expect(out: &ToolOutput, error: bool, expected: &str) -> Result<(), Failed> {
    if out.is_error == error && text(out) == expected {
        Ok(())
    } else {
        Err(format!("expected error={error} {expected:?}\n     got error={} {:?}", out.is_error, text(out)).into())
    }
}

fn read_returns_the_file(a: &Agent) -> Result<(), Failed> {
    a.sh("printf 'one\\ntwo\\n' > ~/f.txt");
    expect(&a.call("read", json!({"path": a.path("f.txt")})), false, "one\ntwo\n")
}

fn read_pages_with_offset_and_limit(a: &Agent) -> Result<(), Failed> {
    a.sh("seq 1 10 > ~/n.txt");
    expect(
        &a.call("read", json!({"path": a.path("n.txt"), "offset": 3, "limit": 2})),
        false,
        "3\n4\n\n[7 more lines in file. Use offset=5 to continue.]",
    )?;
    expect(&a.call("read", json!({"path": a.path("n.txt"), "offset": 10})), false, "10\n")?;
    expect(
        &a.call("read", json!({"path": a.path("n.txt"), "offset": 20})),
        true,
        "Offset 20 is beyond end of file (11 lines total)",
    )
}

fn read_truncates_long_files(a: &Agent) -> Result<(), Failed> {
    a.sh("seq 1 2500 > ~/long.txt; for i in $(seq 1 30); do head -c 2000 /dev/zero | tr '\\0' x; echo; done > ~/wide.txt");
    let out = a.call("read", json!({"path": a.path("long.txt")}));
    let body = text(&out);
    if !body.ends_with("2000\n\n[Showing lines 1-2000 of 2501. Use offset=2001 to continue.]") {
        return Err(format!("{:?}", &body[body.len().saturating_sub(120)..]).into());
    }
    let out = a.call("read", json!({"path": a.path("wide.txt")}));
    let body = text(&out);
    if !body.ends_with("\n\n[Showing lines 1-25 of 31 (50.0KB limit). Use offset=26 to continue.]") {
        return Err(format!("{:?}", &body[body.len().saturating_sub(120)..]).into());
    }
    Ok(())
}

fn read_reports_a_huge_first_line(a: &Agent) -> Result<(), Failed> {
    a.sh("head -c 61440 /dev/zero | tr '\\0' x > ~/big");
    expect(
        &a.call("read", json!({"path": "big"})),
        false,
        "[Line 1 is 60.0KB, exceeds 50.0KB limit. Use bash: sed -n '1p' big | head -c 51200]",
    )
}

fn read_errors_like_node(a: &Agent) -> Result<(), Failed> {
    expect(
        &a.call("read", json!({"path": "missing.txt"})),
        true,
        &format!("ENOENT: no such file or directory, access '{}'", a.path("missing.txt")),
    )?;
    expect(&a.call("read", json!({"path": "/etc"})), true, "EISDIR: illegal operation on a directory, read")
}

fn read_resolves_paths_like_pi(a: &Agent) -> Result<(), Failed> {
    a.sh("mkdir -p ~/sub && echo hi > ~/sub/x");
    for path in ["sub/x".to_owned(), "~/sub/x".into(), "@sub/x".into(), a.path("sub/../sub/x")] {
        expect(&a.call("read", json!({"path": path})), false, "hi\n")?;
    }
    Ok(())
}

fn read_returns_images_as_attachments(a: &Agent) -> Result<(), Failed> {
    a.sh("printf '\\x89PNG\\r\\n\\x1a\\n\\0\\0\\0\\rIHDR' > ~/i.png");
    let out = a.call("read", json!({"path": "i.png"}));
    let image = out.content.iter().any(|c| matches!(c, Content::Image { mime, .. } if mime == "image/png"));
    if image && text(&out) == "Read image file [image/png]" && !out.is_error {
        Ok(())
    } else {
        Err(format!("{out:?}").into())
    }
}

fn write_creates_parents(a: &Agent) -> Result<(), Failed> {
    expect(
        &a.call("write", json!({"path": "deep/er/new.txt", "content": "made\n"})),
        false,
        "Successfully wrote to deep/er/new.txt",
    )?;
    let seen = a.sh("cat ~/deep/er/new.txt");
    if seen == "made\n" { Ok(()) } else { Err(seen.into()) }
}

fn edit_replaces_blocks_preserving_endings(a: &Agent) -> Result<(), Failed> {
    a.sh("printf '\\xef\\xbb\\xbfalpha\\r\\nbeta\\r\\ngamma\\r\\n' > ~/e.txt");
    let out = a.call(
        "edit",
        json!({"path": "e.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}, {"oldText": "gamma\n", "newText": "GAMMA\nDELTA\n"}]}),
    );
    expect(&out, false, "Successfully replaced 2 block(s) in e.txt.")?;
    let patch = out.details["patch"].as_str().unwrap_or_default();
    if !patch.contains("+GAMMA") {
        return Err(format!("patch: {patch:?}").into());
    }
    let bytes = a.sh("od -An -c ~/e.txt | tr -s ' '");
    let expected = " 357 273 277 A L P H A \\r \\n b e t a \\r \\n\n G A M M A \\r \\n D E L T A \\r \\n\n";
    if bytes == expected { Ok(()) } else { Err(format!("{bytes:?}").into()) }
}

fn edit_accepts_legacy_argument_shapes(a: &Agent) -> Result<(), Failed> {
    a.sh("echo 'one two three' > ~/l.txt");
    expect(
        &a.call("edit", json!({"path": "l.txt", "oldText": "one", "newText": "1"})),
        false,
        "Successfully replaced 1 block(s) in l.txt.",
    )?;
    expect(
        &a.call("edit", json!({"path": "l.txt", "edits": "[{\"oldText\":\"two\",\"newText\":\"2\"}]"})),
        false,
        "Successfully replaced 1 block(s) in l.txt.",
    )?;
    expect(
        &a.call("edit", json!({"path": "l.txt", "edits": {"oldText": "three", "newText": "3"}})),
        false,
        "Successfully replaced 1 block(s) in l.txt.",
    )?;
    let seen = a.sh("cat ~/l.txt");
    if seen == "1 2 3\n" { Ok(()) } else { Err(seen.into()) }
}

fn edit_reports_missing_files(a: &Agent) -> Result<(), Failed> {
    expect(
        &a.call("edit", json!({"path": "nope.txt", "edits": [{"oldText": "a", "newText": "b"}]})),
        true,
        "Could not edit file: nope.txt. Error code: ENOENT.",
    )?;
    expect(
        &a.call("edit", json!({"path": "nope.txt", "edits": []})),
        true,
        "Edit tool input is invalid. edits must contain at least one replacement.",
    )
}

fn bash_returns_output(a: &Agent) -> Result<(), Failed> {
    expect(&a.call("bash", json!({"command": "echo out; echo err >&2"})), false, "out\nerr\n")?;
    expect(&a.call("bash", json!({"command": "true"})), false, "(no output)")
}

fn bash_reports_failures(a: &Agent) -> Result<(), Failed> {
    expect(&a.call("bash", json!({"command": "echo bad; exit 3"})), true, "bad\n\n\nCommand exited with code 3")?;
    expect(&a.call("bash", json!({"command": "exit 4"})), true, "(no output)\n\nCommand exited with code 4")
}

fn bash_times_out(a: &Agent) -> Result<(), Failed> {
    let started = std::time::Instant::now();
    let out = a.call("bash", json!({"command": "echo start; sleep 30", "timeout": 1}));
    expect(&out, true, "start\n\n\nCommand timed out after 1 seconds")?;
    if started.elapsed() < Duration::from_secs(5) { Ok(()) } else { Err("timeout ignored".into()) }
}

fn bash_is_cancellable(a: &Agent) -> Result<(), Failed> {
    let cancel = a.context.cancel.clone();
    common::runtime().spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
    });
    expect(&a.call("bash", json!({"command": "sleep 30"})), true, "Command aborted")
}

fn bash_keeps_the_tail_and_saves_full_output(a: &Agent) -> Result<(), Failed> {
    let out = a.call("bash", json!({"command": "seq 1 3000"}));
    let body = text(&out);
    let Some((kept, notice)) = body.split_once("\n\n[") else { return Err(body.into()) };
    if !kept.starts_with("1001\n") || !kept.ends_with("\n3000") {
        return Err(format!("kept {:?}..{:?}", &kept[..10], &kept[kept.len() - 10..]).into());
    }
    let Some(path) =
        notice.strip_prefix("Showing lines 1001-3000 of 3000. Full output: ").and_then(|p| p.strip_suffix(']'))
    else {
        return Err(notice.to_owned().into());
    };
    let counted = a.sh(&format!("wc -l < {path}; head -1 {path}"));
    if counted == "3000\n1\n" { Ok(()) } else { Err(counted.into()) }
}

fn tools_describe_themselves(_: &Agent) -> Result<(), Failed> {
    let names: Vec<String> = builtin().iter().map(|t| t.name().to_owned()).collect();
    if names != ["read", "bash", "edit", "write", "send_message"] {
        return Err(format!("{names:?}").into());
    }
    for tool in builtin() {
        let schema = tool.parameters();
        if schema["type"] != "object" || tool.description().is_empty() {
            return Err(format!("{} schema {schema}", tool.name()).into());
        }
    }
    Ok(())
}
