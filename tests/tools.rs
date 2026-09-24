//! The built-in tools inside a real sandbox. Model-facing text matches Pi 0.87 exactly.

mod common;

use common::{Fixture, block_on};
use erisharness::tools::{Content, Tool, ToolContext, ToolOutput, builtin};
use libtest_mimic::Failed;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn main() {
    common::run(&[
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
    ]);
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
    fn new(f: &Fixture, name: &str) -> Self {
        let context = ToolContext {
            agent: name.into(),
            sandbox: f.sandbox(name, f.spec()),
            cancel: CancellationToken::new(),
            mailbox: Arc::new(NoMail),
        };
        Self { context, tools: builtin() }
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

fn read_returns_the_file(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-file");
    a.sh("printf 'one\\ntwo\\n' > /root/f.txt");
    expect(&a.call("read", json!({"path": "/root/f.txt"})), false, "one\ntwo\n")
}

fn read_pages_with_offset_and_limit(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-page");
    a.sh("seq 1 10 > /root/n.txt");
    expect(
        &a.call("read", json!({"path": "/root/n.txt", "offset": 3, "limit": 2})),
        false,
        "3\n4\n\n[7 more lines in file. Use offset=5 to continue.]",
    )?;
    expect(&a.call("read", json!({"path": "/root/n.txt", "offset": 10})), false, "10\n")?;
    expect(
        &a.call("read", json!({"path": "/root/n.txt", "offset": 20})),
        true,
        "Offset 20 is beyond end of file (11 lines total)",
    )
}

fn read_truncates_long_files(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-long");
    a.sh("seq 1 2500 > /root/long.txt; for i in $(seq 1 30); do head -c 2000 /dev/zero | tr '\\0' x; echo; done > /root/wide.txt");
    let out = a.call("read", json!({"path": "/root/long.txt"}));
    let body = text(&out);
    if !body.ends_with("2000\n\n[Showing lines 1-2000 of 2501. Use offset=2001 to continue.]") {
        return Err(format!("{:?}", &body[body.len().saturating_sub(120)..]).into());
    }
    let out = a.call("read", json!({"path": "/root/wide.txt"}));
    let body = text(&out);
    if !body.ends_with("\n\n[Showing lines 1-25 of 31 (50.0KB limit). Use offset=26 to continue.]") {
        return Err(format!("{:?}", &body[body.len().saturating_sub(120)..]).into());
    }
    Ok(())
}

fn read_reports_a_huge_first_line(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-huge");
    a.sh("head -c 61440 /dev/zero | tr '\\0' x > /root/big");
    expect(
        &a.call("read", json!({"path": "big"})),
        false,
        "[Line 1 is 60.0KB, exceeds 50.0KB limit. Use bash: sed -n '1p' big | head -c 51200]",
    )
}

fn read_errors_like_node(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-errors");
    expect(
        &a.call("read", json!({"path": "missing.txt"})),
        true,
        "ENOENT: no such file or directory, access '/root/missing.txt'",
    )?;
    expect(&a.call("read", json!({"path": "/etc"})), true, "EISDIR: illegal operation on a directory, read")
}

fn read_resolves_paths_like_pi(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-paths");
    a.sh("mkdir -p /root/sub && echo hi > /root/sub/x");
    for path in ["sub/x", "~/sub/x", "@sub/x", "/root/sub/../sub/x"] {
        expect(&a.call("read", json!({"path": path})), false, "hi\n")?;
    }
    Ok(())
}

fn read_returns_images_as_attachments(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "read-image");
    a.sh("printf '\\x89PNG\\r\\n\\x1a\\n\\0\\0\\0\\rIHDR' > /root/i.png");
    let out = a.call("read", json!({"path": "i.png"}));
    let image = out.content.iter().any(|c| matches!(c, Content::Image { mime, .. } if mime == "image/png"));
    if image && text(&out) == "Read image file [image/png]" && !out.is_error {
        Ok(())
    } else {
        Err(format!("{out:?}").into())
    }
}

fn write_creates_parents(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "write");
    expect(
        &a.call("write", json!({"path": "deep/er/new.txt", "content": "made\n"})),
        false,
        "Successfully wrote to deep/er/new.txt",
    )?;
    let seen = a.sh("cat /root/deep/er/new.txt");
    if seen == "made\n" { Ok(()) } else { Err(seen.into()) }
}

fn edit_replaces_blocks_preserving_endings(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "edit");
    a.sh("printf '\\xef\\xbb\\xbfalpha\\r\\nbeta\\r\\ngamma\\r\\n' > /root/e.txt");
    let out = a.call(
        "edit",
        json!({"path": "e.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}, {"oldText": "gamma\n", "newText": "GAMMA\nDELTA\n"}]}),
    );
    expect(&out, false, "Successfully replaced 2 block(s) in e.txt.")?;
    let patch = out.details["patch"].as_str().unwrap_or_default();
    if !patch.contains("+GAMMA") {
        return Err(format!("patch: {patch:?}").into());
    }
    let bytes = a.sh("od -An -c /root/e.txt | tr -s ' '");
    let expected = " 357 273 277 A L P H A \\r \\n b e t a \\r \\n\n G A M M A \\r \\n D E L T A \\r \\n\n";
    if bytes == expected { Ok(()) } else { Err(format!("{bytes:?}").into()) }
}

fn edit_accepts_legacy_argument_shapes(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "edit-legacy");
    a.sh("echo 'one two three' > /root/l.txt");
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
    let seen = a.sh("cat /root/l.txt");
    if seen == "1 2 3\n" { Ok(()) } else { Err(seen.into()) }
}

fn edit_reports_missing_files(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "edit-missing");
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

fn bash_returns_output(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "bash-output");
    expect(&a.call("bash", json!({"command": "echo out; echo err >&2"})), false, "out\nerr\n")?;
    expect(&a.call("bash", json!({"command": "true"})), false, "(no output)")
}

fn bash_reports_failures(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "bash-fail");
    expect(&a.call("bash", json!({"command": "echo bad; exit 3"})), true, "bad\n\n\nCommand exited with code 3")?;
    expect(&a.call("bash", json!({"command": "exit 4"})), true, "(no output)\n\nCommand exited with code 4")
}

fn bash_times_out(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "bash-timeout");
    let started = std::time::Instant::now();
    let out = a.call("bash", json!({"command": "echo start; sleep 30", "timeout": 1}));
    expect(&out, true, "start\n\n\nCommand timed out after 1 seconds")?;
    if started.elapsed() < Duration::from_secs(5) { Ok(()) } else { Err("timeout ignored".into()) }
}

fn bash_is_cancellable(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "bash-cancel");
    let cancel = a.context.cancel.clone();
    common::runtime().spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
    });
    expect(&a.call("bash", json!({"command": "sleep 30"})), true, "Command aborted")
}

fn bash_keeps_the_tail_and_saves_full_output(f: &Fixture) -> Result<(), Failed> {
    let a = Agent::new(f, "bash-long");
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

fn tools_describe_themselves(_: &Fixture) -> Result<(), Failed> {
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
