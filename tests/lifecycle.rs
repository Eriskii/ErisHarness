//! An agent's settings, credentials, context size, compaction, retries and removal.

mod common;

use common::model::{Model, Reply, calls, says};
use common::{block_on, direct, harness, items, settle};
use erisharness::agent::{AgentSpec, AgentState, Item, Observation, Usage};
use erisharness::provider::{Authorization, Credentials, ProviderEvent};
use erisharness::{Harness, tools};
use futures_util::future::BoxFuture;
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn spec(cwd: &Path) -> AgentSpec {
    AgentSpec {
        model: "model-one".into(),
        reasoning_effort: Some("high".into()),
        metadata: json!({"name": "worker", "type": "default"}),
        ..direct(cwd)
    }
}

#[test]
fn each_agent_chooses_its_model_and_effort_and_can_change_them() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![says("one"), says("two")]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "first").unwrap();
    settle(&h, &agent, AgentState::Idle);
    h.update_agent(&agent, |spec| {
        spec.model = "model-two".into();
        spec.reasoning_effort = None;
    })
    .unwrap();
    h.send(&agent, "user", "second").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let requests = model.requests();
    assert_eq!(requests[0]["model"], "model-one");
    assert_eq!(requests[0]["reasoning"], json!({"effort": "high", "summary": "auto"}));
    assert_eq!(requests[1]["model"], "model-two");
    assert!(requests[1].get("reasoning").is_none(), "{}", requests[1]);
    assert_eq!(h.agent(&agent).unwrap().spec.model, "model-two");
    let error = h.update_agent(&agent, |spec| spec.provider = "missing".into()).unwrap_err();
    assert!(error.to_string().contains("unknown provider missing"), "{error}");
    assert_eq!(h.agent(&agent).unwrap().spec.provider, "test");
}

struct Account;

impl Credentials for Account {
    fn authorize(&self) -> BoxFuture<'_, anyhow::Result<Authorization>> {
        Box::pin(async {
            Ok(Authorization {
                token: "account-token".into(),
                headers: vec![("chatgpt-account-id".into(), "acct-1".into())],
            })
        })
    }
}

#[test]
fn credentials_supply_headers_and_requests_carry_a_cache_key() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![says("hi")]);
    let h = block_on(
        Harness::builder(temp.path().join("state"))
            .provider("test", model.provider_with(Arc::new(Account)))
            .tools(tools::builtin())
            .open(),
    )
    .unwrap();
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "hello").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let script = model.script.lock().unwrap();
    assert_eq!(script.auth, ["Bearer account-token"]);
    assert_eq!(script.headers[0]["chatgpt-account-id"], "acct-1");
    assert_eq!(script.headers[0]["x-test"], "1");
    assert_eq!(script.requests[0]["prompt_cache_key"], json!(agent));
}

#[test]
fn the_record_tracks_the_latest_context_size() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "true"})), says("done")]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(spec(temp.path())).unwrap();
    assert_eq!(h.agent(&agent).unwrap().context_tokens, 0);
    h.send(&agent, "user", "go").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let record = h.agent(&agent).unwrap();
    assert_eq!(record.context_tokens, 10);
    assert_eq!(record.usage.input, 20);
}

#[test]
fn a_full_context_is_compacted_into_a_summary_and_the_turn_continues() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![
        calls("c1", "bash", json!({"command": "echo lots of output"})),
        says("The user asked for output; bash printed it."),
        says("Finished after compaction."),
    ]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(AgentSpec { context_window: Some(12), ..spec(temp.path()) }).unwrap();
    h.send(&agent, "user", "print things").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    let compaction = &requests[1];
    // The same tools, so the summary reads the context from the prompt cache.
    assert_eq!(compaction["tools"], requests[0]["tools"]);
    let instruction = compaction["input"].as_array().unwrap().last().unwrap()["content"][0]["text"].clone();
    assert!(instruction.as_str().unwrap().contains("summary"), "{instruction}");
    assert_eq!(
        requests[2]["input"],
        json!([{"role": "user", "content": [{"type": "input_text", "text":
            "The conversation so far was compacted. Summary:\n\nThe user asked for output; bash printed it.\n\nContinue from here."}]}])
    );
    let items = items(&h, &agent);
    assert!(
        matches!(&items[..], [
            Item::Input { .. },
            Item::ToolCall { .. },
            Item::ToolResult { .. },
            Item::Compaction { summary },
            Item::Assistant { text },
        ] if summary == "The user asked for output; bash printed it." && text == "Finished after compaction."),
        "{items:#?}"
    );
}

#[test]
fn a_compaction_without_a_summary_leaves_the_context_whole() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![
        calls("c1", "bash", json!({"command": "true"})),
        calls("c2", "bash", json!({"command": "echo instead of a summary"})),
        says("carried on"),
    ]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(AgentSpec { context_window: Some(12), ..spec(temp.path()) }).unwrap();
    h.send(&agent, "user", "go").unwrap();
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(model.requests().len(), 3);
    let items = items(&h, &agent);
    assert!(
        matches!(&items[..], [
            Item::Input { .. },
            Item::ToolCall { .. },
            Item::ToolResult { .. },
            Item::Assistant { text },
        ] if text == "carried on"),
        "{items:#?}"
    );
}

#[test]
fn each_model_call_reports_its_usage() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "true"})), says("done")]);
    let h = harness(&temp.path().join("state"), &model);
    let mut observations = h.subscribe();
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "go").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let mut calls = Vec::new();
    while let Ok(observation) = observations.try_recv() {
        if let Observation::Usage { usage, .. } = observation {
            calls.push(usage);
        }
    }
    let each = Usage { input: 10, cached_input: 4, cache_write: 0, output: 5, reasoning: 1 };
    assert_eq!(calls, [each, each]);
    assert_eq!(
        h.agent(&agent).unwrap().usage,
        Usage { input: 20, cached_input: 8, cache_write: 0, output: 10, reasoning: 2 }
    );
}

#[test]
fn a_database_without_cache_write_totals_gains_them() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("state");
    std::fs::create_dir_all(dir.join("transcripts/old")).unwrap();
    let db = rusqlite::Connection::open(dir.join("harness.db")).unwrap();
    db.execute_batch(
        "CREATE TABLE agents (id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL, error TEXT,
            held INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0, context_tokens INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL);",
    )
    .unwrap();
    let spec = serde_json::to_string(&spec(temp.path())).unwrap();
    db.execute(
        "INSERT INTO agents (id, spec, state, input_tokens, cached_tokens, created_at) VALUES ('old', ?1, 'idle', 7, 3, 0)",
        [spec],
    )
    .unwrap();
    drop(db);
    let h = harness(&dir, &Model::start(vec![]));
    assert_eq!(h.agent("old").unwrap().usage, Usage { input: 7, cached_input: 3, ..Usage::default() });
}

#[test]
fn retries_are_reported_to_observers() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![Reply::Status(503, vec![("retry-after-ms", "1")]), says("ok")]);
    let h = harness(&temp.path().join("state"), &model);
    let mut observations = h.subscribe();
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "go").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let mut retries = Vec::new();
    while let Ok(observation) = observations.try_recv() {
        if let Observation::Provider { event: event @ ProviderEvent::Retry { .. }, .. } = observation {
            retries.push(event);
        }
    }
    assert!(
        matches!(&retries[..], [ProviderEvent::Retry { attempt: 1, status: Some(503), delay_ms: 1, .. }]),
        "{retries:?}"
    );
}

#[test]
fn removing_an_agent_stops_it_and_deletes_its_state() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "sleep 30"}))]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "sleep").unwrap();
    settle(&h, &agent, AgentState::Running);
    let transcript = h.transcript_path(&agent);
    assert!(transcript.exists());
    block_on(async { tokio::time::timeout(Duration::from_secs(10), h.remove_agent(&agent)).await }).unwrap().unwrap();
    assert!(!transcript.parent().unwrap().exists());
    assert!(h.agent(&agent).is_err());
    assert!(h.send(&agent, "user", "hello").is_err());
    let others = h.create_agent(spec(temp.path())).unwrap();
    assert!(h.agent(&others).is_ok());
}

#[test]
fn reading_a_transcript_changes_nothing_on_disk() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(spec(temp.path())).unwrap();
    assert!(h.transcript(&agent).unwrap().is_empty());
    assert!(!h.transcript_path(&agent).exists(), "a read creates no transcript");
    assert!(h.transcript("no-such-agent").is_err());
}

/// Hosts read transcripts and mail agents at any moment, also while one is being removed. The
/// removal still succeeds, leaves nothing on disk, and no turn starts for the removed agent.
#[test]
fn removing_agents_while_others_read_and_mail_them_always_completes() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![]);
    let h = harness(&temp.path().join("state"), &model);
    for round in 0..40 {
        model.push(calls(&format!("c{round}"), "bash", json!({"command": "sleep 30"})));
        let agent = h.create_agent(spec(temp.path())).unwrap();
        h.send(&agent, "user", "sleep").unwrap();
        settle(&h, &agent, AgentState::Running);
        let asked = model.requests().len();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let busy: Vec<_> = (0..2)
            .map(|n| {
                let (h, agent, done) = (h.clone(), agent.clone(), done.clone());
                std::thread::spawn(move || {
                    while !done.load(std::sync::atomic::Ordering::Relaxed) {
                        let _ = if n == 0 {
                            h.transcript(&agent).map(drop)
                        } else {
                            h.send(&agent, "peer", "hi").map(drop)
                        };
                    }
                })
            })
            .collect();
        let removed = block_on(async { tokio::time::timeout(Duration::from_secs(10), h.remove_agent(&agent)).await });
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        busy.into_iter().for_each(|t| t.join().unwrap());
        removed.unwrap().unwrap_or_else(|e| panic!("round {round}: {e:#}"));
        assert!(!h.transcript_path(&agent).parent().unwrap().exists(), "round {round}");
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(model.requests().len(), asked, "round {round}: a removed agent started a turn");
    }
}

#[test]
fn agents_are_listed_with_the_metadata_their_host_gave_them() {
    let temp = tempfile::tempdir().unwrap();
    let h = harness(&temp.path().join("state"), &Model::start(vec![]));
    let first = h.create_agent(spec(temp.path())).unwrap();
    let second = h
        .create_agent(AgentSpec { metadata: json!({"name": "reviewer", "parent": first}), ..spec(temp.path()) })
        .unwrap();
    let agents = h.agents().unwrap();
    assert_eq!(agents.iter().map(|a| a.id.clone()).collect::<Vec<_>>(), [first.clone(), second.clone()]);
    assert_eq!(agents[1].spec.metadata, json!({"name": "reviewer", "parent": first}));
    h.update_agent(&second, |spec| spec.metadata["name"] = json!("renamed")).unwrap();
    assert_eq!(h.agent(&second).unwrap().spec.metadata["name"], "renamed");
}
