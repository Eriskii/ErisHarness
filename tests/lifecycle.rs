//! An agent's settings, credentials, context size, compaction and removal.

mod common;

use common::block_on;
use common::model::{Model, calls, says};
use erisharness::agent::{AgentSpec, AgentState, Item};
use erisharness::machine::{DirectSpec, MachineSpec};
use erisharness::provider::{Authorization, Credentials};
use erisharness::{Harness, tools};
use futures_util::future::BoxFuture;
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn spec(cwd: &Path) -> AgentSpec {
    AgentSpec {
        system_prompt: "Lifecycle agent.".into(),
        tools: vec!["bash".into()],
        provider: "test".into(),
        model: "model-one".into(),
        reasoning_effort: Some("high".into()),
        context_window: None,
        metadata: json!({"name": "worker", "type": "default"}),
        machine: MachineSpec::Direct(DirectSpec { cwd: cwd.to_str().unwrap().into(), env: None }),
    }
}

fn harness(dir: &Path, model: &Model) -> Arc<Harness> {
    block_on(Harness::builder(dir).provider("test", model.provider()).tools(tools::builtin()).open()).unwrap()
}

fn settle(h: &Harness, agent: &str) {
    block_on(async { tokio::time::timeout(Duration::from_secs(20), h.wait_for(agent, AgentState::Idle)).await })
        .expect("agent settled");
}

#[test]
fn each_agent_chooses_its_model_and_effort_and_can_change_them() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![says("one"), says("two")]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "first").unwrap();
    settle(&h, &agent);
    h.update_agent(&agent, |spec| {
        spec.model = "model-two".into();
        spec.reasoning_effort = None;
    })
    .unwrap();
    h.send(&agent, "user", "second").unwrap();
    settle(&h, &agent);
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
    settle(&h, &agent);
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
    settle(&h, &agent);
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
    settle(&h, &agent);
    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    let compaction = &requests[1];
    assert_eq!(compaction["tools"], json!([]));
    let instruction = compaction["input"].as_array().unwrap().last().unwrap()["content"][0]["text"].clone();
    assert!(instruction.as_str().unwrap().contains("summary"), "{instruction}");
    assert_eq!(
        requests[2]["input"],
        json!([{"role": "user", "content": [{"type": "input_text", "text":
            "The conversation so far was compacted. Summary:\n\nThe user asked for output; bash printed it.\n\nContinue from here."}]}])
    );
    let items: Vec<Item> = h.transcript(&agent).unwrap().into_iter().map(|e| e.item).collect();
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
fn removing_an_agent_stops_it_and_deletes_its_state() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "sleep 30"}))]);
    let h = harness(&temp.path().join("state"), &model);
    let agent = h.create_agent(spec(temp.path())).unwrap();
    h.send(&agent, "user", "sleep").unwrap();
    block_on(async { tokio::time::timeout(Duration::from_secs(10), h.wait_for(&agent, AgentState::Running)).await })
        .unwrap();
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
fn agents_are_listed_with_the_metadata_their_host_gave_them() {
    let temp = tempfile::tempdir().unwrap();
    let model = Model::start(vec![]);
    let h = harness(&temp.path().join("state"), &model);
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
