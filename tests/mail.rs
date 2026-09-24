//! Mail between agents and the host: waiting for replies, and recipients the host serves.

mod common;

use common::block_on;
use common::model::{Model, Reply, calls, says};
use erisharness::agent::{AgentSpec, AgentState, Item};
use erisharness::machine::{DirectSpec, MachineSpec};
use erisharness::{Harness, Recipient, tools};
use serde_json::json;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn spec(cwd: &Path, provider: &str) -> AgentSpec {
    AgentSpec {
        system_prompt: "Mail agent.".into(),
        tools: vec!["send_message".into()],
        provider: provider.into(),
        model: "test-model".into(),
        reasoning_effort: None,
        context_window: None,
        machine: MachineSpec::Direct(DirectSpec { cwd: cwd.to_str().unwrap().into(), env: None }),
    }
}

#[derive(Default)]
struct Outbox(Mutex<Vec<(String, String)>>);

impl Recipient for Outbox {
    fn deliver(&self, from: &str, text: &str) -> anyhow::Result<()> {
        self.0.lock().unwrap().push((from.to_owned(), text.to_owned()));
        Ok(())
    }
}

fn harness(dir: &Path, models: &[(&str, &Model)], outbox: &Arc<Outbox>) -> Arc<Harness> {
    let builder = models.iter().fold(Harness::builder(dir), |b, (name, model)| b.provider(name, model.provider()));
    block_on(builder.tools(tools::builtin()).recipient("user", outbox.clone()).open()).unwrap()
}

fn settle(h: &Harness, agent: &str, state: AgentState) {
    block_on(async { tokio::time::timeout(Duration::from_secs(20), h.wait_for(agent, state)).await })
        .unwrap_or_else(|_| panic!("agent never became {state:?}"));
}

fn results(h: &Harness, agent: &str) -> Vec<(bool, String)> {
    h.transcript(agent)
        .unwrap()
        .into_iter()
        .filter_map(|e| match e.item {
            Item::ToolResult { output, .. } => Some((
                output.is_error,
                output.content.iter().map(|c| if let tools::Content::Text(t) = c { t.as_str() } else { "" }).collect(),
            )),
            _ => None,
        })
        .collect()
}

fn inputs(h: &Harness, agent: &str) -> Vec<(String, String)> {
    h.transcript(agent)
        .unwrap()
        .into_iter()
        .filter_map(|e| match e.item {
            Item::Input { from, text } => Some((from, text)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_waiting_send_returns_the_recipients_reply() {
    let temp = tempfile::tempdir().unwrap();
    let (asker, answerer) = (Model::start(vec![]), Model::start(vec![]));
    let outbox = Arc::new(Outbox::default());
    let h = harness(&temp.path().join("state"), &[("a", &asker), ("b", &answerer)], &outbox);
    let b = h.create_agent(spec(temp.path(), "b")).unwrap();
    let a = h.create_agent(spec(temp.path(), "a")).unwrap();
    asker.push(calls("m1", "send_message", json!({"agent": b, "text": "What is 6*7?", "wait": true})));
    asker.push(says("It is 42."));
    answerer.push(calls("r1", "send_message", json!({"agent": a, "text": "42"})));
    answerer.push(says("answered"));
    h.send(&a, "user", "ask b").unwrap();
    settle(&h, &a, AgentState::Idle);
    settle(&h, &b, AgentState::Idle);
    assert_eq!(results(&h, &a), [(false, format!("Reply from agent {b}:\n42"))]);
    assert_eq!(inputs(&h, &a), [("user".to_owned(), "ask b".to_owned())], "the reply is not delivered twice");
    let second = &asker.requests()[1]["input"];
    assert_eq!(second[2]["output"], json!(format!("Reply from agent {b}:\n42")), "{second}");
    assert_eq!(results(&h, &b), [(false, format!("Message queued for agent {a}."))]);
}

#[test]
fn a_wait_that_times_out_leaves_the_reply_to_arrive_as_mail() {
    let temp = tempfile::tempdir().unwrap();
    let (asker, answerer) = (Model::start(vec![]), Model::start(vec![]));
    let outbox = Arc::new(Outbox::default());
    let h = harness(&temp.path().join("state"), &[("a", &asker), ("b", &answerer)], &outbox);
    let b = h.create_agent(spec(temp.path(), "b")).unwrap();
    let a = h.create_agent(spec(temp.path(), "a")).unwrap();
    asker.push(calls(
        "m1",
        "send_message",
        json!({"agent": b, "text": "slow question", "wait": true, "timeout_seconds": 1}),
    ));
    asker.push(says("moving on"));
    asker.push(says("thanks for the late answer"));
    let late = match calls("r1", "send_message", json!({"agent": a, "text": "late answer"})) {
        Reply::Events(events) => Reply::Delayed(Duration::from_secs(2), events),
        _ => unreachable!(),
    };
    answerer.push(late);
    answerer.push(says("answered"));
    h.send(&a, "user", "ask b").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while asker.requests().len() < 3 {
        assert!(std::time::Instant::now() < deadline, "the late reply never reached the asker");
        std::thread::sleep(Duration::from_millis(20));
    }
    settle(&h, &a, AgentState::Idle);
    assert_eq!(
        results(&h, &a),
        [(false, format!("No reply from agent {b} within 1 second. A reply will arrive as a message."))]
    );
    assert_eq!(inputs(&h, &a), [("user".to_owned(), "ask b".to_owned()), (b.clone(), "late answer".to_owned())]);
    assert_eq!(asker.requests().len(), 3);
}

#[test]
fn interrupting_a_waiting_agent_ends_the_wait() {
    let temp = tempfile::tempdir().unwrap();
    let (asker, answerer) = (Model::start(vec![]), Model::start(vec![]));
    let outbox = Arc::new(Outbox::default());
    let h = harness(&temp.path().join("state"), &[("a", &asker), ("b", &answerer)], &outbox);
    let b = h.create_agent(spec(temp.path(), "b")).unwrap();
    let a = h.create_agent(spec(temp.path(), "a")).unwrap();
    asker.push(calls("m1", "send_message", json!({"agent": b, "text": "hello?", "wait": true})));
    answerer.push(says("not replying"));
    h.send(&a, "user", "ask b").unwrap();
    settle(&h, &b, AgentState::Idle);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(h.agent(&a).unwrap().state, AgentState::Running);
    h.interrupt(&a);
    settle(&h, &a, AgentState::Idle);
    assert_eq!(results(&h, &a), [(true, format!("Stopped waiting for a reply from agent {b}: interrupted."))]);
    assert_eq!(asker.requests().len(), 1);
}

#[test]
fn host_recipients_receive_mail_and_their_replies_end_waits() {
    let temp = tempfile::tempdir().unwrap();
    let asker = Model::start(vec![]);
    let outbox = Arc::new(Outbox::default());
    let h = harness(&temp.path().join("state"), &[("a", &asker)], &outbox);
    let a = h.create_agent(spec(temp.path(), "a")).unwrap();
    asker.push(calls("m1", "send_message", json!({"agent": "user", "text": "Deploy now?", "wait": true})));
    asker.push(calls("m2", "send_message", json!({"agent": "user", "text": "Deployed."})));
    asker.push(says("done"));
    h.send(&a, "user", "deploy when approved").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while outbox.0.lock().unwrap().is_empty() {
        assert!(std::time::Instant::now() < deadline, "the host never received the question");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(outbox.0.lock().unwrap()[0], (a.clone(), "Deploy now?".to_owned()));
    h.send(&a, "user", "yes").unwrap();
    settle(&h, &a, AgentState::Idle);
    assert_eq!(
        results(&h, &a),
        [(false, "Reply from the user:\nyes".to_owned()), (false, "Message sent to the user.".to_owned())]
    );
    assert_eq!(outbox.0.lock().unwrap()[1], (a.clone(), "Deployed.".to_owned()));
    assert_eq!(inputs(&h, &a), [("user".to_owned(), "deploy when approved".to_owned())]);
}

#[test]
fn a_reply_claimed_by_a_wait_is_not_redelivered_after_a_restart() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("state");
    let (asker, answerer) = (Model::start(vec![]), Model::start(vec![]));
    let outbox = Arc::new(Outbox::default());
    let h = harness(&dir, &[("a", &asker), ("b", &answerer)], &outbox);
    let b = h.create_agent(spec(temp.path(), "b")).unwrap();
    let a = h.create_agent(spec(temp.path(), "a")).unwrap();
    asker.push(calls("m1", "send_message", json!({"agent": b, "text": "q", "wait": true})));
    asker.push(says("done"));
    answerer.push(calls("r1", "send_message", json!({"agent": a, "text": "r"})));
    answerer.push(says("answered"));
    h.send(&a, "user", "ask").unwrap();
    settle(&h, &a, AgentState::Idle);
    settle(&h, &b, AgentState::Idle);
    block_on(h.shutdown());
    drop(h);
    let h = harness(&dir, &[("a", &asker), ("b", &answerer)], &outbox);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(asker.requests().len(), 2, "the restarted harness woke the asker again");
    assert_eq!(inputs(&h, &a).len(), 1);
}
