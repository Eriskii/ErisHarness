//! The agent loop end to end: a scripted Responses API server, real sandboxes and tools.

#[macro_use]
mod common;

use common::model::{Model, Reply, calls, completed, says};
use common::{Fixture, block_on, direct, items, results, settle};
use erisharness::Harness;
use erisharness::agent::{AgentSpec, AgentState, Item, Observation};
use erisharness::machine::MachineSpec;
use erisharness::provider::Hold;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn main() {
    common::run(named![
        a_turn_runs_tools_until_the_model_answers,
        assistant_text_streams_to_observers,
        mail_arriving_mid_turn_is_delivered_at_a_tool_boundary,
        mail_arriving_during_a_batch_of_calls_skips_the_rest,
        agents_message_each_other,
        send_message_tool_queues_mail_for_another_agent,
        send_message_respects_held_recipients,
        rate_limits_are_retried_after_the_advertised_delay,
        provider_failures_leave_the_agent_failed_and_retryable,
        interrupt_aborts_the_running_tool_and_holds_mail,
        a_restarted_harness_resumes_unfinished_agents,
        transcripts_are_jsonl_readable_by_other_agents,
        reasoning_is_carried_back_to_the_provider,
        direct_agents_run_as_the_invoking_user_beside_sandboxes,
    ]);
}

fn spec(f: &Fixture) -> AgentSpec {
    AgentSpec { machine: MachineSpec::Sandbox(f.spec()), reasoning_effort: Some("low".into()), ..direct(f.rootfs()) }
}

fn harness(f: &Fixture, dir: &Path, models: &[(&str, &Model)]) -> Arc<Harness> {
    let builder = Harness::builder(dir)
        .sandboxes(&f.host)
        .sandbox_dir(dir.join("elsewhere"))
        .idle_grace(Duration::from_millis(500));
    common::open(builder, models)
}

fn user_text(text: &str) -> Value {
    json!({"role": "user", "content": [{"type": "input_text", "text": text}]})
}

fn a_turn_runs_tools_until_the_model_answers(f: &Fixture) {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "echo hi"})), says("It printed hi.")]);
    let dir = f.host_dir("turn");
    let h = harness(f, &dir, &[("test", &model)]);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "Run echo hi").unwrap();
    settle(&h, &agent, AgentState::Idle);
    assert!(dir.join("elsewhere").join(&agent).exists() && !dir.join("sandboxes").exists(), "sandbox directory");
    let items = items(&h, &agent);
    assert!(
        matches!(&items[..], [
            Item::Input { from, text, .. },
            Item::ToolCall { call_id, name, .. },
            Item::ToolResult { .. },
            Item::Assistant { text: answer },
        ] if from == "user" && text == "Run echo hi" && call_id == "c1" && name == "bash" && answer == "It printed hi."),
        "{items:#?}"
    );
    assert_eq!(results(&h, &agent), [(false, "hi\n".into())]);
    let requests = model.requests();
    let first = &requests[0];
    assert!(first["model"] == "test-model" && first["stream"] == true && first["store"] == false, "{first}");
    let instructions = first["instructions"].as_str().unwrap();
    assert!(instructions.starts_with("You are a test agent."), "{instructions}");
    assert!(instructions.contains(&format!("Your agent id is {agent}.")), "{instructions}");
    assert!(first["tools"].as_array().unwrap().len() == 5 && first["tools"][0]["name"] == "read");
    assert_eq!(first["input"], json!([user_text("Run echo hi")]));
    let second = &requests[1]["input"];
    assert_eq!(
        second[1],
        json!({"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{\"command\":\"echo hi\"}"})
    );
    assert_eq!(second[2], json!({"type": "function_call_output", "call_id": "c1", "output": "hi\n"}));
    assert!(model.script.lock().unwrap().auth.iter().all(|a| a == "Bearer secret-token"));
    let usage = h.agent(&agent).unwrap().usage;
    assert!(usage.input == 20 && usage.cached_input == 8 && usage.output == 10, "{usage:?}");
}

fn assistant_text_streams_to_observers(f: &Fixture) {
    let model = Model::start(vec![says("streamed words")]);
    let h = harness(f, &f.host_dir("stream"), &[("test", &model)]);
    let mut observations = h.subscribe();
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "hello").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let mut deltas = String::new();
    let mut states = Vec::new();
    while let Ok(observation) = observations.try_recv() {
        match observation {
            Observation::TextDelta { agent: a, text } if a == agent => deltas.push_str(&text),
            Observation::State { agent: a, state } if a == agent => states.push(state),
            _ => {}
        }
    }
    assert_eq!(deltas, "streamed words");
    assert!(states.first() == Some(&AgentState::Running) && states.last() == Some(&AgentState::Idle), "{states:?}");
}

fn mail_arriving_mid_turn_is_delivered_at_a_tool_boundary(f: &Fixture) {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "sleep 1"})), says("Noted the update.")]);
    let h = harness(f, &f.host_dir("steer"), &[("test", &model)]);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "start").unwrap();
    settle(&h, &agent, AgentState::Running);
    std::thread::sleep(Duration::from_millis(400));
    h.send(&agent, "user", "also check the logs").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1]["input"].as_array().unwrap().last().unwrap(), &user_text("also check the logs"));
}

fn mail_arriving_during_a_batch_of_calls_skips_the_rest(f: &Fixture) {
    let call = |id: &str, command: &str| {
        json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": id,
            "name": "bash", "arguments": json!({"command": command}).to_string()}})
    };
    let batch = Reply::Events(vec![
        call("c1", "sleep 1; echo first"),
        call("c2", "echo second"),
        call("c3", "echo third"),
        completed(),
    ]);
    let model = Model::start(vec![batch, says("Stopping to read the message.")]);
    let h = harness(f, &f.host_dir("batch"), &[("test", &model)]);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "start").unwrap();
    settle(&h, &agent, AgentState::Running);
    std::thread::sleep(Duration::from_millis(400));
    h.send(&agent, "user", "stop, do this instead").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let outputs: Vec<(bool, String)> = results(&h, &agent);
    assert!(outputs[0].1.contains("first") && !outputs[0].0, "the running call finishes: {outputs:?}");
    for (error, text) in &outputs[1..] {
        assert!(*error && text.contains("a new message arrived"), "the rest are not run: {outputs:?}");
    }
    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1]["input"].as_array().unwrap().last().unwrap(), &user_text("stop, do this instead"));
}

fn agents_message_each_other(f: &Fixture) {
    let model = Model::start(vec![says("pong")]);
    let h = harness(f, &f.host_dir("mail"), &[("test", &model)]);
    let a = h.create_agent(spec(f)).unwrap();
    let b = h.create_agent(spec(f)).unwrap();
    h.send(&b, &a, "ping").unwrap();
    settle(&h, &b, AgentState::Idle);
    assert_eq!(model.requests()[0]["input"][0], user_text(&format!("[Message from agent {a}]\nping")));
}

fn rate_limits_are_retried_after_the_advertised_delay(f: &Fixture) {
    let model = Model::start(vec![
        Reply::Status(429, vec![("retry-after-ms", "300")]),
        Reply::Status(503, vec![]),
        says("finally"),
    ]);
    let h = harness(f, &f.host_dir("retry"), &[("test", &model)]);
    let mut observations = h.subscribe();
    let agent = h.create_agent(spec(f)).unwrap();
    let started = std::time::Instant::now();
    let asked = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
    h.send(&agent, "user", "go").unwrap();
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(model.requests().len(), 3);
    assert!(started.elapsed() >= Duration::from_millis(300), "retry-after ignored");
    assert!(matches!(items(&h, &agent).last(), Some(Item::Assistant { text }) if text == "finally"));
    // Only the 429 is a rate limit; the 503 is retried without one.
    let mut limits = Vec::new();
    while let Ok(observation) = observations.try_recv() {
        if let Observation::Held { agent: a, hold } = observation
            && a == agent
        {
            limits.push(hold);
        }
    }
    assert!(
        matches!(limits.as_slice(), [Some(Hold::RateLimited { until }), None] if (asked + 300..asked + 2000).contains(until)),
        "{limits:?}"
    );
}

fn provider_failures_leave_the_agent_failed_and_retryable(f: &Fixture) {
    let model = Model::start(vec![Reply::Status(400, vec![]), says("recovered")]);
    let h = harness(f, &f.host_dir("fail"), &[("test", &model)]);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "first").unwrap();
    settle(&h, &agent, AgentState::Failed);
    let error = h.agent(&agent).unwrap().error.unwrap_or_default();
    assert!(error.contains("400") && error.contains("scripted"), "{error}");
    h.send(&agent, "user", "try again").unwrap();
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(model.requests()[1]["input"].as_array().unwrap().len(), 2);
}

fn interrupt_aborts_the_running_tool_and_holds_mail(f: &Fixture) {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "sleep 30"})), says("resumed")]);
    let h = harness(f, &f.host_dir("interrupt"), &[("test", &model)]);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "sleep").unwrap();
    std::thread::sleep(Duration::from_millis(800));
    h.send(&agent, "agent-x", "queued mail").unwrap();
    h.interrupt(&agent);
    settle(&h, &agent, AgentState::Idle);
    assert_eq!(results(&h, &agent), [(true, "Command aborted".into())]);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(model.requests().len(), 1, "mail was processed after an interrupt");
    h.send(&agent, "user", "continue").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let input = model.requests()[1]["input"].as_array().unwrap().clone();
    let texts: Vec<&str> = input.iter().filter_map(|i| i["content"][0]["text"].as_str()).collect();
    assert!(texts.ends_with(&["[Message from agent agent-x]\nqueued mail", "continue"]), "{texts:?}");
}

fn a_restarted_harness_resumes_unfinished_agents(f: &Fixture) {
    let dir = f.host_dir("restart");
    let slow = Model::start(vec![Reply::Delayed(Duration::from_secs(30), vec![])]);
    let first = harness(f, &dir, &[("test", &slow)]);
    let agent = first.create_agent(spec(f)).unwrap();
    first.send(&agent, "user", "survive a restart").unwrap();
    settle(&first, &agent, AgentState::Running);
    block_on(first.shutdown());
    drop(first);
    let fast = Model::start(vec![says("back")]);
    let second = harness(f, &dir, &[("test", &fast)]);
    settle(&second, &agent, AgentState::Idle);
    let items = items(&second, &agent);
    assert!(
        matches!(&items[..], [Item::Input { text, .. }, Item::Assistant { text: answer }] if text == "survive a restart" && answer == "back"),
        "{items:#?}"
    );
}

fn transcripts_are_jsonl_readable_by_other_agents(f: &Fixture) {
    let model = Model::start(vec![
        says("recorded"),
        calls("c1", "bash", json!({"command": "cat /records/*/transcript.jsonl | wc -l"})),
        says("ok"),
    ]);
    let h = harness(f, &f.host_dir("records"), &[("test", &model)]);
    let writer = h.create_agent(spec(f)).unwrap();
    h.send(&writer, "user", "say something").unwrap();
    settle(&h, &writer, AgentState::Idle);
    let path = h.transcript_path(&writer);
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(lines.len() == 2 && lines[0]["seq"] == 0 && lines[1]["type"] == "assistant", "{lines:?}");
    let mut reader_spec = spec(f);
    let MachineSpec::Sandbox(sandbox) = &mut reader_spec.machine else { unreachable!() };
    sandbox.binds.push(erissandbox::Bind {
        source: path.parent().unwrap().parent().unwrap().to_path_buf(),
        target: "/records".into(),
        writable: false,
    });
    let reader = h.create_agent(reader_spec).unwrap();
    h.send(&reader, "user", "count lines").unwrap();
    settle(&h, &reader, AgentState::Idle);
    // The writer's two lines plus the reader's own input and tool call so far.
    assert_eq!(results(&h, &reader), [(false, "4\n".into())]);
}

fn reasoning_is_carried_back_to_the_provider(f: &Fixture) {
    let reasoning = json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}], "encrypted_content": "opaque"});
    let model = Model::start(vec![
        Reply::Events(vec![
            json!({"type": "response.output_item.done", "item": reasoning}),
            json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{\"command\":\"true\"}"}}),
            completed(),
        ]),
        says("done"),
    ]);
    let h = harness(f, &f.host_dir("reasoning"), &[("test", &model)]);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "think").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let requests = model.requests();
    assert_eq!(requests[0]["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(requests[0]["reasoning"], json!({"effort": "low", "summary": "auto"}));
    assert_eq!(requests[1]["input"][1], reasoning);
}

fn send_message_tool_queues_mail_for_another_agent(f: &Fixture) {
    let (sender_model, recipient_model) = (Model::start(vec![]), Model::start(vec![says("pong")]));
    let h = harness(f, &f.host_dir("send-tool"), &[("a", &sender_model), ("b", &recipient_model)]);
    let recipient = h.create_agent(AgentSpec { provider: "b".into(), ..spec(f) }).unwrap();
    let sender = h.create_agent(AgentSpec { provider: "a".into(), ..spec(f) }).unwrap();
    sender_model.push(calls("m1", "send_message", json!({"agent": recipient, "text": "ping"})));
    sender_model.push(calls("m2", "send_message", json!({"agent": "nobody", "text": "x"})));
    sender_model.push(says("sent"));
    h.send(&sender, "user", "tell the other agent").unwrap();
    settle(&h, &sender, AgentState::Idle);
    settle(&h, &recipient, AgentState::Idle);
    assert_eq!(
        results(&h, &sender),
        [(false, format!("Message queued for agent {recipient}.")), (true, "No agent nobody".into())]
    );
    assert_eq!(recipient_model.requests()[0]["input"][0], user_text(&format!("[Message from agent {sender}]\nping")));
    assert!(matches!(items(&h, &recipient).last(), Some(Item::Assistant { text }) if text == "pong"));
}

fn send_message_respects_held_recipients(f: &Fixture) {
    let (sender_model, recipient_model) = (Model::start(vec![]), Model::start(vec![]));
    let h = harness(f, &f.host_dir("send-held"), &[("a", &sender_model), ("b", &recipient_model)]);
    let recipient = h.create_agent(AgentSpec { provider: "b".into(), ..spec(f) }).unwrap();
    let sender = h.create_agent(AgentSpec { provider: "a".into(), ..spec(f) }).unwrap();
    h.interrupt(&recipient);
    sender_model.push(calls("m1", "send_message", json!({"agent": recipient, "text": "while held"})));
    sender_model.push(says("sent"));
    h.send(&sender, "user", "go").unwrap();
    settle(&h, &sender, AgentState::Idle);
    std::thread::sleep(Duration::from_millis(300));
    assert!(recipient_model.requests().is_empty(), "held recipient was woken by agent mail");
    recipient_model.push(says("caught up"));
    h.send(&recipient, "user", "resume").unwrap();
    settle(&h, &recipient, AgentState::Idle);
    let input = recipient_model.requests()[0]["input"].clone();
    assert!(input.as_array().unwrap().len() == 2 && input[1]["content"][0]["text"] == "resume", "{input}");
}

fn direct_agents_run_as_the_invoking_user_beside_sandboxes(f: &Fixture) {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "cat /proc/self/uid_map; pwd"})), says("ok")]);
    let dir = f.host_dir("direct-beside");
    let h = harness(f, &dir, &[("test", &model)]);
    let agent = h.create_agent(direct(&dir)).unwrap();
    h.send(&agent, "user", "who am I").unwrap();
    settle(&h, &agent, AgentState::Idle);
    let [(false, text)] = &results(&h, &agent)[..] else { panic!("{:?}", results(&h, &agent)) };
    let lines: Vec<&str> = text.lines().collect();
    let uid_map: Vec<&str> = lines[0].split_whitespace().collect();
    assert_eq!(uid_map, ["0", "0", "4294967295"], "not the host's user namespace: {text:?}");
    assert_eq!(lines[1], dir.to_str().unwrap());
}
