//! The agent loop end to end: a scripted Responses API server, real sandboxes and tools.

mod common;

use common::model::{Model, Reply, calls, completed, says};
use common::{Fixture, block_on};
use erisharness::agent::{AgentSpec, AgentState, Item, Observation};
use erisharness::machine::MachineSpec;
use erisharness::{Harness, tools};
use libtest_mimic::Failed;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

fn main() {
    common::run(&[
        ("a_turn_runs_tools_until_the_model_answers", a_turn_runs_tools_until_the_model_answers),
        ("assistant_text_streams_to_observers", assistant_text_streams_to_observers),
        (
            "mail_arriving_mid_turn_is_delivered_at_a_tool_boundary",
            mail_arriving_mid_turn_is_delivered_at_a_tool_boundary,
        ),
        ("agents_message_each_other", agents_message_each_other),
        ("send_message_tool_queues_mail_for_another_agent", send_message_tool_queues_mail_for_another_agent),
        ("send_message_respects_held_recipients", send_message_respects_held_recipients),
        ("rate_limits_are_retried_after_the_advertised_delay", rate_limits_are_retried_after_the_advertised_delay),
        (
            "provider_failures_leave_the_agent_failed_and_retryable",
            provider_failures_leave_the_agent_failed_and_retryable,
        ),
        ("interrupt_aborts_the_running_tool_and_holds_mail", interrupt_aborts_the_running_tool_and_holds_mail),
        ("a_restarted_harness_resumes_unfinished_agents", a_restarted_harness_resumes_unfinished_agents),
        ("transcripts_are_jsonl_readable_by_other_agents", transcripts_are_jsonl_readable_by_other_agents),
        ("reasoning_is_carried_back_to_the_provider", reasoning_is_carried_back_to_the_provider),
    ]);
}

fn spec(f: &Fixture) -> AgentSpec {
    AgentSpec {
        system_prompt: "You are a test agent.".into(),
        tools: vec!["read".into(), "bash".into(), "edit".into(), "write".into(), "send_message".into()],
        provider: "test".into(),
        model: "test-model".into(),
        reasoning_effort: Some("low".into()),
        context_window: None,
        metadata: serde_json::Value::Null,
        machine: MachineSpec::Sandbox(f.spec()),
    }
}

fn harness(f: &Fixture, dir: &std::path::Path, model: &Model) -> Arc<Harness> {
    harness_with(f, dir, &[("test", model)])
}

fn harness_with(f: &Fixture, dir: &std::path::Path, models: &[(&str, &Model)]) -> Arc<Harness> {
    let builder = models
        .iter()
        .fold(Harness::builder(dir).sandboxes(&f.host).sandbox_dir(dir.join("elsewhere")), |b, (name, model)| {
            b.provider(name, model.provider())
        });
    block_on(builder.tools(tools::builtin()).idle_grace(Duration::from_millis(500)).open()).expect("open harness")
}

fn wait_for(harness: &Harness, agent: &str, state: AgentState) -> Result<(), Failed> {
    block_on(async { tokio::time::timeout(Duration::from_secs(20), harness.wait_for(agent, state)).await })
        .map_err(|_| format!("agent never became {state:?}: {:?}", harness.agent(agent).map(|a| a.state)).into())
}

fn items(harness: &Harness, agent: &str) -> Vec<Item> {
    harness.transcript(agent).unwrap().into_iter().map(|e| e.item).collect()
}

fn check(condition: bool, message: impl Into<String>) -> Result<(), Failed> {
    if condition { Ok(()) } else { Err(message.into().into()) }
}

fn a_turn_runs_tools_until_the_model_answers(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "echo hi"})), says("It printed hi.")]);
    let dir = f.host_dir("turn");
    let h = harness(f, &dir, &model);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "Run echo hi").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    check(dir.join("elsewhere").join(&agent).exists() && !dir.join("sandboxes").exists(), "sandbox directory")?;
    let items = items(&h, &agent);
    check(
        matches!(&items[..], [
            Item::Input { from, text },
            Item::ToolCall { call_id, name, .. },
            Item::ToolResult { output, .. },
            Item::Assistant { text: answer },
        ] if from == "user" && text == "Run echo hi" && call_id == "c1" && name == "bash"
            && output.content == [tools::Content::Text("hi\n".into())] && answer == "It printed hi."),
        format!("{items:#?}"),
    )?;
    let requests = model.requests();
    let first = &requests[0];
    check(first["model"] == "test-model" && first["stream"] == true && first["store"] == false, format!("{first}"))?;
    check(first["instructions"].as_str().unwrap().starts_with("You are a test agent."), "system prompt")?;
    check(first["tools"].as_array().unwrap().len() == 5 && first["tools"][0]["name"] == "read", "tools")?;
    check(first["instructions"].as_str().unwrap().contains(&format!("Your agent id is {agent}.")), "own id in prompt")?;
    check(
        first["input"] == json!([{"role": "user", "content": [{"type": "input_text", "text": "Run echo hi"}]}]),
        format!("{}", first["input"]),
    )?;
    let second = &requests[1]["input"];
    check(
        second[1]
            == json!({"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{\"command\":\"echo hi\"}"})
            && second[2] == json!({"type": "function_call_output", "call_id": "c1", "output": "hi\n"}),
        format!("{second}"),
    )?;
    let auth = model.script.lock().unwrap().auth.clone();
    check(auth.iter().all(|a| a == "Bearer secret-token"), format!("{auth:?}"))?;
    let usage = h.agent(&agent).unwrap().usage;
    check(usage.input == 20 && usage.cached_input == 8 && usage.output == 10, format!("{usage:?}"))
}

fn assistant_text_streams_to_observers(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![says("streamed words")]);
    let h = harness(f, &f.host_dir("stream"), &model);
    let mut observations = h.subscribe();
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "hello").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    let mut deltas = String::new();
    let mut states = Vec::new();
    while let Ok(observation) = observations.try_recv() {
        match observation {
            Observation::TextDelta { agent: a, text } if a == agent => deltas.push_str(&text),
            Observation::State { agent: a, state } if a == agent => states.push(state),
            _ => {}
        }
    }
    check(deltas == "streamed words", deltas)?;
    check(
        states.first() == Some(&AgentState::Running) && states.last() == Some(&AgentState::Idle),
        format!("{states:?}"),
    )
}

fn mail_arriving_mid_turn_is_delivered_at_a_tool_boundary(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "sleep 1"})), says("Noted the update.")]);
    let h = harness(f, &f.host_dir("steer"), &model);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "start").unwrap();
    wait_for(&h, &agent, AgentState::Running)?;
    std::thread::sleep(Duration::from_millis(400));
    h.send(&agent, "user", "also check the logs").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    let requests = model.requests();
    check(requests.len() == 2, format!("{} requests", requests.len()))?;
    let input = requests[1]["input"].as_array().unwrap();
    check(
        input.last().unwrap()
            == &json!({"role": "user", "content": [{"type": "input_text", "text": "also check the logs"}]}),
        format!("{input:?}"),
    )
}

fn agents_message_each_other(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![says("pong")]);
    let h = harness(f, &f.host_dir("mail"), &model);
    let a = h.create_agent(spec(f)).unwrap();
    let b = h.create_agent(spec(f)).unwrap();
    h.send(&b, &a, "ping").unwrap();
    wait_for(&h, &b, AgentState::Idle)?;
    let text = model.requests()[0]["input"][0]["content"][0]["text"].clone();
    check(text == json!(format!("[Message from agent {a}]\nping")), format!("{text}"))
}

fn rate_limits_are_retried_after_the_advertised_delay(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![
        Reply::Status(429, vec![("retry-after-ms", "300")]),
        Reply::Status(503, vec![]),
        says("finally"),
    ]);
    let h = harness(f, &f.host_dir("retry"), &model);
    let agent = h.create_agent(spec(f)).unwrap();
    let started = std::time::Instant::now();
    h.send(&agent, "user", "go").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    check(model.requests().len() == 3, "expected three attempts")?;
    check(started.elapsed() >= Duration::from_millis(300), "retry-after ignored")?;
    check(matches!(items(&h, &agent).last(), Some(Item::Assistant { text }) if text == "finally"), "no answer")
}

fn provider_failures_leave_the_agent_failed_and_retryable(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![Reply::Status(400, vec![]), says("recovered")]);
    let h = harness(f, &f.host_dir("fail"), &model);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "first").unwrap();
    wait_for(&h, &agent, AgentState::Failed)?;
    let error = h.agent(&agent).unwrap().error.unwrap_or_default();
    check(error.contains("400") && error.contains("scripted"), error)?;
    h.send(&agent, "user", "try again").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    let input = model.requests()[1]["input"].clone();
    check(input.as_array().unwrap().len() == 2, format!("{input}"))
}

fn interrupt_aborts_the_running_tool_and_holds_mail(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![calls("c1", "bash", json!({"command": "sleep 30"})), says("resumed")]);
    let h = harness(f, &f.host_dir("interrupt"), &model);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "sleep").unwrap();
    std::thread::sleep(Duration::from_millis(800));
    h.send(&agent, "agent-x", "queued mail").unwrap();
    h.interrupt(&agent);
    wait_for(&h, &agent, AgentState::Idle)?;
    let items = items(&h, &agent);
    let aborted = items.iter().any(|i| matches!(i, Item::ToolResult { output, .. } if output.is_error && output.content == [tools::Content::Text("Command aborted".into())]));
    check(aborted, format!("{items:#?}"))?;
    std::thread::sleep(Duration::from_millis(300));
    check(model.requests().len() == 1, "mail was processed after an interrupt")?;
    h.send(&agent, "user", "continue").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    let last = model.requests()[1]["input"].as_array().unwrap().clone();
    let texts: Vec<&str> = last.iter().filter_map(|i| i["content"][0]["text"].as_str()).collect();
    check(texts.ends_with(&["[Message from agent agent-x]\nqueued mail", "continue"]), format!("{texts:?}"))
}

fn a_restarted_harness_resumes_unfinished_agents(f: &Fixture) -> Result<(), Failed> {
    let dir = f.host_dir("restart");
    let slow = Model::start(vec![Reply::Delayed(Duration::from_secs(30), vec![])]);
    let first = harness(f, &dir, &slow);
    let agent = first.create_agent(spec(f)).unwrap();
    first.send(&agent, "user", "survive a restart").unwrap();
    wait_for(&first, &agent, AgentState::Running)?;
    block_on(first.shutdown());
    drop(first);
    let fast = Model::start(vec![says("back")]);
    let second = harness(f, &dir, &fast);
    wait_for(&second, &agent, AgentState::Idle)?;
    let items = items(&second, &agent);
    check(
        matches!(&items[..], [Item::Input { text, .. }, Item::Assistant { text: answer }] if text == "survive a restart" && answer == "back"),
        format!("{items:#?}"),
    )
}

fn transcripts_are_jsonl_readable_by_other_agents(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![
        says("recorded"),
        calls("c1", "bash", json!({"command": "cat /records/*/transcript.jsonl | wc -l"})),
        says("ok"),
    ]);
    let dir = f.host_dir("records");
    let h = harness(f, &dir, &model);
    let writer = h.create_agent(spec(f)).unwrap();
    h.send(&writer, "user", "say something").unwrap();
    wait_for(&h, &writer, AgentState::Idle)?;
    let path = h.transcript_path(&writer);
    let lines: Vec<Value> = std::fs::read_to_string(&path)?.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    check(lines.len() == 2 && lines[0]["seq"] == 0 && lines[1]["type"] == "assistant", format!("{lines:?}"))?;
    let mut reader_spec = spec(f);
    let MachineSpec::Sandbox(sandbox) = &mut reader_spec.machine else { unreachable!() };
    sandbox.binds.push(erissandbox::Bind {
        source: path.parent().unwrap().parent().unwrap().to_path_buf(),
        target: "/records".into(),
        writable: false,
    });
    let reader = h.create_agent(reader_spec).unwrap();
    h.send(&reader, "user", "count lines").unwrap();
    wait_for(&h, &reader, AgentState::Idle)?;
    let result = items(&h, &reader)
        .into_iter()
        .find_map(|i| match i {
            Item::ToolResult { output, .. } => Some(output.content),
            _ => None,
        })
        .unwrap_or_default();
    // The writer's two lines plus the reader's own input and tool call so far.
    check(result == [tools::Content::Text("4\n".into())], format!("{result:?}"))
}

fn reasoning_is_carried_back_to_the_provider(f: &Fixture) -> Result<(), Failed> {
    let model = Model::start(vec![
        Reply::Events(vec![
            json!({"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}], "encrypted_content": "opaque"}}),
            json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{\"command\":\"true\"}"}}),
            completed(),
        ]),
        says("done"),
    ]);
    let h = harness(f, &f.host_dir("reasoning"), &model);
    let agent = h.create_agent(spec(f)).unwrap();
    h.send(&agent, "user", "think").unwrap();
    wait_for(&h, &agent, AgentState::Idle)?;
    let requests = model.requests();
    check(requests[0]["include"] == json!(["reasoning.encrypted_content"]), "include")?;
    check(requests[0]["reasoning"] == json!({"effort": "low", "summary": "auto"}), "reasoning config")?;
    let carried = &requests[1]["input"][1];
    check(
        carried
            == &json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}], "encrypted_content": "opaque"}),
        format!("{carried}"),
    )
}

fn send_message_tool_queues_mail_for_another_agent(f: &Fixture) -> Result<(), Failed> {
    let (sender_model, recipient_model) = (Model::start(vec![]), Model::start(vec![says("pong")]));
    let h = harness_with(f, &f.host_dir("send-tool"), &[("a", &sender_model), ("b", &recipient_model)]);
    let recipient = h.create_agent(AgentSpec { provider: "b".into(), ..spec(f) }).unwrap();
    let sender = h.create_agent(AgentSpec { provider: "a".into(), ..spec(f) }).unwrap();
    {
        let mut script = sender_model.script.lock().unwrap();
        script.replies.push_back(calls("m1", "send_message", json!({"agent": recipient, "text": "ping"})));
        script.replies.push_back(calls("m2", "send_message", json!({"agent": "nobody", "text": "x"})));
        script.replies.push_back(says("sent"));
    }
    h.send(&sender, "user", "tell the other agent").unwrap();
    wait_for(&h, &sender, AgentState::Idle)?;
    wait_for(&h, &recipient, AgentState::Idle)?;
    let results: Vec<(bool, Vec<tools::Content>)> = items(&h, &sender)
        .into_iter()
        .filter_map(|i| match i {
            Item::ToolResult { output, .. } => Some((output.is_error, output.content)),
            _ => None,
        })
        .collect();
    check(
        results
            == [
                (false, vec![tools::Content::Text(format!("Message queued for agent {recipient}."))]),
                (true, vec![tools::Content::Text("No agent nobody".into())]),
            ],
        format!("{results:?}"),
    )?;
    let received = &recipient_model.requests()[0]["input"][0]["content"][0]["text"];
    check(received == &json!(format!("[Message from agent {sender}]\nping")), format!("{received}"))?;
    check(
        matches!(items(&h, &recipient).last(), Some(Item::Assistant { text }) if text == "pong"),
        "recipient did not answer",
    )
}

fn send_message_respects_held_recipients(f: &Fixture) -> Result<(), Failed> {
    let (sender_model, recipient_model) = (Model::start(vec![]), Model::start(vec![]));
    let h = harness_with(f, &f.host_dir("send-held"), &[("a", &sender_model), ("b", &recipient_model)]);
    let recipient = h.create_agent(AgentSpec { provider: "b".into(), ..spec(f) }).unwrap();
    let sender = h.create_agent(AgentSpec { provider: "a".into(), ..spec(f) }).unwrap();
    h.interrupt(&recipient);
    {
        let mut script = sender_model.script.lock().unwrap();
        script.replies.push_back(calls("m1", "send_message", json!({"agent": recipient, "text": "while held"})));
        script.replies.push_back(says("sent"));
    }
    h.send(&sender, "user", "go").unwrap();
    wait_for(&h, &sender, AgentState::Idle)?;
    std::thread::sleep(Duration::from_millis(300));
    check(recipient_model.requests().is_empty(), "held recipient was woken by agent mail")?;
    recipient_model.script.lock().unwrap().replies.push_back(says("caught up"));
    h.send(&recipient, "user", "resume").unwrap();
    wait_for(&h, &recipient, AgentState::Idle)?;
    let input = recipient_model.requests()[0]["input"].clone();
    check(input.as_array().unwrap().len() == 2 && input[1]["content"][0]["text"] == "resume", format!("{input}"))
}
