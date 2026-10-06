//! The Chat Completions provider against a local HTTP server; no account or Docker needed.

mod common;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::{Router, http::StatusCode, routing::post};
use erisharness::agent::{AgentSpec, AgentState, Item, Usage};
use erisharness::provider::{
    ChatCompletions, ChatCompletionsConfig, Progress, Provider, ProviderEvent, RateGate, Request, StaticToken, ToolSpec,
};
use erisharness::tools::{Content, ToolOutput};
use erisharness::{Harness, tools};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Script = Arc<Mutex<(VecDeque<(u16, String)>, Vec<Value>)>>;

struct Server {
    url: String,
    script: Script,
}

impl Server {
    async fn start(replies: Vec<(u16, String)>) -> Self {
        async fn respond(State(script): State<Script>, body: String) -> Response {
            let (status, text) = {
                let mut script = script.lock().unwrap();
                script.1.push(serde_json::from_str(&body).unwrap());
                script.0.pop_front().unwrap_or((500, "exhausted".into()))
            };
            let mut response = (StatusCode::from_u16(status).unwrap(), text).into_response();
            response.headers_mut().insert("content-type", "text/event-stream".parse().unwrap());
            response.headers_mut().insert("retry-after-ms", "1".parse().unwrap());
            response
        }
        let script: Script = Arc::new(Mutex::new((replies.into(), Vec::new())));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let app = Router::new().route("/v1/chat/completions", post(respond)).with_state(script.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, script }
    }

    fn provider(&self, reasoning_field: Option<&str>) -> ChatCompletions {
        ChatCompletions::new(ChatCompletionsConfig {
            base_url: self.url.clone(),
            headers: Vec::new(),
            max_retries: 2,
            credentials: Arc::new(StaticToken("go-key".into())),
            gate: RateGate::new(4, 4),
            reasoning_field: reasoning_field.map(str::to_owned),
        })
    }

    fn requests(&self) -> Vec<Value> {
        self.script.lock().unwrap().1.clone()
    }
}

/// An SSE body of these chunks, each a `choices[0].delta` unless it has its own shape.
fn stream(chunks: &[Value]) -> (u16, String) {
    let mut body: String = chunks.iter().map(|chunk| format!("data: {chunk}\n\n")).collect();
    body.push_str("data: [DONE]\n\n");
    (200, body)
}

fn delta(delta: Value) -> Value {
    json!({"choices": [{"index": 0, "delta": delta, "finish_reason": null}]})
}

fn finish(reason: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
}

fn usage() -> Value {
    json!({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 60}, "completion_tokens_details": {"reasoning_tokens": 5}}})
}

fn says(text: &str) -> (u16, String) {
    stream(&[delta(json!({"role": "assistant", "content": text})), finish("stop"), usage()])
}

fn request<'a>(items: &'a [Item], tools: &'a [ToolSpec]) -> Request<'a> {
    Request { model: "kimi-k3", reasoning_effort: Some("high"), cache_key: "agent", system: "Review.", tools, items }
}

fn input(text: &str) -> Item {
    Item::Input { from: "user".into(), text: text.into(), images: Vec::new() }
}

#[tokio::test]
async fn streams_reasoning_text_and_tool_calls_and_reports_usage() {
    let server = Server::start(vec![stream(&[
        delta(json!({"role": "assistant", "reasoning_content": "Look "})),
        delta(json!({"reasoning_content": "first."})),
        delta(json!({"content": "Let me "})),
        delta(json!({"content": "look."})),
        delta(json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
            "function": {"name": "read", "arguments": "{\"pa"}}]})),
        delta(json!({"tool_calls": [{"index": 0, "function": {"arguments": "th\":\"a\"}"}}]})),
        finish("tool_calls"),
        usage(),
    ])])
    .await;
    let tools =
        [ToolSpec { name: "read".into(), description: "Read a file".into(), parameters: json!({"type": "object"}) }];
    let streamed = Mutex::new(String::new());
    let completion = server
        .provider(None)
        .complete(request(&[input("hi")], &tools), &|progress| {
            if let Progress::Text(text) = progress {
                streamed.lock().unwrap().push_str(text);
            }
        })
        .await
        .unwrap();
    assert_eq!(
        completion.items,
        [
            Item::Reasoning { summary: vec!["Look first.".into()], encrypted: None },
            Item::Assistant { text: "Let me look.".into() },
            Item::ToolCall { call_id: "c1".into(), name: "read".into(), arguments: "{\"path\":\"a\"}".into() },
        ]
    );
    assert_eq!(completion.usage, Usage { input: 100, cached_input: 60, cache_write: 0, output: 20, reasoning: 5 });
    assert_eq!(*streamed.lock().unwrap(), "Let me look.");
    let body = &server.requests()[0];
    assert_eq!(body["model"], "kimi-k3");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["messages"], json!([{"role": "system", "content": "Review."}, {"role": "user", "content": "hi"}]));
    assert_eq!(
        body["tools"],
        json!([{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object"}}}])
    );
}

#[tokio::test]
async fn replays_a_turn_as_one_assistant_message_then_tool_messages() {
    let items = [
        input("hi"),
        Item::Reasoning { summary: vec!["Look first.".into()], encrypted: None },
        // Another provider's opaque reasoning is not this format's to send.
        Item::Reasoning { summary: vec!["other".into()], encrypted: Some("opaque".into()) },
        Item::Assistant { text: "Let me look.".into() },
        Item::ToolCall { call_id: "c1".into(), name: "read".into(), arguments: "{\"path\":\"a\"}".into() },
        Item::ToolCall { call_id: "c2".into(), name: "read".into(), arguments: "{\"path\":".into() },
        Item::ToolResult { call_id: "c1".into(), output: ToolOutput::text("contents") },
        Item::ToolResult {
            call_id: "c2".into(),
            output: ToolOutput::new(vec![Content::Image { mime: "image/png".into(), data: "AAAA".into() }]),
        },
        Item::Input { from: "agent-b".into(), text: "ping".into(), images: Vec::new() },
    ];
    let server = Server::start(vec![says("ok"), says("ok")]).await;
    server.provider(Some("reasoning_content")).complete(request(&items, &[]), &|_| {}).await.unwrap();
    server.provider(None).complete(request(&items, &[]), &|_| {}).await.unwrap();
    let [with, without] = &server.requests()[..] else { panic!() };
    let call = |id: &str, arguments: &str| json!({"id": id, "type": "function", "function": {"name": "read", "arguments": arguments}});
    assert_eq!(
        with["messages"],
        json!([
            {"role": "system", "content": "Review."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Let me look.", "reasoning_content": "Look first.",
             "tool_calls": [call("c1", "{\"path\":\"a\"}"), call("c2", "{\"INVALID_JSON\":\"{\\\"path\\\":\"}")]},
            {"role": "tool", "tool_call_id": "c1", "content": "contents"},
            {"role": "tool", "tool_call_id": "c2", "content": ""},
            {"role": "user", "content": [
                {"type": "text", "text": "Images from tool call c2:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
            {"role": "user", "content": "[Message from agent agent-b]\nping"},
        ])
    );
    assert!(without["messages"][2].get("reasoning_content").is_none(), "{}", without["messages"][2]);
}

#[tokio::test]
async fn rate_limits_are_retried_and_reported() {
    let server = Server::start(vec![(429, r#"{"error":{"message":"slow down"}}"#.into()), says("ok")]).await;
    let events = Mutex::new(Vec::new());
    let completion = server
        .provider(None)
        .complete(request(&[input("hi")], &[]), &|progress| {
            if let Progress::Event(event @ ProviderEvent::Retry { .. }) = progress {
                events.lock().unwrap().push(event);
            }
        })
        .await
        .unwrap();
    assert_eq!(completion.items, [Item::Assistant { text: "ok".into() }]);
    assert!(matches!(&events.lock().unwrap()[..], [ProviderEvent::Retry { status: Some(429), .. }]));
}

#[tokio::test]
async fn a_reply_cut_off_at_the_output_limit_fails() {
    let server = Server::start(vec![stream(&[delta(json!({"content": "partial"})), finish("length"), usage()])]).await;
    let error = server.provider(None).complete(request(&[input("hi")], &[]), &|_| {}).await.unwrap_err();
    assert!(error.to_string().contains("output limit"), "{error}");
}

#[tokio::test]
async fn an_agent_runs_tools_through_chat_completions() {
    let server = Server::start(vec![
        stream(&[
            delta(json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                "function": {"name": "read", "arguments": "{\"path\":\"notes.txt\"}"}}]})),
            finish("tool_calls"),
            usage(),
        ]),
        says("done"),
    ])
    .await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), "fixture\n").unwrap();
    let harness = Harness::builder(dir.path().join("state"))
        .provider("go", Arc::new(server.provider(Some("reasoning_content"))))
        .tools(tools::builtin())
        .open()
        .await
        .unwrap();
    let id = harness
        .create_agent(AgentSpec { provider: "go".into(), tools: vec!["read".into()], ..common::direct(dir.path()) })
        .unwrap();
    harness.send(&id, "user", "read notes.txt").unwrap();
    tokio::time::timeout(Duration::from_secs(10), harness.wait_for(&id, AgentState::Idle)).await.unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1]["messages"].as_array().unwrap().last().unwrap(),
        &json!({"role": "tool", "tool_call_id": "c1", "content": "fixture\n"})
    );
    let record = harness.agent(&id).unwrap();
    assert_eq!((record.usage.input, record.usage.cached_input), (200, 120));
    harness.shutdown().await;
}

#[tokio::test]
async fn images_sent_with_mail_follow_its_text() {
    let items = [Item::Input {
        from: "user".into(),
        text: "What is here?".into(),
        images: vec![erisharness::agent::Image { mime: "image/jpeg".into(), data: "/9j/".into() }],
    }];
    let server = Server::start(vec![says("ok")]).await;
    server.provider(None).complete(request(&items, &[]), &|_| {}).await.unwrap();
    assert_eq!(
        server.requests()[0]["messages"][1],
        json!({"role": "user", "content": [
            {"type": "text", "text": "What is here?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}},
        ]})
    );
}
