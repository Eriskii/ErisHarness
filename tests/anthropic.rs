//! Protocol tests use a real local HTTP server; no account, subscription or Docker needed.
use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use erisharness::provider::anthropic::fingerprint::{self, DEFAULT_VERSION, IDENTITY};
use erisharness::{
    Harness,
    agent::{AgentSpec, AgentState, Item, Observation},
    machine::{DirectSpec, MachineSpec},
    provider::{
        Anthropic, AnthropicAuth, AnthropicConfig, ClaudeCode, Progress, Provider, ProviderEvent, Request, StaticToken,
        Thinking, ToolSpec,
    },
    tools,
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

struct Reply(u16, String, Vec<(&'static str, &'static str)>);
#[derive(Default)]
struct Script {
    replies: VecDeque<Reply>,
    requests: Vec<(HeaderMap, String, String)>,
}
struct Server {
    url: String,
    script: Arc<Mutex<Script>>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    async fn new(replies: Vec<Reply>) -> Self {
        async fn handle(
            State(state): State<Arc<Mutex<Script>>>,
            uri: axum::http::Uri,
            headers: HeaderMap,
            body: String,
        ) -> Response {
            let reply = {
                let mut state = state.lock().unwrap();
                state.requests.push((headers, body, uri.to_string()));
                state.replies.pop_front().unwrap_or(Reply(500, "exhausted".into(), vec![]))
            };
            let mut response = (StatusCode::from_u16(reply.0).unwrap(), reply.1).into_response();
            response.headers_mut().insert("content-type", "text/event-stream".parse().unwrap());
            response.headers_mut().insert("anthropic-ratelimit-unified-status", "allowed".parse().unwrap());
            response.headers_mut().insert("set-cookie", "never-log-this".parse().unwrap());
            for (k, v) in reply.2 {
                response.headers_mut().insert(k, v.parse().unwrap());
            }
            response
        }
        let script = Arc::new(Mutex::new(Script { replies: replies.into(), ..Script::default() }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let app = Router::new().route("/v1/messages", post(handle)).with_state(script.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, script, task }
    }
    fn config(&self, oauth: bool) -> AnthropicConfig {
        let auth = if oauth {
            AnthropicAuth::ClaudeCode(ClaudeCode {
                install_id: "test-install".into(),
                account_id: Some("account".into()),
                version: None,
            })
        } else {
            AnthropicAuth::ApiKey
        };
        let mut config = AnthropicConfig::new(Arc::new(StaticToken("test-secret".into())), auth);
        config.base_url = self.url.clone();
        config.thinking = Thinking::Adaptive { display: true };
        config.timeout = Duration::from_secs(5);
        config
    }
}
fn events(blocks: Vec<Value>, stop: &str) -> Reply {
    streamed_events(blocks, stop, &[])
}
fn streamed_events(blocks: Vec<Value>, stop: &str, deltas: &[(usize, &str)]) -> Reply {
    let mut events = vec![
        json!({"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":20,"cache_creation_input_tokens":30,"output_tokens":1}}}),
    ];
    for (i, b) in blocks.into_iter().enumerate() {
        events.push(json!({"type":"content_block_start","index":i,"content_block":b}));
        for (_, partial) in deltas.iter().filter(|(index, _)| *index == i) {
            events.push(json!({"type":"content_block_delta","index":i,
                "delta":{"type":"input_json_delta","partial_json":partial}}));
        }
        events.push(json!({"type":"content_block_stop","index":i}));
    }
    events.extend([
        json!({"type":"message_delta","delta":{"stop_reason":stop},"usage":{"output_tokens":7}}),
        json!({"type":"message_stop"}),
    ]);
    Reply(
        200,
        events.into_iter().map(|v| format!("event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap())).collect(),
        vec![],
    )
}
fn says() -> Reply {
    events(vec![json!({"type":"text","text":"done"})], "end_turn")
}
fn request<'a>(items: &'a [Item], tools: &'a [ToolSpec]) -> Request<'a> {
    Request {
        model: "claude-test",
        reasoning_effort: Some("high"),
        cache_key: "session-id",
        system: "Review code.",
        tools,
        items,
    }
}

#[tokio::test]
async fn oauth_wire_checksum_headers_cache_and_tool_names() {
    let server = Server::new(vec![says()]).await;
    let mut config = server.config(true);
    config.max_tokens = 128000;
    config.headers = vec![
        ("Authorization".into(), "wrong".into()),
        ("x-api-key".into(), "wrong".into()),
        ("User-Agent".into(), "wrong".into()),
    ];
    let provider = Anthropic::new(config).unwrap();
    let seen = Mutex::new(Vec::new());
    let result = provider
        .complete(
            request(
                &[Item::Input { from: "user".into(), text: "hello cch=00000 😀".into() }],
                &[ToolSpec { name: "_read".into(), description: "Read".into(), parameters: json!({"type":"object"}) }],
            ),
            &|p| {
                if let Progress::Event(e) = p {
                    seen.lock().unwrap().push(e);
                }
            },
        )
        .await
        .unwrap();
    assert_eq!(result.usage.input, 60);
    assert_eq!(result.usage.cached_input, 20);
    assert_eq!(result.usage.output, 7);
    let script = server.script.lock().unwrap();
    let (headers, raw, uri) = &script.requests[0];
    assert_eq!(uri, "/v1/messages?beta=true");
    assert_eq!(headers["authorization"], "Bearer test-secret");
    assert!(!headers.contains_key("x-api-key"));
    assert_eq!(headers["user-agent"], format!("claude-cli/{DEFAULT_VERSION} (external, cli)"));
    assert_eq!(headers["x-claude-code-session-id"], "session-id");
    assert!(headers["anthropic-beta"].to_str().unwrap().contains("oauth-2025-04-20"));
    assert!(!headers["anthropic-beta"].to_str().unwrap().contains("context-1m"));
    let mut body: Value = serde_json::from_str(raw).unwrap();
    assert_eq!(body["system"][1]["text"], IDENTITY);
    assert_eq!(body["system"][2]["text"], "Review code.");
    assert_eq!(body["system"][1]["cache_control"]["ttl"], "1h");
    assert!(body["system"][0].get("cache_control").is_none());
    assert_eq!(body["tools"][0]["name"], "__read");
    assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(body["messages"][0]["content"][0]["text"], "hello cch=00000 😀");
    assert_eq!(body["messages"][0]["content"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(body["max_tokens"], 64000);
    let metadata: Value = serde_json::from_str(body["metadata"]["user_id"].as_str().unwrap()).unwrap();
    assert_eq!(metadata["session_id"], "session-id");
    assert_eq!(metadata["device_id"], fingerprint::device_id("test-install", Some("account")));
    let billing = body["system"][0]["text"].as_str().unwrap().to_owned();
    let at = billing.find("cch=").unwrap() + 4;
    let hash = &billing[at..at + 5];
    let mut placeholder = billing.clone();
    placeholder.replace_range(at..at + 5, "00000");
    body["system"][0]["text"] = json!(placeholder);
    assert_eq!(hash, fingerprint::checksum(serde_json::to_string(&body).unwrap().as_bytes()));
    assert!(!format!("{:?}", seen.lock().unwrap()).contains("never-log-this"));
}

#[tokio::test]
async fn api_key_requests_have_no_oauth_identity() {
    let server = Server::new(vec![says()]).await;
    let provider = Anthropic::new(server.config(false)).unwrap();
    provider
        .complete(request(&[Item::Input { from: "user".into(), text: "hello".into() }], &[]), &|_| {})
        .await
        .unwrap();
    let script = server.script.lock().unwrap();
    let (headers, raw, uri) = &script.requests[0];
    assert_eq!(uri, "/v1/messages");
    assert_eq!(headers["x-api-key"], "test-secret");
    assert!(!headers.contains_key("authorization"));
    assert!(!raw.contains("billing-header"));
    assert!(!raw.contains(IDENTITY));
}

#[tokio::test]
async fn signed_thinking_and_tools_round_trip_through_the_harness() {
    let signed = json!({"type":"thinking","thinking":"Inspect the file.","signature":"opaque-signature"});
    let server = Server::new(vec![
        events(
            vec![signed.clone(), json!({"type":"tool_use","id":"call_1","name":"_read","input":{"path":"README.md"}})],
            "tool_use",
        ),
        says(),
    ])
    .await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), "fixture contents").unwrap();
    let harness = Harness::builder(dir.path().join("state"))
        .provider("claude", Arc::new(Anthropic::new(server.config(true)).unwrap()))
        .tools(tools::builtin())
        .open()
        .await
        .unwrap();
    let mut observations = harness.subscribe();
    let id = harness
        .create_agent(AgentSpec {
            system_prompt: "Review".into(),
            tools: vec!["read".into()],
            provider: "claude".into(),
            model: "claude-test".into(),
            reasoning_effort: Some("high".into()),
            context_window: None,
            metadata: Value::Null,
            machine: MachineSpec::Direct(DirectSpec { cwd: dir.path().to_str().unwrap().into(), env: None }),
        })
        .unwrap();
    harness.send(&id, "user", "Read README.md").unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Observation::State { state: AgentState::Idle | AgentState::Failed, .. } =
                observations.recv().await.unwrap()
            {
                break;
            }
        }
    })
    .await
    .unwrap();
    let agent = harness.agent(&id).unwrap();
    assert_eq!(agent.state, AgentState::Idle, "{:?}", agent.error);
    {
        let script = server.script.lock().unwrap();
        assert_eq!(script.requests.len(), 2);
        let body: Value = serde_json::from_str(&script.requests[1].1).unwrap();
        assert_eq!(body["messages"][1]["content"][0], signed);
        assert_eq!(body["messages"][1]["content"][1]["name"], "_read");
        assert!(body["messages"][2]["content"][0]["content"][0]["text"].as_str().unwrap().contains("fixture contents"));
    }
    harness.shutdown().await;
}

#[tokio::test]
async fn malformed_tool_arguments_are_returned_to_the_model_and_can_be_corrected() {
    use erisharness::tools::{Tool, ToolContext, ToolOutput};
    use futures_util::future::BoxFuture;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Count(Arc<AtomicUsize>);
    impl Tool for Count {
        fn name(&self) -> &str {
            "count"
        }
        fn description(&self) -> String {
            "Count executions".into()
        }
        fn parameters(&self) -> Value {
            json!({"type":"object"})
        }
        fn call<'a>(&'a self, _: &'a ToolContext, _: Value) -> BoxFuture<'a, ToolOutput> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { ToolOutput::text("executed") })
        }
    }
    // Finished but malformed streams, including valid JSON of the wrong type.
    for raw in [r#"{"path":"fixture","line":}"#, r#"{"path":"fixture""#, r#"{"path":"C:\q"}"#, "[]", "null", "", "   "]
    {
        let (first, second) = raw.split_at(raw.len() / 2);
        let server = Server::new(vec![
            streamed_events(
                vec![
                    json!({"type":"tool_use","id":"bad","name":"_count","input":{}}),
                    json!({"type":"tool_use","id":"good","name":"_count","input":{}}),
                ],
                "tool_use",
                &[(0, first), (0, second)],
            ),
            streamed_events(
                vec![json!({"type":"tool_use","id":"fixed","name":"_count","input":{}})],
                "tool_use",
                &[(0, "{\"path\":"), (0, "\"fixture\"}")],
            ),
            says(),
        ])
        .await;
        let count = Arc::new(AtomicUsize::new(0));
        let dir = tempfile::tempdir().unwrap();
        let harness = Harness::builder(dir.path().join("state"))
            .provider("claude", Arc::new(Anthropic::new(server.config(true)).unwrap()))
            .tools(vec![Arc::new(Count(count.clone())) as Arc<dyn Tool>])
            .open()
            .await
            .unwrap();
        let mut observations = harness.subscribe();
        let id = harness
            .create_agent(AgentSpec {
                system_prompt: "Test recovery".into(),
                tools: vec!["count".into()],
                provider: "claude".into(),
                model: "claude-test".into(),
                reasoning_effort: None,
                context_window: None,
                metadata: Value::Null,
                machine: MachineSpec::Direct(DirectSpec { cwd: dir.path().to_str().unwrap().into(), env: None }),
            })
            .unwrap();
        harness.send(&id, "user", "Call count").unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Observation::State { state: AgentState::Idle | AgentState::Failed, .. } =
                    observations.recv().await.unwrap()
                {
                    break;
                }
            }
        })
        .await
        .unwrap();
        let agent = harness.agent(&id).unwrap();
        assert_eq!(agent.state, AgentState::Idle, "{raw:?}: {:?}", agent.error);
        assert_eq!(count.load(Ordering::SeqCst), 2, "invalid call must not execute: {raw:?}");
        let transcript = harness.transcript(&id).unwrap();
        assert!(transcript.iter().any(|entry| matches!(&entry.item,
            Item::ToolCall {call_id, arguments, ..} if call_id == "bad" && arguments == raw)));
        {
            let script = server.script.lock().unwrap();
            assert_eq!(script.requests.len(), 3);
            let body: Value = serde_json::from_str(&script.requests[1].1).unwrap();
            assert_eq!(body["messages"][1]["content"][0]["input"]["INVALID_JSON"], raw);
            let result = &body["messages"][2]["content"][0];
            assert_eq!(result["tool_use_id"], "bad");
            assert_eq!(result["is_error"], true);
            let diagnostic: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
            assert_eq!(diagnostic["INVALID_JSON"], raw);
            assert!(diagnostic["error"].as_str().unwrap().contains("Send a corrected tool call"));
            assert_eq!(body["messages"][2]["content"][1]["is_error"], false);
        }
        harness.shutdown().await;
    }
}

#[tokio::test]
async fn retries_rate_limits_and_reads_credentials_again() {
    let server = Server::new(vec![
        Reply(
            429,
            json!({"error":{"type":"rate_limit_error","message":"limited"}}).to_string(),
            vec![("retry-after-ms", "1")],
        ),
        says(),
    ])
    .await;
    let provider = Anthropic::new(server.config(true)).unwrap();
    let seen = Mutex::new(Vec::new());
    provider
        .complete(request(&[Item::Input { from: "user".into(), text: "hello".into() }], &[]), &|p| {
            if let Progress::Event(e) = p {
                seen.lock().unwrap().push(e);
            }
        })
        .await
        .unwrap();
    assert_eq!(server.script.lock().unwrap().requests.len(), 2);
    assert!(
        seen.lock().unwrap().iter().any(|e| matches!(e, ProviderEvent::Retry { status: Some(429), delay_ms: 1, .. }))
    );
}

#[tokio::test]
async fn version_upgrade_rebuilds_header_and_attestation_with_zero_normal_retries() {
    let server = Server::new(vec![
        Reply(
            400,
            json!({"error":{"type":"claude_code_version_too_old","message":"version 9.0.0 or newer is required"}})
                .to_string(),
            vec![],
        ),
        says(),
    ])
    .await;
    let mut config = server.config(true);
    config.max_retries = 0;
    let provider = Anthropic::new(config).unwrap();
    provider
        .complete(request(&[Item::Input { from: "user".into(), text: "hello".into() }], &[]), &|_| {})
        .await
        .unwrap();
    let script = server.script.lock().unwrap();
    assert_eq!(script.requests.len(), 2);
    assert_eq!(script.requests[1].0["user-agent"], "claude-cli/9.0.0 (external, cli)");
    assert!(script.requests[1].1.contains("cc_version=9.0.0."));
    assert_ne!(script.requests[0].1, script.requests[1].1);
}

#[tokio::test]
async fn truncated_stream_is_not_replayed_after_content() {
    let mut reply = says();
    reply.1 = reply.1.split("event: message_stop").next().unwrap().to_owned();
    let server = Server::new(vec![reply, says()]).await;
    let provider = Anthropic::new(server.config(true)).unwrap();
    let error = provider
        .complete(request(&[Item::Input { from: "user".into(), text: "hello".into() }], &[]), &|_| {})
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("message_stop"));
    assert_eq!(server.script.lock().unwrap().requests.len(), 1);
}
