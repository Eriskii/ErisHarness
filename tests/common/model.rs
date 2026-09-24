//! A scripted OpenAI Responses server: each request pops the next reply and is recorded.

use super::{block_on, runtime};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use erisharness::provider::{Credentials, Responses, ResponsesConfig, StaticToken};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// One scripted reply: an HTTP status with headers, or an SSE stream of Responses events.
#[derive(Clone)]
pub enum Reply {
    Status(u16, Vec<(&'static str, &'static str)>),
    Events(Vec<Value>),
    Delayed(Duration, Vec<Value>),
}

#[derive(Default)]
pub struct Script {
    pub replies: VecDeque<Reply>,
    pub requests: Vec<Value>,
    pub auth: Vec<String>,
    pub headers: Vec<HeaderMap>,
}

pub type Shared = Arc<Mutex<Script>>;

pub struct Model {
    pub url: String,
    pub script: Shared,
}

impl Model {
    pub fn start(replies: Vec<Reply>) -> Self {
        let script: Shared = Arc::new(Mutex::new(Script { replies: replies.into(), ..Script::default() }));
        let app = axum::Router::new().route("/v1/responses", axum::routing::post(respond)).with_state(script.clone());
        let listener = block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        runtime().spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, script }
    }

    pub fn requests(&self) -> Vec<Value> {
        self.script.lock().unwrap().requests.clone()
    }

    pub fn provider(&self) -> Arc<Responses> {
        self.provider_with(Arc::new(StaticToken("secret-token".into())))
    }

    pub fn provider_with(&self, credentials: Arc<dyn Credentials>) -> Arc<Responses> {
        Arc::new(Responses::new(ResponsesConfig {
            base_url: self.url.clone(),
            headers: vec![("x-test".into(), "1".into())],
            store: false,
            max_retries: 3,
            credentials,
        }))
    }

    pub fn push(&self, reply: Reply) {
        self.script.lock().unwrap().replies.push_back(reply);
    }
}

async fn respond(State(script): State<Shared>, headers: HeaderMap, body: String) -> Response {
    let reply = {
        let mut script = script.lock().unwrap();
        script.requests.push(serde_json::from_str(&body).unwrap());
        script.auth.push(headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned());
        script.headers.push(headers);
        script.replies.pop_front()
    };
    let events = match reply {
        None => return (StatusCode::INTERNAL_SERVER_ERROR, "script exhausted").into_response(),
        Some(Reply::Status(code, headers)) => {
            let mut response =
                (StatusCode::from_u16(code).unwrap(), "{\"error\":{\"message\":\"scripted\"}}").into_response();
            for (key, value) in headers {
                response.headers_mut().insert(key, value.parse().unwrap());
            }
            return response;
        }
        Some(Reply::Events(events)) => events,
        Some(Reply::Delayed(delay, events)) => {
            tokio::time::sleep(delay).await;
            events
        }
    };
    let body: String =
        events.iter().map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap())).collect();
    ([("content-type", "text/event-stream")], body).into_response()
}

pub fn completed() -> Value {
    json!({"type": "response.completed", "response": {"usage": {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 4}, "output_tokens_details": {"reasoning_tokens": 1}}}})
}

pub fn says(text: &str) -> Reply {
    Reply::Events(vec![
        json!({"type": "response.output_text.delta", "delta": &text[..text.len() / 2]}),
        json!({"type": "response.output_text.delta", "delta": &text[text.len() / 2..]}),
        json!({"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}}),
        completed(),
    ])
}

pub fn calls(call_id: &str, name: &str, args: Value) -> Reply {
    Reply::Events(vec![
        json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": call_id, "name": name, "arguments": args.to_string()}}),
        completed(),
    ])
}
