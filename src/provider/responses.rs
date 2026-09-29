//! The OpenAI Responses API, streamed. Requests are stateless by default (`store: false`):
//! the whole visible transcript is sent each time, carrying encrypted reasoning back with it.

use super::transport::{self, Events, Failure};
use super::{Completion, Credentials, Progress, Provider, RateGate, Request, user_text};
use crate::agent::{Item, Usage};
use crate::tools::{Content, ToolOutput};
use anyhow::{Result, anyhow};
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::Arc;

pub struct ResponsesConfig {
    /// Endpoint root; requests go to `{base_url}/responses`.
    pub base_url: String,
    /// Sent with every request, such as account or beta headers.
    pub headers: Vec<(String, String)>,
    /// Whether the provider keeps responses server-side.
    pub store: bool,
    /// Attempts after the first for rate limits, server errors and dropped connections.
    pub max_retries: u32,
    pub credentials: Arc<dyn Credentials>,
    /// Paces calls for the account; share one gate among every provider using the account.
    pub gate: Arc<RateGate>,
}

pub struct Responses {
    config: ResponsesConfig,
    client: reqwest::Client,
}

impl Responses {
    pub fn new(config: ResponsesConfig) -> Self {
        Self { config, client: reqwest::Client::new() }
    }

    fn body(&self, request: &Request) -> Value {
        let mut body = json!({
            "model": request.model,
            "instructions": request.system,
            "input": request.items.iter().filter_map(input_item).collect::<Vec<_>>(),
            "tools": request.tools.iter().map(|t| json!({
                "type": "function", "name": t.name, "description": t.description,
                "parameters": t.parameters, "strict": false,
            })).collect::<Vec<_>>(),
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "stream": true,
            "store": self.config.store,
            "prompt_cache_key": request.cache_key,
        });
        if let Some(effort) = request.reasoning_effort {
            body["reasoning"] = json!({"effort": effort, "summary": "auto"});
        }
        if !self.config.store {
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
        body
    }

    async fn attempt(&self, body: &Value, progress: &(dyn Fn(Progress) + Send + Sync)) -> Result<Completion, Failure> {
        let authorization = self.config.credentials.authorize().await.map_err(Failure::fatal)?;
        let url = format!("{}/responses", self.config.base_url.trim_end_matches('/'));
        let mut request = self.client.post(url).bearer_auth(&authorization.token).json(body);
        for (key, value) in self.config.headers.iter().chain(&authorization.headers) {
            request = request.header(key, value);
        }
        let mut events = Events::new(transport::send(request, progress).await?);
        let mut items = Vec::new();
        while let Some(data) = events.next().await {
            let Ok(event) = serde_json::from_str::<Value>(&data.map_err(Failure::retry)?) else { continue };
            match event["type"].as_str().unwrap_or_default() {
                "response.output_text.delta" => progress(Progress::Text(event["delta"].as_str().unwrap_or_default())),
                "response.output_item.done" => items.extend(output_item(&event["item"])),
                "response.completed" | "response.incomplete" => {
                    return Ok(Completion { items, usage: usage(&event["response"]["usage"]) });
                }
                "response.failed" | "error" => {
                    let message = event["response"]["error"]["message"].as_str().or(event["message"].as_str());
                    return Err(Failure::fatal(anyhow!("{}", message.unwrap_or("response failed"))));
                }
                _ => {}
            }
        }
        Err(Failure::retry(anyhow!("stream ended before the response completed")))
    }
}

impl Provider for Responses {
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        progress: &'a (dyn Fn(Progress) + Send + Sync),
    ) -> BoxFuture<'a, Result<Completion>> {
        Box::pin(async move {
            let body = self.body(&request);
            transport::retrying(&self.config.gate, self.config.max_retries, progress, || self.attempt(&body, progress))
                .await
        })
    }
}

fn usage(value: &Value) -> Usage {
    let n = |v: &Value| v.as_u64().unwrap_or(0);
    Usage {
        input: n(&value["input_tokens"]),
        cached_input: n(&value["input_tokens_details"]["cached_tokens"]),
        cache_write: 0,
        output: n(&value["output_tokens"]),
        reasoning: n(&value["output_tokens_details"]["reasoning_tokens"]),
    }
}

fn output_item(item: &Value) -> Option<Item> {
    let text = |v: &Value| v.as_str().unwrap_or_default().to_owned();
    match item["type"].as_str()? {
        "message" => Some(Item::Assistant {
            text: item["content"]
                .as_array()?
                .iter()
                .filter(|part| part["type"] == "output_text")
                .map(|part| text(&part["text"]))
                .collect(),
        }),
        "reasoning" => Some(Item::Reasoning {
            summary: item["summary"]
                .as_array()
                .map_or_else(Vec::new, |parts| parts.iter().map(|p| text(&p["text"])).collect()),
            encrypted: item["encrypted_content"].as_str().map(str::to_owned),
        }),
        "function_call" => Some(Item::ToolCall {
            call_id: text(&item["call_id"]),
            name: text(&item["name"]),
            arguments: text(&item["arguments"]),
        }),
        _ => None,
    }
}

fn input_item(item: &Item) -> Option<Value> {
    Some(match item {
        Item::Input { .. } | Item::Compaction { .. } => {
            json!({"role": "user", "content": [{"type": "input_text", "text": user_text(item)?}]})
        }
        Item::Assistant { text } => {
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
        }
        Item::Reasoning { summary, encrypted } => json!({
            "type": "reasoning",
            "summary": summary.iter().map(|s| json!({"type": "summary_text", "text": s})).collect::<Vec<_>>(),
            "encrypted_content": encrypted.as_ref()?,
        }),
        Item::ToolCall { call_id, name, arguments } => {
            json!({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})
        }
        Item::ToolResult { call_id, output } => {
            json!({"type": "function_call_output", "call_id": call_id, "output": tool_output(output)})
        }
    })
}

/// Plain text stays a string; results carrying images become a content list.
fn tool_output(output: &ToolOutput) -> Value {
    let mut text = String::new();
    for content in &output.content {
        match content {
            Content::Text(part) => text.push_str(part),
            Content::Image { .. } => return Value::Array(output.content.iter().map(content_part).collect()),
        }
    }
    Value::String(text)
}

fn content_part(content: &Content) -> Value {
    match content {
        Content::Text(text) => json!({"type": "input_text", "text": text}),
        Content::Image { mime, data } => {
            json!({"type": "input_image", "image_url": format!("data:{mime};base64,{data}")})
        }
    }
}
