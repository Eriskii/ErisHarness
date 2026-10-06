//! OpenAI-compatible Chat Completions, streamed: the format most gateways and open-model hosts
//! speak. Each request carries the whole visible transcript.

use super::transport::{self, Events, Failure};
use super::{Completion, Credentials, Progress, Provider, RateGate, Request, user_images, user_text};
use crate::agent::{Item, Usage};
use crate::tools::{Content, parse_arguments};
use anyhow::{Result, anyhow};
use futures_util::future::BoxFuture;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct ChatCompletionsConfig {
    /// Endpoint root; requests go to `{base_url}/chat/completions`.
    pub base_url: String,
    /// Sent with every request.
    pub headers: Vec<(String, String)>,
    /// Attempts after the first for rate limits, server errors and dropped connections.
    pub max_retries: u32,
    pub credentials: Arc<dyn Credentials>,
    /// Paces calls for the account; share one gate among every provider using the account.
    pub gate: Arc<RateGate>,
    /// The assistant-message field in which the model expects its earlier reasoning back, such
    /// as `reasoning_content`. `None` sends no reasoning back.
    pub reasoning_field: Option<String>,
}

pub struct ChatCompletions {
    config: ChatCompletionsConfig,
    client: reqwest::Client,
}

impl ChatCompletions {
    pub fn new(config: ChatCompletionsConfig) -> Self {
        Self { config, client: reqwest::Client::new() }
    }

    fn body(&self, request: &Request) -> Value {
        let mut body = json!({
            "model": request.model,
            "messages": messages(request, self.config.reasoning_field.as_deref()),
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if !request.tools.is_empty() {
            body["tools"] = request
                .tools
                .iter()
                .map(|t| {
                    json!({"type": "function", "function": {
                        "name": t.name, "description": t.description, "parameters": t.parameters,
                    }})
                })
                .collect();
        }
        if let Some(effort) = request.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }
        body
    }

    async fn attempt(&self, body: &Value, progress: &(dyn Fn(Progress) + Send + Sync)) -> Result<Completion, Failure> {
        let authorization = self.config.credentials.authorize().await.map_err(Failure::fatal)?;
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        let mut request = self.client.post(url).bearer_auth(&authorization.token).json(body);
        for (key, value) in self.config.headers.iter().chain(&authorization.headers) {
            request = request.header(key, value);
        }
        let mut events = Events::new(transport::send(request, progress).await?);
        let mut reply = Reply::default();
        while let Some(data) = events.next().await {
            let data = data.map_err(Failure::retry)?;
            if data.trim() == "[DONE]" {
                return reply.finish();
            }
            let Ok(chunk) = serde_json::from_str::<Value>(&data) else { continue };
            if let Some(error) = chunk.get("error").filter(|e| !e.is_null()) {
                let message = error["message"].as_str().unwrap_or("stream error");
                return Err(match error["code"].as_u64() {
                    Some(status @ (429 | 500..=599)) => Failure::status(anyhow!("{status} {message}"), status as u16),
                    _ => Failure::fatal(anyhow!("{message}")),
                });
            }
            reply.take(&chunk, progress);
        }
        // Some servers end the stream without [DONE] once the reply is finished.
        if reply.finish_reason.is_some() {
            return reply.finish();
        }
        Err(Failure::retry(anyhow!("stream ended before the response completed")))
    }
}

impl Provider for ChatCompletions {
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

/// A reply as its chunks arrive: text, reasoning and tool calls, each call's pieces by index.
#[derive(Default)]
struct Reply {
    text: String,
    reasoning: String,
    calls: BTreeMap<u64, (String, String, String)>,
    usage: Value,
    finish_reason: Option<String>,
}

impl Reply {
    fn take(&mut self, chunk: &Value, progress: &(dyn Fn(Progress) + Send + Sync)) {
        if chunk["usage"].is_object() {
            self.usage = chunk["usage"].clone();
        }
        let Some(choice) = chunk["choices"].as_array().and_then(|choices| choices.first()) else { return };
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            self.text.push_str(text);
            progress(Progress::Text(text));
        }
        if let Some(reasoning) = delta["reasoning_content"].as_str().or(delta["reasoning"].as_str()) {
            self.reasoning.push_str(reasoning);
        }
        for (position, call) in delta["tool_calls"].as_array().into_iter().flatten().enumerate() {
            let index = call["index"].as_u64().unwrap_or(position as u64);
            let (id, name, arguments) = self.calls.entry(index).or_default();
            if let Some(value) = call["id"].as_str() {
                *id = value.to_owned();
            }
            if let Some(value) = call["function"]["name"].as_str() {
                name.push_str(value);
            }
            if let Some(value) = call["function"]["arguments"].as_str() {
                arguments.push_str(value);
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_owned());
        }
    }

    fn finish(self) -> Result<Completion, Failure> {
        if self.finish_reason.as_deref() == Some("length") {
            return Err(Failure::fatal(anyhow!("the reply stopped at the output limit")));
        }
        let mut items = Vec::new();
        if !self.reasoning.is_empty() {
            items.push(Item::Reasoning { summary: vec![self.reasoning], encrypted: None });
        }
        if !self.text.is_empty() {
            items.push(Item::Assistant { text: self.text });
        }
        items.extend(self.calls.into_values().map(|(call_id, name, arguments)| Item::ToolCall {
            call_id,
            name,
            arguments,
        }));
        let n = |v: &Value| v.as_u64().unwrap_or(0);
        let usage = &self.usage;
        let details = &usage["prompt_tokens_details"];
        Ok(Completion {
            items,
            usage: Usage {
                input: n(&usage["prompt_tokens"]),
                cached_input: details["cached_tokens"].as_u64().unwrap_or_else(|| n(&usage["prompt_cache_hit_tokens"])),
                cache_write: n(&details["cache_write_tokens"]),
                output: n(&usage["completion_tokens"]),
                reasoning: n(&usage["completion_tokens_details"]["reasoning_tokens"]),
            },
        })
    }
}

/// The transcript as Chat Completions messages. A model turn's reasoning, text and tool calls
/// form one assistant message; tool results follow as tool messages, and images they carry
/// follow those in a user message, since tool messages hold only text.
fn messages(request: &Request, reasoning_field: Option<&str>) -> Vec<Value> {
    let mut messages = Vec::new();
    if !request.system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": request.system}));
    }
    let mut images: Vec<Value> = Vec::new();
    for item in request.items {
        if !matches!(item, Item::ToolResult { .. }) && !images.is_empty() {
            messages.push(json!({"role": "user", "content": std::mem::take(&mut images)}));
        }
        if let Some(text) = user_text(item) {
            let attached = user_images(item);
            let content = if attached.is_empty() {
                json!(text)
            } else {
                let images = attached.iter().map(|i| {
                    json!({"type": "image_url", "image_url": {"url": format!("data:{};base64,{}", i.mime, i.data)}})
                });
                json!(std::iter::once(json!({"type": "text", "text": text})).chain(images).collect::<Vec<_>>())
            };
            messages.push(json!({"role": "user", "content": content}));
            continue;
        }
        match item {
            Item::Assistant { text } => {
                let message = assistant(&mut messages);
                let content = message.get("content").and_then(Value::as_str).unwrap_or_default();
                message.insert("content".into(), json!(format!("{content}{text}")));
            }
            Item::Reasoning { summary, encrypted: None } => {
                if let Some(field) = reasoning_field {
                    assistant(&mut messages).insert(field.into(), json!(summary.join("\n")));
                }
            }
            Item::ToolCall { call_id, name, arguments } => {
                // A call whose arguments were not valid JSON goes back as an object holding them.
                let arguments = match parse_arguments(arguments) {
                    Ok(_) => arguments.clone(),
                    Err(_) => json!({"INVALID_JSON": arguments}).to_string(),
                };
                let call =
                    json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}});
                let message = assistant(&mut messages);
                match message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    Some(calls) => calls.push(call),
                    None => _ = message.insert("tool_calls".into(), json!([call])),
                }
            }
            Item::ToolResult { call_id, output } => {
                let mut text = String::new();
                for content in &output.content {
                    match content {
                        Content::Text(part) => text.push_str(part),
                        Content::Image { mime, data } => {
                            if images.is_empty() {
                                images
                                    .push(json!({"type": "text", "text": format!("Images from tool call {call_id}:")}));
                            }
                            images.push(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{data}")}}));
                        }
                    }
                }
                messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": text}));
            }
            _ => {}
        }
    }
    if !images.is_empty() {
        messages.push(json!({"role": "user", "content": images}));
    }
    messages
}

/// The assistant message a model turn's items join: the last message if it is one, or a new one.
fn assistant(messages: &mut Vec<Value>) -> &mut Map<String, Value> {
    if messages.last().is_none_or(|m| m["role"] != "assistant") {
        messages.push(json!({"role": "assistant", "content": null}));
    }
    messages.last_mut().and_then(Value::as_object_mut).expect("an assistant message")
}
