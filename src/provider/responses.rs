//! OpenAI Responses API, streamed. Requests are stateless (`store: false` by default): the
//! whole transcript is sent each time, and encrypted reasoning is carried back in it.

use super::sse::Parser;
use super::{Completion, Credentials, Progress, Provider, Request};
use crate::agent::{Item, Usage};
use crate::tools::{Content, ToolOutput};
use anyhow::{Result, anyhow, bail};
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use reqwest::StatusCode;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
        let authorization = self.config.credentials.authorize().await.map_err(Failure::Fatal)?;
        let mut builder = self
            .client
            .post(format!("{}/responses", self.config.base_url.trim_end_matches('/')))
            .bearer_auth(&authorization.token)
            .json(body);
        for (key, value) in self.config.headers.iter().chain(&authorization.headers) {
            builder = builder.header(key, value);
        }
        let response = builder.send().await.map_err(|e| Failure::Retry(anyhow!(e), None))?;
        let status = response.status();
        if !status.is_success() {
            let wait = retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_owned))
                .unwrap_or(text);
            let error = anyhow!("{} {message}", status.as_u16());
            return Err(if status == StatusCode::TOO_MANY_REQUESTS {
                Failure::RateLimited(error, wait)
            } else if status.is_server_error() {
                Failure::Retry(error, wait)
            } else {
                Failure::Fatal(error)
            });
        }
        let mut stream = response.bytes_stream();
        let mut parser = Parser::default();
        let mut items = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Failure::Retry(anyhow!(e), None))?;
            for data in parser.feed(&chunk) {
                let Ok(event) = serde_json::from_str::<Value>(&data) else { continue };
                match event["type"].as_str().unwrap_or_default() {
                    "response.output_text.delta" => {
                        progress(Progress::Text(event["delta"].as_str().unwrap_or_default()))
                    }
                    "response.output_item.done" => items.extend(output_item(&event["item"])),
                    "response.completed" | "response.incomplete" => {
                        return Ok(Completion { items, usage: usage(&event["response"]["usage"]) });
                    }
                    "response.failed" | "error" => {
                        let error = &event["response"]["error"];
                        let message =
                            error["message"].as_str().or(event["message"].as_str()).unwrap_or("response failed");
                        return Err(Failure::Fatal(anyhow!("{message}")));
                    }
                    _ => {}
                }
            }
        }
        Err(Failure::Retry(anyhow!("stream ended before the response completed"), None))
    }
}

enum Failure {
    Retry(anyhow::Error, Option<Duration>),
    RateLimited(anyhow::Error, Option<Duration>),
    Fatal(anyhow::Error),
}

impl Provider for Responses {
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        progress: &'a (dyn Fn(Progress) + Send + Sync),
    ) -> BoxFuture<'a, Result<Completion>> {
        Box::pin(async move {
            let body = self.body(&request);
            let mut backoff = Duration::from_millis(250);
            for attempt in 0..=self.config.max_retries {
                let (error, wait, limited) = match self.attempt(&body, progress).await {
                    Ok(completion) => return Ok(completion),
                    Err(Failure::Fatal(error)) => return Err(error),
                    Err(Failure::Retry(error, wait)) => (error, wait, false),
                    Err(Failure::RateLimited(error, wait)) => (error, wait, true),
                };
                if attempt == self.config.max_retries {
                    return Err(error);
                }
                let wait = wait.unwrap_or(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(30));
                if limited {
                    let until = (SystemTime::now() + wait).duration_since(UNIX_EPOCH).unwrap_or_default();
                    progress(Progress::RateLimited { until: Some(until.as_millis() as u64) });
                }
                tokio::time::sleep(wait).await;
                if limited {
                    progress(Progress::RateLimited { until: None });
                }
            }
            bail!("no attempts made")
        })
    }
}

fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let number = |name: &str| headers.get(name)?.to_str().ok()?.trim().parse::<f64>().ok();
    number("retry-after-ms")
        .map(|ms| Duration::from_secs_f64(ms / 1000.0))
        .or_else(|| number("retry-after").map(Duration::from_secs_f64))
}

fn usage(value: &Value) -> Usage {
    let n = |v: &Value| v.as_u64().unwrap_or(0);
    Usage {
        input: n(&value["input_tokens"]),
        cached_input: n(&value["input_tokens_details"]["cached_tokens"]),
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
        Item::Input { from, text } => {
            let text = if from == "user" { text.clone() } else { format!("[Message from agent {from}]\n{text}") };
            json!({"role": "user", "content": [{"type": "input_text", "text": text}]})
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
        Item::Compaction { summary } => json!({"role": "user", "content": [{"type": "input_text", "text":
            format!("The conversation so far was compacted. Summary:\n\n{summary}\n\nContinue from here.")}]}),
    })
}

/// Plain text stays a string; results carrying images become a content list.
fn tool_output(output: &ToolOutput) -> Value {
    if output.content.iter().all(|c| matches!(c, Content::Text(_))) {
        let text: String = output
            .content
            .iter()
            .filter_map(|c| if let Content::Text(t) = c { Some(t.as_str()) } else { None })
            .collect();
        return Value::String(text);
    }
    Value::Array(
        output
            .content
            .iter()
            .map(|c| match c {
                Content::Text(text) => json!({"type": "input_text", "text": text}),
                Content::Image { mime, data } => {
                    json!({"type": "input_image", "image_url": format!("data:{mime};base64,{data}")})
                }
            })
            .collect(),
    )
}
