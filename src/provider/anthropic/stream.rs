//! Decodes a streamed Anthropic message. Text streams to observers as it arrives; items are
//! built only from finished blocks. Tool arguments stay verbatim, so malformed JSON reaches
//! the harness, which answers it with a tool error.

use crate::agent::{Item, Usage};
use crate::provider::{Completion, Progress};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub struct Stream<'a> {
    model: &'a str,
    /// Tool names arrive in Claude Code's prefixed form.
    oauth: bool,
    blocks: BTreeMap<u64, Block>,
    usage: Value,
    started: bool,
    stop: Option<String>,
}

struct Block {
    value: Value,
    json: Option<String>,
    stopped: bool,
}

impl<'a> Stream<'a> {
    pub fn new(model: &'a str, oauth: bool) -> Self {
        Self { model, oauth, blocks: BTreeMap::new(), usage: Value::Null, started: false, stop: None }
    }

    /// Takes the next event; returns the completion once the message stops.
    pub fn event(&mut self, event: Value, progress: &(dyn Fn(Progress) + Send + Sync)) -> Result<Option<Completion>> {
        match event["type"].as_str().context("Anthropic event is missing type")? {
            "ping" => {}
            "message_start" => {
                ensure!(!self.started, "duplicate message_start");
                self.started = true;
                self.usage = event["message"]["usage"].clone();
            }
            "content_block_start" => {
                ensure!(self.started, "content before message_start");
                let index = event["index"].as_u64().context("content block is missing index")?;
                ensure!(!self.blocks.contains_key(&index), "duplicate content block");
                let value = event["content_block"].clone();
                if value["type"] == "text"
                    && let Some(text) = value["text"].as_str().filter(|t| !t.is_empty())
                {
                    progress(Progress::Text(text));
                }
                self.blocks.insert(index, Block { value, json: None, stopped: false });
            }
            "content_block_delta" => {
                let index = event["index"].as_u64().context("content delta is missing index")?;
                let block = self.blocks.get_mut(&index).context("delta before content_block_start")?;
                ensure!(!block.stopped, "delta after content_block_stop");
                let delta = &event["delta"];
                let (field, value) = match delta["type"].as_str() {
                    Some("text_delta") => {
                        let text = delta["text"].as_str().context("text delta is missing text")?;
                        progress(Progress::Text(text));
                        ("text", text)
                    }
                    Some("thinking_delta") => {
                        ("thinking", delta["thinking"].as_str().context("missing thinking delta")?)
                    }
                    Some("signature_delta") => {
                        ("signature", delta["signature"].as_str().context("missing signature delta")?)
                    }
                    Some("input_json_delta") => {
                        block
                            .json
                            .get_or_insert_default()
                            .push_str(delta["partial_json"].as_str().context("missing input JSON delta")?);
                        return Ok(None);
                    }
                    _ => return Ok(None),
                };
                let mut text = block.value[field].as_str().unwrap_or_default().to_owned();
                text.push_str(value);
                block.value[field] = json!(text);
            }
            "content_block_stop" => {
                let index = event["index"].as_u64().context("content stop is missing index")?;
                let block = self.blocks.get_mut(&index).context("stop before content_block_start")?;
                ensure!(!block.stopped, "duplicate content_block_stop");
                block.stopped = true;
            }
            "message_delta" => {
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    self.stop = Some(reason.into());
                }
                if let Some(usage) = event["usage"].as_object() {
                    for (k, v) in usage {
                        self.usage[k] = v.clone();
                    }
                }
            }
            "message_stop" => {
                ensure!(self.started && self.stop.is_some(), "incomplete Anthropic message");
                ensure!(self.blocks.values().all(|b| b.stopped), "unfinished Anthropic content block");
                if matches!(self.stop.as_deref(), Some("max_tokens" | "model_context_window_exceeded" | "refusal")) {
                    bail!("Anthropic stopped with {}", self.stop.as_deref().unwrap());
                }
                let mut items = Vec::new();
                for block in self.blocks.values() {
                    let v = &block.value;
                    match v["type"].as_str() {
                        Some("text") => {
                            let text = v["text"].as_str().context("text block is missing text")?;
                            if !text.is_empty() {
                                items.push(Item::Assistant { text: text.into() });
                            }
                        }
                        Some("thinking" | "redacted_thinking") => {
                            if v["type"] == "thinking" {
                                ensure!(
                                    v["signature"].as_str().is_some_and(|s| !s.is_empty()),
                                    "thinking block is missing signature"
                                );
                            }
                            items.push(Item::Reasoning {
                                summary: v["thinking"].as_str().map(|s| vec![s.into()]).unwrap_or_default(),
                                encrypted: Some(
                                    json!({"provider":"anthropic","model":self.model,"block":v}).to_string(),
                                ),
                            });
                        }
                        Some("tool_use") => {
                            let id = v["id"].as_str().context("tool_use is missing id")?;
                            let name = v["name"].as_str().context("tool_use is missing name")?;
                            items.push(Item::ToolCall {
                                call_id: id.into(),
                                name: if self.oauth { super::fingerprint::decode_tool(name) } else { name }.into(),
                                arguments: block.json.clone().unwrap_or_else(|| v["input"].to_string()),
                            });
                        }
                        other => bail!("unsupported Anthropic content block: {other:?}"),
                    }
                }
                ensure!(!items.is_empty(), "Anthropic returned an empty completion");
                let n = |key| self.usage[key].as_u64().unwrap_or_default();
                return Ok(Some(Completion {
                    items,
                    usage: Usage {
                        input: n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens"),
                        cached_input: n("cache_read_input_tokens"),
                        cache_write: n("cache_creation_input_tokens"),
                        output: n("output_tokens"),
                        reasoning: 0,
                    },
                }));
            }
            _ => {}
        }
        Ok(None)
    }
}
