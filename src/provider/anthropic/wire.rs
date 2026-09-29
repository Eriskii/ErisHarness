//! The request body: transcript items as Anthropic messages, the system prompt, tools, cache
//! breakpoints and thinking settings, in the field order Claude Code sends them.

use super::{AnthropicAuth, AnthropicConfig, CacheRetention, Thinking, fingerprint};
use crate::agent::Item;
use crate::provider::{Request, user_text};
use crate::tools::Content;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub fn body(config: &AnthropicConfig, request: &Request, version: &str) -> Result<Value> {
    let oauth = config.oauth();
    let cache = match config.cache_retention {
        CacheRetention::None => None,
        CacheRetention::Short => Some(json!({"type":"ephemeral"})),
        CacheRetention::Long => Some(json!({"type":"ephemeral","ttl":"1h"})),
    };
    let mut messages: Vec<Value> = Vec::new();
    let mut user_indices = Vec::new();
    let mut first_user = None;
    for item in request.items {
        let (role, blocks) = match item {
            Item::Input { .. } | Item::Compaction { .. } => {
                let Some(text) = user_text(item) else { continue };
                if matches!(item, Item::Input { .. }) {
                    first_user.get_or_insert_with(|| text.clone());
                    // Where this input lands: the open user message, or a new one.
                    let index = messages.len() - usize::from(messages.last().is_some_and(|m| m["role"] == "user"));
                    if user_indices.last() != Some(&index) {
                        user_indices.push(index);
                    }
                }
                ("user", vec![json!({"type":"text","text":text})])
            }
            Item::Assistant { text } if !text.is_empty() => ("assistant", vec![json!({"type":"text","text":text})]),
            Item::Reasoning { encrypted: Some(state), .. } => {
                let Ok(state) = serde_json::from_str::<Value>(state) else { continue };
                if state["provider"] != "anthropic" || state["model"] != request.model {
                    continue;
                }
                let block = &state["block"];
                if !matches!(block["type"].as_str(), Some("thinking" | "redacted_thinking")) {
                    continue;
                }
                ("assistant", vec![block.clone()])
            }
            Item::ToolCall { call_id, name, arguments } => {
                // Anthropic requires an object even when replaying a failed tool
                // call. Preserve the original text without repairing or executing it.
                let input =
                    crate::tools::parse_arguments(arguments).unwrap_or_else(|_| json!({"INVALID_JSON": arguments}));
                (
                    "assistant",
                    vec![json!({"type":"tool_use","id":call_id,
                    "name":if oauth { fingerprint::tool_name(name) } else { name.clone() },"input":input})],
                )
            }
            Item::ToolResult { call_id, output } => {
                let content: Vec<Value> = output
                    .content
                    .iter()
                    .map(|part| match part {
                        Content::Text(text) => json!({"type":"text","text":text}),
                        Content::Image { mime, data } => {
                            json!({"type":"image","source":{"type":"base64","media_type":mime,"data":data}})
                        }
                    })
                    .collect();
                (
                    "user",
                    vec![
                        json!({"type":"tool_result","tool_use_id":call_id,"content":content,"is_error":output.is_error}),
                    ],
                )
            }
            _ => continue,
        };
        if let Some(last) = messages.last_mut().filter(|m| m["role"] == role) {
            last["content"].as_array_mut().unwrap().extend(blocks);
        } else {
            messages.push(json!({"role":role,"content":blocks}));
        }
    }
    ensure!(!messages.is_empty(), "Anthropic requests need at least one message");
    let mut system = Vec::new();
    if oauth {
        system.push(json!({"type":"text","text":fingerprint::billing(first_user.as_deref().unwrap_or(""), version)}));
        let mut identity = json!({"type":"text","text":fingerprint::IDENTITY});
        // OMP always marks the identity, even when other prompt caching is disabled.
        identity["cache_control"] = cache.clone().unwrap_or(json!({"type":"ephemeral"}));
        system.push(identity);
    }
    if !request.system.trim().is_empty() {
        system.push(json!({"type":"text","text":request.system}));
    }
    if !oauth && let (Some(last), Some(cache)) = (system.last_mut(), &cache) {
        last["cache_control"] = cache.clone();
    }

    let mut tools: Vec<Value> = request
        .tools
        .iter()
        .map(|t| {
            json!({
                "name":if oauth { fingerprint::tool_name(&t.name) } else { t.name.clone() },
                "description":t.description,"input_schema":t.parameters,"eager_input_streaming":true
            })
        })
        .collect();
    if let (Some(last), Some(cache)) = (tools.last_mut(), &cache) {
        last["cache_control"] = cache.clone();
    }

    if let Some(cache) = &cache {
        let head = system.iter().chain(&tools).filter(|v| v.get("cache_control").is_some()).count();
        let mut candidates = vec![messages.len() - 1];
        candidates.extend(user_indices.iter().enumerate().filter(|(n, _)| (n + 1) % 15 == 0).map(|(_, &i)| i).rev());
        if messages.len() > 1 {
            candidates.push(messages.len() - 2);
        }
        let mut used = Vec::new();
        for index in candidates {
            if used.len() >= 4usize.saturating_sub(head) {
                break;
            }
            if used.contains(&index) {
                continue;
            }
            if let Some(block) = messages[index]["content"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .rev()
                .find(|b| matches!(b["type"].as_str(), Some("text" | "tool_use" | "tool_result" | "image")))
            {
                block["cache_control"] = cache.clone();
                used.push(index);
            }
        }
    }
    if messages.last().is_some_and(|m| m["role"] == "assistant") {
        messages.push(json!({"role":"user","content":"Continue."}));
    }

    // Insertion order is intentional; serialize once, then attest those exact bytes.
    let mut body = json!({"model":request.model,"messages":messages});
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if oauth || !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    if let AnthropicAuth::ClaudeCode(identity) = &config.auth {
        body["metadata"] =
            fingerprint::metadata(&identity.install_id, identity.account_id.as_deref(), request.cache_key);
    }
    let ceiling = if oauth { config.max_tokens.min(64000) } else { config.max_tokens };
    ensure!(ceiling > 0, "max_tokens must be positive");
    body["max_tokens"] = json!(ceiling);
    match config.thinking {
        Thinking::Disabled => {}
        Thinking::Adaptive { display } => {
            body["thinking"] = json!({"type":"adaptive"});
            if display {
                body["thinking"]["display"] = json!("summarized");
            }
            if let Some(effort) = request.reasoning_effort {
                ensure!(
                    ["low", "medium", "high", "xhigh", "max"].contains(&effort),
                    "unsupported Anthropic effort: {effort}"
                );
                body["output_config"] = json!({"effort":effort});
            }
        }
        Thinking::Budget { tokens, display } => {
            ensure!(tokens >= 1024 && tokens < ceiling, "thinking budget must be >= 1024 and less than max_tokens");
            body["thinking"] = json!({"type":"enabled","budget_tokens":tokens});
            if display {
                body["thinking"]["display"] = json!("summarized");
            }
        }
    }
    if config.thinking != Thinking::Disabled {
        body["context_management"] = json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]});
        // Keep the OMP field order: context management before output_config.
        if let Some(output) = body.as_object_mut().unwrap().shift_remove("output_config") {
            body["output_config"] = output;
        }
    }
    body["stream"] = json!(true);
    Ok(body)
}
