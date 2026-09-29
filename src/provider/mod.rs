//! Model providers. A [`Provider`] turns an agent's transcript into the model's next items;
//! the harness knows nothing about wire formats. Transport, credentials and model policy all
//! stay inside the provider.
//!
//! Two are built in: [`Responses`] for the OpenAI Responses API and [`Anthropic`] for the
//! Anthropic Messages API, by API key or Claude subscription.

pub mod anthropic;
mod gate;
pub mod responses;
mod transport;

pub use anthropic::{Anthropic, AnthropicAuth, AnthropicConfig, CacheRetention, ClaudeCode, Thinking};
pub use gate::{Permit, RateGate};
pub use responses::{Responses, ResponsesConfig};

use crate::agent::{Item, Usage};
use futures_util::future::BoxFuture;
use serde_json::Value;

pub trait Provider: Send + Sync {
    /// Runs one model call, reporting its [`Progress`]. Dropping the future cancels the call.
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        progress: &'a (dyn Fn(Progress) + Send + Sync),
    ) -> BoxFuture<'a, anyhow::Result<Completion>>;
}

pub struct Request<'a> {
    pub model: &'a str,
    pub reasoning_effort: Option<&'a str>,
    /// Stable per conversation, so the provider can reuse its prompt cache.
    pub cache_key: &'a str,
    pub system: &'a str,
    pub tools: &'a [ToolSpec],
    pub items: &'a [Item],
}

pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub struct Completion {
    pub items: Vec<Item>,
    pub usage: Usage,
}

/// What a model call reports while it runs.
pub enum Progress<'a> {
    /// Assistant text as it streams.
    Text(&'a str),
    /// The call is waiting, or with `None`, goes on after waiting.
    Held(Option<Hold>),
    Event(ProviderEvent),
}

/// Why a model call is waiting instead of running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    /// Every slot of the account's [`RateGate`] is taken; the call waits its turn.
    Queued,
    /// The provider rate-limited the account; calls wait until this time (Unix milliseconds).
    RateLimited { until: u64 },
}

/// Transport diagnostics, the same for every provider.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderEvent {
    /// Only rate-limit, retry and request-id headers; never authentication or cookies.
    Response {
        status: u16,
        headers: Vec<(String, String)>,
    },
    Retry {
        attempt: u32,
        status: Option<u16>,
        message: String,
        delay_ms: u64,
    },
}

/// Authorizes each request, so logins, storage and refresh stay with the host.
pub trait Credentials: Send + Sync {
    fn authorize(&self) -> BoxFuture<'_, anyhow::Result<Authorization>>;
}

pub struct Authorization {
    pub token: String,
    /// Sent along with the token, such as an account id.
    pub headers: Vec<(String, String)>,
}

/// A token that never changes, such as an API key.
pub struct StaticToken(pub String);

impl Credentials for StaticToken {
    fn authorize(&self) -> BoxFuture<'_, anyhow::Result<Authorization>> {
        Box::pin(async move { Ok(Authorization { token: self.0.clone(), headers: Vec::new() }) })
    }
}

/// What the model reads for an item in the user's role: the user's mail as written, another
/// agent's mail under a line naming it, and a compaction's summary with how to take it.
pub fn user_text(item: &Item) -> Option<String> {
    match item {
        Item::Input { from, text } if from == "user" => Some(text.clone()),
        Item::Input { from, text } => Some(format!("[Message from agent {from}]\n{text}")),
        Item::Compaction { summary } => {
            Some(format!("The conversation so far was compacted. Summary:\n\n{summary}\n\nContinue from here."))
        }
        _ => None,
    }
}
