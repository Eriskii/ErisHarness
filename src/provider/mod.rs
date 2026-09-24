//! Model providers. A provider turns an agent's transcript into model output items. The
//! harness knows nothing about wire formats; [`Responses`] speaks OpenAI's Responses API.

mod responses;
mod sse;

pub use responses::{Responses, ResponsesConfig};

use crate::agent::{Item, Usage};
use futures_util::future::BoxFuture;
use serde_json::Value;

pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
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

pub struct Completion {
    pub items: Vec<Item>,
    pub usage: Usage,
}

pub trait Provider: Send + Sync {
    /// Runs one model call. `on_text` receives assistant text as it streams. Dropping the
    /// future cancels the call.
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        on_text: &'a (dyn Fn(&str) + Send + Sync),
    ) -> BoxFuture<'a, anyhow::Result<Completion>>;
}

pub struct Authorization {
    pub token: String,
    /// Sent along with the token, such as an account id.
    pub headers: Vec<(String, String)>,
}

/// Authorizes each request, so refreshable credentials stay outside the harness.
pub trait Credentials: Send + Sync {
    fn authorize(&self) -> BoxFuture<'_, anyhow::Result<Authorization>>;
}

pub struct StaticToken(pub String);

impl Credentials for StaticToken {
    fn authorize(&self) -> BoxFuture<'_, anyhow::Result<Authorization>> {
        Box::pin(async move { Ok(Authorization { token: self.0.clone(), headers: Vec::new() }) })
    }
}
