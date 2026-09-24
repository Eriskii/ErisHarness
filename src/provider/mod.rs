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

/// Supplies the bearer token for each request, so refreshable credentials stay outside.
pub trait Credentials: Send + Sync {
    fn token(&self) -> BoxFuture<'_, anyhow::Result<String>>;
}

pub struct StaticToken(pub String);

impl Credentials for StaticToken {
    fn token(&self) -> BoxFuture<'_, anyhow::Result<String>> {
        Box::pin(async move { Ok(self.0.clone()) })
    }
}
