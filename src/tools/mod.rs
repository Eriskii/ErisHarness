//! Tools an agent can call. Anything implementing [`Tool`] can be registered; [`builtin`]
//! provides Pi's coding set (read, bash, edit, write) running on the agent's machine,
//! plus `send_message` for mail between agents.

mod bash;
mod edit;
mod errors;
mod message;
mod path;
mod read;
pub mod text;
mod write;

pub use path::resolve;

use crate::machine::Machine;
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// What a tool call returns to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Content {
    Text(String),
    /// Base64 image data.
    Image {
        mime: String,
        data: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: Vec<Content>,
    pub is_error: bool,
    /// For observers such as a UI; never sent to the model.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub details: Value,
    /// Inbox mail this result hands to the agent, which is then not delivered again.
    #[serde(skip)]
    pub(crate) delivers: Option<i64>,
}

impl ToolOutput {
    pub fn new(content: Vec<Content>) -> Self {
        Self { content, is_error: false, details: Value::Null, delivers: None }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::new(vec![Content::Text(text.into())])
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self { is_error: true, ..Self::text(text) }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }
}

/// Delivers mail between agents. The harness implements it over its inboxes.
pub trait Mailbox: Send + Sync {
    /// Queues mail. Returns a position in the inbox order: replies to it come after it.
    fn send(&self, from: &str, to: &str, text: &str) -> anyhow::Result<i64>;
    /// Waits for unread mail to `agent` from `from` that came after `after`.
    fn reply<'a>(&'a self, agent: &'a str, from: &'a str, after: i64) -> BoxFuture<'a, anyhow::Result<Reply>>;
}

pub struct Reply {
    pub id: i64,
    pub text: String,
}

/// Everything a tool call may touch.
pub struct ToolContext {
    /// The calling agent's id.
    pub agent: String,
    pub machine: Arc<dyn Machine>,
    pub mailbox: Arc<dyn Mailbox>,
    /// Cancelled when the agent is interrupted; long-running tools should stop.
    pub cancel: CancellationToken,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> String;
    /// JSON Schema of the arguments object.
    fn parameters(&self) -> Value;
    /// One line for the system prompt's tool list.
    fn snippet(&self) -> &str {
        ""
    }
    /// Usage rules added to the system prompt.
    fn guidelines(&self) -> &[&str] {
        &[]
    }
    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput>;
}

/// Pi's coding tools in Pi's order, then agent messaging.
pub fn builtin() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(read::Read),
        Arc::new(bash::Bash),
        Arc::new(edit::Edit),
        Arc::new(write::Write),
        Arc::new(message::SendMessage),
    ]
}

fn string_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| format!("Missing required string argument: {key}"))
}

/// JavaScript prints whole numbers without a fraction; notices echo numbers the model sent.
fn js_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 { format!("{}", value as i64) } else { value.to_string() }
}
