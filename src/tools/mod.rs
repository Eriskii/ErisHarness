//! Tools an agent can call. Anything implementing [`Tool`] can be registered. [`builtin`] is
//! Pi's coding set (read, bash, edit, write), acting on the agent's machine, plus
//! `send_message` for mail.

mod bash;
mod edit;
mod errors;
mod message;
mod path;
mod read;
mod text;
mod write;

pub use path::resolve;

use crate::machine::{Machine, OpenMode};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> String;
    /// JSON Schema of the arguments object.
    fn parameters(&self) -> Value;
    /// One line for the system prompt's tool list; unlisted when empty.
    fn snippet(&self) -> &str {
        ""
    }
    /// Usage rules added to the system prompt.
    fn guidelines(&self) -> &[&str] {
        &[]
    }
    /// Runs the call. Failures belong in an error output, which the model reads.
    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput>;
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

/// Delivers mail between agents. The harness implements it over its inboxes.
pub trait Mailbox: Send + Sync {
    /// Queues mail to `to`, as [`Harness::send`](crate::Harness::send) does. Returns its place
    /// in inbox order: replies to it come after it.
    fn send(&self, to: &str, from: &str, text: &str) -> anyhow::Result<i64>;
    /// Waits for unread mail to `agent` from `from` that came after `after`.
    fn reply<'a>(&'a self, agent: &'a str, from: &'a str, after: i64) -> BoxFuture<'a, anyhow::Result<Reply>>;
}

pub struct Reply {
    pub id: i64,
    pub text: String,
}

/// What a tool call returns to the model.
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

/// Parses a model's tool arguments, which must be a JSON object: argument-free tools get
/// `{}`. The harness answers invalid arguments with a tool error rather than a failed turn.
pub fn parse_arguments(arguments: &str) -> anyhow::Result<Value> {
    let input: Value = serde_json::from_str(arguments)?;
    anyhow::ensure!(input.is_object(), "tool arguments must be a JSON object");
    Ok(input)
}

fn string_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| format!("Missing required string argument: {key}"))
}

/// Opens an absolute path on the machine as an async file.
async fn open(machine: &dyn Machine, path: &str, mode: OpenMode) -> std::io::Result<tokio::fs::File> {
    Ok(tokio::fs::File::from_std(machine.open(path, mode).await?.into()))
}

/// Resolves after `seconds`, or never.
async fn deadline(seconds: Option<f64>) {
    match seconds {
        Some(seconds) => tokio::time::sleep(Duration::from_secs_f64(seconds)).await,
        None => std::future::pending().await,
    }
}

/// A number as JavaScript prints it, whole numbers without a fraction. Notices echo numbers
/// the model sent.
fn js_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 { format!("{}", value as i64) } else { value.to_string() }
}
