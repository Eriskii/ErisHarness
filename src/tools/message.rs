//! `send_message`: mail to another agent, or to a recipient the host serves such as the
//! user. The recipient gets it as its next input: an idle agent starts a turn, a busy one
//! sees it at its next tool boundary, and one held by an interrupt keeps it until the user
//! writes again. With `wait`, the call returns the recipient's next message to the sender
//! as its result instead of it arriving as mail.

use super::{Tool, ToolContext, ToolOutput, js_number, string_arg};
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::time::Duration;

pub struct SendMessage;

fn describe(to: &str) -> String {
    if to == "user" { "the user".into() } else { format!("agent {to}") }
}

impl Tool for SendMessage {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> String {
        "Send a message to another agent by id, or to \"user\". It is queued: an idle agent starts working on it, a busy agent receives it at its next tool call boundary. Replies arrive as messages. Set wait to block until the recipient replies; the reply is then this call's result. timeout_seconds limits the wait, after which a reply arrives as a message instead.".into()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "Id of the receiving agent, or \"user\""},
                "text": {"type": "string", "description": "Message text"},
                "wait": {"type": "boolean", "description": "Wait for the recipient's reply (default false)"},
                "timeout_seconds": {"type": "number", "description": "Longest wait for a reply; no limit when omitted"}
            },
            "required": ["agent", "text"]
        })
    }

    fn snippet(&self) -> &str {
        "Message another agent or the user, optionally waiting for the reply"
    }

    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput> {
        Box::pin(async move {
            let (to, text) = match (string_arg(&args, "agent"), string_arg(&args, "text")) {
                (Ok(to), Ok(text)) => (to, text),
                (Err(e), _) | (_, Err(e)) => return ToolOutput::error(e),
            };
            if to == context.agent {
                return ToolOutput::error("You cannot message yourself.");
            }
            let sent = match context.mailbox.send(&context.agent, to, text) {
                Ok(position) => position,
                Err(error) => return ToolOutput::error(format!("{error:#}")),
            };
            if !args.get("wait").and_then(Value::as_bool).unwrap_or(false) {
                return ToolOutput::text(if to == "user" {
                    "Message sent to the user.".into()
                } else {
                    format!("Message queued for agent {to}.")
                });
            }
            let timeout = args.get("timeout_seconds").and_then(Value::as_f64).filter(|s| *s > 0.0);
            let limit = async {
                match timeout {
                    Some(seconds) => tokio::time::sleep(Duration::from_secs_f64(seconds)).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                reply = context.mailbox.reply(&context.agent, to, sent) => match reply {
                    Ok(reply) => {
                        let mut output = ToolOutput::text(format!("Reply from {}:\n{}", describe(to), reply.text));
                        output.delivers = Some(reply.id);
                        output
                    }
                    Err(error) => ToolOutput::error(format!("{error:#}")),
                },
                _ = limit => {
                    let seconds = timeout.unwrap_or_default();
                    let unit = if seconds == 1.0 { "second" } else { "seconds" };
                    ToolOutput::text(format!(
                        "No reply from {} within {} {unit}. A reply will arrive as a message.",
                        describe(to),
                        js_number(seconds)
                    ))
                }
                _ = context.cancel.cancelled() => {
                    ToolOutput::error(format!("Stopped waiting for a reply from {}: interrupted.", describe(to)))
                }
            }
        })
    }
}
