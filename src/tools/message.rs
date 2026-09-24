//! `send_message`: mail to another agent through the harness inbox. The recipient gets it
//! as its next input: an idle agent starts a turn, a busy one sees it at its next tool
//! boundary, and one held by an interrupt keeps it until the user writes again.

use super::{Tool, ToolContext, ToolOutput, string_arg};
use futures_util::future::BoxFuture;
use serde_json::{Value, json};

pub struct SendMessage;

impl Tool for SendMessage {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> String {
        "Send a message to another agent by id. It is queued: an idle agent starts working on it, a busy agent receives it at its next tool call boundary. Replies arrive as messages from that agent.".into()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "Id of the receiving agent"},
                "text": {"type": "string", "description": "Message text"}
            },
            "required": ["agent", "text"]
        })
    }

    fn snippet(&self) -> &str {
        "Message another agent by id"
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
            match context.mailbox.send(&context.agent, to, text) {
                Ok(()) => ToolOutput::text(format!("Message queued for agent {to}.")),
                Err(error) => ToolOutput::error(format!("{error:#}")),
            }
        })
    }
}
