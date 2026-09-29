//! Pi's `write` tool.

use super::{Tool, ToolContext, ToolOutput, errors, open, resolve, string_arg};
use crate::machine::OpenMode;
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

pub struct Write;

impl Tool for Write {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> String {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.".into()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"}
            },
            "required": ["path", "content"]
        })
    }

    fn snippet(&self) -> &str {
        "Create or overwrite files"
    }

    fn guidelines(&self) -> &[&str] {
        &["Use write only for new files or complete rewrites."]
    }

    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput> {
        Box::pin(async move { write(context, &args).await.unwrap_or_else(ToolOutput::error) })
    }
}

async fn write(context: &ToolContext, args: &Value) -> Result<ToolOutput, String> {
    let path = string_arg(args, "path")?;
    let content = string_arg(args, "content")?;
    let machine = &context.machine;
    let absolute = resolve(path, machine.cwd(), machine.home());
    let mut file = open(machine.as_ref(), &absolute, OpenMode::Write { create_parents: true })
        .await
        .map_err(|e| errors::node(&e, "open", &absolute))?;
    file.write_all(content.as_bytes()).await.map_err(|e| errors::node(&e, "write", ""))?;
    file.flush().await.map_err(|e| errors::node(&e, "write", ""))?;
    Ok(ToolOutput::text(format!("Successfully wrote to {path}")))
}
