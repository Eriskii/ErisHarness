//! Pi's `write` tool.

use super::{Tool, ToolContext, ToolOutput, errors, resolve, string_arg};
use crate::sandbox::OpenMode;
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
        Box::pin(async move {
            match write(context, &args).await {
                Ok(output) => output,
                Err(message) => ToolOutput::error(message),
            }
        })
    }
}

async fn write(context: &ToolContext, args: &Value) -> Result<ToolOutput, String> {
    let path = string_arg(args, "path")?;
    let content = string_arg(args, "content")?;
    let spec = context.sandbox.spec();
    let absolute = resolve(path, &spec.cwd, spec.home());
    let fd = context
        .sandbox
        .open(&absolute, OpenMode::Write { create_parents: true })
        .await
        .map_err(|e| errors::node(&e, "open", &absolute))?;
    let mut file = tokio::fs::File::from_std(std::fs::File::from(fd));
    file.write_all(content.as_bytes()).await.map_err(|e| errors::node(&e, "write", ""))?;
    file.flush().await.map_err(|e| errors::node(&e, "write", ""))?;
    Ok(ToolOutput::text(format!("Successfully wrote to {path}")))
}
