//! Pi's `bash` tool. Each call is a fresh `bash -c` in the machine's working directory.
//! Output memory is bounded: once output passes Pi's limits the full log streams to a file
//! on the machine and only a tail window stays in memory.

use super::text::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, format_size, truncate_tail};
use super::{Tool, ToolContext, ToolOutput, js_number, string_arg};
use crate::machine::{DRAIN_GRACE, Machine, OpenMode};
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MAX_TIMEOUT_SECONDS: f64 = 2_147_483.647;
/// Tail kept in memory once output overflows; twice the limit so a whole final page fits.
const TAIL_WINDOW: usize = 2 * DEFAULT_MAX_BYTES;

pub struct Bash;

impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> String {
        format!(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            DEFAULT_MAX_BYTES / 1024
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}
            },
            "required": ["command"]
        })
    }

    fn snippet(&self) -> &str {
        "Execute bash commands (ls, grep, find, etc.)"
    }

    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput> {
        Box::pin(async move {
            match bash(context, &args).await {
                Ok(output) => output,
                Err(message) => ToolOutput::error(message),
            }
        })
    }
}

enum Stopped {
    TimedOut,
    Aborted,
}

async fn bash(context: &ToolContext, args: &Value) -> Result<ToolOutput, String> {
    let command = string_arg(args, "command")?;
    let timeout = match args.get("timeout").and_then(Value::as_f64) {
        None => None,
        Some(t) if !t.is_finite() || t <= 0.0 => {
            return Err("Invalid timeout: must be a finite number of seconds".into());
        }
        Some(t) if t > MAX_TIMEOUT_SECONDS => {
            return Err(format!("Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"));
        }
        Some(t) => Some(t),
    };
    let machine = &context.machine;
    // Found through PATH: not every host has /bin/bash (NixOS, for one).
    let argv = ["bash".to_owned(), "-c".to_owned(), command.to_owned()];
    let mut process = machine.spawn(&argv, None).await.map_err(|e| {
        let message = format!("{e:#}");
        if message.starts_with("Working directory does not exist") {
            format!("{message}\nCannot execute bash commands.")
        } else {
            message
        }
    })?;
    let mut output = process.take_output();
    let killer = process.killer();
    let exit = process.wait();
    tokio::pin!(exit);
    let expired = async {
        match timeout {
            Some(seconds) => tokio::time::sleep(Duration::from_secs_f64(seconds)).await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(expired);
    let mut collected = Collected::new(machine.clone());
    let mut buffer = vec![0u8; 16 * 1024];
    let mut stopped = None;
    let mut open = true;
    // Until the command exits. A timeout or cancellation kills it and keeps waiting.
    let status = loop {
        tokio::select! {
            status = &mut exit => break status,
            _ = &mut expired, if stopped.is_none() => {
                killer.kill().await;
                stopped = Some(Stopped::TimedOut);
            }
            _ = context.cancel.cancelled(), if stopped.is_none() => {
                killer.kill().await;
                stopped = Some(Stopped::Aborted);
            }
            read = output.read(&mut buffer), if open => match read {
                Ok(n) if n > 0 => collected.push(&buffer[..n]).await,
                _ => open = false,
            },
        }
    };
    // Then whatever is already buffered. Background processes may hold the pipe open.
    let grace = tokio::time::Instant::now() + DRAIN_GRACE;
    while open {
        match tokio::time::timeout_at(grace, output.read(&mut buffer)).await {
            Ok(Ok(n)) if n > 0 => collected.push(&buffer[..n]).await,
            _ => open = false,
        }
    }
    let rendered = collected.finish().await;
    let status_text = |status: &str, empty: &str| {
        let text = rendered.text(empty);
        if text.is_empty() { status.to_owned() } else { format!("{text}\n\n{status}") }
    };
    match stopped {
        Some(Stopped::TimedOut) => {
            Err(status_text(&format!("Command timed out after {} seconds", js_number(timeout.unwrap_or_default())), ""))
        }
        Some(Stopped::Aborted) => Err(status_text("Command aborted", "")),
        None if !status.success() => {
            Err(status_text(&format!("Command exited with code {}", status.code()), "(no output)"))
        }
        None => Ok(ToolOutput::text(rendered.text("(no output)")).with_details(rendered.details())),
    }
}

/// Output of one command: everything while small, then a log file plus a tail window.
struct Collected {
    machine: Arc<dyn Machine>,
    kept: Vec<u8>,
    trimmed: bool,
    log: Option<(String, tokio::fs::File)>,
    total_bytes: usize,
    newlines: usize,
    last_byte: Option<u8>,
    line_bytes: usize,
    previous_line_bytes: usize,
}

struct Rendered {
    content: String,
    notice: Option<String>,
    log: Option<String>,
}

impl Rendered {
    fn text(&self, empty: &str) -> String {
        let body = if self.content.is_empty() { empty.to_owned() } else { self.content.clone() };
        match &self.notice {
            Some(notice) => format!("{body}\n\n{notice}"),
            None => body,
        }
    }

    fn details(&self) -> Value {
        self.log.as_ref().map_or(Value::Null, |path| json!({"fullOutputPath": path}))
    }
}

impl Collected {
    fn new(machine: Arc<dyn Machine>) -> Self {
        Self {
            machine,
            kept: Vec::new(),
            trimmed: false,
            log: None,
            total_bytes: 0,
            newlines: 0,
            last_byte: None,
            line_bytes: 0,
            previous_line_bytes: 0,
        }
    }

    /// Lines as Pi counts them: a final newline does not begin another line.
    fn total_lines(&self) -> usize {
        self.newlines + usize::from(self.last_byte.is_some_and(|b| b != b'\n'))
    }

    async fn push(&mut self, bytes: &[u8]) {
        self.total_bytes += bytes.len();
        for &byte in bytes {
            if byte == b'\n' {
                self.newlines += 1;
                self.previous_line_bytes = self.line_bytes;
                self.line_bytes = 0;
            } else {
                self.line_bytes += 1;
            }
        }
        self.last_byte = bytes.last().copied().or(self.last_byte);
        if self.log.is_none() && (self.total_bytes > DEFAULT_MAX_BYTES || self.total_lines() > DEFAULT_MAX_LINES) {
            self.open_log().await;
        }
        if let Some((_, file)) = &mut self.log {
            let _ = file.write_all(bytes).await;
        }
        self.kept.extend_from_slice(bytes);
        if self.log.is_some() && self.kept.len() > 2 * TAIL_WINDOW {
            self.kept.drain(..self.kept.len() - TAIL_WINDOW);
            self.trimmed = true;
        }
    }

    async fn open_log(&mut self) {
        let path = format!("/tmp/bash-{}.log", uuid::Uuid::now_v7().simple());
        if let Ok(fd) = self.machine.open(&path, OpenMode::Write { create_parents: true }).await {
            let mut file = tokio::fs::File::from_std(std::fs::File::from(fd));
            let _ = file.write_all(&self.kept).await;
            self.log = Some((path, file));
        }
    }

    async fn finish(mut self) -> Rendered {
        if let Some((_, file)) = &mut self.log {
            let _ = file.flush().await;
        }
        let mut window = String::from_utf8_lossy(&self.kept).into_owned();
        // A trimmed window starts mid-line; that partial line is not the command's output.
        if self.trimmed
            && let Some(at) = window.find('\n')
        {
            window.drain(..=at);
        }
        let truncation = truncate_tail(&window, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        let overflowed = self.total_bytes > DEFAULT_MAX_BYTES || self.total_lines() > DEFAULT_MAX_LINES;
        if !overflowed {
            return Rendered { content: truncation.content, notice: None, log: None };
        }
        let path = self.log.as_ref().map_or("(unavailable)".to_owned(), |(p, _)| p.clone());
        let total = self.total_lines();
        let start = total - truncation.output_lines + 1;
        let notice = if truncation.last_line_partial {
            let last_line = if self.line_bytes > 0 { self.line_bytes } else { self.previous_line_bytes };
            format!(
                "[Showing last {} of line {total} (line is {}). Full output: {path}]",
                format_size(truncation.output_bytes),
                format_size(last_line)
            )
        } else if truncation.truncated_by == Some(TruncatedBy::Lines) || self.total_bytes <= DEFAULT_MAX_BYTES {
            format!("[Showing lines {start}-{total} of {total}. Full output: {path}]")
        } else {
            format!(
                "[Showing lines {start}-{total} of {total} ({} limit). Full output: {path}]",
                format_size(DEFAULT_MAX_BYTES)
            )
        };
        Rendered { content: truncation.content, notice: Some(notice), log: Some(path) }
    }
}
