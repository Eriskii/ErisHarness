//! Pi's `read` tool. Text is scanned as a stream, so reading a page of a huge file costs no
//! more memory than the page itself.

use super::text::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head};
use super::{Content, Tool, ToolContext, ToolOutput, errors, js_number, resolve, string_arg};
use crate::machine::OpenMode;
use base64::Engine;
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::io;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

/// Images larger than this are refused rather than sent.
const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;

pub struct Read;

impl Tool for Read {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> String {
        format!(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            DEFAULT_MAX_BYTES / 1024
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "number", "description": "Maximum number of lines to read"}
            },
            "required": ["path"]
        })
    }

    fn snippet(&self) -> &str {
        "Read file contents"
    }

    fn guidelines(&self) -> &[&str] {
        &["Use read to examine files instead of cat or sed."]
    }

    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput> {
        Box::pin(async move {
            match read(context, &args).await {
                Ok(output) => output,
                Err(message) => ToolOutput::error(message),
            }
        })
    }
}

async fn read(context: &ToolContext, args: &Value) -> Result<ToolOutput, String> {
    let path = string_arg(args, "path")?;
    let offset = args.get("offset").and_then(Value::as_f64);
    let limit = args.get("limit").and_then(Value::as_f64);
    let machine = &context.machine;
    let absolute = resolve(path, machine.cwd(), machine.home());
    let fd = machine.open(&absolute, OpenMode::Read).await.map_err(|e| errors::node(&e, "access", &absolute))?;
    let file = tokio::fs::File::from_std(std::fs::File::from(fd));
    let metadata = file.metadata().await.map_err(|e| errors::node(&e, "read", ""))?;
    if metadata.is_dir() {
        return Err(errors::node(&io::Error::from_raw_os_error(libc::EISDIR), "read", ""));
    }
    if !metadata.is_file() {
        return Err(format!("Not a regular file: {path}"));
    }
    let mut reader = BufReader::new(file);
    if let Some(mime) = image_type(reader.fill_buf().await.map_err(|e| errors::node(&e, "read", ""))?) {
        if metadata.len() > MAX_IMAGE_BYTES {
            return Err(format!(
                "Image {path} is {}, larger than the {} limit.",
                format_size(metadata.len() as usize),
                format_size(MAX_IMAGE_BYTES as usize)
            ));
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        reader.read_to_end(&mut bytes).await.map_err(|e| errors::node(&e, "read", ""))?;
        return Ok(ToolOutput::new(vec![
            Content::Text(format!("Read image file [{mime}]")),
            Content::Image { mime: mime.to_owned(), data: base64::engine::general_purpose::STANDARD.encode(bytes) },
        ]));
    }
    let start = offset.map_or(0, |o| (o - 1.0).max(0.0) as usize);
    let wanted = limit.map(|l| l.max(0.0) as usize);
    let page = scan(reader, start, wanted).await.map_err(|e| errors::node(&e, "read", ""))?;
    if start >= page.total_lines {
        return Err(format!(
            "Offset {} is beyond end of file ({} lines total)",
            js_number(offset.unwrap_or(0.0)),
            page.total_lines
        ));
    }
    Ok(ToolOutput::text(format_page(path, start, wanted, &page)))
}

fn image_type(head: &[u8]) -> Option<&'static str> {
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        Some("image/webp")
    } else if head.starts_with(b"BM") && head.len() >= 14 {
        Some("image/bmp")
    } else {
        None
    }
}

/// The selected lines, kept only up to just past the output limits.
struct Page {
    lines: Vec<Vec<u8>>,
    /// Every line in the file, counting the empty one after a trailing newline, as
    /// JavaScript's `split("\n")` does.
    total_lines: usize,
    first_line_bytes: usize,
}

async fn scan(mut reader: BufReader<tokio::fs::File>, start: usize, wanted: Option<usize>) -> io::Result<Page> {
    let store_limit = DEFAULT_MAX_BYTES + 1;
    let mut page = Page { lines: Vec::new(), total_lines: 0, first_line_bytes: 0 };
    let mut stored_bytes = 0usize;
    let mut storing = true;
    let mut current: Vec<u8> = Vec::new();
    let mut current_len = 0usize;
    let end = wanted.map(|w| start.saturating_add(w));
    loop {
        let chunk = reader.fill_buf().await?;
        if chunk.is_empty() {
            break;
        }
        let index = page.total_lines;
        let selected = index >= start && end.is_none_or(|e| index < e);
        let (piece, newline) = match chunk.iter().position(|&b| b == b'\n') {
            Some(at) => (&chunk[..at], true),
            None => (chunk, false),
        };
        if selected && storing && current.len() < store_limit {
            let room = store_limit - current.len();
            current.extend_from_slice(&piece[..piece.len().min(room)]);
        }
        current_len += piece.len();
        let consumed = piece.len() + usize::from(newline);
        reader.consume(consumed);
        if newline {
            finish_line(&mut page, &mut current, &mut current_len, selected, &mut storing, &mut stored_bytes, start);
        }
    }
    let index = page.total_lines;
    let selected = index >= start && end.is_none_or(|e| index < e);
    finish_line(&mut page, &mut current, &mut current_len, selected, &mut storing, &mut stored_bytes, start);
    Ok(page)
}

fn finish_line(
    page: &mut Page,
    current: &mut Vec<u8>,
    current_len: &mut usize,
    selected: bool,
    storing: &mut bool,
    stored_bytes: &mut usize,
    start: usize,
) {
    if page.total_lines == start {
        page.first_line_bytes = *current_len;
    }
    if selected && *storing {
        *stored_bytes += current.len() + usize::from(!page.lines.is_empty());
        page.lines.push(std::mem::take(current));
        // Past either limit the page is certainly truncated; the rest need only be counted.
        if page.lines.len() > DEFAULT_MAX_LINES + 1 || *stored_bytes > DEFAULT_MAX_BYTES + 1 {
            *storing = false;
        }
    }
    current.clear();
    *current_len = 0;
    page.total_lines += 1;
}

fn format_page(path: &str, start: usize, wanted: Option<usize>, page: &Page) -> String {
    let selected: Vec<String> = page.lines.iter().map(|l| String::from_utf8_lossy(l).into_owned()).collect();
    let content = selected.join("\n");
    let truncation = truncate_head(&content, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let start_display = start + 1;
    if truncation.first_line_exceeds_limit {
        return format!(
            "[Line {start_display} is {}, exceeds {} limit. Use bash: sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]",
            format_size(page.first_line_bytes),
            format_size(DEFAULT_MAX_BYTES)
        );
    }
    if truncation.truncated() {
        let end_display = start_display + truncation.output_lines - 1;
        let next = end_display + 1;
        let limit_note = match truncation.truncated_by {
            Some(super::text::TruncatedBy::Bytes) => format!(" ({} limit)", format_size(DEFAULT_MAX_BYTES)),
            _ => String::new(),
        };
        return format!(
            "{}\n\n[Showing lines {start_display}-{end_display} of {}{limit_note}. Use offset={next} to continue.]",
            truncation.content, page.total_lines
        );
    }
    if let Some(wanted) = wanted {
        let shown = wanted.min(page.total_lines - start);
        if start + shown < page.total_lines {
            let remaining = page.total_lines - (start + shown);
            return format!(
                "{}\n\n[{remaining} more lines in file. Use offset={} to continue.]",
                truncation.content,
                start + shown + 1
            );
        }
    }
    truncation.content
}
