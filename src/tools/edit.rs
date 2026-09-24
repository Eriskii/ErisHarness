//! Pi's `edit` tool: exact (then whitespace- and typography-tolerant) replacement of unique,
//! non-overlapping blocks, preserving the file's BOM and line endings.

use super::text::{self, Edit as Replacement};
use super::{Tool, ToolContext, ToolOutput, errors, resolve};
use crate::machine::OpenMode;
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::io::SeekFrom;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

pub struct Edit;

impl Tool for Edit {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> String {
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.".into()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
                            "newText": {"type": "string", "description": "Replacement text for this targeted edit."}
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }

    fn snippet(&self) -> &str {
        "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
    }

    fn guidelines(&self) -> &[&str] {
        &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ]
    }

    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput> {
        Box::pin(async move {
            match edit(context, args).await {
                Ok(output) => output,
                Err(message) => ToolOutput::error(message),
            }
        })
    }
}

fn single(value: &Value) -> Option<Replacement> {
    Some(Replacement {
        old_text: value.get("oldText")?.as_str()?.to_owned(),
        new_text: value.get("newText")?.as_str()?.to_owned(),
    })
}

/// Models send `edits` as a JSON string, as one object, or as top-level `oldText`/`newText`.
fn replacements(args: &Value) -> Result<Vec<Replacement>, String> {
    let invalid = || "Edit tool input is invalid. edits must contain at least one replacement.".to_owned();
    let mut edits: Vec<Replacement> = match args.get("edits") {
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Array(items)) => items.iter().map(single).collect::<Option<_>>().ok_or_else(invalid)?,
            Ok(one) => single(&one).into_iter().collect(),
            Err(_) => Vec::new(),
        },
        Some(Value::Array(items)) => items.iter().map(single).collect::<Option<_>>().ok_or_else(invalid)?,
        Some(one @ Value::Object(_)) => single(one).into_iter().collect(),
        _ => Vec::new(),
    };
    edits.extend(single(args));
    if edits.is_empty() { Err(invalid()) } else { Ok(edits) }
}

async fn edit(context: &ToolContext, args: Value) -> Result<ToolOutput, String> {
    let edits = replacements(&args)?;
    let path = args.get("path").and_then(Value::as_str).ok_or("Edit tool input is invalid. path must be a string.")?;
    let machine = &context.machine;
    let absolute = resolve(path, machine.cwd(), machine.home());
    let fd = machine
        .open(&absolute, OpenMode::Update)
        .await
        .map_err(|e| format!("Could not edit file: {path}. Error code: {}.", errors::code(&e)))?;
    let mut file = tokio::fs::File::from_std(std::fs::File::from(fd));
    let mut raw = Vec::new();
    file.read_to_end(&mut raw).await.map_err(|e| errors::node(&e, "read", ""))?;
    let raw = String::from_utf8_lossy(&raw);
    let (bom, content) = text::split_bom(&raw);
    let ending = text::detect_line_ending(content);
    let normalized = text::normalize_to_lf(content);
    let (base, new) = text::apply_edits(&normalized, &edits, path)?;
    let output = format!("{bom}{}", text::restore_line_endings(&new, ending));
    file.seek(SeekFrom::Start(0)).await.map_err(|e| errors::node(&e, "write", ""))?;
    file.set_len(0).await.map_err(|e| errors::node(&e, "write", ""))?;
    file.write_all(output.as_bytes()).await.map_err(|e| errors::node(&e, "write", ""))?;
    file.flush().await.map_err(|e| errors::node(&e, "write", ""))?;
    let diff = similar::TextDiff::from_lines(&base, &new);
    let patch = diff.unified_diff().context_radius(4).header(path, path).to_string();
    let first_changed_line =
        diff.ops().iter().find(|op| op.tag() != similar::DiffTag::Equal).map(|op| op.new_range().start + 1);
    Ok(ToolOutput::text(format!("Successfully replaced {} block(s) in {path}.", edits.len()))
        .with_details(json!({"patch": patch, "firstChangedLine": first_changed_line})))
}
