//! Text handling shared by the file and shell tools, matching Pi's `truncate.js` and
//! `edit-diff.js` so models see the same limits, notices and edit semantics.

use unicode_normalization::UnicodeNormalization;

pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Truncation {
    pub content: String,
    pub truncated_by: Option<TruncatedBy>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub last_line_partial: bool,
    pub first_line_exceeds_limit: bool,
}

impl Truncation {
    pub fn truncated(&self) -> bool {
        self.truncated_by.is_some()
    }

    fn whole(content: &str, lines: usize) -> Self {
        Self {
            content: content.to_owned(),
            truncated_by: None,
            total_lines: lines,
            total_bytes: content.len(),
            output_lines: lines,
            output_bytes: content.len(),
            last_line_partial: false,
            first_line_exceeds_limit: false,
        }
    }
}

/// Lines as Pi counts them: a trailing newline does not start another line.
pub fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Keeps the first complete lines within both limits. Never returns a partial line.
pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = split_lines_for_counting(content);
    if lines.len() <= max_lines && content.len() <= max_bytes {
        return Truncation::whole(content, lines.len());
    }
    if lines[0].len() > max_bytes {
        return Truncation {
            content: String::new(),
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines: lines.len(),
            total_bytes: content.len(),
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
        };
    }
    let mut kept = 0;
    let mut bytes = 0;
    let mut by = TruncatedBy::Lines;
    for (index, line) in lines.iter().take(max_lines).enumerate() {
        let cost = line.len() + usize::from(index > 0);
        if bytes + cost > max_bytes {
            by = TruncatedBy::Bytes;
            break;
        }
        kept += 1;
        bytes += cost;
    }
    if kept >= max_lines && bytes <= max_bytes {
        by = TruncatedBy::Lines;
    }
    let output = lines[..kept].join("\n");
    Truncation {
        output_bytes: output.len(),
        content: output,
        truncated_by: Some(by),
        total_lines: lines.len(),
        total_bytes: content.len(),
        output_lines: kept,
        last_line_partial: false,
        first_line_exceeds_limit: false,
    }
}

/// Keeps the last complete lines within both limits. When the final line alone exceeds the
/// byte limit, its tail is returned and `last_line_partial` is set.
pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = split_lines_for_counting(content);
    if lines.len() <= max_lines && content.len() <= max_bytes {
        return Truncation::whole(content, lines.len());
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0;
    let mut by = TruncatedBy::Lines;
    let mut partial = false;
    for line in lines.iter().rev() {
        if kept.len() >= max_lines {
            break;
        }
        let cost = line.len() + usize::from(!kept.is_empty());
        if bytes + cost > max_bytes {
            by = TruncatedBy::Bytes;
            if kept.is_empty() {
                let tail = tail_within_bytes(line, max_bytes);
                bytes = tail.len();
                kept.push(tail);
                partial = true;
            }
            break;
        }
        kept.push(line);
        bytes += cost;
    }
    if kept.len() >= max_lines && bytes <= max_bytes {
        by = TruncatedBy::Lines;
    }
    kept.reverse();
    let output = kept.join("\n");
    Truncation {
        output_bytes: output.len(),
        content: output,
        truncated_by: Some(by),
        total_lines: lines.len(),
        total_bytes: content.len(),
        output_lines: kept.len(),
        last_line_partial: partial,
        first_line_exceeds_limit: false,
    }
}

fn tail_within_bytes(text: &str, max_bytes: usize) -> &str {
    let mut start = text.len().saturating_sub(max_bytes);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

pub fn split_bom(content: &str) -> (&'static str, &str) {
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => ("\u{FEFF}", text),
        None => ("", content),
    }
}

pub fn detect_line_ending(content: &str) -> &'static str {
    match (content.find("\r\n"), content.find('\n')) {
        (Some(crlf), Some(lf)) if crlf < lf => "\r\n",
        _ => "\n",
    }
}

pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" { text.replace('\n', "\r\n") } else { text.to_owned() }
}

/// Pi's fuzzy form: NFKC, trailing whitespace stripped per line, and typographic quotes,
/// dashes and spaces folded to ASCII.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let trimmed = nfkc.split('\n').map(str::trim_end).collect::<Vec<_>>().join("\n");
    trimmed
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

#[derive(Clone, Debug)]
struct Matched {
    edit_index: usize,
    index: usize,
    length: usize,
    new_text: String,
}

/// Applies every edit against the same LF-normalized original. Returns `(base, new)`.
/// Errors carry Pi's exact wording because models learn to react to it.
pub fn apply_edits(normalized: &str, edits: &[Edit], path: &str) -> Result<(String, String), String> {
    let total = edits.len();
    let edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit { old_text: normalize_to_lf(&edit.old_text), new_text: normalize_to_lf(&edit.new_text) })
        .collect();
    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(if total == 1 {
                format!("oldText must not be empty in {path}.")
            } else {
                format!("edits[{index}].oldText must not be empty in {path}.")
            });
        }
    }
    let normalized_fuzzy = normalize_for_fuzzy_match(normalized);
    let fuzzy = edits.iter().any(|edit| {
        !normalized.contains(&edit.old_text) && normalized_fuzzy.contains(&normalize_for_fuzzy_match(&edit.old_text))
    });
    let base_for_replacement = if fuzzy { normalized_fuzzy } else { normalized.to_owned() };
    let fuzzy_base = normalize_for_fuzzy_match(&base_for_replacement);
    let mut matched = Vec::with_capacity(total);
    for (index, edit) in edits.iter().enumerate() {
        let found = base_for_replacement.find(&edit.old_text).map(|at| (at, edit.old_text.len())).or_else(|| {
            let needle = normalize_for_fuzzy_match(&edit.old_text);
            fuzzy_base.find(&needle).map(|at| (at, needle.len()))
        });
        let Some((at, length)) = found else {
            return Err(if total == 1 {
                format!(
                    "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
                )
            } else {
                format!(
                    "Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace and newlines."
                )
            });
        };
        let needle = normalize_for_fuzzy_match(&edit.old_text);
        let occurrences = fuzzy_base.matches(needle.as_str()).count();
        if occurrences > 1 {
            return Err(if total == 1 {
                format!(
                    "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
                )
            });
        }
        matched.push(Matched { edit_index: index, index: at, length, new_text: edit.new_text.clone() });
    }
    matched.sort_by_key(|m| m.index);
    for pair in matched.windows(2) {
        if pair[0].index + pair[0].length > pair[1].index {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                pair[0].edit_index, pair[1].edit_index
            ));
        }
    }
    let new = if fuzzy {
        preserve_unchanged_lines(normalized, &base_for_replacement, &matched)?
    } else {
        replace_all(&base_for_replacement, &matched, 0)
    };
    if new == normalized {
        return Err(if total == 1 {
            format!(
                "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
            )
        } else {
            format!("No changes made to {path}. The replacements produced identical content.")
        });
    }
    Ok((normalized.to_owned(), new))
}

fn replace_all(content: &str, matched: &[Matched], offset: usize) -> String {
    let mut result = content.to_owned();
    for m in matched.iter().rev() {
        let at = m.index - offset;
        result.replace_range(at..at + m.length, &m.new_text);
    }
    result
}

fn line_spans(content: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for (index, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            spans.push((start, index + 1));
            start = index + 1;
        }
    }
    if start < content.len() {
        spans.push((start, content.len()));
    }
    spans
}

/// Fuzzy edits are computed in normalized space. Only the lines they touch are rewritten
/// from that space; every other line keeps the original bytes.
fn preserve_unchanged_lines(original: &str, base: &str, matched: &[Matched]) -> Result<String, String> {
    let original_lines = line_spans(original);
    let base_lines = line_spans(base);
    if original_lines.len() != base_lines.len() {
        return Err("Cannot preserve unchanged lines because the base content has a different line count.".into());
    }
    let mut groups: Vec<(usize, usize, Vec<Matched>)> = Vec::new();
    for m in matched {
        let (start, end) = replacement_line_range(&base_lines, m)?;
        if let Some(group) = groups.last_mut().filter(|group| start < group.1) {
            group.1 = group.1.max(end);
            group.2.push(m.clone());
        } else {
            groups.push((start, end, vec![m.clone()]));
        }
    }
    let mut result = String::with_capacity(original.len());
    let mut next = 0;
    for (start, end, group) in &groups {
        for &(a, b) in &original_lines[next..*start] {
            result.push_str(&original[a..b]);
        }
        let from = base_lines[*start].0;
        let to = base_lines[end - 1].1;
        result.push_str(&replace_all(&base[from..to], group, from));
        next = *end;
    }
    for &(a, b) in &original_lines[next..] {
        result.push_str(&original[a..b]);
    }
    Ok(result)
}

fn replacement_line_range(lines: &[(usize, usize)], m: &Matched) -> Result<(usize, usize), String> {
    let outside = || "Replacement range is outside the base content.".to_owned();
    let start = lines.iter().position(|&(a, b)| m.index >= a && m.index < b).ok_or_else(outside)?;
    let end_offset = m.index + m.length;
    let mut end = start;
    while end < lines.len() && lines[end].1 < end_offset {
        end += 1;
    }
    if end >= lines.len() {
        return Err(outside());
    }
    Ok((start, end + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(old: &str, new: &str) -> Edit {
        Edit { old_text: old.into(), new_text: new.into() }
    }

    #[test]
    fn head_truncation_keeps_whole_lines_within_limits() {
        let t = truncate_head("a\nb\nc\n", 2, 100);
        assert_eq!(t.content, "a\nb");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!((t.total_lines, t.output_lines), (3, 2));

        let t = truncate_head("aaaa\nbbbb\ncccc", 10, 9);
        assert_eq!(t.content, "aaaa\nbbbb");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Bytes));

        let t = truncate_head("x".repeat(20).as_str(), 10, 5);
        assert!(t.first_line_exceeds_limit);
        assert_eq!(t.content, "");

        assert!(!truncate_head("a\nb\n", 2, 100).truncated());
    }

    #[test]
    fn tail_truncation_keeps_the_end_and_can_split_a_huge_last_line() {
        let t = truncate_tail("a\nb\nc", 2, 100);
        assert_eq!(t.content, "b\nc");
        assert_eq!(t.truncated_by, Some(TruncatedBy::Lines));

        let t = truncate_tail("short\n0123456789", 10, 4);
        assert_eq!(t.content, "6789");
        assert!(t.last_line_partial);

        let t = truncate_tail("é".repeat(4).as_str(), 10, 3);
        assert_eq!(t.content, "é");
    }

    #[test]
    fn sizes_format_like_pi() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(50 * 1024), "50.0KB");
        assert_eq!(format_size(3 * 1024 * 1024 / 2), "1.5MB");
    }

    #[test]
    fn line_endings_and_bom_are_detected() {
        assert_eq!(detect_line_ending("a\r\nb\n"), "\r\n");
        assert_eq!(detect_line_ending("a\nb\r\n"), "\n");
        assert_eq!(normalize_to_lf("a\r\nb\rc"), "a\nb\nc");
        assert_eq!(restore_line_endings("a\nb", "\r\n"), "a\r\nb");
        assert_eq!(split_bom("\u{FEFF}x"), ("\u{FEFF}", "x"));
    }

    #[test]
    fn exact_edits_apply_against_the_original() {
        let (_, new) = apply_edits("one\ntwo\nthree\n", &[edit("one", "1"), edit("three", "3")], "f").unwrap();
        assert_eq!(new, "1\ntwo\n3\n");
    }

    #[test]
    fn edit_errors_use_pi_wording() {
        let err = apply_edits("abc", &[edit("zzz", "y")], "f.txt").unwrap_err();
        assert_eq!(
            err,
            "Could not find the exact text in f.txt. The old text must match exactly including all whitespace and newlines."
        );
        let err = apply_edits("a a", &[edit("a", "b")], "f.txt").unwrap_err();
        assert_eq!(
            err,
            "Found 2 occurrences of the text in f.txt. The text must be unique. Please provide more context to make it unique."
        );
        let err = apply_edits("abcdef", &[edit("abcd", "x"), edit("cdef", "y")], "f").unwrap_err();
        assert_eq!(err, "edits[0] and edits[1] overlap in f. Merge them into one edit or target disjoint regions.");
        let err = apply_edits("abc", &[edit("", "y")], "f").unwrap_err();
        assert_eq!(err, "oldText must not be empty in f.");
        let err = apply_edits("abc", &[edit("b", "b")], "f").unwrap_err();
        assert!(err.starts_with("No changes made to f."));
        let err = apply_edits("abc", &[edit("q", "x"), edit("b", "y")], "f").unwrap_err();
        assert!(err.starts_with("Could not find edits[0] in f."));
    }

    #[test]
    fn fuzzy_edits_rewrite_only_touched_lines() {
        let original = "keep   \nsay \u{201C}hi\u{201D}  \nalso keep   \n";
        let (_, new) = apply_edits(original, &[edit("say \"hi\"", "say \"bye\"")], "f").unwrap();
        assert_eq!(new, "keep   \nsay \"bye\"\nalso keep   \n");
    }
}
