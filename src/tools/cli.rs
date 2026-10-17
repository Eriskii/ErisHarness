//! For tools that are command lines, as the model calls them with one `command` string: splitting
//! it into words as a shell would (quotes, backslashes, no expansion), parsing it with clap,
//! `--grep`, and stopping when the agent is interrupted. There is no shell behind them.

use super::{ToolContext, ToolOutput};
use clap::Parser;

/// Splits a command line into words. Single quotes keep everything literally, double quotes
/// allow `\"` and `\\`, a backslash outside quotes escapes the next character, and quoted and
/// unquoted parts next to each other make one word, so `'it'\''s'` is `it's`.
pub fn words(line: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut word: Option<String> = None;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => words.extend(word.take()),
            '\'' => {
                let w = word.get_or_insert_default();
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(c) => w.push(c),
                        None => return Err("unclosed ' quote".into()),
                    }
                }
            }
            '"' => {
                let w = word.get_or_insert_default();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(c @ ('"' | '\\')) => w.push(c),
                            Some(c) => {
                                w.push('\\');
                                w.push(c);
                            }
                            None => return Err("unclosed \" quote".into()),
                        },
                        Some(c) => w.push(c),
                        None => return Err("unclosed \" quote".into()),
                    }
                }
            }
            '\\' => word.get_or_insert_default().extend(chars.next()),
            c => word.get_or_insert_default().push(c),
        }
    }
    words.extend(word);
    Ok(words)
}

/// Parses `line` as a command of `program`, with or without the program's name in front.
/// Help is an answer; a parse error is a tool error with clap's message.
pub fn parse<C: Parser>(program: &str, line: &str) -> Result<C, ToolOutput> {
    let mut argv = words(line).map_err(|e| ToolOutput::error(format!("{program}: {e}")))?;
    if argv.first().is_some_and(|w| w == program) {
        argv.remove(0);
    }
    argv.insert(0, program.into());
    C::try_parse_from(argv).map_err(|error| {
        let text = error.render().to_string();
        if error.use_stderr() { ToolOutput::error(text) } else { ToolOutput::text(text) }
    })
}

/// The `command` argument every CLI tool takes.
pub fn line(args: &serde_json::Value, program: &str, example: &str) -> Result<String, ToolOutput> {
    args["command"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| ToolOutput::error(format!("Give the command line as `command`, such as: {program} {example}")))
}

pub fn parameters(example: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {"command": {"type": "string", "description": format!("The command line, such as: {example}")}},
        "required": ["command"],
    })
}

/// The lines of `text` matching `pattern` (a case-insensitive regular expression), each with
/// `context` lines around it and its line number, groups apart separated by `--`.
pub fn grep(text: &str, pattern: &str, context: usize) -> Result<String, String> {
    let regex = regex::RegexBuilder::new(pattern)
        .case_insensitive(true)
        .size_limit(1 << 20)
        .build()
        .map_err(|e| format!("--grep {pattern}: {e}"))?;
    let lines: Vec<&str> = text.lines().collect();
    let hits: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| regex.is_match(l)).map(|(i, _)| i).collect();
    if hits.is_empty() {
        return Ok(format!("no line of {} matches {pattern}", lines.len()));
    }
    let mut out = format!("{} of {} lines match {pattern}\n", hits.len(), lines.len());
    let mut shown_to: Option<usize> = None;
    for &hit in &hits {
        let start = hit.saturating_sub(context);
        let end = (hit + context).min(lines.len() - 1);
        let from = match shown_to {
            Some(to) if start <= to + 1 => to + 1,
            Some(_) => {
                out.push_str("--\n");
                start
            }
            None => start,
        };
        for (i, line) in lines.iter().enumerate().take(end + 1).skip(from) {
            out.push_str(&format!("{}{} {line}\n", i + 1, if i == hit { ':' } else { '-' }));
        }
        shown_to = Some(shown_to.map_or(end, |to| to.max(end)));
    }
    Ok(out)
}

/// Runs a command until it finishes or the agent is interrupted.
pub async fn until_stopped(context: &ToolContext, command: impl Future<Output = ToolOutput>) -> ToolOutput {
    tokio::select! {
        output = command => output,
        _ = context.cancel.cancelled() => ToolOutput::error("stopped: the agent was interrupted"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_split_like_a_shell() {
        assert_eq!(words(r#"get "a b" c\ d 'e "f" g'"#).unwrap(), ["get", "a b", "c d", r#"e "f" g"#]);
        assert_eq!(words(r#"'it'\''s'"#).unwrap(), ["it's"]);
        assert_eq!(words(r#""say \"hi\" \n""#).unwrap(), [r#"say "hi" \n"#]);
        assert_eq!(words("  ").unwrap(), Vec::<String>::new());
        assert!(words("'open").is_err());
    }

    #[test]
    fn grep_shows_matches_with_context_and_line_numbers() {
        let text = "a\nParcels\nb\nc\nd\nTAX parcels\ne";
        assert_eq!(
            grep(text, "parcels", 1).unwrap(),
            "2 of 7 lines match parcels\n1- a\n2: Parcels\n3- b\n--\n5- d\n6: TAX parcels\n7- e\n"
        );
        assert_eq!(grep(text, "^[bc]$", 0).unwrap(), "2 of 7 lines match ^[bc]$\n3: b\n4: c\n");
        assert!(grep(text, "(", 0).is_err());
        assert!(grep(text, "zzz", 0).unwrap().starts_with("no line"));
    }
}
