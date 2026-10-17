//! Turning what servers answer into what agents read: pages as text with their links, JSON one
//! value per line, windows of long answers, DuckDuckGo's results.

use serde_json::Value;

/// An answer as agents read it: HTML as text, JSON pretty-printed, anything else as is.
pub fn shown(body: &str, content_type: &str, url: &str) -> String {
    if content_type.contains("html") || body.trim_start().starts_with('<') {
        return readable(body, url);
    }
    match serde_json::from_str::<Value>(body) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| body.to_owned()),
        Err(_) => body.to_owned(),
    }
}

/// The `size`-character window of `text` starting at `offset`, with where it sits.
pub fn window_of(text: &str, offset: usize, size: usize) -> String {
    let total = text.chars().count();
    let body: String = text.chars().skip(offset).take(size).collect();
    let end = (offset + size).min(total);
    if offset == 0 && end == total {
        body
    } else {
        format!("[characters {offset}–{end} of {total}; pass offset {end} for more]\n{body}")
    }
}

/// A page's text for reading: scripts and styles dropped, links kept as `[text](href)`, tags
/// removed, whitespace collapsed.
pub fn readable(html: &str, base: &str) -> String {
    let base = url::Url::parse(base).ok();
    let lower = html.to_ascii_lowercase();
    let mut out = String::new();
    let mut i = 0;
    let mut link: Option<String> = None;
    while i < html.len() {
        if html.as_bytes()[i] == b'<' {
            let end = html[i..].find('>').map_or(html.len(), |e| i + e + 1);
            let tag = &lower[i..end];
            for skip in ["<script", "<style", "<noscript", "<svg"] {
                if tag.starts_with(skip) {
                    let close = format!("</{}", &skip[1..]);
                    let after = lower[end..].find(&close).map_or(html.len(), |e| end + e);
                    i = lower[after..].find('>').map_or(html.len(), |e| after + e + 1);
                    break;
                }
            }
            if i > end || i >= html.len() {
                continue;
            }
            if tag.starts_with("<a ") || tag.starts_with("<a\n") {
                let raw = &html[i..end];
                link = attribute(raw, "href").map(|h| match &base {
                    Some(base) => base.join(&h).map_or(h, |u| u.to_string()),
                    None => h,
                });
                out.push('[');
            } else if tag.starts_with("</a") {
                match link.take() {
                    Some(href) => out.push_str(&format!("]({href})")),
                    None => out.push(']'),
                }
            } else if ["<br", "<p", "</p", "<div", "</div", "<li", "<tr", "<h", "</h", "<table"]
                .iter()
                .any(|t| tag.starts_with(t))
            {
                out.push('\n');
            } else if tag.starts_with("<td") || tag.starts_with("<th") {
                out.push(' ');
            }
            i = end;
        } else {
            let next = html[i..].find('<').map_or(html.len(), |n| i + n);
            out.push_str(&decode_entities(&html[i..next]));
            i = next;
        }
    }
    let mut collapsed = String::new();
    for line in out.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() && line != "[]" {
            collapsed.push_str(&line);
            collapsed.push('\n');
        }
    }
    collapsed
}

fn attribute(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let at = lower.find(&format!("{name}="))? + name.len() + 1;
    let rest = &tag[at..];
    let value = match rest.chars().next()? {
        q @ ('"' | '\'') => rest[1..].split(q).next()?,
        _ => rest.split(|c: char| c.is_whitespace() || c == '>').next()?,
    };
    Some(decode_entities(value))
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

/// Results from DuckDuckGo's HTML page: title, URL and snippet.
pub fn duckduckgo_results(html: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    for block in html.split("class=\"result__a\"").skip(1) {
        let href = attribute(&format!(" {block}"), "href").unwrap_or_default();
        let target = url::Url::parse(&format!("https:{href}"))
            .ok()
            .and_then(|u| u.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.into_owned()))
            .unwrap_or(href);
        let title =
            block.split_once('>').map(|(_, rest)| rest.split("</a>").next().unwrap_or_default()).unwrap_or_default();
        let snippet = block
            .split_once("class=\"result__snippet\"")
            .and_then(|(_, rest)| rest.split_once('>'))
            .map(|(_, rest)| rest.split("</a>").next().unwrap_or_default())
            .unwrap_or_default();
        let clean = |s: &str| readable(s, "https://duckduckgo.com/").trim().replace('\n', " ");
        results.push((clean(title), target, clean(snippet)));
    }
    results
}

pub fn short(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        format!("{}…", text.chars().take(max).collect::<String>())
    } else {
        text.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_read_as_text_with_links() {
        let html = r#"<html><head><style>p{}</style><script>var x="<a>";</script></head><body>
            <h1>Fulton&nbsp;County GIS</h1><p>Open <a href="/datasets/parcels">Tax Parcels</a> &amp; more</p>
            <table><tr><td>A</td><td>B</td></tr></table></body></html>"#;
        let text = readable(html, "https://gis.example.gov/hub/");
        assert_eq!(
            text,
            "Fulton County GIS\nOpen [Tax Parcels](https://gis.example.gov/datasets/parcels) & more\nA B\n"
        );
    }

    #[test]
    fn duckduckgo_results_are_read() {
        let html = r#"<div><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgis.fulton.gov%2Frest&amp;rut=1">Fulton <b>GIS</b></a>
            <a class="result__snippet" href="x">Tax <b>parcels</b> service</a></div>"#;
        assert_eq!(
            duckduckgo_results(html),
            [("Fulton GIS".into(), "https://gis.fulton.gov/rest".into(), "Tax parcels service".into())]
        );
    }

    #[test]
    fn long_text_comes_in_windows() {
        let text = "x".repeat(110);
        assert!(window_of(&text, 0, 100).starts_with("[characters 0–100 of 110; pass offset 100 for more]"));
        assert_eq!(window_of("short", 0, 100), "short");
    }

    #[test]
    fn json_reads_one_value_per_line() {
        assert_eq!(
            shown(r#"{"a":[1,2]}"#, "application/json", "https://x.org"),
            "{\n  \"a\": [\n    1,\n    2\n  ]\n}"
        );
        assert_eq!(shown("plain", "text/plain", "https://x.org"), "plain");
    }
}
