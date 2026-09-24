//! Server-sent events: blocks separated by a blank line, payload in `data:` lines.

#[derive(Default)]
pub struct Parser {
    buffer: Vec<u8>,
}

impl Parser {
    /// Feeds bytes and returns the data payloads of every complete event.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((end, separator)) = find_boundary(&self.buffer) {
            let block: Vec<u8> = self.buffer.drain(..end + separator).collect();
            let text = String::from_utf8_lossy(&block[..end]);
            let data: Vec<&str> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|data| data.strip_prefix(' ').unwrap_or(data))
                .collect();
            if !data.is_empty() {
                events.push(data.join("\n"));
            }
        }
        events
    }
}

fn find_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|w| w == b"\n\n").map(|at| (at, 2));
    let crlf = buffer.windows(4).position(|w| w == b"\r\n\r\n").map(|at| (at, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::Parser;

    #[test]
    fn splits_events_across_chunks() {
        let mut parser = Parser::default();
        assert!(parser.feed(b"event: a\ndata: {\"x\"").is_empty());
        assert_eq!(parser.feed(b":1}\n\ndata: two\r\n\r\n"), vec!["{\"x\":1}", "two"]);
        assert_eq!(parser.feed(b": comment\n\n"), Vec::<String>::new());
    }
}
