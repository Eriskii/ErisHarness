//! What the HTTP providers share: sending a request and classifying its failure, reading
//! server-sent events, and retrying calls through the account's [`RateGate`].

use super::{Completion, Progress, ProviderEvent, RateGate};
use anyhow::anyhow;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use reqwest::header::HeaderMap;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

const FIRST_BACKOFF: Duration = Duration::from_millis(500);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Why an attempt failed, and whether trying again can help.
pub(super) struct Failure {
    pub error: anyhow::Error,
    pub status: Option<u16>,
    /// How long the provider asked to wait; the backoff when `None`.
    pub wait: Option<Duration>,
    pub kind: Kind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Fatal,
    /// A server error or dropped connection: try again after the wait.
    Retry,
    /// The account's gate pauses every call until the wait ends, then this one tries again.
    RateLimited,
    /// Try again at once, outside the retry allowance. Once per call.
    Again,
}

impl Kind {
    fn of_status(status: u16) -> Self {
        match status {
            429 => Kind::RateLimited,
            500.. => Kind::Retry,
            _ => Kind::Fatal,
        }
    }
}

impl Failure {
    pub fn fatal(error: impl Into<anyhow::Error>) -> Self {
        Self { error: error.into(), status: None, wait: None, kind: Kind::Fatal }
    }

    pub fn retry(error: impl Into<anyhow::Error>) -> Self {
        Self { kind: Kind::Retry, ..Self::fatal(error) }
    }

    /// A failure the provider reported for `status` itself, such as in a stream's error event.
    pub fn status(error: anyhow::Error, status: u16) -> Self {
        Self { error, status: Some(status), wait: None, kind: Kind::of_status(status) }
    }
}

/// Sends a request and reports its response to observers. A connection failure is retried;
/// an error status fails as `"{status} {type}: {message}"` from the provider's error body.
pub(super) async fn send(
    request: reqwest::RequestBuilder,
    progress: &(dyn Fn(Progress) + Send + Sync),
) -> Result<reqwest::Response, Failure> {
    let response = request.send().await.map_err(Failure::retry)?;
    let status = response.status().as_u16();
    progress(Progress::Event(response_event(status, response.headers())));
    if response.status().is_success() {
        return Ok(response);
    }
    let wait = retry_after(response.headers());
    let text = response.text().await.unwrap_or_default();
    let body: Value = serde_json::from_str(&text).unwrap_or_default();
    let message = body["error"]["message"].as_str().unwrap_or(&text);
    let error = match body["error"]["type"].as_str() {
        Some(kind) => anyhow!("{status} {kind}: {message}"),
        None => anyhow!("{status} {message}"),
    };
    Err(Failure { wait, ..Failure::status(error, status) })
}

/// Runs `attempt` until it succeeds, holding a gate slot for each try. Rate limits pause the
/// gate; other retryable failures wait out their backoff. Each retry is reported.
pub(super) async fn retrying<F>(
    gate: &Arc<RateGate>,
    max_retries: u32,
    progress: &(dyn Fn(Progress) + Send + Sync),
    mut attempt: impl FnMut() -> F,
) -> anyhow::Result<Completion>
where
    F: Future<Output = Result<Completion, Failure>>,
{
    let report = |hold| progress(Progress::Held(hold));
    let (mut retries, mut again) = (0, false);
    let mut backoff = FIRST_BACKOFF;
    loop {
        let permit = gate.acquire(&report).await;
        let failure = match attempt().await {
            Ok(completion) => {
                permit.succeeded();
                return Ok(completion);
            }
            Err(failure) => failure,
        };
        let wait = failure.wait.unwrap_or(backoff);
        if failure.kind == Kind::RateLimited {
            permit.rate_limited(wait);
        } else {
            drop(permit);
        }
        let allowed = match failure.kind {
            Kind::Fatal => false,
            Kind::Again => !std::mem::replace(&mut again, true),
            Kind::Retry | Kind::RateLimited => {
                retries += 1;
                retries <= max_retries
            }
        };
        if !allowed {
            return Err(failure.error);
        }
        progress(Progress::Event(ProviderEvent::Retry {
            attempt: retries + u32::from(again),
            status: failure.status,
            message: failure.error.to_string(),
            delay_ms: wait.as_millis() as u64,
        }));
        // A rate-limited call waits at the gate instead.
        if failure.kind != Kind::RateLimited {
            tokio::time::sleep(wait).await;
        }
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}

/// The data of each server-sent event in a response body: blocks separated by a blank line,
/// payload in `data:` lines.
pub(super) struct Events {
    body: BoxStream<'static, reqwest::Result<bytes::Bytes>>,
    buffer: Vec<u8>,
    ready: VecDeque<String>,
}

impl Events {
    pub fn new(response: reqwest::Response) -> Self {
        Self { body: response.bytes_stream().boxed(), buffer: Vec::new(), ready: VecDeque::new() }
    }

    /// The next event's data, or `None` once the body ends.
    pub async fn next(&mut self) -> Option<reqwest::Result<String>> {
        loop {
            if let Some(data) = self.ready.pop_front() {
                return Some(Ok(data));
            }
            match self.body.next().await? {
                Ok(chunk) => self.feed(&chunk),
                Err(error) => return Some(Err(error)),
            }
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        while let Some((end, separator)) = boundary(&self.buffer) {
            let block: Vec<u8> = self.buffer.drain(..end + separator).collect();
            let text = String::from_utf8_lossy(&block[..end]);
            let data: Vec<&str> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|data| data.strip_prefix(' ').unwrap_or(data))
                .collect();
            if !data.is_empty() {
                self.ready.push_back(data.join("\n"));
            }
        }
    }
}

/// Where the first event ends, and the length of the blank line that ends it.
fn boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|w| w == b"\n\n").map(|at| (at, 2));
    let crlf = buffer.windows(4).position(|w| w == b"\r\n\r\n").map(|at| (at, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// `retry-after-ms`, else `retry-after` in seconds or as an HTTP date.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let number = |name: &str| {
        let n = headers.get(name)?.to_str().ok()?.trim().parse::<f64>().ok()?;
        (n.is_finite() && n >= 0.0).then_some(n)
    };
    number("retry-after-ms")
        .and_then(|ms| Duration::try_from_secs_f64(ms / 1000.0).ok())
        .or_else(|| number("retry-after").and_then(|s| Duration::try_from_secs_f64(s).ok()))
        .or_else(|| {
            let date = httpdate::parse_http_date(headers.get("retry-after")?.to_str().ok()?).ok()?;
            Some(date.duration_since(SystemTime::now()).unwrap_or_default())
        })
}

/// The status and the rate-limit, retry and request-id headers; never credentials or cookies.
fn response_event(status: u16, headers: &HeaderMap) -> ProviderEvent {
    let headers = headers
        .iter()
        .filter(|(name, _)| {
            let name = name.as_str();
            name.starts_with("anthropic-ratelimit-")
                || name.starts_with("x-ratelimit-")
                || matches!(name, "retry-after" | "retry-after-ms" | "request-id" | "x-request-id")
        })
        .filter_map(|(name, value)| Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned())))
        .collect();
    ProviderEvent::Response { status, headers }
}

#[cfg(test)]
mod tests {
    use super::Events;
    use futures_util::StreamExt;

    #[test]
    fn events_split_across_chunks() {
        let mut events =
            Events { body: futures_util::stream::empty().boxed(), buffer: Vec::new(), ready: Default::default() };
        events.feed(b"event: a\ndata: {\"x\"");
        assert!(events.ready.is_empty());
        events.feed(b":1}\n\ndata: two\r\n\r\n");
        events.feed(b": comment\n\n");
        assert_eq!(events.ready, ["{\"x\":1}", "two"]);
    }
}
