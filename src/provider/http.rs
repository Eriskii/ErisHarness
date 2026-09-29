//! Shared HTTP diagnostics and retry timing for independently implemented transports.

use super::ProviderEvent;
use reqwest::header::HeaderMap;
use std::time::{Duration, SystemTime};

pub(super) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
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

pub(super) fn response_event(status: u16, headers: &HeaderMap) -> ProviderEvent {
    let headers = headers
        .iter()
        .filter_map(|(key, value)| {
            let name = key.as_str();
            (name.starts_with("anthropic-ratelimit-")
                || name.starts_with("x-ratelimit-")
                || matches!(name, "retry-after" | "retry-after-ms" | "request-id" | "x-request-id"))
            .then(|| value.to_str().ok().map(|value| (name.to_owned(), value.to_owned())))
            .flatten()
        })
        .collect();
    ProviderEvent::Response { status, headers }
}
