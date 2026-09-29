//! Anthropic Messages transport with explicit API-key or Claude subscription OAuth mode.
//! Claude Code fingerprinting is confined to this provider. Credentials are supplied by
//! the host through `Credentials`, exactly as with the Responses provider.

pub mod fingerprint;
mod stream;
mod wire;

use super::http::{response_event, retry_after};
use super::sse::Parser;
use super::{Completion, Credentials, Progress, Provider, ProviderEvent, RateGate, Request};
use anyhow::{Result, anyhow, ensure};
use futures_util::{StreamExt, future::BoxFuture};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct ClaudeCode {
    /// Stable installation identity, used with account_id to derive OMP's device hash.
    pub install_id: String,
    pub account_id: Option<String>,
    /// None: pinned fallback, with server-directed version upgrades. An explicit version
    /// (or PI_AI_CLAUDE_CODE_VERSION) disables upgrades, as in OMP.
    pub version: Option<String>,
}

#[derive(Clone, Debug)]
pub enum AnthropicAuth {
    ApiKey,
    ClaudeCode(ClaudeCode),
}

/// Model capabilities are explicit, so adding a model never changes the harness runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thinking {
    Disabled,
    Adaptive { display: bool },
    Budget { tokens: u64, display: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

pub struct AnthropicConfig {
    /// Endpoint root including /v1. OAuth calls append /messages?beta=true.
    pub base_url: String,
    pub auth: AnthropicAuth,
    pub credentials: Arc<dyn Credentials>,
    pub gate: Arc<RateGate>,
    pub max_retries: u32,
    pub max_tokens: u64,
    pub thinking: Thinking,
    pub cache_retention: CacheRetention,
    pub timeout: Duration,
    /// Extra headers; identity and authorization headers remain owned by this provider.
    pub headers: Vec<(String, String)>,
}

impl AnthropicConfig {
    pub fn new(credentials: Arc<dyn Credentials>, auth: AnthropicAuth) -> Self {
        let oauth = matches!(auth, AnthropicAuth::ClaudeCode(_));
        Self {
            base_url: "https://api.anthropic.com/v1".into(),
            auth,
            credentials,
            gate: RateGate::new(16, 256),
            max_retries: 3,
            max_tokens: 64000,
            thinking: Thinking::Disabled,
            cache_retention: if oauth { CacheRetention::Long } else { CacheRetention::Short },
            timeout: Duration::from_secs(600),
            headers: Vec::new(),
        }
    }
}

pub struct Anthropic {
    config: AnthropicConfig,
    client: reqwest::Client,
    version: Mutex<String>,
    pinned: bool,
}

impl Anthropic {
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let explicit = match &config.auth {
            AnthropicAuth::ClaudeCode(identity) => {
                ensure!(!identity.install_id.is_empty(), "Claude Code install_id must not be empty");
                identity
                    .version
                    .clone()
                    .or_else(|| std::env::var("PI_AI_CLAUDE_CODE_VERSION").ok().filter(|v| !v.is_empty()))
            }
            AnthropicAuth::ApiKey => None,
        };
        let pinned = explicit.is_some();
        let version = explicit.unwrap_or_else(|| fingerprint::DEFAULT_VERSION.into());
        fingerprint::validate_version(&version)?;
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self { config, client, version: Mutex::new(version), pinned })
    }

    async fn attempt(
        &self,
        request: &Request<'_>,
        progress: &(dyn Fn(Progress) + Send + Sync),
    ) -> Result<Completion, Failure> {
        let oauth = matches!(self.config.auth, AnthropicAuth::ClaudeCode(_));
        let version = self.version.lock().unwrap().clone();
        let body = wire::body(&self.config, request, &version).map_err(Failure::fatal)?;
        let bytes = fingerprint::serialize(&body, oauth).map_err(Failure::fatal)?;
        let authorization = self.config.credentials.authorize().await.map_err(Failure::fatal)?;
        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.headers.iter().chain(&authorization.headers) {
            let key = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|e| Failure::fatal(e.into()))?;
            let value = reqwest::header::HeaderValue::from_str(value).map_err(|e| Failure::fatal(e.into()))?;
            headers.insert(key, value);
        }
        // Never allow another auth mode to leak alongside the selected credential.
        headers.remove("authorization");
        headers.remove("x-api-key");
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        if oauth {
            for (key, value) in fingerprint::headers(
                &version,
                request.cache_key,
                !request.tools.is_empty() || self.config.thinking != Thinking::Disabled,
                self.config.thinking != Thinking::Disabled,
            ) {
                headers.insert(
                    reqwest::header::HeaderName::from_bytes(key.as_bytes()).unwrap(),
                    value.parse().map_err(|e: reqwest::header::InvalidHeaderValue| Failure::fatal(e.into()))?,
                );
            }
        } else {
            headers.insert("accept", "text/event-stream".parse().unwrap());
            if self.config.thinking != Thinking::Disabled {
                headers.insert(
                    "anthropic-beta",
                    "interleaved-thinking-2025-05-14,context-management-2025-06-27,effort-2025-11-24".parse().unwrap(),
                );
            }
        }
        let mut auth = reqwest::header::HeaderValue::from_str(&if oauth {
            format!("Bearer {}", authorization.token)
        } else {
            authorization.token
        })
        .map_err(|e| Failure::fatal(e.into()))?;
        auth.set_sensitive(true);
        headers.insert(if oauth { "authorization" } else { "x-api-key" }, auth);
        let url =
            format!("{}/messages{}", self.config.base_url.trim_end_matches('/'), if oauth { "?beta=true" } else { "" });
        let response = self
            .client
            .post(url)
            .headers(headers)
            .body(bytes)
            .send()
            .await
            .map_err(|e| Failure::retry(e.into(), None, None))?;
        let status = response.status();
        let wait = retry_after(response.headers());
        progress(Progress::Event(response_event(status.as_u16(), response.headers())));
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let value: Value = serde_json::from_str(&text).unwrap_or_default();
            let message = format!(
                "{} {}: {}",
                status.as_u16(),
                value["error"]["type"].as_str().unwrap_or("http_error"),
                value["error"]["message"].as_str().unwrap_or(&text)
            );
            if oauth && !self.pinned {
                let mut current = self.version.lock().unwrap();
                if let Some(required) = fingerprint::required_version(&message, &current) {
                    *current = required;
                    return Err(Failure {
                        error: anyhow!(message),
                        status: Some(status.as_u16()),
                        wait: Some(Duration::ZERO),
                        kind: Kind::Version,
                    });
                }
                // Another in-flight request may already have adopted the required version.
                if *current != version && fingerprint::required_version(&message, &version).is_some() {
                    return Err(Failure {
                        error: anyhow!(message),
                        status: Some(status.as_u16()),
                        wait: Some(Duration::ZERO),
                        kind: Kind::Version,
                    });
                }
            }
            return Err(Failure {
                error: anyhow!(message),
                status: Some(status.as_u16()),
                wait,
                kind: if status.as_u16() == 429 {
                    Kind::RateLimit
                } else if status.is_server_error() {
                    Kind::Retry
                } else {
                    Kind::Fatal
                },
            });
        }
        let mut bytes = response.bytes_stream();
        let mut parser = Parser::default();
        let mut stream = stream::Stream::default();
        let mut emitted = false;
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk
                .map_err(|e| if emitted { Failure::fatal(e.into()) } else { Failure::retry(e.into(), None, None) })?;
            for data in parser.feed(&chunk) {
                let event: Value = serde_json::from_str(&data).map_err(|e| Failure::fatal(e.into()))?;
                if event["type"] == "error" {
                    let kind = event["error"]["type"].as_str().unwrap_or("api_error");
                    let message = event["error"]["message"].as_str().unwrap_or("Anthropic stream error");
                    let status = match kind {
                        "rate_limit_error" => Some(429),
                        "overloaded_error" => Some(529),
                        "api_error" => Some(500),
                        _ => None,
                    };
                    return Err(Failure {
                        error: anyhow!("{kind}: {message}"),
                        status,
                        wait: None,
                        kind: if emitted {
                            Kind::Fatal
                        } else if status == Some(429) {
                            Kind::RateLimit
                        } else if status.is_some() {
                            Kind::Retry
                        } else {
                            Kind::Fatal
                        },
                    });
                }
                emitted |= matches!(event["type"].as_str(), Some("content_block_start" | "content_block_delta"));
                if let Some(completion) = stream.event(event, request.model, oauth, progress).map_err(Failure::fatal)? {
                    return Ok(completion);
                }
            }
        }
        let error = anyhow!("Anthropic stream ended before message_stop");
        Err(if emitted { Failure::fatal(error) } else { Failure::retry(error, None, None) })
    }
}

#[derive(PartialEq)]
enum Kind {
    Fatal,
    Retry,
    RateLimit,
    Version,
}
struct Failure {
    error: anyhow::Error,
    status: Option<u16>,
    wait: Option<Duration>,
    kind: Kind,
}
impl Failure {
    fn fatal(error: anyhow::Error) -> Self {
        Self { error, status: None, wait: None, kind: Kind::Fatal }
    }
    fn retry(error: anyhow::Error, status: Option<u16>, wait: Option<Duration>) -> Self {
        Self { error, status, wait, kind: Kind::Retry }
    }
}

impl Provider for Anthropic {
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        progress: &'a (dyn Fn(Progress) + Send + Sync),
    ) -> BoxFuture<'a, Result<Completion>> {
        Box::pin(async move {
            let mut retries = 0;
            let mut version_retries = 0;
            let mut backoff = Duration::from_millis(500);
            loop {
                let report = |hold| progress(Progress::Held(hold));
                let permit = self.config.gate.acquire(&report).await;
                let failure = match self.attempt(&request, progress).await {
                    Ok(completion) => {
                        permit.succeeded();
                        return Ok(completion);
                    }
                    Err(failure) => failure,
                };
                let wait = failure.wait.unwrap_or(backoff);
                if failure.kind == Kind::RateLimit {
                    permit.rate_limited(wait);
                } else {
                    drop(permit);
                }
                if failure.kind == Kind::Fatal {
                    return Err(failure.error);
                }
                if failure.kind == Kind::Version {
                    if version_retries >= 1 {
                        return Err(failure.error);
                    }
                    version_retries += 1;
                } else {
                    if retries >= self.config.max_retries {
                        return Err(failure.error);
                    }
                    retries += 1;
                }
                progress(Progress::Event(ProviderEvent::Retry {
                    attempt: retries + version_retries,
                    status: failure.status,
                    message: failure.error.to_string(),
                    delay_ms: wait.as_millis() as u64,
                }));
                if failure.kind != Kind::RateLimit {
                    tokio::time::sleep(wait).await;
                }
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        })
    }
}
