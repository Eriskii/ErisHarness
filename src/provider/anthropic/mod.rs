//! The Anthropic Messages API, by API key or by Claude subscription (OAuth). A subscription
//! request carries Claude Code's identity, which lives entirely in [`fingerprint`]; [`wire`]
//! converts transcripts to messages and [`stream`] decodes the streamed reply.

pub mod fingerprint;
mod stream;
mod wire;

use super::transport::{self, Events, Failure, Kind};
use super::{Authorization, Completion, Credentials, Progress, Provider, RateGate, Request};
use anyhow::{Result, anyhow, ensure};
use futures_util::future::BoxFuture;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Betas an API-key request with thinking asks for.
const API_KEY_BETAS: &str = "interleaved-thinking-2025-05-14,context-management-2025-06-27,effort-2025-11-24";

#[derive(Clone, Debug)]
pub enum AnthropicAuth {
    /// The credential's token is an API key, sent as `x-api-key`.
    ApiKey,
    /// The credential's token is a Claude subscription's OAuth token, sent as a bearer token
    /// on requests that identify as Claude Code.
    ClaudeCode(ClaudeCode),
}

#[derive(Clone, Debug)]
pub struct ClaudeCode {
    /// Stable per installation; with `account_id`, it derives the device id.
    pub install_id: String,
    pub account_id: Option<String>,
    /// The Claude Code version to claim. `None` claims [`fingerprint::DEFAULT_VERSION`] (or
    /// `PI_AI_CLAUDE_CODE_VERSION`) and adopts a newer one when the server requires it; a
    /// version given here or in that variable is kept.
    pub version: Option<String>,
}

/// How the model thinks. It is set per provider rather than inferred from the model name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thinking {
    Disabled,
    /// The model decides how much to think; the agent's reasoning effort guides it.
    Adaptive {
        display: bool,
    },
    /// At most `tokens` of thinking: at least 1024 and below `max_tokens`.
    Budget {
        tokens: u64,
        display: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRetention {
    None,
    /// Five minutes.
    Short,
    /// One hour.
    Long,
}

pub struct AnthropicConfig {
    /// Endpoint root including `/v1`; requests go to `{base_url}/messages`.
    pub base_url: String,
    pub auth: AnthropicAuth,
    pub credentials: Arc<dyn Credentials>,
    /// Paces calls for the account; share one gate among every provider using the account.
    pub gate: Arc<RateGate>,
    /// Attempts after the first for rate limits, server errors and dropped connections.
    pub max_retries: u32,
    /// Output ceiling. Subscriptions are held to 64,000.
    pub max_tokens: u64,
    pub thinking: Thinking,
    pub cache_retention: CacheRetention,
    /// Longest a whole request may take, streaming included.
    pub timeout: Duration,
    /// Sent with every request. Authentication and Claude Code identity headers override them.
    pub headers: Vec<(String, String)>,
}

impl AnthropicConfig {
    /// Defaults: the public API, 3 retries, 64,000 output tokens, no thinking, a 10-minute
    /// timeout, and prompt caching for an hour on a subscription or five minutes on a key.
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

    fn oauth(&self) -> bool {
        matches!(self.auth, AnthropicAuth::ClaudeCode(_))
    }
}

pub struct Anthropic {
    config: AnthropicConfig,
    client: reqwest::Client,
    /// The Claude Code version requests claim.
    version: Mutex<String>,
    /// Whether the server may move `version` forward: a subscription with no version pinned.
    upgradable: bool,
}

impl Anthropic {
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let pinned = match &config.auth {
            AnthropicAuth::ClaudeCode(identity) => {
                ensure!(!identity.install_id.is_empty(), "Claude Code install_id must not be empty");
                identity
                    .version
                    .clone()
                    .or_else(|| std::env::var("PI_AI_CLAUDE_CODE_VERSION").ok().filter(|v| !v.is_empty()))
            }
            AnthropicAuth::ApiKey => None,
        };
        let upgradable = config.oauth() && pinned.is_none();
        let version = pinned.unwrap_or_else(|| fingerprint::DEFAULT_VERSION.into());
        fingerprint::validate_version(&version)?;
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self { config, client, version: Mutex::new(version), upgradable })
    }

    async fn attempt(
        &self,
        request: &Request<'_>,
        progress: &(dyn Fn(Progress) + Send + Sync),
    ) -> Result<Completion, Failure> {
        let oauth = self.config.oauth();
        let version = self.version.lock().unwrap().clone();
        let body = wire::body(&self.config, request, &version).map_err(Failure::fatal)?;
        let bytes = fingerprint::serialize(&body, oauth).map_err(Failure::fatal)?;
        let authorization = self.config.credentials.authorize().await.map_err(Failure::fatal)?;
        let headers = self.headers(request, &version, authorization).map_err(Failure::fatal)?;
        let url =
            format!("{}/messages{}", self.config.base_url.trim_end_matches('/'), if oauth { "?beta=true" } else { "" });
        let response = match transport::send(self.client.post(url).headers(headers).body(bytes), progress).await {
            Err(failure) if self.upgrade(&failure.error.to_string(), &version) => {
                return Err(Failure { kind: Kind::Again, wait: Some(Duration::ZERO), ..failure });
            }
            response => response?,
        };
        // Once content has streamed, a failure is final: a retry would stream it again.
        let mut streamed = false;
        let fail =
            |error: anyhow::Error, streamed| if streamed { Failure::fatal(error) } else { Failure::retry(error) };
        let mut events = Events::new(response);
        let mut stream = stream::Stream::new(request.model, oauth);
        while let Some(data) = events.next().await {
            let data = data.map_err(|e| fail(e.into(), streamed))?;
            let event: Value = serde_json::from_str(&data).map_err(Failure::fatal)?;
            if event["type"] == "error" {
                let failure = stream_error(&event["error"]);
                return Err(if streamed { Failure { kind: Kind::Fatal, ..failure } } else { failure });
            }
            streamed |= matches!(event["type"].as_str(), Some("content_block_start" | "content_block_delta"));
            if let Some(completion) = stream.event(event, progress).map_err(Failure::fatal)? {
                return Ok(completion);
            }
        }
        Err(fail(anyhow!("Anthropic stream ended before message_stop"), streamed))
    }

    fn headers(&self, request: &Request, version: &str, authorization: Authorization) -> Result<HeaderMap> {
        let oauth = self.config.oauth();
        let thinking = self.config.thinking != Thinking::Disabled;
        // Only the selected credential is sent, whatever the other headers carry.
        let mut pairs: Vec<(String, String)> = (self.config.headers.iter().chain(&authorization.headers))
            .filter(|(key, _)| !key.eq_ignore_ascii_case("authorization") && !key.eq_ignore_ascii_case("x-api-key"))
            .cloned()
            .collect();
        pairs.push(("content-type".into(), "application/json".into()));
        pairs.push(("anthropic-version".into(), "2023-06-01".into()));
        if oauth {
            pairs.extend(fingerprint::headers(
                version,
                request.cache_key,
                thinking || !request.tools.is_empty(),
                thinking,
            ));
        } else {
            pairs.push(("accept".into(), "text/event-stream".into()));
            if thinking {
                pairs.push(("anthropic-beta".into(), API_KEY_BETAS.into()));
            }
        }
        let mut headers = HeaderMap::new();
        for (key, value) in pairs {
            headers.insert(HeaderName::from_bytes(key.as_bytes())?, HeaderValue::from_str(&value)?);
        }
        let (name, token) = match oauth {
            true => ("authorization", format!("Bearer {}", authorization.token)),
            false => ("x-api-key", authorization.token),
        };
        let mut credential = HeaderValue::from_str(&token)?;
        credential.set_sensitive(true);
        headers.insert(name, credential);
        Ok(headers)
    }

    /// Whether a failure asks for a newer Claude Code version. The first request to see it
    /// adopts the version; others that raced it retry with the version it adopted.
    fn upgrade(&self, message: &str, sent: &str) -> bool {
        if !self.upgradable {
            return false;
        }
        let mut current = self.version.lock().unwrap();
        if let Some(required) = fingerprint::required_version(message, &current) {
            *current = required;
            return true;
        }
        *current != sent && fingerprint::required_version(message, sent).is_some()
    }
}

/// A stream's `error` event, retried like the HTTP status its type stands for.
fn stream_error(error: &Value) -> Failure {
    let kind = error["type"].as_str().unwrap_or("api_error");
    let message = anyhow!("{kind}: {}", error["message"].as_str().unwrap_or("Anthropic stream error"));
    match kind {
        "rate_limit_error" => Failure::status(message, 429),
        "overloaded_error" => Failure::status(message, 529),
        "api_error" => Failure::status(message, 500),
        _ => Failure::fatal(message),
    }
}

impl Provider for Anthropic {
    fn complete<'a>(
        &'a self,
        request: Request<'a>,
        progress: &'a (dyn Fn(Progress) + Send + Sync),
    ) -> BoxFuture<'a, Result<Completion>> {
        Box::pin(async move {
            transport::retrying(&self.config.gate, self.config.max_retries, progress, || {
                self.attempt(&request, progress)
            })
            .await
        })
    }
}
