//! The client `web` reads the public web with: GET only, the app's User-Agent, timeouts, a size
//! cap on every body, a few requests at a time per host, and no way into private networks,
//! since agents choose the URLs.

use anyhow::{Result, anyhow, bail, ensure};
use futures_util::StreamExt;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Semaphore;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub per_host: usize,
    pub timeout: Duration,
    /// Whether loopback and private addresses may be reached; only tests allow them.
    pub allow_private: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self { per_host: 3, timeout: Duration::from_secs(60), allow_private: false }
    }
}

#[derive(Clone)]
pub struct Http(Arc<Inner>);

struct Inner {
    client: reqwest::Client,
    limits: Limits,
    hosts: Mutex<HashMap<String, Arc<Semaphore>>>,
}

pub struct Fetched {
    pub status: u16,
    pub content_type: String,
    pub url: String,
    pub body: Vec<u8>,
    /// The body was longer than the cap and was cut.
    pub truncated: bool,
}

impl Fetched {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Whether an address belongs to this machine or a private network.
pub fn private(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1])
                || v4.octets()[0] == 0
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                || v6.to_ipv4_mapped().is_some_and(|v4| private(IpAddr::V4(v4)))
        }
    }
}

/// Resolves names as the system does, leaving out private addresses.
struct PublicOnly;

impl Resolve for PublicOnly {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((name.as_str(), 0)).await?.collect();
            let public: Vec<SocketAddr> = addrs.into_iter().filter(|a| !private(a.ip())).collect();
            if public.is_empty() {
                return Err(format!("{} resolves only to private addresses", name.as_str()).into());
            }
            Ok(Box::new(public.into_iter()) as Addrs)
        })
    }
}

fn literal_private(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(v4)) => private(IpAddr::V4(v4)),
        Some(url::Host::Ipv6(v6)) => private(IpAddr::V6(v6)),
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost") || d.ends_with(".localhost"),
        None => true,
    }
}

impl Http {
    /// A client that names itself `user_agent`, such as `MyApp/1.0 (+https://example.org)`, so
    /// servers' administrators know who is asking.
    pub fn new(limits: Limits, user_agent: &str) -> Result<Self> {
        let allow_private = limits.allow_private;
        let mut builder = reqwest::Client::builder()
            .user_agent(user_agent)
            .connect_timeout(Duration::from_secs(15))
            .timeout(limits.timeout)
            .redirect(reqwest::redirect::Policy::custom(move |attempt| {
                if attempt.previous().len() >= 5 {
                    attempt.error("too many redirects")
                } else if !allow_private && literal_private(attempt.url()) {
                    attempt.error("redirected to a private address")
                } else {
                    attempt.follow()
                }
            }));
        if !allow_private {
            builder = builder.dns_resolver(Arc::new(PublicOnly));
        }
        Ok(Self(Arc::new(Inner { client: builder.build()?, limits, hosts: Mutex::new(HashMap::new()) })))
    }

    fn slot(&self, host: &str) -> Arc<Semaphore> {
        self.0
            .hosts
            .lock()
            .unwrap()
            .entry(host.to_owned())
            .or_insert_with(|| Arc::new(Semaphore::new(self.0.limits.per_host)))
            .clone()
    }

    /// GETs `url`, keeping at most `cap` bytes of the body.
    pub async fn get(&self, url: &str, cap: usize) -> Result<Fetched> {
        let parsed = url::Url::parse(url).map_err(|e| anyhow!("{url} is not a URL: {e}"))?;
        ensure!(matches!(parsed.scheme(), "http" | "https"), "only http and https URLs can be fetched");
        if !self.0.limits.allow_private && literal_private(&parsed) {
            bail!("{url} is a private address");
        }
        let host = parsed.host_str().unwrap_or_default().to_owned();
        let slot = self.slot(&host);
        let _permit = slot.acquire().await?;
        let response = self.0.client.get(parsed).send().await.map_err(|e| anyhow!("fetching {url} failed: {e}"))?;
        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let mut stream = response.bytes_stream();
        let mut body = Vec::new();
        let mut truncated = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow!("reading {url} failed: {e}"))?;
            if body.len() + chunk.len() > cap {
                body.extend_from_slice(&chunk[..cap - body.len()]);
                truncated = true;
                break;
            }
            body.extend_from_slice(&chunk);
        }
        Ok(Fetched { status, content_type, url: final_url, body, truncated })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_addresses_are_recognised() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "192.168.0.1",
            "172.16.5.4",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "fd00::1",
            "fe80::1",
            "::ffff:10.0.0.1",
            "0.0.0.0",
        ] {
            assert!(private(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["8.8.8.8", "52.1.2.3", "2606:4700::1111"] {
            assert!(!private(ip.parse().unwrap()), "{ip}");
        }
    }

    #[tokio::test]
    async fn private_urls_are_refused() {
        let http = Http::new(Limits::default(), "erisharness-test").unwrap();
        for url in ["http://127.0.0.1:9/", "http://localhost/", "http://[::1]/", "file:///etc/passwd"] {
            assert!(http.get(url, 100).await.is_err(), "{url}");
        }
    }
}
