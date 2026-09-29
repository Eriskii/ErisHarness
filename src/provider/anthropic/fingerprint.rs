//! Claude Code request identity, following oh-my-pi at
//! 25097b1be3d9b06dc38ca5e20fc7058ad5fd65f4. Keep versioned wire details here.

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const DEFAULT_VERSION: &str = "2.1.280";
pub const IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
pub const CCH_SEED: u64 = 0x4d659218e32a3268;
const PLACEHOLDER: &str = "cch=00000";

pub fn billing(first_user: &str, version: &str) -> String {
    // JavaScript indexes UTF-16 code units, including lone surrogate replacement on UTF-8 encoding.
    let units: Vec<u16> = first_user.encode_utf16().collect();
    let selected: Vec<u16> = [4, 7, 20].iter().map(|&i| units.get(i).copied().unwrap_or(b'0' as u16)).collect();
    let digest = Sha256::digest(format!("59cf53e54c78{}{version}", String::from_utf16_lossy(&selected)));
    let suffix = format!("{digest:x}");
    format!("x-anthropic-billing-header: cc_version={version}.{}; cc_entrypoint=cli; {PLACEHOLDER};", &suffix[..3])
}

pub fn device_id(install_id: &str, account: Option<&str>) -> String {
    let input = match account.filter(|s| !s.is_empty()) {
        Some(account) => format!("omp-claude-device-id-v2\0{install_id}\0{account}"),
        None => format!("omp-claude-device-id-v1:{install_id}"),
    };
    format!("{:x}", Sha256::digest(input))
}

pub fn metadata(install_id: &str, account: Option<&str>, session: &str) -> Value {
    let mut id = json!({"device_id": device_id(install_id, account), "session_id": session});
    if let Some(account) = account.filter(|s| !s.is_empty()) {
        id["account_uuid"] = json!(account);
    }
    json!({"user_id": id.to_string()})
}

pub fn tool_name(name: &str) -> String {
    if matches!(name.to_ascii_lowercase().as_str(), "web_search" | "code_execution" | "text_editor" | "computer") {
        name.to_owned()
    } else {
        format!("_{name}")
    }
}

pub fn decode_tool(name: &str) -> &str {
    name.strip_prefix('_').unwrap_or(name)
}

pub fn checksum(body: &[u8]) -> String {
    format!("{:05x}", xxhash_rust::xxh64::xxh64(body, CCH_SEED) & 0xfffff)
}

/// Serialize once and patch only our first system block. Never replace matching user text.
pub fn serialize(body: &Value, oauth: bool) -> Result<Vec<u8>> {
    let mut encoded = serde_json::to_vec(body)?;
    if oauth {
        let block = serde_json::to_string(&body["system"][0])?;
        let marker = format!("\"system\":[{block}");
        let text = std::str::from_utf8(&encoded)?;
        let start = text.find(&marker).context("missing Claude Code billing system block")?;
        let offset = marker.find(PLACEHOLDER).context("missing Claude Code checksum placeholder")?;
        let at = start + offset + 4;
        let hash = checksum(&encoded);
        encoded[at..at + 5].copy_from_slice(hash.as_bytes());
    }
    Ok(encoded)
}

pub fn headers(version: &str, session: &str, agent: bool, thinking: bool) -> Vec<(String, String)> {
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    };
    let os = match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "MacOS",
        "windows" => "Windows",
        other => other,
    };
    let mut betas = Vec::new();
    if agent {
        betas.push("claude-code-20250219");
    }
    betas.extend([
        "oauth-2025-04-20",
        "interleaved-thinking-2025-05-14",
        "thinking-token-count-2026-05-13",
        "context-management-2025-06-27",
        "prompt-caching-scope-2026-01-05",
    ]);
    if agent {
        betas.push("mid-conversation-system-2026-04-07");
        if thinking {
            betas.push("effort-2025-11-24");
        }
        betas.push("fallback-credit-2026-06-01");
    } else {
        betas.push("structured-outputs-2025-12-15");
    }
    let mut headers: Vec<(String, String)> = [
        ("Accept", "application/json"),
        ("Content-Type", "application/json"),
        ("X-Stainless-Arch", arch),
        ("X-Stainless-Lang", "js"),
        ("X-Stainless-OS", os),
        ("X-Stainless-Package-Version", "0.112.1"),
        ("X-Stainless-Retry-Count", "0"),
        ("X-Stainless-Runtime", "node"),
        ("X-Stainless-Runtime-Version", "v26.3.0"),
        ("X-Stainless-Timeout", "600"),
        ("anthropic-dangerous-direct-browser-access", "true"),
        ("anthropic-version", "2023-06-01"),
        ("x-app", "cli"),
        ("Connection", "keep-alive"),
        ("Accept-Encoding", "gzip, deflate, br, zstd"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect();
    headers.extend([
        ("User-Agent".into(), format!("claude-cli/{version} (external, cli)")),
        ("X-Claude-Code-Session-Id".into(), session.into()),
        ("anthropic-beta".into(), betas.join(",")),
    ]);
    headers
}

pub fn validate_version(version: &str) -> Result<()> {
    ensure!(semver(version).is_some(), "Claude Code version must be major.minor.patch");
    Ok(())
}

fn semver(version: &str) -> Option<[u64; 3]> {
    let v: Vec<_> = version.split('.').map(str::parse::<u64>).collect::<Result<_, _>>().ok()?;
    v.try_into().ok()
}

pub fn required_version(message: &str, current: &str) -> Option<String> {
    if !message.contains("claude_code_version_too_old") {
        return None;
    }
    let lower = message.to_ascii_lowercase();
    lower.split("version ").skip(1).find_map(|tail| {
        let (version, suffix) = tail.split_once(' ')?;
        (suffix.starts_with("or newer is required") && semver(version)? > semver(current)?).then(|| version.to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_xxhash_vectors() {
        for (body, expected) in [
            ("cch=00000", "a47f7"),
            ("{\"messages\":[],\"cch=00000\",\"x\":1}", "3073d"),
            ("x-anthropic-billing-header: cc_version=2.1.158; cc_entrypoint=cli; cch=00000;", "f2b0b"),
        ] {
            assert_eq!(checksum(body.as_bytes()), expected);
        }
    }
    #[test]
    fn patches_only_billing_and_hashes_final_bytes() {
        let user = "cch=00000 😀";
        let body = json!({"messages":[{"role":"user","content":user}], "system":[{"type":"text","text":billing(user, DEFAULT_VERSION)}]});
        let bytes = serialize(&body, true).unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["messages"][0]["content"], user);
        let original = serde_json::to_vec(&body).unwrap();
        assert!(value["system"][0]["text"].as_str().unwrap().contains(&format!("cch={}", checksum(&original))));
    }
    #[test]
    fn version_retry_only_moves_forward() {
        assert_eq!(
            required_version("claude_code_version_too_old: version 2.2.0 or newer is required", "2.1.280"),
            Some("2.2.0".into())
        );
        assert_eq!(required_version("claude_code_version_too_old: version 2.2.0 or newer is required", "2.2.0"), None);
    }
}
