# ErisHarness

A Rust library for running thousands of LLM agents on one Linux machine. An agent is a row in
SQLite plus a transcript file. It takes memory only while it works, and works on a machine of
its own: an isolated [ErisSandbox](https://github.com/Eriskii/ErisSandbox) sandbox, or this
computer directly.

```rust,no_run
use erisharness::agent::{AgentSpec, AgentState};
use erisharness::machine::{DirectSpec, MachineSpec};
use erisharness::provider::{RateGate, Responses, ResponsesConfig, StaticToken};
use erisharness::{Harness, tools};
use std::sync::Arc;

async fn fix_the_tests() -> anyhow::Result<()> {
    let openai = Arc::new(Responses::new(ResponsesConfig {
        base_url: "https://api.openai.com/v1".into(),
        headers: Vec::new(),
        store: false,
        max_retries: 3,
        credentials: Arc::new(StaticToken(std::env::var("OPENAI_API_KEY")?)),
        gate: RateGate::new(16, 256),
    }));
    let harness = Harness::builder("state").provider("openai", openai).tools(tools::builtin()).open().await?;
    let agent = harness.create_agent(AgentSpec {
        system_prompt: "You are a careful engineer.".into(),
        tools: vec!["read".into(), "bash".into(), "edit".into(), "write".into()],
        provider: "openai".into(),
        model: "gpt-5".into(),
        reasoning_effort: Some("medium".into()),
        context_window: Some(400_000),
        metadata: serde_json::Value::Null,
        machine: MachineSpec::Direct(DirectSpec { cwd: "/home/me/project".into(), env: None }),
    })?;
    harness.send(&agent, "user", "Make the tests pass.")?;
    harness.wait_for(&agent, AgentState::Idle).await;
    Ok(())
}
```

## How it works

```text
Harness::send ──▶ inbox (SQLite) ──wake──▶ turn ──▶ provider ──▶ items ──▶ transcript.jsonl
                        ▲                    │
                        └── send_message ◀───┴──▶ tools ──▶ machine (sandbox or direct)
```

**Agents.** `create_agent` stores an `AgentSpec`: provider, model, reasoning effort, tools,
system prompt, machine, and `metadata` the host keeps with the agent and the harness never
reads. The transcript at `transcripts/<id>/transcript.jsonl` holds every finished item, one
JSON line each. `agents` lists every agent, `update_agent` changes a spec for the next turn,
and `remove_agent` stops an agent and deletes its record, mail, transcript and sandbox. An
`AgentRecord` shows its state (idle, running, or failed with the error), how full its context
is, and its token usage: input, input read from and written to the prompt cache, output, and
reasoning.

**Mail** is the only way to make an agent act. `Harness::send(agent, from, text)` queues a
message from `"user"` or another agent's id. An idle agent starts a turn; a busy one reads the
message at its next tool boundary. `interrupt` stops the current turn and holds mail from
agents until the user writes again. Agents write to each other with the `send_message` tool,
which takes the same path.

**Waiting for a reply.** `send_message` with `wait` blocks until the recipient's next message
to the sender, which becomes the call's result instead of arriving as mail. `timeout_seconds`
or an interrupt ends the wait early, and the reply then arrives as mail. A reply is marked
delivered only once its result is in the transcript, so a crash neither loses nor repeats it.

**Recipients** are names outside the harness, such as the user, registered with
`HarnessBuilder::recipient`. Mail to that name goes to the `Recipient`. Its replies come back
through `Harness::send` with the name as sender, and end waits the same way.

**Turns** load the transcript, call the provider, run the tool calls, and repeat until the
model answers without calls. Then everything is dropped. `max_concurrent_turns` caps how many
run at once. A restarted harness resumes agents that were mid-turn: tool calls left without
results get an interruption result, and mail already in the transcript is not delivered again.

**Compaction.** With `context_window` set, a turn whose context has passed 80% of it first
asks the model to summarize the conversation. The request offers the usual tools, so the
conversation is read from the prompt cache. From then on the model sees the summary and what
follows it; the transcript keeps everything. A reply without a summary leaves the context
whole.

**Observers** `subscribe` to live events: state changes, finished items, streamed text, each
model call's token usage, calls held by rate limits, and provider diagnostics. Streamed text
is never stored.

## Machines

Tools see only the `Machine` trait, and the tool tests run on both machines, so the model
sees the same behavior either way.

**Direct** (`MachineSpec::Direct`) runs commands on this computer as this user, in `cwd`, with
the harness's environment or the one given. Each command is `bash -c` (found through `PATH`)
leading its own process group, so an interrupt kills pipelines and children. Direct agents
need no setup. When the harness also runs sandboxes, the program lives in ErisSandbox's
namespace, so direct commands start outside it through `Host::spawn_outside` and still see
the machine as the user does.

**Sandbox** (`MachineSpec::Sandbox`) runs commands in an ErisSandbox sandbox with a private
copy-on-write filesystem, process tree, loopback network, hostname and cgroup. Inside it the
agent is root and can install and run anything. Call `erissandbox::bootstrap()` first in
`main`, and pass its host to `HarnessBuilder::sandboxes`. Each sandbox is named for its agent
and keeps its filesystem under `sandbox_dir` (by default `<dir>/sandboxes`). It runs only
while the agent runs commands, and hibernates after `idle_grace`. To let agents read each
other's transcripts, bind the `transcripts` directory into their sandboxes.

## Providers

A `Provider` turns a request (model, system prompt, tools, visible items) into the model's
next items. The runtime knows no wire format. Two providers are built in, and they share:

- `Credentials`, asked for a token before every request, so the host owns login and refresh.
- A `RateGate` per account, shared by every provider using it. Calls wait for one of its
  slots. A rate limit halves the slots and pauses every call until the provider's retry time;
  each success adds back a fraction of a slot.
- Retries of rate limits, server errors and dropped connections, honoring `retry-after-ms`
  and `retry-after`, up to `max_retries`.
- Diagnostics, as `ProviderEvent`s: each response's status with only its rate-limit, retry and
  request-id headers, and each retry.
- The same text for mail from other agents (`[Message from agent <id>]`) and for summaries.

**Responses** speaks the OpenAI Responses API, streamed, at any endpoint with extra headers.
It sends the whole visible transcript with `store: false`, carries encrypted reasoning
forward, and uses the agent id as `prompt_cache_key`.

**Anthropic** speaks the Messages API. `AnthropicAuth::ApiKey` sends the token as
`x-api-key`. `AnthropicAuth::ClaudeCode` uses a Claude subscription's OAuth token and makes
each request as Claude Code does. `Thinking` is set per provider: `Disabled`,
`Adaptive { display }`, or `Budget { tokens, display }`. Signed and redacted thinking is kept
as opaque state in reasoning items and sent back only to the same model. A transport failure
is retried only before any content streams, so streamed output is never repeated.

```rust,no_run
use erisharness::provider::{Anthropic, AnthropicAuth, AnthropicConfig, ClaudeCode, StaticToken, Thinking};
use std::sync::Arc;

fn claude() -> anyhow::Result<Arc<Anthropic>> {
    let token = Arc::new(StaticToken(std::env::var("CLAUDE_CODE_OAUTH_TOKEN")?));
    let identity = ClaudeCode { install_id: "a-stable-installation-id".into(), account_id: None, version: None };
    let mut config = AnthropicConfig::new(token, AnthropicAuth::ClaudeCode(identity));
    config.thinking = Thinking::Adaptive { display: false };
    Ok(Arc::new(Anthropic::new(config)?))
}
```

A subscription request follows oh-my-pi at commit
[`25097b1`](https://github.com/can1357/oh-my-pi/tree/25097b1be3d9b06dc38ca5e20fc7058ad5fd65f4):
Claude Code's identity and billing system blocks, a SHA-256 fingerprint of the first user
message, an XXHash64 checksum of the exact body bytes, the CLI's headers and beta flags,
stable device and session metadata, prefixed tool names, one-hour cache breakpoints, and a
64,000-token output ceiling. It claims version 2.1.280 unless `ClaudeCode::version` or
`PI_AI_CLAUDE_CODE_VERSION` names one. Without a pinned version, it adopts a newer one the
server requires and retries once, outside the retry allowance.

## Tools

`tools::builtin()` is Pi 0.87's `read`, `bash`, `edit` and `write`, with Pi's exact
model-facing text, limits and error wording, plus `send_message`. Any `Tool` can be
registered. A tool call's arguments must be a JSON object (`{}` when there are none).
Anything else, such as JSON cut off mid-stream, stays verbatim in the transcript and gets an
error result showing the raw input, so the model can send the call again.

## Cost

`cargo run --release --example scale -- <rootfs> 50000 3000 2000` on a 32 GB desktop:

| What | Cost |
| --- | --- |
| Idle agent (record and transcript) | 0.05 KiB of harness memory |
| Live sandbox with one background process | 442 KiB PSS, 741 KiB cgroup charge (with kernel memory), 4 KiB in the harness |
| Turn in flight (model call and one bash command) | 19 KiB in the harness, 646 KiB cgroup charge |

Costs stay flat from 1,000 to 3,000 live sandboxes and from 500 to 2,000 concurrent turns.

## Requirements

Direct agents need Linux and `bash`. Sandboxed agents need what
[ErisSandbox needs](https://github.com/Eriskii/ErisSandbox#requirements).

## Testing

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The sandbox tests (`agent`, `tools`) export `debian:bookworm-slim` from Docker once. The rest
need no Docker, account or subscription:

```sh
cargo test --lib --test anthropic --test direct --test lifecycle --test mail
cargo test --doc
```

Sandbox isolation itself is tested in ErisSandbox.
