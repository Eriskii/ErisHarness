# ErisHarness

A Rust agent harness for running thousands of agents on one Linux machine. An agent is its
transcript on disk. A live agent costs under a megabyte; an idle one costs a database row.

Each agent runs on a machine chosen per agent:

- **Sandbox**: an ErisSandbox sandbox, with a private copy-on-write filesystem, process tree, loopback network, hostname
  and cgroup. Inside it the agent is root and can install packages and run any program;
  nothing of the host is visible except kernel and hardware facts.
- **Direct**: this machine, as this user, in a working directory, with the harness's
  environment or one given. No isolation and no setup: a harness with only direct agents
  needs no `bootstrap`, subordinate ids or cgroups.

Tools see only the `Machine` trait, and the tool tests run on both machines, so what the
model sees is identical either way.

## Measured

`cargo run --release --example scale -- <rootfs> 50000 3000 2000` on a 32 GB desktop:

| What | Cost |
| --- | --- |
| Idle agent (record + transcript) | 0.05 KiB of harness memory |
| Live sandbox holding one background process | 442 KiB PSS, 741 KiB cgroup charge (includes kernel memory), 4 KiB in the harness |
| Turn in flight (model call + one bash command) | 19 KiB in the harness, 646 KiB cgroup charge |

Costs are flat from 1,000 to 3,000 live sandboxes and 500 to 2,000 concurrent turns.

## Model

```
inbox (SQLite)  ──wake──▶  turn task  ──▶ provider ──▶ items ──▶ transcript.jsonl
     ▲                         │
     └── send_message ◀────────┴── tools ──▶ machine: sandbox init (while live) or direct
```

- **Agents** are rows in `harness.db` plus `transcripts/<id>/transcript.jsonl`. The
  transcript is the whole context. Only finished items are written; streaming text goes to
  observers only. Each agent's spec names its provider, model, reasoning effort, tools,
  prompt and machine, plus `metadata` the host keeps with it and the harness never reads.
  `agents()` lists them; `update_agent` changes it for the next turn, and `remove_agent` stops
  the agent and deletes everything it had.
- **Mail** is the only way to make an agent act. `Harness::send` from `"user"` or another
  agent's id queues a message. An idle agent starts a turn. A busy agent receives it at its
  next tool boundary. After `interrupt`, mail is held until the user writes again. The
  `send_message` tool takes the same path.
- **Waiting for replies.** `send_message` with `wait` blocks until the recipient's next
  message to the sender, which becomes the call's result instead of arriving as mail. An
  optional `timeout_seconds` ends the wait early, and an interrupt ends it too; the reply
  then arrives as mail. The claimed message is marked delivered only after the result is in
  the transcript, and the result records its event id, so a crash neither loses nor repeats it.
- **Recipients** outside the harness, such as the user, are registered with
  `HarnessBuilder::recipient(name, recipient)`. Mail to that name goes to the `Recipient`;
  replies come back through `Harness::send` with the name as the sender, and end waits the
  same way.
- **Turns** load the transcript, call the provider, run tool calls, and repeat until the
  model answers with no calls. Then everything is dropped. A restarted harness resumes agents
  that were mid-turn: calls without results get an interruption result, and delivery is
  idempotent via event ids in the transcript.
- **Compaction.** With `context_window` set, a request that would start above 80% of it is
  preceded by one that asks the model to summarize the conversation. The summary is appended
  as a `Compaction` item; from then on the model sees the summary and what follows it. The
  transcript keeps everything. `AgentRecord::context_tokens` is the latest request's input
  size.
- **Providers** implement `Provider`. `Responses` speaks the OpenAI Responses API (streaming,
  `store: false`, encrypted reasoning carried forward, the agent id as `prompt_cache_key`)
  with a configurable endpoint and headers. `Credentials::authorize` supplies the token and
  any per-account headers for each request, so refreshable logins stay outside the harness.
  It retries 429s and 5xx, honoring `retry-after-ms` and `retry-after`.
- **Tools** implement `Tool`. `tools::builtin()` is Pi 0.87's `read`, `bash`, `edit`, `write`,
  with Pi's exact model-facing text, plus `send_message`.

## Direct machines

`MachineSpec::Direct(DirectSpec { cwd, env })` runs each command as `bash -c` (found through
`PATH`) in `cwd`, in its own process group so an interrupt kills pipelines and children.
`env: None` inherits the harness's environment. Files open with the harness user's
permissions. Children are reaped through pidfds, so running commands cost no threads.

When the harness also has sandboxes, the program runs inside ErisSandbox's supervisor
namespace, where the user appears as root. Direct commands then start through
`Host::spawn_outside`, so they run as the user with the host's view of users and files, just
as in a harness without sandboxes.

## Sandboxes

Sandboxed agents run in [ErisSandbox](https://github.com/Eriskii/ErisSandbox) sandboxes. Enable
them with `HarnessBuilder::sandboxes(&host)`, where `host` comes from `erissandbox::bootstrap()`,
the first call in `main`. Each agent's sandbox has the agent's id, and its filesystem lives
under `HarnessBuilder::sandbox_dir` (by default `<dir>/sandboxes`). It is live only while the
agent runs commands and hibernates after the harness's `idle_grace`. The harness implements
`Machine` for `erissandbox::Sandbox`.

## Requirements

Direct agents need only Linux and `bash`. Sandboxed agents need what
[ErisSandbox](https://github.com/Eriskii/ErisSandbox#requirements) needs.

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test          # needs Docker once, to export debian:bookworm-slim as the test image
```

The sandbox isolation tests live in ErisSandbox.

