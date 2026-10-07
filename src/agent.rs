//! What an agent is: a spec saying how it runs, a state, and a transcript of items. The
//! transcript is the agent's whole context.

use crate::machine::MachineSpec;
use crate::tools::ToolOutput;
use serde::{Deserialize, Serialize};

/// How an agent runs, stored with it. [`Harness::update_agent`](crate::Harness::update_agent)
/// changes it for the next turn.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentSpec {
    /// The agent's own instructions. The harness appends its id and the tools' prompt lines.
    pub system_prompt: String,
    /// Names of registered tools this agent may call.
    pub tools: Vec<String>,
    /// Name of a registered provider.
    pub provider: String,
    /// Passed to the provider as is.
    pub model: String,
    pub reasoning_effort: Option<String>,
    /// The model's context size in tokens. When a request would start above 80% of it, the
    /// transcript so far is compacted into a summary first.
    pub context_window: Option<u64>,
    /// Whatever the host keeps with the agent, such as its name. The harness never reads it.
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub machine: MachineSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    /// Nothing to do, or mail held after an interrupt.
    Idle,
    /// A turn is in progress: waiting on the model or running tools.
    Running,
    /// The last turn ended in an error, kept in [`AgentRecord::error`]. The next message
    /// tries again.
    Failed,
}

impl AgentState {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            AgentState::Idle => "idle",
            AgentState::Running => "running",
            AgentState::Failed => "failed",
        }
    }

    pub(crate) fn parse(text: &str) -> Self {
        match text {
            "running" => AgentState::Running,
            "failed" => AgentState::Failed,
            _ => AgentState::Idle,
        }
    }
}

/// One step of an agent's history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    /// A message delivered to the agent: from `"user"` or from another agent's id, with any
    /// images sent along.
    Input {
        from: String,
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<Image>,
    },
    Assistant {
        text: String,
    },
    /// Model reasoning. `encrypted` is opaque provider state sent back in later requests.
    Reasoning {
        summary: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted: Option<String>,
    },
    ToolCall {
        call_id: String,
        name: String,
        /// Exactly what the model sent, which may not be valid JSON. The harness runs the tool
        /// only for a JSON object, and otherwise answers with an error result.
        arguments: String,
    },
    ToolResult {
        call_id: String,
        output: ToolOutput,
    },
    /// Everything before this item, summarized. The model sees only the summary and what
    /// follows it.
    Compaction {
        summary: String,
    },
}

/// An image sent with mail: base64 data and its media type, such as `image/png`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub mime: String,
    pub data: String,
}

/// A transcript line: one JSON object per line of `transcript.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub seq: u64,
    /// Milliseconds since the Unix epoch.
    pub at: u64,
    /// The inbox mail this entry delivered. A restarted harness finds it here instead of
    /// delivering it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<i64>,
    #[serde(flatten)]
    pub item: Item,
}

/// Tokens a model call used, or an agent's calls together. The share of input served from
/// the prompt cache is `cached_input / input`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Every prompt token: fresh, read from the cache, and written to it.
    pub input: u64,
    /// Prompt tokens read from the cache.
    pub cached_input: u64,
    /// Prompt tokens written to the cache, which Anthropic bills above plain input. Providers
    /// that cache on their own, such as OpenAI's, report none.
    #[serde(default)]
    pub cache_write: u64,
    pub output: u64,
    /// Output tokens spent reasoning, when the provider reports them.
    pub reasoning: u64,
}

/// An agent as stored: its spec, where it stands, and what it has used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentRecord {
    pub id: String,
    pub spec: AgentSpec,
    pub state: AgentState,
    pub error: Option<String>,
    /// Interrupted: mail from agents waits until the user writes again.
    pub held: bool,
    pub usage: Usage,
    /// Input tokens of the latest model request: how full the context is.
    pub context_tokens: u64,
}

/// Live events for observers. Text deltas are never persisted; items are.
#[derive(Clone, Debug, PartialEq)]
pub enum Observation {
    /// Transport diagnostics from the agent's provider.
    Provider {
        agent: String,
        event: crate::provider::ProviderEvent,
    },
    State {
        agent: String,
        state: AgentState,
    },
    Item {
        agent: String,
        entry: Entry,
    },
    TextDelta {
        agent: String,
        text: String,
    },
    /// What one model call used, a compaction's included.
    Usage {
        agent: String,
        usage: Usage,
    },
    /// The agent's model call is waiting, queued for a slot or rate-limited; `None` once it
    /// stops waiting, however the call then ends.
    Held {
        agent: String,
        hold: Option<crate::provider::Hold>,
    },
    /// Mail was sent: queued for an agent, or delivered to a recipient. `from` is `"user"`,
    /// another sender the host names, or an agent's id; `to` is an agent's id or a recipient.
    Mail {
        from: String,
        to: String,
    },
}
