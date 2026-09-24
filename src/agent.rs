//! What an agent is: a spec, a state, and a transcript of items. The transcript is the
//! agent's entire context; everything else can be rebuilt from it.

use crate::sandbox::SandboxSpec;
use crate::tools::ToolOutput;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentSpec {
    pub system_prompt: String,
    /// Names of registered tools this agent may call.
    pub tools: Vec<String>,
    /// Name of a registered provider.
    pub provider: String,
    pub sandbox: SandboxSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    /// Nothing to do, or mail held after an interrupt.
    Idle,
    /// A turn is in progress: waiting on the model or running tools.
    Running,
    /// The last turn ended in a provider error. The next message retries.
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
    /// A message delivered to the agent: from `"user"` or from another agent's id.
    Input {
        from: String,
        text: String,
    },
    Assistant {
        text: String,
    },
    /// Model reasoning. `encrypted` is opaque provider state carried into later requests.
    Reasoning {
        summary: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted: Option<String>,
    },
    ToolCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        call_id: String,
        output: ToolOutput,
    },
}

/// A transcript line: one JSON object per line of `transcript.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub seq: u64,
    /// Milliseconds since the Unix epoch.
    pub at: u64,
    /// The inbox event this entry delivered, which makes delivery idempotent across crashes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<i64>,
    #[serde(flatten)]
    pub item: Item,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub cached_input: u64,
    pub output: u64,
    pub reasoning: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentRecord {
    pub id: String,
    pub spec: AgentSpec,
    pub state: AgentState,
    pub error: Option<String>,
    /// Interrupted: mail waits until the user writes again.
    pub held: bool,
    pub usage: Usage,
}

/// Live events for observers. Text deltas are never persisted; items are.
#[derive(Clone, Debug, PartialEq)]
pub enum Observation {
    State { agent: String, state: AgentState },
    Item { agent: String, entry: Entry },
    TextDelta { agent: String, text: String },
}
