#![allow(dead_code, unused_macros)]

pub mod model;

/// `(name, function)` for each test function, for [`run`].
macro_rules! named {
    ($($test:ident),* $(,)?) => { &[$((stringify!($test), $test)),*] };
}

use erisharness::agent::{AgentSpec, AgentState, Item};
use erisharness::machine::{DirectSpec, MachineSpec};
use erisharness::tools::{self, Content};
use erisharness::{Harness, HarnessBuilder};
use model::Model;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Sandboxes, and the fixture their tests share, need Linux.
#[cfg(target_os = "linux")]
mod sandbox;
#[cfg(target_os = "linux")]
pub use sandbox::*;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

pub fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap())
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    runtime().block_on(future)
}

/// Opens `builder` with the built-in tools and one provider per named model.
pub fn open(builder: HarnessBuilder, models: &[(&str, &Model)]) -> Arc<Harness> {
    let builder = models.iter().fold(builder, |b, (name, model)| b.provider(name, model.provider()));
    block_on(builder.tools(tools::builtin()).open()).expect("open harness")
}

/// A harness in `dir` whose only provider, `test`, is `model`.
pub fn harness(dir: &Path, model: &Model) -> Arc<Harness> {
    open(Harness::builder(dir), &[("test", model)])
}

/// An agent working directly in `cwd` with every built-in tool and the `test` provider.
pub fn direct(cwd: &Path) -> AgentSpec {
    AgentSpec {
        system_prompt: "You are a test agent.".into(),
        tools: tools::builtin().iter().map(|t| t.name().to_owned()).collect(),
        provider: "test".into(),
        model: "test-model".into(),
        reasoning_effort: None,
        context_window: None,
        metadata: serde_json::Value::Null,
        machine: MachineSpec::Direct(DirectSpec { cwd: cwd.to_str().unwrap().into(), env: None }),
    }
}

/// Waits until `agent` is in `state`, panicking after 20 seconds.
pub fn settle(h: &Harness, agent: &str, state: AgentState) {
    block_on(async { tokio::time::timeout(Duration::from_secs(20), h.wait_for(agent, state)).await })
        .unwrap_or_else(|_| panic!("agent never became {state:?}: {:?}", h.agent(agent).map(|a| a.state)));
}

pub fn items(h: &Harness, agent: &str) -> Vec<Item> {
    h.transcript(agent).unwrap().into_iter().map(|e| e.item).collect()
}

pub fn text(content: &[Content]) -> String {
    content.iter().map(|c| if let Content::Text(t) = c { t.as_str() } else { "" }).collect()
}

/// Every tool result: whether it is an error, and its text.
pub fn results(h: &Harness, agent: &str) -> Vec<(bool, String)> {
    items(h, agent)
        .into_iter()
        .filter_map(|item| match item {
            Item::ToolResult { output, .. } => Some((output.is_error, text(&output.content))),
            _ => None,
        })
        .collect()
}

/// Every message delivered: its sender and text.
pub fn inputs(h: &Harness, agent: &str) -> Vec<(String, String)> {
    items(h, agent)
        .into_iter()
        .filter_map(|item| match item {
            Item::Input { from, text, .. } => Some((from, text)),
            _ => None,
        })
        .collect()
}
