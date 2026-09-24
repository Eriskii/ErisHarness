//! The runtime. An idle agent costs a database row and a transcript file; nothing of it is
//! in memory. Mail wakes an agent: a task loads its transcript, runs model calls and tools
//! until there is nothing left to answer, then drops everything and exits.

use crate::Host;
use crate::agent::{AgentRecord, AgentSpec, AgentState, Entry, Item, Observation};
use crate::provider::{Provider, Request, ToolSpec};
use crate::sandbox::Sandboxes;
use crate::store::{Store, Transcript};
use crate::tools::{Mailbox, Tool, ToolContext, ToolOutput};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Semaphore, broadcast};
use tokio_util::sync::CancellationToken;

pub struct HarnessBuilder {
    host: Host,
    dir: PathBuf,
    transcripts: Option<PathBuf>,
    providers: HashMap<String, Arc<dyn Provider>>,
    tools: HashMap<String, Arc<dyn Tool>>,
    max_turns: usize,
    idle_grace: Duration,
}

impl HarnessBuilder {
    /// Where transcripts live, one `<agent>/transcript.jsonl` each. Defaults to
    /// `<dir>/transcripts`. Bind this into sandboxes to let agents read each other.
    pub fn transcripts(mut self, dir: impl Into<PathBuf>) -> Self {
        self.transcripts = Some(dir.into());
        self
    }

    pub fn provider(mut self, name: &str, provider: Arc<dyn Provider>) -> Self {
        self.providers.insert(name.to_owned(), provider);
        self
    }

    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(tool.name().to_owned(), tool);
        self
    }

    pub fn tools(self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        tools.into_iter().fold(self, Self::tool)
    }

    /// Turns that may run at once; further woken agents wait for a slot.
    pub fn max_concurrent_turns(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }

    /// How long an agent's sandbox stays live after its last command.
    pub fn idle_grace(mut self, grace: Duration) -> Self {
        self.idle_grace = grace;
        self
    }

    /// Opens the store and resumes every agent that was mid-turn or had unheld mail.
    pub async fn open(self) -> Result<Arc<Harness>> {
        let transcripts = self.transcripts.unwrap_or_else(|| self.dir.join("transcripts"));
        let store = Store::open(&self.dir.join("harness.db"), &transcripts)?;
        let sandboxes = Sandboxes::new(&self.host, self.dir.join("sandboxes"))?.idle_grace(self.idle_grace);
        let (observations, _) = broadcast::channel(4096);
        let harness = Arc::new_cyclic(|this| Harness {
            mailbox: Arc::new(Inboxes(this.clone())),
            store,
            sandboxes,
            providers: self.providers,
            tools: self.tools,
            turns: Arc::new(Semaphore::new(self.max_turns)),
            running: Mutex::default(),
            observations,
            closing: CancellationToken::new(),
            runtime: tokio::runtime::Handle::current(),
        });
        for agent in harness.store.unfinished()? {
            harness.wake(&agent);
        }
        Ok(harness)
    }
}

struct Run {
    cancel: CancellationToken,
    task: Option<tokio::task::JoinHandle<()>>,
}

pub struct Harness {
    store: Store,
    sandboxes: Sandboxes,
    providers: HashMap<String, Arc<dyn Provider>>,
    tools: HashMap<String, Arc<dyn Tool>>,
    turns: Arc<Semaphore>,
    running: Mutex<HashMap<String, Run>>,
    observations: broadcast::Sender<Observation>,
    closing: CancellationToken,
    /// Captured at open so agents can be woken from any thread.
    runtime: tokio::runtime::Handle,
    mailbox: Arc<dyn Mailbox>,
}

/// Tools send mail through the same path as [`Harness::send`], so queueing is identical.
struct Inboxes(std::sync::Weak<Harness>);

impl Mailbox for Inboxes {
    fn send(&self, from: &str, to: &str, text: &str) -> Result<()> {
        let harness = self.0.upgrade().context("harness stopped")?;
        harness.send(to, from, text).map(drop)
    }
}

enum Ended {
    Idle,
    Interrupted,
    Closing,
}

const INTERRUPTED_CALL: &str = "Tool call was interrupted before it finished.";
const SKIPPED_CALL: &str = "Tool call skipped: the turn was interrupted.";

impl Harness {
    pub fn builder(host: &Host, dir: impl AsRef<Path>) -> HarnessBuilder {
        HarnessBuilder {
            host: host.clone(),
            dir: dir.as_ref().to_owned(),
            transcripts: None,
            providers: HashMap::new(),
            tools: HashMap::new(),
            max_turns: 1024,
            idle_grace: Duration::from_secs(10),
        }
    }

    pub fn create_agent(&self, spec: AgentSpec) -> Result<String> {
        if !self.providers.contains_key(&spec.provider) {
            bail!("unknown provider {}", spec.provider);
        }
        if let Some(missing) = spec.tools.iter().find(|t| !self.tools.contains_key(*t)) {
            bail!("unknown tool {missing}");
        }
        let id = uuid::Uuid::now_v7().simple().to_string();
        self.store.create_agent(&id, &spec)?;
        Ok(id)
    }

    pub fn agent(&self, id: &str) -> Result<AgentRecord> {
        self.store.agent(id)?.with_context(|| format!("No agent {id}"))
    }

    pub fn transcript(&self, id: &str) -> Result<Vec<Entry>> {
        Ok(self.store.transcript(id)?.entries)
    }

    pub fn transcript_path(&self, id: &str) -> PathBuf {
        self.store.transcript_path(id)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Observation> {
        self.observations.subscribe()
    }

    /// Sandboxes currently running processes.
    pub fn live_sandboxes(&self) -> usize {
        self.sandboxes.live_count()
    }

    /// Queues a message for `agent` from `"user"` or another agent's id. A user message
    /// releases mail held by an interrupt.
    pub fn send(self: &Arc<Self>, agent: &str, from: &str, text: &str) -> Result<i64> {
        let record = self.agent(agent)?;
        let id = self.store.enqueue(agent, from, text)?;
        if from == "user" && record.held {
            self.store.set_held(agent, false)?;
        }
        if from == "user" || !record.held {
            self.wake(agent);
        }
        Ok(id)
    }

    /// Stops the agent's current turn. Mail stays queued until the user writes again.
    pub fn interrupt(&self, agent: &str) {
        let _ = self.store.set_held(agent, true);
        if let Some(run) = self.running.lock().unwrap().get(agent) {
            run.cancel.cancel();
        }
    }

    pub async fn wait_for(&self, agent: &str, state: AgentState) {
        let mut observations = self.subscribe();
        loop {
            let running = self.running.lock().unwrap().contains_key(agent);
            let current = self.agent(agent).map(|a| a.state).ok();
            // Idle means settled: a woken agent may still show its previous state.
            if current == Some(state) && (state == AgentState::Running || !running) {
                return;
            }
            match tokio::time::timeout(Duration::from_millis(200), observations.recv()).await {
                Err(_) | Ok(Ok(_)) | Ok(Err(broadcast::error::RecvError::Lagged(_))) => {}
                Ok(Err(broadcast::error::RecvError::Closed)) => return,
            }
        }
    }

    /// Stops all turns without changing agent state, so a new harness resumes them, and
    /// stops every sandbox.
    pub async fn shutdown(&self) {
        self.closing.cancel();
        let tasks: Vec<_> = self.running.lock().unwrap().values_mut().filter_map(|r| r.task.take()).collect();
        for task in tasks {
            let _ = task.await;
        }
        self.sandboxes.shutdown_all().await;
    }

    fn observe(&self, observation: Observation) {
        let _ = self.observations.send(observation);
    }

    fn set_state(&self, agent: &str, state: AgentState, error: Option<&str>) {
        let _ = self.store.set_state(agent, state, error);
        self.observe(Observation::State { agent: agent.to_owned(), state });
    }

    fn wake(self: &Arc<Self>, agent: &str) {
        if self.closing.is_cancelled() {
            return;
        }
        let mut running = self.running.lock().unwrap();
        if running.contains_key(agent) {
            return;
        }
        let cancel = self.closing.child_token();
        let harness = self.clone();
        let id = agent.to_owned();
        let task = self.runtime.spawn({
            let cancel = cancel.clone();
            async move { harness.run(&id, cancel).await }
        });
        running.insert(agent.to_owned(), Run { cancel, task: Some(task) });
    }

    async fn run(self: Arc<Self>, agent: &str, cancel: CancellationToken) {
        let Ok(_slot) = self.turns.clone().acquire_owned().await else { return };
        loop {
            let ended = self.turn(agent, &cancel).await;
            if self.closing.is_cancelled() {
                return;
            }
            let mut running = self.running.lock().unwrap();
            match ended {
                // Mail that arrived after the last check is picked up here, under the lock
                // `wake` takes, so none is stranded.
                Ok(Ended::Idle) if self.store.pending(agent).is_ok_and(|p| !p.is_empty()) => continue,
                Ok(Ended::Idle | Ended::Interrupted) => self.set_state(agent, AgentState::Idle, None),
                Ok(Ended::Closing) => return,
                Err(error) => self.set_state(agent, AgentState::Failed, Some(&format!("{error:#}"))),
            }
            running.remove(agent);
            return;
        }
    }

    /// Delivers mail and runs model calls and tools until nothing awaits an answer.
    async fn turn(&self, agent: &str, cancel: &CancellationToken) -> Result<Ended> {
        let record = self.agent(agent)?;
        let spec = &record.spec;
        let provider =
            self.providers.get(&spec.provider).with_context(|| format!("unknown provider {}", spec.provider))?;
        let tools: Vec<Arc<dyn Tool>> = spec.tools.iter().filter_map(|name| self.tools.get(name).cloned()).collect();
        let tool_specs: Vec<ToolSpec> = tools
            .iter()
            .map(|t| ToolSpec { name: t.name().to_owned(), description: t.description(), parameters: t.parameters() })
            .collect();
        let system = system_prompt(&spec.system_prompt, agent, &tools);
        let mut transcript = self.store.transcript(agent)?;
        self.close_dangling_calls(agent, &mut transcript)?;
        let sandbox = self.sandboxes.sandbox(agent, spec.sandbox.clone())?;
        let mut announced = record.state == AgentState::Running;
        loop {
            if cancel.is_cancelled() {
                return Ok(if self.closing.is_cancelled() { Ended::Closing } else { Ended::Interrupted });
            }
            self.deliver(agent, &mut transcript)?;
            if !matches!(transcript.items().last(), Some(Item::Input { .. } | Item::ToolResult { .. })) {
                return Ok(Ended::Idle);
            }
            if !announced {
                self.set_state(agent, AgentState::Running, None);
                announced = true;
            }
            let on_text =
                |text: &str| self.observe(Observation::TextDelta { agent: agent.to_owned(), text: text.to_owned() });
            let items: Vec<Item> = transcript.items().cloned().collect();
            let request = Request { system: &system, tools: &tool_specs, items: &items };
            let completion = tokio::select! {
                completion = provider.complete(request, &on_text) => completion?,
                _ = cancel.cancelled() => continue,
            };
            drop(items);
            self.store.add_usage(agent, completion.usage)?;
            let mut calls = Vec::new();
            for item in completion.items {
                if let Item::ToolCall { call_id, name, arguments } = &item {
                    calls.push((call_id.clone(), name.clone(), arguments.clone()));
                }
                self.append(agent, &mut transcript, item, None)?;
            }
            let context = ToolContext {
                agent: agent.to_owned(),
                sandbox: sandbox.clone(),
                mailbox: self.mailbox.clone(),
                cancel: cancel.child_token(),
            };
            for (call_id, name, arguments) in calls {
                let output = if cancel.is_cancelled() {
                    ToolOutput::error(SKIPPED_CALL)
                } else {
                    call_tool(&tools, &context, &name, &arguments).await
                };
                self.append(agent, &mut transcript, Item::ToolResult { call_id, output }, None)?;
            }
        }
    }

    fn append(&self, agent: &str, transcript: &mut Transcript, item: Item, event: Option<i64>) -> Result<()> {
        let entry = transcript.append(item, event)?;
        self.observe(Observation::Item { agent: agent.to_owned(), entry });
        Ok(())
    }

    /// Appends queued mail as input. Mail already in the transcript (a crash between the
    /// append and the database update) is only marked delivered.
    fn deliver(&self, agent: &str, transcript: &mut Transcript) -> Result<()> {
        let pending = self.store.pending(agent)?;
        if pending.is_empty() {
            return Ok(());
        }
        let present: HashMap<i64, u64> = transcript.entries.iter().filter_map(|e| Some((e.event?, e.seq))).collect();
        let mut delivered = Vec::with_capacity(pending.len());
        for mail in pending {
            let seq = match present.get(&mail.id) {
                Some(&seq) => seq,
                None => {
                    let item = Item::Input { from: mail.from, text: mail.text };
                    let seq = transcript.entries.len() as u64;
                    self.append(agent, transcript, item, Some(mail.id))?;
                    seq
                }
            };
            delivered.push((mail.id, seq));
        }
        self.store.delivered(&delivered)
    }

    /// A harness that stopped mid-tool leaves calls without results; the model needs one
    /// for every call.
    fn close_dangling_calls(&self, agent: &str, transcript: &mut Transcript) -> Result<()> {
        let answered: std::collections::HashSet<String> = transcript
            .items()
            .filter_map(|i| if let Item::ToolResult { call_id, .. } = i { Some(call_id.clone()) } else { None })
            .collect();
        let dangling: Vec<String> = transcript
            .items()
            .filter_map(|i| match i {
                Item::ToolCall { call_id, .. } if !answered.contains(call_id) => Some(call_id.clone()),
                _ => None,
            })
            .collect();
        for call_id in dangling {
            self.append(
                agent,
                transcript,
                Item::ToolResult { call_id, output: ToolOutput::error(INTERRUPTED_CALL) },
                None,
            )?;
        }
        Ok(())
    }
}

async fn call_tool(tools: &[Arc<dyn Tool>], context: &ToolContext, name: &str, arguments: &str) -> ToolOutput {
    let Some(tool) = tools.iter().find(|t| t.name() == name) else {
        return ToolOutput::error(format!("Tool {name} not found"));
    };
    let args = if arguments.trim().is_empty() { Ok(serde_json::json!({})) } else { serde_json::from_str(arguments) };
    match args {
        Ok(args) => tool.call(context, args).await,
        Err(error) => ToolOutput::error(format!("Invalid JSON arguments for {name}: {error}")),
    }
}

/// The agent's own prompt and id, then the tool list and usage rules as Pi arranges them.
fn system_prompt(base: &str, agent: &str, tools: &[Arc<dyn Tool>]) -> String {
    let mut prompt = format!("{}\n\nYour agent id is {agent}.", base.trim_end());
    let listed: Vec<String> =
        tools.iter().filter(|t| !t.snippet().is_empty()).map(|t| format!("- {}: {}", t.name(), t.snippet())).collect();
    if !listed.is_empty() {
        prompt.push_str("\n\nAvailable tools:\n");
        prompt.push_str(&listed.join("\n"));
    }
    let guidelines: Vec<String> = tools.iter().flat_map(|t| t.guidelines()).map(|g| format!("- {g}")).collect();
    if !guidelines.is_empty() {
        prompt.push_str("\n\nGuidelines:\n");
        prompt.push_str(&guidelines.join("\n"));
    }
    prompt
}
