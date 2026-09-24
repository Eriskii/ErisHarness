//! Durable state. SQLite (WAL, `synchronous=NORMAL`) holds agent records and inboxes and is
//! never exposed to sandboxes. Each transcript is an append-only JSONL file that other agents
//! may read through a bind mount; readers take no locks, so they cannot stall the writer.
//! Only finished items are written, never streaming deltas.

use crate::agent::{AgentRecord, AgentSpec, AgentState, Entry, Item, Usage};
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct Store {
    db: Mutex<Connection>,
    transcripts: PathBuf,
}

/// A message waiting in an agent's inbox.
#[derive(Clone, Debug)]
pub struct Mail {
    pub id: i64,
    pub from: String,
    pub text: String,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    spec TEXT NOT NULL,
    state TEXT NOT NULL,
    error TEXT,
    held INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL REFERENCES agents(id),
    sender TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    delivered_seq INTEGER
);
CREATE INDEX IF NOT EXISTS inbox_pending ON inbox(agent) WHERE delivered_seq IS NULL;
";

pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

impl Store {
    pub fn open(db: &Path, transcripts: &Path) -> Result<Self> {
        if let Some(parent) = db.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir_all(transcripts)?;
        let connection = Connection::open(db).with_context(|| format!("opening {}", db.display()))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.execute_batch(SCHEMA)?;
        Ok(Self { db: Mutex::new(connection), transcripts: transcripts.to_owned() })
    }

    pub fn create_agent(&self, id: &str, spec: &AgentSpec) -> Result<()> {
        fs::create_dir_all(self.transcripts.join(id))?;
        self.db.lock().unwrap().execute(
            "INSERT INTO agents (id, spec, state, created_at) VALUES (?1, ?2, 'idle', ?3)",
            params![id, serde_json::to_string(spec)?, now_ms() as i64],
        )?;
        Ok(())
    }

    pub fn agent(&self, id: &str) -> Result<Option<AgentRecord>> {
        let db = self.db.lock().unwrap();
        db.query_row(
            "SELECT spec, state, error, held, input_tokens, cached_tokens, output_tokens, reasoning_tokens FROM agents WHERE id = ?1",
            [id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, bool>(3)?,
                    Usage {
                        input: row.get::<_, i64>(4)? as u64,
                        cached_input: row.get::<_, i64>(5)? as u64,
                        output: row.get::<_, i64>(6)? as u64,
                        reasoning: row.get::<_, i64>(7)? as u64,
                    },
                ))
            },
        )
        .optional()?
        .map(|(spec, state, error, held, usage)| {
            Ok(AgentRecord {
                id: id.to_owned(),
                spec: serde_json::from_str(&spec)?,
                state: AgentState::parse(&state),
                error,
                held,
                usage,
            })
        })
        .transpose()
    }

    pub fn set_state(&self, id: &str, state: AgentState, error: Option<&str>) -> Result<()> {
        self.db
            .lock()
            .unwrap()
            .execute("UPDATE agents SET state = ?2, error = ?3 WHERE id = ?1", params![id, state.as_str(), error])?;
        Ok(())
    }

    pub fn set_held(&self, id: &str, held: bool) -> Result<()> {
        self.db.lock().unwrap().execute("UPDATE agents SET held = ?2 WHERE id = ?1", params![id, held])?;
        Ok(())
    }

    pub fn add_usage(&self, id: &str, usage: Usage) -> Result<()> {
        self.db.lock().unwrap().execute(
            "UPDATE agents SET input_tokens = input_tokens + ?2, cached_tokens = cached_tokens + ?3,
             output_tokens = output_tokens + ?4, reasoning_tokens = reasoning_tokens + ?5 WHERE id = ?1",
            params![id, usage.input as i64, usage.cached_input as i64, usage.output as i64, usage.reasoning as i64],
        )?;
        Ok(())
    }

    /// Agents a restarted harness must wake: those mid-turn, and those with unheld mail.
    pub fn unfinished(&self) -> Result<Vec<String>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare(
            "SELECT id FROM agents WHERE state = 'running'
             OR (held = 0 AND EXISTS (SELECT 1 FROM inbox WHERE inbox.agent = agents.id AND delivered_seq IS NULL))",
        )?;
        let ids = statement.query_map([], |row| row.get(0))?.collect::<Result<_, _>>()?;
        Ok(ids)
    }

    pub fn enqueue(&self, agent: &str, from: &str, text: &str) -> Result<i64> {
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT INTO inbox (agent, sender, body, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![agent, from, text, now_ms() as i64],
        )?;
        Ok(db.last_insert_rowid())
    }

    pub fn pending(&self, agent: &str) -> Result<Vec<Mail>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare_cached(
            "SELECT id, sender, body FROM inbox WHERE agent = ?1 AND delivered_seq IS NULL ORDER BY id",
        )?;
        let mail = statement
            .query_map([agent], |row| Ok(Mail { id: row.get(0)?, from: row.get(1)?, text: row.get(2)? }))?
            .collect::<Result<_, _>>()?;
        Ok(mail)
    }

    pub fn delivered(&self, deliveries: &[(i64, u64)]) -> Result<()> {
        let mut db = self.db.lock().unwrap();
        let transaction = db.transaction()?;
        for (id, seq) in deliveries {
            transaction.execute("UPDATE inbox SET delivered_seq = ?2 WHERE id = ?1", params![id, *seq as i64])?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn transcript_path(&self, agent: &str) -> PathBuf {
        self.transcripts.join(agent).join("transcript.jsonl")
    }

    pub fn transcript(&self, agent: &str) -> Result<Transcript> {
        Transcript::open(self.transcript_path(agent))
    }
}

/// An open transcript: its entries, and the file for appending more.
pub struct Transcript {
    pub entries: Vec<Entry>,
    file: fs::File,
}

impl Transcript {
    fn open(path: PathBuf) -> Result<Self> {
        let file = fs::OpenOptions::new().create(true).read(true).append(true).open(&path)?;
        let mut entries = Vec::new();
        let mut intact = 0u64;
        let mut torn = false;
        for line in BufReader::new(&file).split(b'\n') {
            let line = line?;
            match serde_json::from_slice::<Entry>(&line) {
                Ok(entry) => {
                    entries.push(entry);
                    intact += line.len() as u64 + 1;
                }
                Err(_) => {
                    torn = true;
                    break;
                }
            }
        }
        // A crash mid-append leaves a partial last line. Drop it so the next append starts clean.
        if torn {
            file.set_len(intact)?;
        }
        Ok(Self { entries, file })
    }

    pub fn items(&self) -> impl Iterator<Item = &Item> {
        self.entries.iter().map(|e| &e.item)
    }

    pub fn append(&mut self, item: Item, event: Option<i64>) -> Result<Entry> {
        let entry = Entry { seq: self.entries.len() as u64, at: now_ms(), event, item };
        let mut line = serde_json::to_vec(&entry)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.entries.push(entry.clone());
        Ok(entry)
    }
}
