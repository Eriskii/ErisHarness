//! Durable state. SQLite (WAL, `synchronous=NORMAL`) holds agent records and inboxes, and is
//! never exposed to sandboxes. Each transcript is an append-only JSONL file of finished items,
//! which other agents may read through a bind mount; readers take no locks, so they cannot
//! stall the writer.

use crate::agent::{AgentRecord, AgentSpec, AgentState, Entry, Image, Item, Usage};
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
    pub images: Vec<Image>,
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
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    context_tokens INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL REFERENCES agents(id),
    sender TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    delivered_seq INTEGER,
    images TEXT
);
CREATE INDEX IF NOT EXISTS inbox_pending ON inbox(agent) WHERE delivered_seq IS NULL;
";

const COLUMNS: &str = "id, spec, state, error, held, input_tokens, cached_tokens, output_tokens, reasoning_tokens, \
    context_tokens, cache_write_tokens";

fn record(row: &rusqlite::Row) -> rusqlite::Result<AgentRecord> {
    let count = |index| row.get::<_, i64>(index).map(|n| n as u64);
    let spec: String = row.get(1)?;
    Ok(AgentRecord {
        id: row.get(0)?,
        spec: serde_json::from_str(&spec)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, e.into()))?,
        state: AgentState::parse(&row.get::<_, String>(2)?),
        error: row.get(3)?,
        held: row.get(4)?,
        usage: Usage {
            input: count(5)?,
            cached_input: count(6)?,
            cache_write: count(10)?,
            output: count(7)?,
            reasoning: count(8)?,
        },
        context_tokens: count(9)?,
    })
}

fn mail(row: &rusqlite::Row) -> rusqlite::Result<Mail> {
    let images: Option<String> = row.get(3)?;
    let images = match images {
        Some(json) => serde_json::from_str(&json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, e.into()))?,
        None => Vec::new(),
    };
    Ok(Mail { id: row.get(0)?, from: row.get(1)?, text: row.get(2)?, images })
}

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
        // A database without the cache-write column gains it, starting at zero.
        let counted = connection
            .prepare("SELECT 1 FROM pragma_table_info('agents') WHERE name = 'cache_write_tokens'")?
            .exists([])?;
        if !counted {
            connection.execute_batch("ALTER TABLE agents ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0")?;
        }
        // An inbox without the images column gains it; earlier mail has none.
        let imaged = connection.prepare("SELECT 1 FROM pragma_table_info('inbox') WHERE name = 'images'")?.exists([])?;
        if !imaged {
            connection.execute_batch("ALTER TABLE inbox ADD COLUMN images TEXT")?;
        }
        Ok(Self { db: Mutex::new(connection), transcripts: transcripts.to_owned() })
    }

    fn execute(&self, sql: &str, params: impl rusqlite::Params) -> Result<()> {
        self.db.lock().unwrap().execute(sql, params)?;
        Ok(())
    }

    pub fn create_agent(&self, id: &str, spec: &AgentSpec) -> Result<()> {
        fs::create_dir_all(self.transcripts.join(id))?;
        let sql = "INSERT INTO agents (id, spec, state, created_at) VALUES (?1, ?2, 'idle', ?3)";
        self.execute(sql, params![id, serde_json::to_string(spec)?, now_ms() as i64])
    }

    pub fn agent(&self, id: &str) -> Result<Option<AgentRecord>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare_cached(&format!("SELECT {COLUMNS} FROM agents WHERE id = ?1"))?;
        Ok(statement.query_row([id], record).optional()?)
    }

    pub fn agents(&self) -> Result<Vec<AgentRecord>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare_cached(&format!("SELECT {COLUMNS} FROM agents ORDER BY created_at, id"))?;
        Ok(statement.query_map([], record)?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn set_state(&self, id: &str, state: AgentState, error: Option<&str>) -> Result<()> {
        self.execute("UPDATE agents SET state = ?2, error = ?3 WHERE id = ?1", params![id, state.as_str(), error])
    }

    pub fn set_held(&self, id: &str, held: bool) -> Result<()> {
        self.execute("UPDATE agents SET held = ?2 WHERE id = ?1", params![id, held])
    }

    pub fn set_spec(&self, id: &str, spec: &AgentSpec) -> Result<()> {
        self.execute("UPDATE agents SET spec = ?2 WHERE id = ?1", params![id, serde_json::to_string(spec)?])
    }

    /// Adds a request's usage to the totals and records how full the context now is.
    pub fn add_usage(&self, id: &str, usage: Usage, context: u64) -> Result<()> {
        self.execute(
            "UPDATE agents SET input_tokens = input_tokens + ?2, cached_tokens = cached_tokens + ?3,
             output_tokens = output_tokens + ?4, reasoning_tokens = reasoning_tokens + ?5, context_tokens = ?6,
             cache_write_tokens = cache_write_tokens + ?7
             WHERE id = ?1",
            params![
                id,
                usage.input as i64,
                usage.cached_input as i64,
                usage.output as i64,
                usage.reasoning as i64,
                context as i64,
                usage.cache_write as i64
            ],
        )
    }

    pub fn remove_agent(&self, id: &str) -> Result<()> {
        let mut db = self.db.lock().unwrap();
        let transaction = db.transaction()?;
        transaction.execute("DELETE FROM inbox WHERE agent = ?1", [id])?;
        transaction.execute("DELETE FROM agents WHERE id = ?1", [id])?;
        transaction.commit()?;
        drop(db);
        match fs::remove_dir_all(self.transcripts.join(id)) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
            _ => Ok(()),
        }
    }

    /// The first unread mail for `agent` from `from` newer than `after`.
    pub fn reply(&self, agent: &str, from: &str, after: i64) -> Result<Option<Mail>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare_cached(
            "SELECT id, sender, body, images FROM inbox WHERE agent = ?1 AND sender = ?2 AND id > ?3 AND delivered_seq IS NULL
             ORDER BY id LIMIT 1",
        )?;
        Ok(statement.query_row(params![agent, from, after], mail).optional()?)
    }

    /// Agents a restarted harness must wake: those mid-turn, and those with unheld mail.
    pub fn unfinished(&self) -> Result<Vec<String>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare(
            "SELECT id FROM agents WHERE state = 'running'
             OR (held = 0 AND EXISTS (SELECT 1 FROM inbox WHERE inbox.agent = agents.id AND delivered_seq IS NULL))",
        )?;
        Ok(statement.query_map([], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?)
    }

    pub fn enqueue(&self, agent: &str, from: &str, text: &str, images: &[Image]) -> Result<i64> {
        let images = if images.is_empty() { None } else { Some(serde_json::to_string(images)?) };
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT INTO inbox (agent, sender, body, created_at, images) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![agent, from, text, now_ms() as i64, images],
        )?;
        Ok(db.last_insert_rowid())
    }

    /// The newest mail id so far.
    pub fn last_mail(&self) -> Result<i64> {
        Ok(self.db.lock().unwrap().query_row("SELECT COALESCE(MAX(id), 0) FROM inbox", [], |row| row.get(0))?)
    }

    /// Undelivered mail for `agent`, oldest first.
    pub fn pending(&self, agent: &str) -> Result<Vec<Mail>> {
        let db = self.db.lock().unwrap();
        let mut statement = db.prepare_cached(
            "SELECT id, sender, body, images FROM inbox WHERE agent = ?1 AND delivered_seq IS NULL ORDER BY id",
        )?;
        Ok(statement.query_map([agent], mail)?.collect::<rusqlite::Result<_>>()?)
    }

    /// Marks mail delivered, each id at the transcript entry that holds it.
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

    /// The agent's transcript, opened for its turn to append to.
    pub fn transcript(&self, agent: &str) -> Result<Transcript> {
        Transcript::open(self.transcript_path(agent))
    }

    /// The agent's entries so far, read without creating or changing anything: a host may read
    /// while the agent is being removed. A transcript not yet written has none.
    pub fn entries(&self, agent: &str) -> Result<Vec<Entry>> {
        match fs::File::open(self.transcript_path(agent)) {
            Ok(file) => Ok(read_entries(&file)?.0),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(error.into()),
        }
    }
}

/// The intact entries at the start of a transcript, and how many bytes they take. A crash
/// mid-append leaves a partial last line, which is not an entry.
fn read_entries(file: &fs::File) -> Result<(Vec<Entry>, u64)> {
    let mut entries = Vec::new();
    let mut intact = 0u64;
    for line in BufReader::new(file).split(b'\n') {
        let line = line?;
        match serde_json::from_slice::<Entry>(&line) {
            Ok(entry) => {
                entries.push(entry);
                intact += line.len() as u64 + 1;
            }
            Err(_) => break,
        }
    }
    Ok((entries, intact))
}

/// An open transcript: its entries, and the file for appending more.
pub struct Transcript {
    pub entries: Vec<Entry>,
    file: fs::File,
}

impl Transcript {
    fn open(path: PathBuf) -> Result<Self> {
        let file = fs::OpenOptions::new().create(true).read(true).append(true).open(&path)?;
        let (entries, intact) = read_entries(&file)?;
        // Drop a partial last line so the next append starts clean.
        if file.metadata()?.len() > intact {
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
