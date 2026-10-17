//! `web`: search the web, read a page by asking a question about it, and GET endpoints, as one
//! command-line tool. Searching and reading run on models of their own, outside the agent's
//! context, so only their few lines reach the agent: search is a provider's hosted web search
//! (OpenAI's on a Responses provider, Claude's on an Anthropic one), falling back to
//! DuckDuckGo's results page; a page is read whole by a reader model that answers the question.
//! Every request goes through [`Http`], which keeps agents off private networks. The host learns
//! the tokens these calls spend through [`Web::on_usage`].

mod http;
mod text;

pub use http::{Fetched, Http, Limits, private};
pub use text::{duckduckgo_results, readable, short, shown, window_of};

use super::cli;
use super::{Tool, ToolContext, ToolOutput};
use crate::agent::{Item, Usage};
use crate::provider::{Hosted, Provider, Request, Source};
use anyhow::{Result, anyhow, ensure};
use clap::{CommandFactory, Parser, Subcommand};
use futures_util::future::BoxFuture;
use serde_json::Value;
use std::sync::Arc;

/// Most results a search shows.
pub const SEARCH_RESULTS: usize = 8;
/// Most characters of a page the reader is given.
pub const PAGE_CHARS: usize = 120_000;
/// Characters of a `web get` answer shown per call.
pub const GET_WINDOW: usize = 4_000;
/// Biggest answer `web get` and `web read` take.
const CAP: usize = 8 << 20;
/// DuckDuckGo's HTML results page, which search falls back to.
pub const DUCKDUCKGO: &str = "https://html.duckduckgo.com/html/";

const SEARCH_SYSTEM: &str = "You run one web search for a researcher and report what it found. Search once \
with the query you are given. Then reply with one line per useful result, at most 8, each exactly \
`<url> — <one short line on what the page holds>`, best first. No other text.";

const READ_SYSTEM: &str = "You read one web page for a researcher and answer their question about it from \
the page alone. Answer in at most a few short lines. Quote every URL you rely on exactly as the page gives \
it (links appear as [text](url)). If the page does not answer the question, say so in one line and name \
the links on it most likely to lead to the answer.";

#[derive(Parser)]
#[command(
    name = "web",
    disable_version_flag = true,
    about = "Read the public web: search it, ask a question of a page, and GET JSON endpoints."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Search the web; prints up to 8 results, each a title, URL and one line on what it holds.
    Search {
        #[arg(required = true)]
        query: Vec<String>,
    },
    /// Read a web page and answer a question about it: a model reads the whole page and
    /// answers in a few lines, quoting the URLs it found. Use this for HTML pages.
    #[command(
        after_help = "Example: web read https://data.example.gov/ which links lead to the datasets or their APIs?"
    )]
    Read {
        url: String,
        /// The question, such as: which links lead to the datasets or their APIs?
        #[arg(required = true)]
        question: Vec<String>,
    },
    /// GET an endpoint as it is, for JSON and REST answers: JSON is pretty-printed, one value
    /// per line, so --grep can pick lines out of it. For HTML pages, use `web read`.
    #[command(after_help = format!("Answers come in windows of {GET_WINDOW} characters; pass --offset to read on. \
        Example: web get https://data.example.gov/api/datasets --grep 'title|url' --context 1"))]
    Get {
        url: String,
        /// Character of the answer (after --grep) to start from.
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Show only lines matching this case-insensitive regular expression.
        #[arg(long)]
        grep: Option<String>,
        /// Lines shown around each --grep match.
        #[arg(long, default_value_t = 0, requires = "grep")]
        context: usize,
    },
}

/// A model and the provider that serves it, for a side call.
#[derive(Clone)]
pub struct Model {
    pub provider: Arc<dyn Provider>,
    pub model: String,
    pub effort: Option<String>,
}

/// The models `web` works with. Without a searcher, search goes straight to `fallback_search`;
/// without a reader, pages cannot be read and `web read` points to `web get`.
#[derive(Clone)]
pub struct Models {
    /// Runs the provider's hosted web search. A provider without one fails the call, and
    /// search falls back.
    pub searcher: Option<Model>,
    /// Reads a page and answers a question about it.
    pub reader: Option<Model>,
    /// An HTML results page in DuckDuckGo's form, searched with `?q=`.
    pub fallback_search: String,
}

impl Default for Models {
    fn default() -> Self {
        Self { searcher: None, reader: None, fallback_search: DUCKDUCKGO.into() }
    }
}

/// One search result as agents read it.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub title: String,
    pub url: String,
    pub about: String,
}

/// Called with the agent, the command (`web search` or `web read`) and the tokens it spent.
pub type OnUsage = Arc<dyn Fn(&str, &str, &Usage) + Send + Sync>;
/// Called with the agent and each command line it runs, before it runs.
pub type OnCommand = Arc<dyn Fn(&str, &str) + Send + Sync>;

pub struct Web {
    http: Http,
    models: Models,
    on_usage: Option<OnUsage>,
    on_command: Option<OnCommand>,
}

impl Web {
    pub fn new(http: Http, models: Models) -> Self {
        Self { http, models, on_usage: None, on_command: None }
    }

    /// Reports the tokens each search and read spends.
    pub fn on_usage(mut self, report: impl Fn(&str, &str, &Usage) + Send + Sync + 'static) -> Self {
        self.on_usage = Some(Arc::new(report));
        self
    }

    /// Reports each command line as it starts.
    pub fn on_command(mut self, report: impl Fn(&str, &str) + Send + Sync + 'static) -> Self {
        self.on_command = Some(Arc::new(report));
        self
    }

    pub fn http(&self) -> &Http {
        &self.http
    }

    fn charge(&self, agent: &str, command: &str, usage: &Usage) {
        if let Some(report) = &self.on_usage {
            report(agent, command, usage);
        }
    }

    /// Runs one command line for `agent`.
    pub async fn run(&self, agent: &str, line: &str) -> ToolOutput {
        if let Some(report) = &self.on_command {
            report(agent, line);
        }
        let cli: Cli = match cli::parse("web", line) {
            Ok(cli) => cli,
            Err(output) => return output,
        };
        match cli.command {
            Command::Get { url, offset, grep, context } => {
                let fetched = match self.http.get(&url, CAP).await {
                    Ok(fetched) => fetched,
                    Err(error) => return ToolOutput::error(format!("{error:#}")),
                };
                let mut text = shown(&fetched.text(), &fetched.content_type, &fetched.url);
                if let Some(pattern) = grep {
                    text = match cli::grep(&text, &pattern, context) {
                        Ok(text) => text,
                        Err(error) => return ToolOutput::error(error),
                    };
                }
                let cut = if fetched.truncated { " (cut at 8 MB)" } else { "" };
                ToolOutput::text(format!(
                    "HTTP {} {} from {}{cut}\n\n{}",
                    fetched.status,
                    fetched.content_type,
                    fetched.url,
                    window_of(&text, offset, GET_WINDOW)
                ))
            }
            Command::Read { url, question } => self.read_page(agent, &url, &question.join(" ")).await,
            Command::Search { query } => self.search_web(agent, &query.join(" ")).await,
        }
    }

    async fn read_page(&self, agent: &str, url: &str, question: &str) -> ToolOutput {
        let fetched = match self.http.get(url, CAP).await {
            Ok(fetched) => fetched,
            Err(error) => return ToolOutput::error(format!("{error:#}")),
        };
        let page = shown(&fetched.text(), &fetched.content_type, &fetched.url);
        match self.read(&fetched.url, &page, question).await {
            Ok((answer, usage)) => {
                self.charge(agent, "web read", &usage);
                ToolOutput::text(format!("{answer}\n\n(HTTP {} from {})", fetched.status, fetched.url))
            }
            Err(error) => ToolOutput::error(format!(
                "reading the page failed: {error:#}. `web get {url} --grep <pattern>` shows lines of it instead"
            )),
        }
    }

    async fn search_web(&self, agent: &str, query: &str) -> ToolOutput {
        let note = match self.search(query).await {
            Ok((found, usage)) if !found.is_empty() => {
                self.charge(agent, "web search", &usage);
                return ToolOutput::text(listing(&found));
            }
            Ok((_, usage)) => {
                self.charge(agent, "web search", &usage);
                "the search found nothing".to_owned()
            }
            Err(error) => format!("{error:#}"),
        };
        let mut url = match url::Url::parse(&self.models.fallback_search) {
            Ok(url) => url,
            Err(error) => {
                return ToolOutput::error(format!("no search: {note}; the fallback search is not a URL: {error}"));
            }
        };
        url.query_pairs_mut().append_pair("q", query);
        match self.http.get(url.as_str(), 2 << 20).await {
            Ok(fetched) => {
                let found: Vec<Found> = duckduckgo_results(&fetched.text())
                    .into_iter()
                    .map(|(title, url, about)| Found { title, url, about })
                    .collect();
                if found.is_empty() {
                    return ToolOutput::error(format!(
                        "no results ({note}; DuckDuckGo answered HTTP {}); try other words",
                        fetched.status
                    ));
                }
                ToolOutput::text(format!("{}\n(from DuckDuckGo: {note})", listing(&found)))
            }
            Err(error) => ToolOutput::error(format!("no search: {note}; DuckDuckGo: {error:#}")),
        }
    }

    /// Answers `question` from `page` (the text of the page at `url`), with the tokens it took.
    pub async fn read(&self, url: &str, page: &str, question: &str) -> Result<(String, Usage)> {
        let reader = self.models.reader.as_ref().ok_or_else(|| anyhow!("no model can read pages here"))?;
        let mut text: String = page.chars().take(PAGE_CHARS).collect();
        if text.len() < page.len() {
            text.push_str("\n[the page goes on; this is its beginning]");
        }
        let prompt = format!("The page at {url}:\n\n{text}\n\nQuestion: {question}");
        let (answer, usage, _) = side_call(reader, READ_SYSTEM, prompt, &[]).await?;
        ensure!(!answer.is_empty(), "the reader gave no answer");
        Ok((answer, usage))
    }

    /// Searches the web for `query` with the searcher's hosted search: up to [`SEARCH_RESULTS`]
    /// results, with the tokens it took.
    pub async fn search(&self, query: &str) -> Result<(Vec<Found>, Usage)> {
        let searcher = self.models.searcher.as_ref().ok_or_else(|| anyhow!("no model can search the web here"))?;
        let hosted = [Hosted::WebSearch { max_uses: 1 }];
        let (answer, usage, sources) =
            side_call(searcher, SEARCH_SYSTEM, format!("Search the web for: {query}"), &hosted).await?;
        Ok((found(&answer, &sources), usage))
    }
}

/// One model call outside any agent: its text, tokens and sources.
async fn side_call(
    model: &Model,
    system: &str,
    prompt: String,
    hosted: &[Hosted],
) -> Result<(String, Usage, Vec<Source>)> {
    let items = [Item::Input { from: "user".into(), text: prompt, images: Vec::new() }];
    let cache_key = uuid::Uuid::now_v7().to_string();
    let completion = model
        .provider
        .complete(
            Request {
                model: &model.model,
                reasoning_effort: model.effort.as_deref(),
                cache_key: &cache_key,
                system,
                tools: &[],
                hosted,
                items: &items,
            },
            &|_| {},
        )
        .await?;
    let text: Vec<&str> = completion
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Assistant { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    Ok((text.join("\n").trim().to_owned(), completion.usage, completion.sources))
}

/// The results of a search: the model's `<url> — <about>` lines, titled from the search's own
/// sources, or those sources as they came when the model wrote no usable line.
pub fn found(answer: &str, sources: &[Source]) -> Vec<Found> {
    let mut out: Vec<Found> = Vec::new();
    for line in answer.lines() {
        let line = line.trim().trim_start_matches(['-', '*', ' ']).trim();
        let Some((url, about)) = line.split_once(" — ").or_else(|| line.split_once(" - ")) else { continue };
        let url = url.trim().trim_matches(['<', '>', '`']);
        if !url.starts_with("http") || out.iter().any(|f| f.url == url) {
            continue;
        }
        let title = sources.iter().find(|s| s.url == url).map(|s| s.title.clone()).unwrap_or_default();
        out.push(Found { title, url: url.to_owned(), about: about.trim().to_owned() });
    }
    if out.is_empty() {
        out = sources
            .iter()
            .map(|s| Found { title: s.title.clone(), url: s.url.clone(), about: String::new() })
            .collect();
    }
    out.truncate(SEARCH_RESULTS);
    out
}

/// Search results as agents read them: at most [`SEARCH_RESULTS`], each its title (when the
/// search gave one), URL and one line.
fn listing(found: &[Found]) -> String {
    found
        .iter()
        .take(SEARCH_RESULTS)
        .map(|f| {
            let about = short(&f.about, 200);
            let mut lines = vec![if f.title.is_empty() { f.url.clone() } else { format!("{}\n  {}", f.title, f.url) }];
            if !about.is_empty() {
                lines.push(about);
            }
            format!("- {}", lines.join("\n  "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl Tool for Web {
    fn name(&self) -> &str {
        "web"
    }
    fn description(&self) -> String {
        format!(
            "Runs a `web` command, given as a command line. There is no shell: no pipes, no redirects. \
             Search, then ask questions of pages with `web read`; `web get` is for JSON endpoints, with \
             --grep to cut long answers down.\n\n{}",
            Cli::command().render_long_help()
        )
    }
    fn parameters(&self) -> Value {
        cli::parameters("search county zoning map open data")
    }
    fn snippet(&self) -> &str {
        "web: search the web, ask a question of a page, and GET JSON endpoints"
    }
    fn call<'a>(&'a self, context: &'a ToolContext, args: Value) -> BoxFuture<'a, ToolOutput> {
        Box::pin(async move {
            match cli::line(&args, "web", "search county zoning map open data") {
                Ok(line) => cli::until_stopped(context, self.run(&context.agent, &line)).await,
                Err(output) => output,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_without_a_title_shows_its_url_once() {
        let found = [Found { title: String::new(), url: "https://a.gov/x".into(), about: "the layer".into() }];
        assert_eq!(listing(&found), "- https://a.gov/x\n  the layer");
    }

    #[test]
    fn a_search_answer_reads_as_titled_lines() {
        let sources = [
            Source { url: "https://a.gov/parcels".into(), title: "A Parcels".into() },
            Source { url: "https://b.org/gis".into(), title: "B GIS".into() },
        ];
        let answer =
            "- https://a.gov/parcels — the county's parcel viewer\nnoise line\nhttps://b.org/gis — open data hub";
        assert_eq!(
            found(answer, &sources),
            [
                Found {
                    title: "A Parcels".into(),
                    url: "https://a.gov/parcels".into(),
                    about: "the county's parcel viewer".into()
                },
                Found { title: "B GIS".into(), url: "https://b.org/gis".into(), about: "open data hub".into() },
            ]
        );
        assert_eq!(
            found("nothing usable", &sources[..1]),
            [Found { title: "A Parcels".into(), url: "https://a.gov/parcels".into(), about: String::new() }]
        );
    }
}
