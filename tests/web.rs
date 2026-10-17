//! The `web` tool against a stand-in site and a stand-in Responses endpoint: hosted search with
//! titled results, a page read whole by a reader model while only its answer reaches the agent,
//! the tokens both spend, DuckDuckGo when there is no search or the endpoint refuses it, and
//! `web get` with --grep and windows.

use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use erisharness::agent::Usage;
use erisharness::provider::{RateGate, Responses, ResponsesConfig, StaticToken};
use erisharness::tools::web::{Http, Limits, Model, Models, Web};
use erisharness::tools::{Content, Tool, ToolOutput};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Words no answer may carry: they stand for the bulk of a page, which stays out of agents'
/// context.
const FILLER: &str = "LONG-PAGE-BODY";

#[derive(Default)]
struct Endpoint {
    /// Whether hosted web search is refused, as an endpoint without it answers.
    refuses_search: bool,
    requests: Vec<Value>,
}

type Shared = Arc<Mutex<Endpoint>>;

fn sse(events: &[Value]) -> Response {
    let body: String =
        events.iter().map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap())).collect();
    ([("content-type", "text/event-stream")], body).into_response()
}

fn usage(input: u64, output: u64) -> Value {
    json!({"type": "response.completed", "response": {"usage": {"input_tokens": input, "output_tokens": output, "input_tokens_details": {"cached_tokens": 0}}}})
}

/// The stand-in Responses endpoint: a search answers ten lines citing ten pages; a read
/// answers with the link the page names.
async fn responses(State(endpoint): State<Shared>, body: String) -> Response {
    let body: Value = serde_json::from_str(&body).unwrap();
    let searching = body["tools"].as_array().unwrap().iter().any(|t| t["type"] == "web_search");
    let refuses = {
        let mut endpoint = endpoint.lock().unwrap();
        endpoint.requests.push(body.clone());
        endpoint.refuses_search
    };
    if searching && refuses {
        return (
            StatusCode::BAD_REQUEST,
            r#"{"error":{"type":"invalid_request_error","message":"Unsupported tool type: web_search"}}"#,
        )
            .into_response();
    }
    if searching {
        let urls: Vec<String> = (0..10).map(|i| format!("https://county{i}.example.gov/gis")).collect();
        let text: Vec<String> = urls.iter().enumerate().map(|(i, u)| format!("{u} — county {i}'s open data")).collect();
        let annotations: Vec<Value> =
            urls.iter().map(|u| json!({"type": "url_citation", "url": u, "title": format!("Title of {u}")})).collect();
        return sse(&[
            json!({"type": "response.output_item.done", "item": {"type": "web_search_call", "status": "completed", "action": {"type": "search", "query": "q"}}}),
            json!({"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text.join("\n"), "annotations": annotations}]}}),
            usage(300, 40),
        ]);
    }
    let page = body["input"][0]["content"][0]["text"].as_str().unwrap().to_owned();
    assert!(page.contains(FILLER), "the reader gets the whole page");
    let link = page.split("[Zoning service](").nth(1).unwrap().split(')').next().unwrap().to_owned();
    sse(&[
        json!({"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": format!("The zoning layer is {link}")}]}}),
        usage(1000, 20),
    ])
}

/// A county's open-data page: a link to its zoning service among a long body.
async fn page() -> Html<String> {
    let filler = format!("<p>{FILLER} about meetings and minutes.</p>").repeat(2_000);
    Html(format!(
        "<html><body><h1>County GIS</h1>{filler}<a href=\"/rest/Zoning/FeatureServer/0\">Zoning service</a></body></html>"
    ))
}

async fn services() -> axum::Json<Value> {
    let services: Vec<Value> = (0..400)
        .map(|i| json!({"name": format!("Layer{i}"), "type": if i % 50 == 0 { "FeatureServer" } else { "MapServer" }}))
        .collect();
    axum::Json(json!({"currentVersion": 11.1, "services": services}))
}

/// DuckDuckGo's HTML results page, with ten results.
async fn duckduckgo(Query(q): Query<HashMap<String, String>>) -> Html<String> {
    let query = q.get("q").cloned().unwrap_or_default();
    Html(
        (0..10)
            .map(|i| {
                format!(
                    "<div class=\"result\"><a class=\"result__a\" href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fcounty{i}.example.gov%2Fgis\">County {i} GIS for {query}</a>\
                     <a class=\"result__snippet\" href=\"x\">Maps of county {i}.</a></div>"
                )
            })
            .collect(),
    )
}

struct Stand {
    root: String,
    endpoint: Shared,
}

async fn stand() -> Stand {
    let endpoint: Shared = Arc::default();
    let app = Router::new()
        .route("/v1/responses", post(responses))
        .route("/page", get(page))
        .route("/rest/services", get(services))
        .route("/ddg", get(duckduckgo))
        .with_state(endpoint.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let root = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Stand { root, endpoint }
}

fn http() -> Http {
    Http::new(Limits { allow_private: true, ..Limits::default() }, "erisharness-test/1").unwrap()
}

/// A small model on the stand-in endpoint, as an app with a ChatGPT login would give `web`.
fn model(root: &str) -> Model {
    let provider = Arc::new(Responses::new(ResponsesConfig {
        base_url: format!("{root}/v1"),
        headers: Vec::new(),
        store: false,
        max_retries: 0,
        credentials: Arc::new(StaticToken("token".into())),
        gate: RateGate::new(4, 16),
    }));
    Model { provider, model: "small".into(), effort: Some("low".into()) }
}

fn text(output: &ToolOutput) -> String {
    output.content.iter().map(|c| if let Content::Text(t) = c { t.as_str() } else { "" }).collect()
}

type Spent = Arc<Mutex<Vec<(String, String, Usage)>>>;

fn web(models: Models) -> (Web, Spent) {
    let spent: Spent = Arc::default();
    let record = spent.clone();
    let web = Web::new(http(), models)
        .on_usage(move |agent, command, usage| record.lock().unwrap().push((agent.into(), command.into(), *usage)));
    (web, spent)
}

#[tokio::test]
async fn searching_and_reading_bring_back_a_few_lines_and_report_their_tokens() {
    let stand = stand().await;
    let small = model(&stand.root);
    let (web, spent) = web(Models { searcher: Some(small.clone()), reader: Some(small), ..Models::default() });

    let search = web.run("agent", "search county zoning gis").await;
    let shown = text(&search);
    assert!(!search.is_error, "{shown}");
    assert_eq!(shown.lines().filter(|l| l.starts_with("- ")).count(), 8, "{shown}");
    assert!(
        shown.contains(
            "- Title of https://county0.example.gov/gis\n  https://county0.example.gov/gis\n  county 0's open data"
        ),
        "{shown}"
    );
    assert!(!shown.contains("DuckDuckGo"));

    let read = web.run("agent", &format!("read {}/page where is the zoning service", stand.root)).await;
    let answer = text(&read);
    assert!(!read.is_error, "{answer}");
    assert!(answer.contains(&format!("{}/rest/Zoning/FeatureServer/0", stand.root)), "{answer}");
    assert!(!answer.contains(FILLER) && answer.len() < 500, "the page stays out: {answer}");

    let requests = stand.endpoint.lock().unwrap().requests.clone();
    assert_eq!(requests[0]["tools"], json!([{"type": "web_search"}]), "search is the endpoint's own");
    assert_eq!(
        (requests[0]["model"].as_str(), requests[0]["reasoning"]["effort"].as_str()),
        (Some("small"), Some("low"))
    );
    assert_eq!(requests[1]["tools"], json!([]), "reading searches nothing");
    let spent = spent.lock().unwrap();
    let commands: Vec<(&str, &str, u64, u64)> =
        spent.iter().map(|(a, c, u)| (a.as_str(), c.as_str(), u.input, u.output)).collect();
    assert_eq!(commands, [("agent", "web search", 300, 40), ("agent", "web read", 1000, 20)]);
}

#[tokio::test]
async fn without_models_search_falls_back_to_duckduckgo_and_reading_points_to_web_get() {
    let stand = stand().await;
    let (web, spent) = web(Models { fallback_search: format!("{}/ddg", stand.root), ..Models::default() });
    let search = web.run("agent", "search county zoning").await;
    let shown = text(&search);
    assert!(!search.is_error, "{shown}");
    assert_eq!(shown.lines().filter(|l| l.starts_with("- ")).count(), 8, "{shown}");
    assert!(
        shown.contains("- County 0 GIS for county zoning\n  https://county0.example.gov/gis\n  Maps of county 0."),
        "{shown}"
    );
    assert!(shown.contains("(from DuckDuckGo: no model can search the web here)"), "{shown}");
    let read = web.run("agent", &format!("read {}/page where is the zoning service", stand.root)).await;
    assert!(read.is_error && text(&read).contains("web get"), "{}", text(&read));
    assert!(spent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_endpoint_that_refuses_hosted_search_falls_back_to_duckduckgo_saying_why() {
    let stand = stand().await;
    stand.endpoint.lock().unwrap().refuses_search = true;
    let small = model(&stand.root);
    let (web, _) = web(Models { searcher: Some(small), reader: None, fallback_search: format!("{}/ddg", stand.root) });
    let search = web.run("agent", "search county zoning").await;
    let shown = text(&search);
    assert!(!search.is_error, "{shown}");
    assert!(shown.contains("https://county0.example.gov/gis"), "{shown}");
    assert!(shown.contains("from DuckDuckGo: 400 invalid_request_error: Unsupported tool type: web_search"), "{shown}");
}

#[tokio::test]
async fn web_get_shows_json_one_value_per_line_greps_it_and_comes_in_windows() {
    let stand = stand().await;
    let (web, _) = web(Models::default());
    let url = format!("{}/rest/services", stand.root);
    let grepped = text(&web.run("agent", &format!("get {url} --grep 'featureserver' --context 1")).await);
    assert!(grepped.contains("8 of ") && grepped.contains("lines match featureserver"), "{grepped}");
    assert!(grepped.contains(":       \"type\": \"FeatureServer\""), "pretty-printed, one value per line: {grepped}");
    let whole = text(&web.run("agent", &format!("get {url}")).await);
    assert!(whole.starts_with("HTTP 200 application/json from"), "{whole}");
    assert!(whole.contains("[characters 0–4000 of "), "{whole}");
    let next = text(&web.run("agent", &format!("get {url} --offset 4000")).await);
    assert!(next.contains("[characters 4000–8000 of "), "{next}");
    let page = text(&web.run("agent", &format!("get {}/page", stand.root)).await);
    assert!(page.contains("County GIS"), "HTML reads as text: {page}");
}

#[tokio::test]
async fn help_is_an_answer_mistakes_are_errors_and_private_addresses_are_refused() {
    let commands = Arc::new(Mutex::new(Vec::new()));
    let seen = commands.clone();
    let web = Web::new(Http::new(Limits::default(), "erisharness-test/1").unwrap(), Models::default())
        .on_command(move |agent, line| seen.lock().unwrap().push(format!("{agent}: {line}")));
    let help = web.run("agent", "--help").await;
    assert!(!help.is_error && text(&help).contains("search"), "{}", text(&help));
    let get_help = web.run("agent", "get --help").await;
    assert!(!get_help.is_error && text(&get_help).contains("--grep"));
    assert!(web.run("agent", "get").await.is_error, "a missing URL");
    assert!(web.run("agent", "fetch https://x.org").await.is_error, "an unknown command");
    assert!(web.run("agent", "get 'unclosed").await.is_error);
    assert!(web.run("agent", "get https://x.org --context 2").await.is_error, "--context needs --grep");
    let local = web.run("agent", "get http://127.0.0.1:9/").await;
    assert!(local.is_error && text(&local).contains("private address"), "{}", text(&local));
    assert!(web.description().contains("Usage: web"));
    assert_eq!(commands.lock().unwrap()[0], "agent: --help", "each command line is reported");
}
