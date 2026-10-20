//! The Induction payload log, driven through the real request path.
//!
//! `clippy::await_holding_lock` is allowed for the file for the reason
//! `bodycapture.rs` gives: each test holds `ONE_AT_A_TIME` for its whole body
//! on a current-thread runtime.
#![allow(clippy::await_holding_lock)]
//!
//! Its own test binary, and serial within it, because
//! `KOMPRESSOR_INDUCTION_LOG_FILE` is process-wide: a test that sets it arms
//! the log for every request running beside it.

mod support;

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::matchers::any;
use wiremock::{Mock, Request, ResponseTemplate};

use support::{Harness, first_handle, log_payload, responses_body, round_of};

const VARIABLE: &str = "KOMPRESSOR_INDUCTION_LOG_FILE";

static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

const HARNESS_HEADERS: [(&str, &str); 2] = [
    ("x-induction-trajectory-id", "traj_1"),
    ("x-induction-group-id", "grp_1"),
];

/// The log, armed for one test.
struct Armed {
    _serial: std::sync::MutexGuard<'static, ()>,
    _dir: tempfile::TempDir,
    file: PathBuf,
}

impl Armed {
    fn new() -> Armed {
        let serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("payloads.jsonl");
        // Safe here: std orders its own reads of the environment against this
        // write, every test in this file takes `ONE_AT_A_TIME` first, and nothing
        // these tests run reads the environment from C.
        unsafe { std::env::set_var(VARIABLE, &file) };
        Armed {
            _serial: serial,
            _dir: dir,
            file,
        }
    }

    fn records(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.file)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is JSON"))
            .collect()
    }

    /// The records once `requests` requests have each logged what their client
    /// received, which a streamed request does as its relay finishes, a moment
    /// after the client has read the last byte.
    async fn answered(&self, requests: usize) -> Vec<Value> {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let records = self.records();
            let answered = records
                .iter()
                .filter(|record| record["name"] == "payload_to_client")
                .count();
            if answered >= requests {
                return records;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{answered} of {requests} requests logged what the client received: {records:#?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        // Safe here for the reason given where it is set.
        unsafe { std::env::remove_var(VARIABLE) };
    }
}

fn names(records: &[Value]) -> Vec<&str> {
    records
        .iter()
        .map(|record| record["name"].as_str().unwrap())
        .collect()
}

fn body_text(request: &Request) -> String {
    String::from_utf8_lossy(&request.body).into_owned()
}

fn handle_in(request: &Request) -> String {
    first_handle(&body_text(request)).expect("the outgoing prompt carries a handle")
}

fn streaming(mut body: Value) -> Value {
    body["stream"] = json!(true);
    body
}

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body.to_string(), "text/event-stream")
}

fn answer() -> Value {
    json!({
        "id": "resp_1",
        "object": "response",
        "model": "gpt-5-2026-08-01",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}]}],
        "usage": {"input_tokens": 100, "output_tokens": 5}
    })
}

fn asks_to_retrieve(handle: &str) -> Value {
    json!({
        "id": "resp_1",
        "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_r1",
                    "name": "optscale_kompressor_retrieve",
                    "arguments": json!({"ref": handle}).to_string()}],
        "usage": {"input_tokens": 1000, "output_tokens": 20}
    })
}

fn answers_after_retrieving() -> Value {
    json!({
        "id": "resp_final",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "The disk filled."}]}],
        "usage": {"input_tokens": 1200, "output_tokens": 30}
    })
}

const STREAM_ASKS: &str = concat!(
    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
    "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_r1\",\"name\":\"optscale_kompressor_retrieve\",\"arguments\":\"\"}}\n\n",
    "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_r1\",\"name\":\"optscale_kompressor_retrieve\",\"arguments\":\"{\\\"ref\\\":\\\"HANDLE\\\"}\"}}\n\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":1000,\"output_tokens\":20}}}\n\n",
);

const STREAM_ANSWERS: &str = concat!(
    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\"}}\n\n",
    "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"The disk filled.\"}\n\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"usage\":{\"input_tokens\":1200,\"output_tokens\":30}}}\n\n",
);

/// Ask to retrieve on the first request, then answer `then` on the rest.
async fn mount_retrieve(harness: &Harness, stream: bool, then: ResponseTemplate) {
    Mock::given(any())
        .respond_with(move |request: &Request| {
            if round_of(&body_text(request)) > 0 {
                return then.clone();
            }
            if stream {
                sse(&STREAM_ASKS.replace("HANDLE", &handle_in(request)))
            } else {
                ResponseTemplate::new(200).set_body_json(asks_to_retrieve(&handle_in(request)))
            }
        })
        .mount(&harness.server)
        .await;
}

/// The Response a client read off a stream: the last one a frame carried.
fn response_in_stream(stream: &str) -> Value {
    stream
        .lines()
        .rev()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find_map(|event| event.get("response").cloned())
        .expect("the stream carried a response")
}

#[tokio::test]
async fn a_request_is_logged_as_the_client_sent_it_the_provider_got_it_and_the_client_got_it_back()
{
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(answer()))
        .mount(&harness.server)
        .await;
    let body = responses_body("the build is red");

    let reply = harness
        .post_with("/v1/responses", &body, &HARNESS_HEADERS)
        .await;
    let records = armed.answered(1).await;

    assert_eq!(
        names(&records),
        [
            "payload_from_client",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_client"
        ]
    );
    assert_eq!(records[0]["payload"], body);
    assert_eq!(records[1]["payload"], harness.upstream_json().await[0]);
    assert_eq!(records[2]["payload"], answer());
    assert_eq!(records[3]["payload"], reply.json());
    assert_eq!(
        records[0]["kompressor_request_id"],
        reply.header("x-request-id"),
        "the id the client was told, when the proxy chose it"
    );
    assert_eq!(records[1]["llm_call_id"], records[2]["llm_call_id"]);
    assert_eq!(records[1]["proxy_turn"], 0);
    assert!(records[1].get("recursion").is_none());
    assert_eq!(records[0]["model"], "gpt-5");
    assert_eq!(records[2]["model"], "gpt-5-2026-08-01");
    for record in &records {
        assert_eq!(
            record["kompressor_request_id"],
            records[0]["kompressor_request_id"]
        );
        assert_eq!(record["client_body_hash"], records[0]["client_body_hash"]);
        assert_eq!(record["trajectory_id"], "traj_1");
        assert_eq!(record["group_id"], "grp_1");
    }

    let forwarded = harness.upstream_headers().await;
    assert!(
        forwarded
            .keys()
            .all(|name| !name.as_str().starts_with("x-induction-")),
        "the harness's headers are not the provider's business: {forwarded:?}"
    );
}

#[tokio::test]
async fn a_retrieve_round_the_client_never_sees_is_logged_as_a_second_call() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    mount_retrieve(
        &harness,
        false,
        ResponseTemplate::new(200).set_body_json(answers_after_retrieving()),
    )
    .await;

    let reply = harness
        .post_with(
            "/v1/responses",
            &responses_body(&log_payload(400)),
            &HARNESS_HEADERS,
        )
        .await;
    let records = armed.answered(1).await;

    assert_eq!(
        names(&records),
        [
            "payload_from_client",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_client"
        ]
    );
    let request_id = records[0]["kompressor_request_id"].as_str().unwrap();
    assert_eq!(records[1]["llm_call_id"], format!("{request_id}:1"));
    assert_eq!(records[2]["llm_call_id"], format!("{request_id}:1"));
    assert_eq!(records[3]["llm_call_id"], format!("{request_id}:2"));
    assert_eq!(records[4]["llm_call_id"], format!("{request_id}:2"));
    assert_eq!(records[3]["proxy_turn"], 1);
    assert_eq!(records[3]["recursion"], "retrieve");
    assert_eq!(records[4]["recursion"], "retrieve");
    assert!(records[2].get("recursion").is_none());

    let sent = harness.upstream_json().await;
    assert_eq!(records[1]["payload"], sent[0]);
    assert_eq!(records[3]["payload"], sent[1]);
    assert_eq!(records[2]["payload"]["output"][0]["call_id"], "call_r1");
    assert!(
        records[3]["payload"]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "function_call_output" && item["call_id"] == "call_r1"),
        "the re-send answers the call the first reply made"
    );
    assert_eq!(records[4]["payload"], answers_after_retrieving());
    assert_eq!(records[5]["payload"], reply.json());
    assert_eq!(
        records[5]["payload"]["usage"]["input_tokens"], 2200,
        "the client is told what both rounds cost"
    );
}

#[tokio::test]
async fn a_streamed_retrieve_round_is_logged_as_the_responses_each_round_ended_on() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    mount_retrieve(&harness, true, sse(STREAM_ANSWERS)).await;

    let reply = harness
        .post_with(
            "/v1/responses",
            &streaming(responses_body(&log_payload(400))),
            &HARNESS_HEADERS,
        )
        .await;
    let records = armed.answered(1).await;

    assert_eq!(
        names(&records),
        [
            "payload_from_client",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_client"
        ]
    );
    assert_eq!(
        records[2]["payload"],
        json!({"id": "resp_1", "usage": {"input_tokens": 1000, "output_tokens": 20}}),
        "the round the relay swallowed is logged too"
    );
    assert_eq!(records[3]["recursion"], "retrieve");
    assert_eq!(
        records[4]["payload"],
        json!({"id": "resp_2", "usage": {"input_tokens": 1200, "output_tokens": 30}})
    );
    assert_eq!(records[5]["payload"], response_in_stream(&reply.text()));
    assert_eq!(records[5]["payload"]["id"], "resp_2");
}

#[tokio::test]
async fn a_client_reusing_its_request_id_on_a_retry_is_logged_as_two_requests() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(answer()))
        .mount(&harness.server)
        .await;
    let body = responses_body("the build is red");
    let headers = [("x-request-id", "client-1")];

    let first = harness.post_with("/v1/responses", &body, &headers).await;
    let second = harness.post_with("/v1/responses", &body, &headers).await;
    let records = armed.answered(2).await;

    let clients: Vec<&Value> = records
        .iter()
        .filter(|record| record["name"] == "payload_from_client")
        .collect();
    assert_eq!(clients.len(), 2);
    assert_ne!(
        clients[0]["kompressor_request_id"],
        clients[1]["kompressor_request_id"]
    );
    assert!(
        clients
            .iter()
            .all(|client| client["kompressor_request_id"] != "client-1"),
        "{clients:#?}"
    );
    assert_eq!(
        clients[0]["client_body_hash"],
        clients[1]["client_body_hash"]
    );
    assert_eq!(
        (first.header("x-request-id"), second.header("x-request-id")),
        ("client-1".to_string(), "client-1".to_string()),
        "the client's own id still comes back to it"
    );
}

#[tokio::test]
async fn a_compressed_request_is_logged_as_the_json_it_carries() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(answer()))
        .mount(&harness.server)
        .await;
    let body = responses_body(&log_payload(400));

    harness.post_zstd("/v1/responses", &body).await;
    let records = armed.answered(1).await;

    assert_eq!(records[0]["payload"], body);
    let wire = &harness.upstream_raw_bodies().await[0];
    let sent: Value =
        serde_json::from_slice(&zstd::stream::decode_all(&wire[..]).unwrap()).unwrap();
    assert_eq!(records[1]["payload"], sent);
}

#[tokio::test]
async fn a_body_the_proxy_cannot_decode_is_logged_as_null_rather_than_as_binary() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(answer()))
        .mount(&harness.server)
        .await;
    let encoded =
        zstd::stream::encode_all(responses_body(&log_payload(400)).to_string().as_bytes(), 3)
            .unwrap();
    let truncated = encoded[..encoded.len() / 2].to_vec();

    let reply = harness
        .send(
            http::Request::post("/v1/responses")
                .header("content-type", "application/json")
                .header("content-encoding", "zstd")
                .body(axum::body::Body::from(truncated))
                .unwrap(),
        )
        .await;
    let records = armed.answered(1).await;

    assert_eq!(reply.status, 200, "forwarded as it arrived");
    assert_eq!(
        names(&records),
        [
            "payload_from_client",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_client"
        ]
    );
    assert_eq!(records[0]["payload"], Value::Null);
    assert_eq!(records[1]["payload"], Value::Null);
    assert_eq!(records[3]["payload"], reply.json());
}

#[tokio::test]
async fn an_unreachable_provider_leaves_the_call_unanswered_and_logs_the_502() {
    for stream in [false, true] {
        let armed = Armed::new();
        // Nothing listens on the discard port.
        let harness = Harness::with(|config| {
            config.proxy.upstreams[0].base_url = "http://127.0.0.1:9".to_string()
        })
        .await;
        let mut body = responses_body("the build is red");
        if stream {
            body = streaming(body);
        }

        let reply = harness.post("/v1/responses", &body).await;
        let records = armed.answered(1).await;

        assert_eq!(reply.status, 502, "stream: {stream}");
        assert_eq!(
            names(&records),
            ["payload_from_client", "payload_to_llm", "payload_to_client"],
            "stream: {stream}"
        );
        assert_eq!(records[2]["payload"], reply.json(), "stream: {stream}");
    }
}

#[tokio::test]
async fn a_provider_error_is_logged_as_the_provider_wrote_it() {
    let refusal = json!({"error": {"message": "rate limited", "type": "rate_limit_error"}});
    for stream in [false, true] {
        let armed = Armed::new();
        let harness = Harness::new().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(429).set_body_json(refusal.clone()))
            .mount(&harness.server)
            .await;
        let mut body = responses_body("the build is red");
        if stream {
            body = streaming(body);
        }

        let reply = harness.post("/v1/responses", &body).await;
        let records = armed.answered(1).await;

        assert_eq!(reply.status, 429, "stream: {stream}");
        assert_eq!(
            names(&records),
            [
                "payload_from_client",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_client"
            ],
            "stream: {stream}"
        );
        assert_eq!(records[2]["payload"], refusal, "stream: {stream}");
        assert_eq!(records[3]["payload"], refusal, "stream: {stream}");
    }
}

#[tokio::test]
async fn a_failed_retrieve_round_is_logged_with_the_providers_answer() {
    let outage = json!({"error": {"message": "overloaded"}});
    for stream in [false, true] {
        let armed = Armed::new();
        let harness = Harness::new().await;
        mount_retrieve(
            &harness,
            stream,
            ResponseTemplate::new(503).set_body_json(outage.clone()),
        )
        .await;
        let mut body = responses_body(&log_payload(400));
        if stream {
            body = streaming(body);
        }

        let reply = harness.post("/v1/responses", &body).await;
        let records = armed.answered(1).await;

        assert_eq!(
            names(&records),
            [
                "payload_from_client",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_client"
            ],
            "stream: {stream}"
        );
        assert_eq!(records[4]["payload"], outage, "stream: {stream}");
        if stream {
            assert_eq!(
                records[5]["payload"]["code"],
                "retrieve_round_failed",
                "the client was told the loop failed: {}",
                reply.text()
            );
        } else {
            assert_eq!(records[5]["payload"], reply.json());
        }
    }
}

#[tokio::test]
async fn a_client_that_hangs_up_mid_stream_still_has_its_side_logged_last() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(sse(STREAM_ANSWERS).set_delay(Duration::from_millis(200)))
        .mount(&harness.server)
        .await;

    harness
        .post_and_abandon(
            "/v1/responses",
            &streaming(responses_body("the build is red")),
        )
        .await;
    let records = armed.answered(1).await;

    assert_eq!(names(&records).last(), Some(&"payload_to_client"));
    assert_eq!(
        names(&records)[..2],
        ["payload_from_client", "payload_to_llm"]
    );
}

#[tokio::test]
async fn a_client_that_leaves_before_its_answer_starts_still_ends_its_request() {
    for stream in [false, true] {
        let armed = Armed::new();
        let harness = Harness::new().await;
        Mock::given(any())
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(answer())
                    .set_delay(Duration::from_secs(2)),
            )
            .mount(&harness.server)
            .await;
        let mut body = responses_body("the build is red");
        if stream {
            body = streaming(body);
        }
        let request = http::Request::post("/v1/responses")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap();

        // Dropping the request mid-call is what a client hanging up does to it.
        let gave_up = tokio::time::timeout(
            Duration::from_millis(200),
            harness.router().oneshot(request),
        )
        .await;
        assert!(gave_up.is_err(), "stream: {stream}");
        let records = armed.answered(1).await;

        assert_eq!(
            names(&records),
            ["payload_from_client", "payload_to_llm", "payload_to_client"],
            "stream: {stream}"
        );
        assert_eq!(
            records[2]["payload"],
            json!({"raw": ""}),
            "stream: {stream}"
        );
    }
}

#[tokio::test]
async fn a_retrieve_round_whose_error_never_finishes_does_not_hold_the_client() {
    let armed = Armed::new();
    // Asks to retrieve, then fails the round with headers and a body that
    // never arrives.
    let provider = axum::Router::new().fallback(|body: bytes::Bytes| async move {
        let text = String::from_utf8_lossy(&body).into_owned();
        if round_of(&text) > 0 {
            let never = futures_util::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
            return http::Response::builder()
                .status(503)
                .header("content-type", "application/json")
                .body(axum::body::Body::from_stream(never))
                .unwrap();
        }
        let handle = first_handle(&text).expect("the outgoing prompt carries a handle");
        http::Response::builder()
            .header("content-type", "text/event-stream")
            .body(axum::body::Body::from(
                STREAM_ASKS.replace("HANDLE", &handle),
            ))
            .unwrap()
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, provider).await });
    let harness = Harness::with(|config| {
        config.proxy.upstreams[0].base_url = format!("http://{address}");
        config.proxy.stream_keepalive = Duration::from_millis(100);
    })
    .await;

    let reply = tokio::time::timeout(
        Duration::from_secs(5),
        harness.post(
            "/v1/responses",
            &streaming(responses_body(&log_payload(400))),
        ),
    )
    .await
    .expect("the client is told the round failed without waiting on its body");
    let records = armed.answered(1).await;

    assert!(
        reply.text().contains("retrieve_round_failed"),
        "{}",
        reply.text()
    );
    assert_eq!(
        names(&records),
        [
            "payload_from_client",
            "payload_to_llm",
            "payload_from_llm",
            "payload_to_llm",
            "payload_to_client"
        ],
        "a reply that never finished arriving is not logged"
    );
}

#[tokio::test]
async fn requests_in_flight_together_each_keep_their_records_in_order() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(move |request: &Request| {
            let streamed = body_text(request).contains("\"stream\":true");
            if round_of(&body_text(request)) > 0 {
                if streamed {
                    return sse(STREAM_ANSWERS);
                }
                return ResponseTemplate::new(200).set_body_json(answers_after_retrieving());
            }
            if streamed {
                sse(&STREAM_ASKS.replace("HANDLE", &handle_in(request)))
            } else {
                ResponseTemplate::new(200).set_body_json(asks_to_retrieve(&handle_in(request)))
            }
        })
        .mount(&harness.server)
        .await;

    let requests = (0..8).map(|n| {
        let mut body = responses_body(&log_payload(400 + n));
        if n % 2 == 1 {
            body = streaming(body);
        }
        let harness = &harness;
        async move { harness.post("/v1/responses", &body).await }
    });
    futures_util::future::join_all(requests).await;
    let records = armed.answered(8).await;

    let mut by_request: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
    for record in &records {
        by_request
            .entry(record["kompressor_request_id"].as_str().unwrap())
            .or_default()
            .push(record["name"].as_str().unwrap());
    }
    assert_eq!(by_request.len(), 8);
    for (request, names) in by_request {
        assert_eq!(
            names,
            [
                "payload_from_client",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_client"
            ],
            "{request}"
        );
    }
}

#[tokio::test]
async fn a_request_outside_the_responses_api_is_not_logged() {
    let armed = Armed::new();
    let harness = Harness::new().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_json(answer()))
        .mount(&harness.server)
        .await;

    harness
        .post(
            "/v1/chat/completions",
            &support::chat_body("the build is red"),
        )
        .await;
    harness
        .send(
            http::Request::get("/v1/responses")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await;

    assert!(armed.records().is_empty(), "{:#?}", armed.records());
}
