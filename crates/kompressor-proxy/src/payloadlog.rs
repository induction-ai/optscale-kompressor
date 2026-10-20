//! The Induction payload log: each OpenAI Responses request as it looked at
//! every hop, for the Induction benchmark harness to price.
//!
//! Off unless `KOMPRESSOR_INDUCTION_LOG_FILE` names a file. When on, every HTTP
//! POST on the Responses route appends one JSON line per event: the body the
//! client sent, each upstream call and its reply — the retrieve rounds the
//! client never sees included — and what the client received. The records
//! follow the contract of the Headroom fork's log, which the harness reads in
//! `third_party_payload_logs.ts`; `fixtures/headroom_fork_payloads.jsonl` is
//! that fork's output.
//!
//! Best-effort, like `bodycapture`: a record that cannot be written is lost,
//! never the request.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use http::HeaderMap;
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};

use crate::bodycapture;
use crate::sse::{SseDecoder, SseFrame};

const TRAJECTORY_ID: &str = "x-induction-trajectory-id";
const GROUP_ID: &str = "x-induction-group-id";

/// Past this size what the client received is logged as `null`, as the
/// Headroom fork does.
const CLIENT_PAYLOAD_LIMIT: usize = 10 * 1024 * 1024;

/// Where to write, when `KOMPRESSOR_INDUCTION_LOG_FILE` names a file.
pub(crate) fn path() -> Option<PathBuf> {
    std::env::var_os("KOMPRESSOR_INDUCTION_LOG_FILE")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

/// One client request's records.
pub(crate) struct RequestLog {
    path: PathBuf,
    request_id: String,
    client_body_hash: String,
    trajectory_id: Option<String>,
    group_id: Option<String>,
    /// The client body's model, for records whose payload names none.
    model: String,
    /// Upstream calls sent so far. They go out one at a time, so a reply
    /// always answers the latest.
    calls: AtomicU32,
    answered: AtomicBool,
}

impl RequestLog {
    /// Start a request's log by writing the body the client sent.
    ///
    /// `request_id` has to be unique per proxied request: the harness groups
    /// records by it, so a value the client chose and reused on a retry would
    /// merge two attempts into one.
    pub(crate) fn begin(
        path: PathBuf,
        request_id: String,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Arc<RequestLog> {
        let payload = payload_from_bytes(body, None);
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        };
        let log = Arc::new(RequestLog {
            path,
            request_id,
            client_body_hash: client_body_hash(&payload),
            trajectory_id: header(TRAJECTORY_ID),
            group_id: header(GROUP_ID),
            model: payload
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            calls: AtomicU32::new(0),
            answered: AtomicBool::new(false),
        });
        log.write("payload_from_client", None, payload);
        log
    }

    /// Record an upstream call about to go out, given its body before any
    /// content coding.
    pub(crate) fn payload_to_llm(&self, body: &[u8]) {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        self.write("payload_to_llm", Some(call), payload_from_bytes(body, None));
    }

    /// Record the reply to the latest upstream call.
    pub(crate) fn payload_from_llm(&self, payload: Value) {
        let call = self.calls.load(Ordering::Relaxed);
        self.write("payload_from_llm", Some(call), payload);
    }

    pub(crate) fn payload_from_llm_body(&self, body: &[u8], headers: &HeaderMap) {
        self.payload_from_llm(payload_from_bytes(body, content_type(headers)));
    }

    pub(crate) fn payload_to_client(&self, payload: Value) {
        self.answered.store(true, Ordering::Relaxed);
        self.write("payload_to_client", None, payload);
    }

    pub(crate) fn payload_to_client_body(&self, body: &[u8], headers: &HeaderMap) {
        self.payload_to_client(client_payload(body, content_type(headers)));
    }

    /// One record, its fields in the order the Headroom fork writes them.
    fn write(&self, name: &str, call: Option<u32>, payload: Value) {
        // The harness orders records by time, then event id. Taking the time
        // from the id keeps the two in step when requests race for the id.
        let event_id = uuid::Uuid::now_v7();
        let (secs, nanos) = event_id
            .get_timestamp()
            .map_or((0, 0), |stamp| stamp.to_unix());
        let now = UNIX_EPOCH + Duration::new(secs, nanos);
        let model = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&self.model)
            .to_string();
        let mut record = Map::new();
        record.insert("level".to_string(), json!(20));
        record.insert(
            "time".to_string(),
            json!(humantime::format_rfc3339_millis(now).to_string()),
        );
        record.insert("module".to_string(), json!("openai_responses"));
        record.insert("name".to_string(), json!(name));
        record.insert("event_id".to_string(), json!(event_id.to_string()));
        record.insert("kompressor_request_id".to_string(), json!(self.request_id));
        if let Some(call) = call {
            record.insert(
                "llm_call_id".to_string(),
                json!(format!("{}:{call}", self.request_id)),
            );
            record.insert("proxy_turn".to_string(), json!(call.saturating_sub(1)));
            if call > 1 {
                record.insert("recursion".to_string(), json!("retrieve"));
            }
        }
        record.insert("client_body_hash".to_string(), json!(self.client_body_hash));
        if let Some(trajectory_id) = &self.trajectory_id {
            record.insert("trajectory_id".to_string(), json!(trajectory_id));
        }
        if let Some(group_id) = &self.group_id {
            record.insert("group_id".to_string(), json!(group_id));
        }
        record.insert("provider".to_string(), json!("openai"));
        record.insert("model".to_string(), json!(model));
        record.insert("payload".to_string(), payload);
        bodycapture::append_line(&self.path, &Value::Object(record).to_string());
    }
}

/// A client that hangs up before its answer starts takes the handler down with
/// it, so the request still ends on what reached that client: nothing.
impl Drop for RequestLog {
    fn drop(&mut self) {
        if !*self.answered.get_mut() {
            self.payload_to_client(client_payload(&[], None));
        }
    }
}

/// What a streamed reply put in front of the client.
///
/// Shared by the relay task and the client's body stream, and written when the
/// last of them lets go. Only then has the relay logged its final upstream
/// reply, so the record is the request's last, and only then has every byte the
/// client will get passed through here, a stream it walked away from included.
pub(crate) struct StreamToClient {
    log: Arc<RequestLog>,
    /// `None` once the stream has passed `CLIENT_PAYLOAD_LIMIT`.
    seen: Mutex<Option<Vec<u8>>>,
}

impl StreamToClient {
    pub(crate) fn new(log: Arc<RequestLog>) -> Arc<StreamToClient> {
        Arc::new(StreamToClient {
            log,
            seen: Mutex::new(Some(Vec::new())),
        })
    }

    pub(crate) fn push(&self, chunk: &[u8]) {
        let Ok(mut seen) = self.seen.lock() else {
            return;
        };
        match seen.as_mut() {
            Some(bytes) if bytes.len() + chunk.len() <= CLIENT_PAYLOAD_LIMIT => {
                bytes.extend_from_slice(chunk);
            }
            Some(_) => *seen = None,
            None => {}
        }
    }
}

impl Drop for StreamToClient {
    fn drop(&mut self) {
        let seen = self.seen.get_mut().ok().and_then(Option::take);
        let payload = match seen {
            Some(bytes) => payload_from_bytes(&bytes, Some("text/event-stream")),
            None => Value::Null,
        };
        self.log.payload_to_client(payload);
    }
}

fn content_type(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
}

/// A body the client received, as the log records it.
fn client_payload(body: &[u8], content_type: Option<&str>) -> Value {
    if body.len() > CLIENT_PAYLOAD_LIMIT {
        return Value::Null;
    }
    payload_from_bytes(body, content_type)
}

/// A body as the log records it: an event stream as its final Response, JSON
/// as itself, any other JSON value as `{"value": …}` and other text as
/// `{"raw": …}`. Bytes that are not text at all, such as a body still in a
/// content coding this proxy could not undo, are `null` rather than mangled.
fn payload_from_bytes(body: &[u8], content_type: Option<&str>) -> Value {
    let Ok(text) = std::str::from_utf8(body) else {
        return Value::Null;
    };
    let opening = text.trim_start();
    let event_stream = content_type
        .is_some_and(|value| value.to_ascii_lowercase().contains("text/event-stream"))
        || opening.starts_with("event:")
        || opening.starts_with("data:");
    if event_stream && let Some(response) = final_response(body) {
        return response;
    }
    match serde_json::from_str::<Value>(text) {
        Ok(value @ (Value::Object(_) | Value::Array(_))) => value,
        Ok(value) => json!({"value": value}),
        Err(_) => json!({"raw": text}),
    }
}

fn final_response(body: &[u8]) -> Option<Value> {
    let mut decoder = SseDecoder::new();
    decoder.push(body);
    let mut response = FinalResponse::default();
    while let Some(frame) = decoder.next_frame() {
        response.observe(&frame);
    }
    if let Some(frame) = decoder.flush() {
        response.observe(&frame);
    }
    response.into_payload()
}

/// The Response a stream ended on: the `response` of its last
/// `response.completed`, `.incomplete` or `.failed` event, or failing that
/// whichever came last of another event's `response` and an `error` event.
#[derive(Default)]
pub(crate) struct FinalResponse {
    terminal: Option<Value>,
    fallback: Option<Value>,
}

impl FinalResponse {
    pub(crate) fn observe(&mut self, frame: &SseFrame) {
        // Only an event with a `response` or of type `error` is kept, and
        // providers write both names plainly, so text deltas skip the parse.
        let data = frame.data();
        if !data.contains("\"response\"") && !data.contains("\"error\"") {
            return;
        }
        let Some(mut event) = frame.json().filter(Value::is_object) else {
            return;
        };
        let kind = event.get("type").and_then(Value::as_str);
        let terminal = matches!(
            kind,
            Some("response.completed" | "response.incomplete" | "response.failed")
        );
        let error = kind == Some("error");
        match event
            .get_mut("response")
            .filter(|response| response.is_object())
            .map(Value::take)
        {
            Some(response) if terminal => self.terminal = Some(response),
            Some(response) => self.fallback = Some(response),
            None if error => self.fallback = Some(event),
            None => {}
        }
    }

    pub(crate) fn into_payload(self) -> Option<Value> {
        self.terminal.or(self.fallback)
    }
}

/// `sha256:` and the hex digest of the payload's canonical text, exactly as the
/// harness computes it from its own log with `canonicalJson`. A hash that
/// differs drops the request from the harness's fold without a word, so this
/// follows JavaScript rather than serde: keys sorted by UTF-16 code unit, and
/// numbers printed the way `JSON.stringify` prints them after `JSON.parse`.
fn client_body_hash(payload: &Value) -> String {
    let mut text = String::new();
    canonical_json(payload, &mut text);
    let hex: String = Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("sha256:{hex}")
}

fn canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.encode_utf16().cmp(right.encode_utf16()));
            out.push('{');
            for (position, (key, item)) in entries.into_iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                out.push_str(&json_string(key));
                out.push(':');
                canonical_json(item, out);
            }
            out.push('}');
        }
        Value::String(text) => out.push_str(&json_string(text)),
        Value::Number(number) => out.push_str(&js_number(number)),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Null => out.push_str("null"),
    }
}

/// Escaped as `JSON.stringify` escapes a string, which is how serde does too.
fn json_string(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_default()
}

/// A number as JavaScript prints it once `JSON.parse` has made it a double.
///
/// The workspace keeps each number's literal text (`arbitrary_precision`), so
/// `1.0`, `1e5` and `12345678901234567890` would otherwise be hashed as
/// written. Rust's `{:e}` already gives the shortest digits that round-trip;
/// this places them by the rules of ECMAScript's `Number::toString`.
fn js_number(number: &Number) -> String {
    // `None` only for a literal past the double range, which JavaScript reads
    // as an infinity and `JSON.stringify` writes as `null`.
    let Some(value) = number.as_f64() else {
        return "null".to_string();
    };
    if value == 0.0 {
        return "0".to_string();
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let scientific = format!("{:e}", value.abs());
    let Some((mantissa, exponent)) = scientific.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return scientific;
    };
    // The spec's k and n: the k digits d1..dk, and the value is 0.d1..dk × 10^n.
    let n = exponent + 1;
    let digits = even_on_tie(
        mantissa.chars().filter(|c| *c != '.').collect(),
        n,
        value.abs(),
    );
    let k = digits.len() as i32;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        let (whole, fraction) = digits.split_at(n as usize);
        format!("{whole}.{fraction}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat(n.unsigned_abs() as usize))
    } else {
        let exponent_sign = if n > 0 { "+" } else { "-" };
        let magnitude = (n - 1).unsigned_abs();
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            format!("{first}e{exponent_sign}{magnitude}")
        } else {
            format!("{first}.{rest}e{exponent_sign}{magnitude}")
        }
    };
    format!("{sign}{body}")
}

/// Of two shortest digit strings exactly as close to `value`, the even one.
///
/// That is ECMAScript's choice, and node's; Rust's `{:e}` takes the upper one.
/// Over 300,000 random doubles checked against node, the 63 that printed
/// differently were all such ties, `260240775020034.125` among them.
fn even_on_tie(digits: String, n: i32, value: f64) -> String {
    let Some(last) = digits.bytes().last().map(|byte| byte - b'0') else {
        return digits;
    };
    if last % 2 == 0 || (digits.len() == 1 && last == 1) {
        return digits;
    }
    let lower = format!("{}{}", &digits[..digits.len() - 1], last - 1);
    let k = digits.len() as i32;
    if format!("{lower}e{}", n - k).parse::<f64>() != Ok(value) {
        return digits;
    }
    // Both read back as `value`, and Rust chose the upper, so the upper is the
    // closer unless `value` is exactly halfway: its full expansion is `lower`
    // followed by a 5. No double needs more than 767 digits to write out.
    let exact = format!("{value:.800e}");
    let Some((mantissa, exponent)) = exact.split_once('e') else {
        return digits;
    };
    let expansion: String = mantissa.chars().filter(|c| *c != '.').collect();
    if exponent.parse::<i32>() == Ok(n - 1)
        && expansion.trim_end_matches('0') == format!("{lower}5")
    {
        lower
    } else {
        digits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_of(body: &str) -> (String, String) {
        let payload: Value = serde_json::from_str(body).expect("a JSON body");
        let mut text = String::new();
        canonical_json(&payload, &mut text);
        (text, client_body_hash(&payload))
    }

    /// Computed by running the harness's own `canonicalJson` in node 24.
    #[test]
    fn a_body_hashes_as_the_harness_hashes_it() {
        let vectors = [
            (
                r#"{"model":"gpt-5.4","temperature":1.0,"top_p":0.5,"n":1e5,"tiny":1e-7,"small":0.000001,"big":1e21,"neg":-0.0,"huge":12345678901234567890}"#,
                r#"{"big":1e+21,"huge":12345678901234567000,"model":"gpt-5.4","n":100000,"neg":0,"small":0.000001,"temperature":1,"tiny":1e-7,"top_p":0.5}"#,
                "sha256:93250fd8d04c1361720871ceea20a6e65b080e49a36c9797228e9f3ccccef79f",
            ),
            // U+1F600 sorts before U+FF21 by UTF-16 code unit, after it by code
            // point.
            (
                r#"{"b":{"z":1,"a":[2,{"y":"x","é":true}]},"a":null,"😀":1,"Ａ":2}"#,
                "{\"a\":null,\"b\":{\"a\":[2,{\"y\":\"x\",\"\u{e9}\":true}],\"z\":1},\"\u{1f600}\":1,\"\u{ff21}\":2}",
                "sha256:bcb2aff50a3d853f5e64db088d5b8ce90c8f0d8d478ab6f0c095ef82954d7558",
            ),
            (
                r#"{"s":"line\nbreak\ttab \u0001 \u001f \" \\ /   \u007f café 😀"}"#,
                "{\"s\":\"line\\nbreak\\ttab \\u0001 \\u001f \\\" \\\\ / \u{2028} \u{7f} caf\u{e9} \u{1f600}\"}",
                "sha256:45221eb1d02fca05d7f3ee750402e71934d49509e527a07a8f7018bb9e33509f",
            ),
        ];
        for (body, canonical, hash) in vectors {
            assert_eq!(hash_of(body), (canonical.to_string(), hash.to_string()));
        }
    }

    #[test]
    fn every_record_of_the_headroom_fixture_carries_its_client_bodys_hash() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/headroom_fork_payloads.jsonl");
        let records: Vec<Value> = std::fs::read_to_string(path)
            .expect("the fixture")
            .lines()
            .map(|line| serde_json::from_str(line).expect("a JSON record"))
            .collect();
        let bodies: std::collections::HashMap<&str, &Value> = records
            .iter()
            .filter(|record| record["name"] == "payload_from_client")
            .map(|record| {
                (
                    record["headroom_request_id"].as_str().unwrap(),
                    &record["payload"],
                )
            })
            .collect();
        assert_eq!(bodies.len(), 3);
        for record in &records {
            let body = bodies[record["headroom_request_id"].as_str().unwrap()];
            assert_eq!(
                client_body_hash(body),
                record["client_body_hash"],
                "{}",
                record["name"]
            );
        }
    }

    /// Expectations from `JSON.stringify(JSON.parse(literal))` in node 24.
    #[test]
    fn a_number_prints_as_javascript_prints_it() {
        let cases = [
            ("1.0", "1"),
            ("100.0", "100"),
            ("1E2", "100"),
            ("1e5", "100000"),
            ("-42", "-42"),
            ("-0", "0"),
            ("-0.0", "0"),
            ("0.1", "0.1"),
            ("-0.5", "-0.5"),
            ("4.35", "4.35"),
            ("123.456", "123.456"),
            ("0.30000000000000004", "0.30000000000000004"),
            ("260240775020034.125", "260240775020034.12"),
            ("19874369315706.5625", "19874369315706.562"),
            ("-19874369315706.5625", "-19874369315706.562"),
            ("0.00001", "0.00001"),
            ("1e-6", "0.000001"),
            ("2.5e-6", "0.0000025"),
            ("1e-7", "1e-7"),
            ("0.0000001234", "1.234e-7"),
            ("-1.5e-10", "-1.5e-10"),
            ("123e-20", "1.23e-18"),
            ("5e-324", "5e-324"),
            ("1e-400", "0"),
            ("1e20", "100000000000000000000"),
            ("1.5e20", "150000000000000000000"),
            ("1e21", "1e+21"),
            ("1e300", "1e+300"),
            ("1.7976931348623157e308", "1.7976931348623157e+308"),
            ("9007199254740993", "9007199254740992"),
            ("12345678901234567890", "12345678901234567000"),
            ("1e400", "null"),
            ("-1e400", "null"),
        ];
        for (literal, expected) in cases {
            let Value::Number(number) = serde_json::from_str(literal).expect(literal) else {
                panic!("{literal} is not a number");
            };
            assert_eq!(js_number(&number), expected, "{literal}");
        }
    }

    #[test]
    fn a_streamed_reply_is_logged_as_the_response_it_ended_on() {
        let stream = concat!(
            ": kompressor retrieving\n\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.completed\r\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":5}}}\r\n\r\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            payload_from_bytes(stream.as_bytes(), Some("text/event-stream; charset=utf-8")),
            json!({"id": "resp_1", "status": "completed", "usage": {"input_tokens": 5}})
        );
    }

    #[test]
    fn a_stream_that_never_finished_is_logged_as_the_last_response_it_carried() {
        let stream = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n",
            "data: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"seq\":2}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}",
        );
        assert_eq!(
            payload_from_bytes(stream.as_bytes(), Some("text/event-stream")),
            json!({"id": "resp_1", "status": "in_progress", "seq": 2})
        );
    }

    #[test]
    fn a_completed_response_outranks_a_later_event_carrying_another() {
        let stream = concat!(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\"}}\n\n",
        );
        assert_eq!(
            payload_from_bytes(stream.as_bytes(), Some("text/event-stream")),
            json!({"id": "resp_1"})
        );
    }

    #[test]
    fn a_stream_that_ended_in_an_error_is_logged_as_that_error() {
        let stream =
            "event: error\ndata: {\"type\":\"error\",\"code\":\"retrieve_round_failed\"}\n\n";
        assert_eq!(
            payload_from_bytes(stream.as_bytes(), None),
            json!({"type": "error", "code": "retrieve_round_failed"}),
            "recognised as a stream by its first line when no content type says so"
        );
    }

    #[test]
    fn a_body_that_is_not_an_event_stream_is_logged_as_json_or_as_text() {
        assert_eq!(
            payload_from_bytes(
                br#"{"error":{"message":"rate limited"}}"#,
                Some("application/json")
            ),
            json!({"error": {"message": "rate limited"}})
        );
        assert_eq!(payload_from_bytes(b"[1,2]", None), json!([1, 2]));
        assert_eq!(payload_from_bytes(b"42", None), json!({"value": 42}));
        assert_eq!(payload_from_bytes(b"\"ok\"", None), json!({"value": "ok"}));
        assert_eq!(
            payload_from_bytes(b"Bad Gateway", Some("text/plain")),
            json!({"raw": "Bad Gateway"})
        );
        assert_eq!(payload_from_bytes(b"", None), json!({"raw": ""}));
        assert_eq!(
            payload_from_bytes(b": only a keepalive\n\n", Some("text/event-stream")),
            json!({"raw": ": only a keepalive\n\n"}),
            "a stream with no event in it is kept as the text it was"
        );
    }

    #[test]
    fn bytes_that_are_not_text_are_logged_as_null() {
        let compressed = zstd::stream::encode_all(&br#"{"model":"gpt-5"}"#[..], 3).unwrap();
        assert_eq!(payload_from_bytes(&compressed, None), Value::Null);
        assert_eq!(
            payload_from_bytes(&compressed, Some("text/event-stream")),
            Value::Null
        );
        assert_eq!(
            payload_from_bytes("{\"text\":\"caf\u{e9}\"}".as_bytes(), None),
            json!({"text": "caf\u{e9}"}),
            "text beyond ASCII is still text"
        );
    }

    #[test]
    fn what_the_client_received_past_ten_mebibytes_is_logged_as_null() {
        let mut body = br#"{"pad":""#.to_vec();
        body.extend(std::iter::repeat_n(b'x', CLIENT_PAYLOAD_LIMIT));
        body.extend(br#""}"#);
        assert_eq!(client_payload(&body, Some("application/json")), Value::Null);
        assert_eq!(
            client_payload(br#"{"id":"resp_1"}"#, None),
            json!({"id": "resp_1"})
        );
    }

    #[test]
    fn what_a_streaming_client_received_past_ten_mebibytes_is_logged_as_null() {
        fn logged(chunks: &[&[u8]]) -> Value {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("payloads.jsonl");
            let log = RequestLog::begin(path.clone(), "kp-1".to_string(), &HeaderMap::new(), b"{}");
            let to_client = StreamToClient::new(log);
            for chunk in chunks {
                to_client.push(chunk);
            }
            drop(to_client);
            let record = written(&path).pop().unwrap();
            assert_eq!(record["name"], "payload_to_client");
            record["payload"].clone()
        }
        fn comment(len: usize) -> Vec<u8> {
            let mut frame = b":".to_vec();
            frame.resize(len - 2, b' ');
            frame.extend(b"\n\n");
            frame
        }
        let completed: &[u8] =
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n";
        let fill = comment(CLIENT_PAYLOAD_LIMIT - completed.len());
        assert_eq!(
            logged(&[&fill, completed]),
            json!({"id": "resp_1"}),
            "exactly ten mebibytes"
        );
        assert_eq!(
            logged(&[&fill, completed, b"\n"]),
            Value::Null,
            "one byte more"
        );
        assert_eq!(
            logged(&[&comment(CLIENT_PAYLOAD_LIMIT + 1), completed]),
            Value::Null,
            "a frame past the limit on its own, then one that would fit"
        );
    }

    #[test]
    fn a_record_is_stamped_in_utc_to_the_millisecond_its_event_id_carries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads.jsonl");
        RequestLog::begin(path.clone(), "kp-1".to_string(), &HeaderMap::new(), b"{}")
            .payload_to_client(Value::Null);

        for record in written(&path) {
            let time = record["time"].as_str().unwrap();
            let template = "0000-00-00T00:00:00.000Z";
            assert!(
                time.len() == template.len()
                    && time.chars().zip(template.chars()).all(|(c, t)| match t {
                        '0' => c.is_ascii_digit(),
                        _ => c == t,
                    }),
                "{time}"
            );
            let stamped = humantime::parse_rfc3339(time)
                .unwrap()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();
            let event_id = record["event_id"].as_str().unwrap().replace('-', "");
            assert_eq!(
                u128::from_str_radix(&event_id[..12], 16).unwrap(),
                stamped,
                "{record}"
            );
        }
    }

    #[test]
    fn an_event_id_is_a_version_7_uuid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads.jsonl");
        RequestLog::begin(path.clone(), "kp-1".to_string(), &HeaderMap::new(), b"{}")
            .payload_to_client(Value::Null);

        for record in written(&path) {
            let id = record["event_id"].as_str().unwrap();
            let groups: Vec<&str> = id.split('-').collect();
            assert_eq!(
                groups.iter().map(|group| group.len()).collect::<Vec<_>>(),
                [8, 4, 4, 4, 12],
                "{id}"
            );
            assert!(
                id.chars()
                    .all(|c| c == '-' || c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            );
            assert!(groups[2].starts_with('7'), "version 7: {id}");
            assert!(
                matches!(groups[3].chars().next(), Some('8' | '9' | 'a' | 'b')),
                "variant 10: {id}"
            );
        }
    }

    #[test]
    fn records_sort_by_time_then_event_id_in_the_order_they_were_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads.jsonl");
        let log = RequestLog::begin(path.clone(), "kp-1".to_string(), &HeaderMap::new(), b"{}");
        for _ in 0..500 {
            log.payload_to_llm(b"{}");
            log.payload_from_llm(json!({}));
        }
        log.payload_to_client(Value::Null);

        let stamps: Vec<(String, String)> = written(&path)
            .iter()
            .map(|record| {
                (
                    record["time"].as_str().unwrap().to_string(),
                    record["event_id"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert!(
            stamps.windows(2).any(|pair| pair[0].0 == pair[1].0),
            "some records share a millisecond"
        );
        for pair in stamps.windows(2) {
            assert!(pair[0] < pair[1], "{pair:?}");
        }
    }

    fn written(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn keys(record: &Value) -> Vec<String> {
        record.as_object().unwrap().keys().cloned().collect()
    }

    /// The Headroom fork's records with this fork's names, which is what the
    /// harness reader is given.
    #[test]
    fn records_carry_the_fields_of_the_headroom_fork_in_its_order() {
        let fixture: Vec<Value> = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/headroom_fork_payloads.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        // The fixture's second request: one call, a re-send, then the client.
        let expected: Vec<Vec<String>> = fixture[4..10]
            .iter()
            .map(|record| {
                keys(record)
                    .into_iter()
                    .map(|key| key.replace("headroom_request_id", "kompressor_request_id"))
                    .collect()
            })
            .collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads.jsonl");
        let mut headers = HeaderMap::new();
        headers.insert(TRAJECTORY_ID, "traj_1".parse().unwrap());
        headers.insert(GROUP_ID, "grp_1".parse().unwrap());
        let log = RequestLog::begin(
            path.clone(),
            "kp-1".to_string(),
            &headers,
            br#"{"model":"gpt-5.4","input":[]}"#,
        );
        log.payload_to_llm(br#"{"model":"gpt-5.4","input":[]}"#);
        log.payload_from_llm(json!({"id": "resp_1", "model": "gpt-5.4-2026-08-01"}));
        log.payload_to_llm(br#"{"model":"gpt-5.4","input":[{"type":"function_call_output"}]}"#);
        log.payload_from_llm(json!({"id": "resp_2"}));
        log.payload_to_client(json!({"id": "resp_2"}));

        let records = written(&path);
        assert_eq!(records.iter().map(keys).collect::<Vec<_>>(), expected);
        let names: Vec<&str> = records
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "payload_from_client",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_llm",
                "payload_from_llm",
                "payload_to_client"
            ]
        );
        assert_eq!(records[1]["llm_call_id"], "kp-1:1");
        assert_eq!(records[2]["llm_call_id"], "kp-1:1");
        assert_eq!(records[3]["llm_call_id"], "kp-1:2");
        assert_eq!(records[4]["proxy_turn"], 1);
        assert_eq!(records[4]["recursion"], "retrieve");
        assert_eq!(
            records[2]["model"], "gpt-5.4-2026-08-01",
            "the reply's own model"
        );
        assert_eq!(
            records[4]["model"], "gpt-5.4",
            "the client's, where the reply names none"
        );
        for record in &records {
            assert_eq!(record["level"], 20);
            assert_eq!(record["module"], "openai_responses");
            assert_eq!(record["provider"], "openai");
            assert_eq!(record["kompressor_request_id"], "kp-1");
            assert_eq!(record["client_body_hash"], records[0]["client_body_hash"]);
            assert_eq!(record["trajectory_id"], "traj_1");
            assert_eq!(record["group_id"], "grp_1");
        }
    }

    #[test]
    fn a_request_without_harness_headers_or_a_model_leaves_them_out() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads.jsonl");
        let mut headers = HeaderMap::new();
        headers.insert(TRAJECTORY_ID, "".parse().unwrap());
        RequestLog::begin(path.clone(), "kp-1".to_string(), &headers, b"not json")
            .payload_to_client(Value::Null);

        let records = written(&path);
        for record in &records {
            assert!(record.get("trajectory_id").is_none(), "{record}");
            assert!(record.get("group_id").is_none(), "{record}");
            assert_eq!(record["model"], "unknown");
        }
        assert_eq!(records[0]["payload"], json!({"raw": "not json"}));
        assert_eq!(records[1]["payload"], Value::Null);
    }
}
