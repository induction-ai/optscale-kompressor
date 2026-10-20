//! What crosses the proxy in each direction.
//!
//! A reverse proxy that rewrites bodies has to be pedantic about headers,
//! because two of them describe a body we are about to change and three more
//! describe a connection that ends here. The rules are small and each one
//! prevents a concrete failure:
//!
//! - Hop-by-hop headers describe *this* connection. Forwarding `Connection` or
//!   `Transfer-Encoding` to the upstream is how a proxy produces a request that
//!   frames itself two different ways.
//! - `Content-Length` is recomputed, because a compressed body is a different
//!   length and a stale length truncates it.
//! - `Accept-Encoding` is stripped on the way out. A gzipped response is bytes
//!   we cannot look inside, and the retrieve loop has to read the response to
//!   know whether the model asked for a fragment back.
//! - The client's provider credentials go through untouched unless the upstream
//!   config replaces them. The proxy authenticates nothing on the client's
//!   behalf; it carries what the client presented, to the one host it was
//!   configured to carry it to.

use http::header::{HeaderMap, HeaderName, HeaderValue};

use kompressor_core::config::Upstream;

use crate::authmode::AuthMode;

/// Headers that describe the connection rather than the message.
const HOP_BY_HOP: [&str; 9] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
];

/// Our own control headers, which are instructions to the proxy and mean
/// nothing to a provider.
const CONTROL: [&str; 4] = [
    "x-kompressor-mode",
    "x-kompressor-key",
    "x-kompressor-secret",
    "x-kompressor-tenant",
];

/// The Induction benchmark harness's headers, which label a request for the
/// payload log and mean nothing to a provider.
const INDUCTION_PREFIX: &str = "x-induction-";

/// Credential headers, dropped when the upstream supplies its own.
const CREDENTIAL: [&str; 3] = ["authorization", "x-api-key", "api-key"];

/// The request id header, read inbound and written in both directions.
pub const REQUEST_ID: &str = "x-request-id";
/// The mode this request was handled in.
pub const MODE: &str = "x-kompressor-mode";
/// Estimated tokens the compression removed from the prompt.
pub const SAVED_TOKENS: &str = "x-kompressor-saved-tokens";
/// Retrieval rounds the proxy ran inside this request.
pub const RETRIEVE_ROUNDS: &str = "x-kompressor-retrieve-rounds";

/// Build the header map for the upstream request.
///
/// `mode` decides only what the proxy refrains from adding. A subscription CLI
/// is recognised by the provider it is talking to, so a correlation header it
/// never sends is drift the caller did not ask for — see [`crate::authmode`].
pub fn upstream_headers(
    client: &HeaderMap,
    upstream: &Upstream,
    request_id: &str,
    mode: AuthMode,
) -> HeaderMap {
    let listed = connection_listed(client);
    let mut out = HeaderMap::new();

    for (name, value) in client.iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if is_hop_by_hop(&lower) || listed.contains(&lower) {
            continue;
        }
        if lower == "content-length" || lower == "accept-encoding" {
            continue;
        }
        if CONTROL.contains(&lower.as_str()) || lower.starts_with(INDUCTION_PREFIX) {
            continue;
        }
        if !upstream.forward_auth && CREDENTIAL.contains(&lower.as_str()) {
            continue;
        }
        out.append(name.clone(), value.clone());
    }

    // identity rather than nothing: an upstream that defaults to gzip when the
    // header is absent would hand us a body we cannot read.
    out.insert(
        http::header::ACCEPT_ENCODING,
        HeaderValue::from_static("identity"),
    );

    for (name, value) in &upstream.headers {
        let (Ok(name), Ok(value)) = (name.parse::<HeaderName>(), HeaderValue::from_str(value))
        else {
            // a header the config spelled wrong is not worth failing a request
            // the provider would have answered
            tracing::warn!(header = %name, upstream = %upstream.name, "skipping an unusable forced header");
            continue;
        };
        out.insert(name, value);
    }

    // Our own correlation id, except where adding a header the real client
    // never sends would make the request identifiably not that client. The id
    // still reaches every stats row and log line; only the wire is left alone.
    if !mode.is_subscription()
        && let Ok(value) = HeaderValue::from_str(request_id)
    {
        out.insert(HeaderName::from_static(REQUEST_ID), value);
    }
    out
}

/// Build the header map relayed back to the client.
///
/// `Content-Length` is dropped in every case: a retrieve loop returns a body
/// the upstream never sent, and a streamed relay has no length at all.
pub fn client_headers(upstream: &HeaderMap) -> HeaderMap {
    let listed = connection_listed(upstream);
    let mut out = HeaderMap::new();
    for (name, value) in upstream.iter() {
        let lower = name.as_str().to_ascii_lowercase();
        if is_hop_by_hop(&lower) || listed.contains(&lower) || lower == "content-length" {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    out
}

/// Add the headers that say what the proxy did.
pub fn annotate(headers: &mut HeaderMap, request_id: &str, mode: &str, saved: u32, rounds: u32) {
    let mut set = |name: &'static str, value: String| {
        if let Ok(value) = HeaderValue::from_str(&value) {
            headers.insert(HeaderName::from_static(name), value);
        }
    };
    set(REQUEST_ID, request_id.to_string());
    set(MODE, mode.to_string());
    set(SAVED_TOKENS, saved.to_string());
    set(RETRIEVE_ROUNDS, rounds.to_string());
}

fn is_hop_by_hop(lower: &str) -> bool {
    HOP_BY_HOP.contains(&lower)
}

/// The header names a `Connection` header nominates as hop-by-hop.
///
/// `Connection: x-internal-hint` means that header ends here too; relaying it
/// leaks a routing decision from one hop into the next.
fn connection_listed(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(http::header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|name| name.trim().to_ascii_lowercase())
        .filter(|name| !name.is_empty() && name != "close")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client_request_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer sk-live"));
        headers.insert("x-api-key", HeaderValue::from_static("ant-key"));
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert("anthropic-beta", HeaderValue::from_static("tools-2024"));
        headers.insert("openai-organization", HeaderValue::from_static("org_1"));
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert("content-length", HeaderValue::from_static("12"));
        headers.insert("accept-encoding", HeaderValue::from_static("gzip, br"));
        headers.insert("connection", HeaderValue::from_static("keep-alive, x-hint"));
        headers.insert("x-hint", HeaderValue::from_static("internal"));
        headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        headers.insert("host", HeaderValue::from_static("proxy.internal"));
        headers.insert("x-kompressor-mode", HeaderValue::from_static("aggressive"));
        headers
    }

    #[test]
    fn provider_credentials_and_versions_reach_the_upstream() {
        let out = upstream_headers(
            &client_request_headers(),
            &Upstream::default(),
            "req-1",
            AuthMode::Payg,
        );
        assert_eq!(out["authorization"], "Bearer sk-live");
        assert_eq!(out["x-api-key"], "ant-key");
        assert_eq!(out["anthropic-version"], "2023-06-01");
        assert_eq!(out["anthropic-beta"], "tools-2024");
        assert_eq!(out["openai-organization"], "org_1");
        assert_eq!(out["x-request-id"], "req-1");
    }

    #[test]
    fn hop_by_hop_headers_and_a_stale_length_end_at_the_proxy() {
        let out = upstream_headers(
            &client_request_headers(),
            &Upstream::default(),
            "req-1",
            AuthMode::Payg,
        );
        for ended in [
            "connection",
            "transfer-encoding",
            "host",
            "content-length",
            "x-hint",
        ] {
            assert!(!out.contains_key(ended), "{ended} must not be forwarded");
        }
        assert_eq!(
            out["accept-encoding"], "identity",
            "a compressed response is one the retrieve loop cannot read"
        );
    }

    /// The literal header set Claude Code 2.1.278 sends. A subscription CLI is
    /// recognised by the provider it is talking to, so the proxy must not be
    /// the reason the request stops looking like that client.
    #[test]
    fn a_subscription_client_gets_no_header_of_ours() {
        let mut client = HeaderMap::new();
        client.insert(
            "user-agent",
            HeaderValue::from_static("claude-cli/2.1.278 (external, claude-desktop)"),
        );
        client.insert(
            "authorization",
            HeaderValue::from_static("Bearer sk-ant-oat01-EXAMPLE"),
        );
        let mode = crate::authmode::classify(&client);
        assert_eq!(mode, AuthMode::Subscription);

        let out = upstream_headers(&client, &Upstream::default(), "req-1", mode);
        assert!(
            !out.contains_key("x-request-id"),
            "a header the real client never sends is drift the caller did not ask for"
        );
        assert_eq!(
            out["authorization"], "Bearer sk-ant-oat01-EXAMPLE",
            "the credential still goes through untouched"
        );

        // and the pay-as-you-go path is unchanged
        let payg = upstream_headers(&client, &Upstream::default(), "req-1", AuthMode::Payg);
        assert_eq!(payg["x-request-id"], "req-1");
    }

    #[test]
    fn our_own_control_headers_are_not_a_providers_business() {
        let out = upstream_headers(
            &client_request_headers(),
            &Upstream::default(),
            "req-1",
            AuthMode::Payg,
        );
        assert!(!out.contains_key("x-kompressor-mode"));
    }

    #[test]
    fn the_benchmark_harness_headers_are_not_a_providers_business() {
        let mut client = client_request_headers();
        client.insert(
            "x-induction-trajectory-id",
            HeaderValue::from_static("traj_1"),
        );
        client.insert("X-Induction-Group-Id", HeaderValue::from_static("grp_1"));
        client.insert(
            "x-induction-config",
            HeaderValue::from_static("passthrough=false"),
        );
        let out = upstream_headers(&client, &Upstream::default(), "req-1", AuthMode::Payg);
        for name in [
            "x-induction-trajectory-id",
            "x-induction-group-id",
            "x-induction-config",
        ] {
            assert!(!out.contains_key(name), "{name} must not be forwarded");
        }
        assert_eq!(out["authorization"], "Bearer sk-live");
    }

    #[test]
    fn an_upstream_that_supplies_its_own_auth_drops_the_clients() {
        let upstream = Upstream {
            forward_auth: false,
            headers: [("Authorization".to_string(), "Bearer sk-ours".to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let out = upstream_headers(
            &client_request_headers(),
            &upstream,
            "req-1",
            AuthMode::Payg,
        );
        assert_eq!(out["authorization"], "Bearer sk-ours");
        assert!(!out.contains_key("x-api-key"));
    }

    #[test]
    fn the_response_keeps_its_own_headers_but_never_its_length() {
        let mut upstream = HeaderMap::new();
        upstream.insert("content-type", HeaderValue::from_static("application/json"));
        upstream.insert("content-length", HeaderValue::from_static("40"));
        upstream.insert("x-ratelimit-remaining", HeaderValue::from_static("59"));
        upstream.insert("connection", HeaderValue::from_static("keep-alive"));

        let mut out = client_headers(&upstream);
        assert_eq!(out["content-type"], "application/json");
        assert_eq!(out["x-ratelimit-remaining"], "59");
        assert!(!out.contains_key("content-length"));
        assert!(!out.contains_key("connection"));

        annotate(&mut out, "req-1", "recoverable", 4096, 2);
        assert_eq!(out["x-request-id"], "req-1");
        assert_eq!(out["x-kompressor-mode"], "recoverable");
        assert_eq!(out["x-kompressor-saved-tokens"], "4096");
        assert_eq!(out["x-kompressor-retrieve-rounds"], "2");
    }
}
