//! A live Stirrup conversation through the engine.
//!
//! `fixtures/stirrup_cad_model.json.gz` is the last request of a terminal-bench
//! cad-model run through the proxy (`kp-01a15b27cc11-00000025`): its user turn
//! and its 37 shell results in order, each with its index in the Responses
//! `input`. Those are all the walker hands the engine; reasoning and calls are
//! relayed untouched.

use std::io::Read;
use std::path::PathBuf;

use kompressor_engine::EngineOptions;
use kompressor_engine::envelope::split_envelope;
use kompressor_engine::messages::{Message, MessageContent, RequestOptions, compress_messages};
use serde::Deserialize;

#[derive(Deserialize)]
struct Item {
    item: usize,
    role: String,
    text: String,
}

fn load() -> Vec<Item> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/stirrup_cad_model.json.gz")
        .canonicalize()
        .expect("the conversation is committed next to the corpus");
    let file = std::fs::File::open(path).expect("conversation readable");
    let mut text = String::new();
    flate2::read::GzDecoder::new(file)
        .read_to_string(&mut text)
        .expect("the conversation is gzipped json");
    serde_json::from_str(&text).expect("the conversation parses")
}

/// Each item beside the text the engine ships for it.
fn shipped() -> Vec<(Item, String)> {
    let items = load();
    let opts = EngineOptions::default();
    let mut messages: Vec<Message> = items
        .iter()
        .map(|item| Message::text(item.role.clone(), item.text.clone()))
        .collect();
    let mut refs = opts.collector();
    compress_messages(&mut messages, &opts, &RequestOptions::default(), &mut refs);
    items
        .into_iter()
        .zip(messages)
        .map(|(item, message)| match message.content {
            MessageContent::Text(text) => (item, text),
            other => panic!("item {}: text in, text out, got {other:?}", item.item),
        })
        .collect()
}

#[test]
fn every_shell_result_keeps_its_envelope_byte_for_byte() {
    let mut rewritten = 0;
    for (item, out) in shipped() {
        if item.role != "tool" {
            continue;
        }
        let envelope = split_envelope(&item.text).expect("every output is one shell result");
        assert!(
            out.starts_with(envelope.head) && out.ends_with(envelope.tail),
            "item {}: the envelope broke:\n{out}",
            item.item
        );
        if out != item.text {
            rewritten += 1;
        }
    }
    assert!(
        rewritten > 0,
        "some of the conversation is still compressed"
    );
}

#[test]
fn stderr_and_the_closing_tags_reach_the_model_as_the_tool_wrote_them() {
    let shipped = shipped();
    let out = |index: usize| {
        let (item, out) = shipped
            .iter()
            .find(|(item, _)| item.item == index)
            .expect("the item is in the conversation");
        (item.text.as_str(), out.as_str())
    };

    // A uv install log on stderr, under a few lines of stdout.
    let (input, output) = out(10);
    let log = &input[input.find("<stderr>").unwrap()..];
    assert!(output.ends_with(log), "{output}");

    // A traceback an earlier result already showed, on stderr, with stdout
    // empty.
    let (input, output) = out(40);
    assert_eq!(output, input);

    // debconf lines on stderr that earlier results also printed.
    let (input, output) = out(106);
    let stderr = &input[input.find("</stdout>").unwrap()..];
    assert!(output.ends_with(stderr), "{output}");
}
