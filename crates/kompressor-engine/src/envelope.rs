//! The envelope Stirrup wraps around every tool result.
//!
//! Stirrup (0.2.0) returns each tool result as one XML element: a command's
//! `<shell_results>`, a fetched `<web_fetch>` page, search `<results>`, a
//! browser `<page_snapshot>`, a `<sub_agent_result>`. Left on, the tags are
//! content to whatever reads the payload, so detection misreads it and a fold
//! can cut the tags themselves. Like the codex frame in [`crate::preamble`],
//! the envelope is held out and goes back byte for byte; DESIGN.md §7.1a has
//! the measurements.

use once_cell::sync::Lazy;
use regex::Regex;

/// A tool result split around its payload; `head + payload + tail` is the
/// result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Envelope<'a> {
    pub head: &'a str,
    pub payload: &'a str,
    pub tail: &'a str,
}

impl Envelope<'_> {
    /// The same envelope around another payload.
    pub fn wrap(&self, payload: &str) -> String {
        format!("{}{payload}{}", self.head, self.tail)
    }
}

/// How many envelopes nested in one another are peeled. Each layer rescans
/// what it holds, so the bound keeps a pathological nest linear.
pub const MAX_LAYERS: usize = 8;

/// Split `content` around its payload, peeling nested envelopes, or `None` when
/// it is not exactly one complete envelope.
///
/// A nest deeper than [`MAX_LAYERS`] comes back all head, so nothing in it is
/// touched.
pub fn split_envelope(content: &str) -> Option<Envelope<'_>> {
    let (mut start, mut stop) = payload_range(content)?;
    let mut layers = 1;
    while let Some((inner_start, inner_stop)) = payload_range(&content[start..stop]) {
        if layers == MAX_LAYERS {
            return Some(Envelope {
                head: content,
                payload: "",
                tail: "",
            });
        }
        stop = start + inner_stop;
        start += inner_start;
        layers += 1;
    }
    Some(Envelope {
        head: &content[..start],
        payload: &content[start..stop],
        tail: &content[stop..],
    })
}

/// What the compressors are given for `content`: its envelope's payload, or
/// all of it when it has none. Whatever reports on a result without
/// compressing it sizes and classifies this.
pub fn payload_of(content: &str) -> &str {
    split_envelope(content).map_or(content, |envelope| envelope.payload)
}

/// One shape an envelope takes.
struct Shape {
    /// The element it is, which has to close exactly at the end.
    tag: &'static str,
    head: Regex,
    /// Where the tail starts when part of the element after the payload is
    /// kept as it is: stderr after stdout, advice after an error. It is the
    /// first one outside any envelope nested in the payload, so a nested
    /// result peels whole and a quoted boundary leaves the kept part whole.
    boundary: Option<&'static str>,
    end: &'static str,
    /// What a non-empty payload opens and closes with, for a wrapper plain
    /// enough that another tool's output may carry it too.
    holds: Option<(&'static str, &'static str)>,
}

impl Shape {
    fn new(
        tag: &'static str,
        head: &str,
        boundary: Option<&'static str>,
        end: &'static str,
    ) -> Self {
        Shape {
            tag,
            head: Regex::new(&format!(r"\A{head}")).unwrap(),
            boundary,
            end,
            holds: None,
        }
    }

    fn holding(self, open: &'static str, close: &'static str) -> Self {
        Shape {
            holds: Some((open, close)),
            ..self
        }
    }

    fn payload_range(&self, text: &str) -> Option<(usize, usize)> {
        let start = self.head.find(text)?.end();
        if !text.ends_with(self.end) || start > text.len() - self.end.len() {
            return None;
        }
        let mut stop = text.len() - self.end.len();
        if let Some(boundary) = self.boundary {
            let at = find_outside_nesting(text, self.tag, start, boundary)?;
            if at + boundary.len() > stop {
                return None;
            }
            stop = at;
        }
        let payload = &text[start..stop];
        let fits = self.holds.is_none_or(|(open, close)| {
            payload.is_empty() || (payload.starts_with(open) && payload.ends_with(close))
        });
        fits.then_some((start, stop))
    }
}

/// Stirrup 0.2.0's tool results: `format_result` in
/// `tools/code_backends/base.py`, `tools/web.py`, `tools/browser_use.py`, and
/// the sub-agent result in `core/agent.py`.
static SHAPES: Lazy<Vec<Shape>> = Lazy::new(|| {
    let shell_error = r"<shell_results>\n<error_kind>[^<\n]*</error_kind>\n<details>";
    vec![
        Shape::new(
            "shell_results",
            r"<shell_results>\n<exit_code>-?\d+</exit_code>\n<stdout>",
            Some("</stdout>\n<stderr>"),
            "</stderr>\n</shell_results>",
        ),
        Shape::new(
            "shell_results",
            shell_error,
            Some("</details>\n<advice>"),
            "</advice>\n</shell_results>",
        ),
        Shape::new(
            "shell_results",
            shell_error,
            None,
            "</details>\n</shell_results>",
        ),
        Shape::new(
            "web_fetch",
            r"<web_fetch><url>[^<\n]*</url><body>",
            None,
            "</body></web_fetch>",
        ),
        Shape::new(
            "web_fetch",
            r"<web_fetch><url>[^<\n]*</url><error>",
            None,
            "</error></web_fetch>",
        ),
        Shape::new("results", r"<results>\n", None, "\n</results>")
            .holding("<result>\n<title>", "\n</result>"),
        Shape::new(
            "page_snapshot",
            r"<page_snapshot>\n",
            None,
            "\n</page_snapshot>",
        ),
        Shape::new(
            "sub_agent_result",
            r"<sub_agent_result>\n<response>",
            None,
            "</response>\n<finished>True</finished>\n</sub_agent_result>",
        ),
        Shape::new(
            "sub_agent_result",
            r"<sub_agent_result>\n<response>",
            None,
            "</response>\n<finished>False</finished>\n</sub_agent_result>",
        ),
        Shape::new(
            "sub_agent_result",
            r"<sub_agent_result>\n<error>",
            None,
            "</error>\n</sub_agent_result>",
        ),
    ]
});

static OUTER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\A<(shell_results|web_fetch|results|page_snapshot|sub_agent_result)>").unwrap()
});

/// Where the payload sits when `text` is exactly one envelope.
fn payload_range(text: &str) -> Option<(usize, usize)> {
    let tag = OUTER.captures(text)?.get(1)?.as_str();
    if !closes_at_end(text, tag) {
        return None;
    }
    SHAPES
        .iter()
        .filter(|shape| shape.tag == tag)
        .find_map(|shape| shape.payload_range(text))
}

/// Whether the `<tag>` opening `text` is first closed at its very end. Nested
/// `<tag>` elements balance out; a sibling, or anything after the first one,
/// closes it early.
fn closes_at_end(text: &str, tag: &str) -> bool {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut depth = 0usize;
    for (at, _) in text.match_indices('<') {
        let rest = &text[at..];
        if rest.starts_with(&open) {
            depth += 1;
        } else if rest.starts_with(&close) {
            depth -= 1;
            if depth == 0 {
                return at + close.len() == text.len();
            }
        }
    }
    false
}

/// The first `needle` at or after `from` that sits in the `<tag>` element
/// opening `text` itself, not in one nested inside it.
fn find_outside_nesting(text: &str, tag: &str, from: usize, needle: &str) -> Option<usize> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut depth = 0usize;
    for (at, _) in text.match_indices('<') {
        let rest = &text[at..];
        if rest.starts_with(&open) {
            depth += 1;
        } else if rest.starts_with(&close) {
            depth -= 1;
        } else if depth == 1 && at >= from && rest.starts_with(needle) {
            return Some(at);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(stdout: &str, stderr: &str) -> String {
        format!(
            "<shell_results>\n<exit_code>0</exit_code>\n<stdout>{stdout}</stdout>\n<stderr>{stderr}</stderr>\n</shell_results>"
        )
    }

    fn assert_split(content: &str, payload: &str) {
        let envelope = split_envelope(content).expect("one whole envelope");
        assert_eq!(envelope.payload, payload);
        assert_eq!(envelope.wrap(envelope.payload), content);
    }

    #[test]
    fn a_shell_result_is_split_around_its_stdout() {
        let content = shell("line one\nline two\n", "warning: deprecated\n");
        let envelope = split_envelope(&content).unwrap();
        assert_eq!(
            envelope.head,
            "<shell_results>\n<exit_code>0</exit_code>\n<stdout>"
        );
        assert_eq!(envelope.payload, "line one\nline two\n");
        assert_eq!(
            envelope.tail, "</stdout>\n<stderr>warning: deprecated\n</stderr>\n</shell_results>",
            "stderr stays with the tags"
        );
    }

    #[test]
    fn a_negative_exit_code_is_still_a_shell_result() {
        let content = shell("killed\n", "").replace("<exit_code>0<", "<exit_code>-9<");
        assert_split(&content, "killed\n");
    }

    #[test]
    fn stdout_that_quotes_the_stderr_tags_leaves_stderr_whole() {
        let content = shell("before</stdout>\n<stderr>quoted\n", "real stderr\n");
        let envelope = split_envelope(&content).unwrap();
        assert_eq!(envelope.payload, "before");
        assert!(
            envelope
                .tail
                .ends_with("real stderr\n</stderr>\n</shell_results>")
        );
        assert_eq!(envelope.wrap(envelope.payload), content);
    }

    #[test]
    fn every_other_shape_is_split_around_its_payload() {
        for (content, payload) in [
            (
                "<shell_results>\n<error_kind>timeout</error_kind>\n<details>still running\n</details>\n</shell_results>",
                "still running\n",
            ),
            (
                "<shell_results>\n<error_kind>command_not_allowed</error_kind>\n<details>rm -rf /</details>\n<advice>Use a narrower path.</advice>\n</shell_results>",
                "rm -rf /",
            ),
            (
                "<web_fetch><url>https://example.com/a?b=1</url><body># Title\n\nText.</body></web_fetch>",
                "# Title\n\nText.",
            ),
            (
                "<web_fetch><url>https://example.com/</url><error>HTTP 404</error></web_fetch>",
                "HTTP 404",
            ),
            (
                "<results>\n<result>\n<title>A</title>\n<url>https://a</url>\n<description>d</description>\n</result>\n</results>",
                "<result>\n<title>A</title>\n<url>https://a</url>\n<description>d</description>\n</result>",
            ),
            (
                "<page_snapshot>\n[1] button \"OK\"\n</page_snapshot>",
                "[1] button \"OK\"",
            ),
            (
                "<sub_agent_result>\n<response>All done.</response>\n<finished>True</finished>\n</sub_agent_result>",
                "All done.",
            ),
            (
                "<sub_agent_result>\n<response>Ran out of turns.</response>\n<finished>False</finished>\n</sub_agent_result>",
                "Ran out of turns.",
            ),
            (
                "<sub_agent_result>\n<error>boom</error>\n</sub_agent_result>",
                "boom",
            ),
        ] {
            assert_split(content, payload);
        }
    }

    #[test]
    fn an_empty_payload_is_still_an_envelope() {
        assert_split("<results>\n\n</results>", "");
        assert_split(&shell("", "Traceback\n"), "");
    }

    #[test]
    fn anything_but_exactly_one_envelope_is_content() {
        let one = shell("out\n", "");
        for content in [
            format!("{one}\n{one}"),
            format!("{one}{one}"),
            format!("ran it:\n{one}"),
            format!("{one}\n"),
            one.replace("</stdout>\n<stderr>", "</stdout><stderr>"),
            one.replace("<exit_code>0</exit_code>", "<exit_code>zero</exit_code>"),
            one[..one.len() - 1].to_string(),
            shell("cat of a truncated result: </shell_results>\n", ""),
            "<web_fetch><url>https://a</url><body>text</error></web_fetch>".to_string(),
            "<results>\n{\"hits\": 3}\n</results>".to_string(),
            "<sub_agent_result>\n<response>x</response>\n<finished>Maybe</finished>\n</sub_agent_result>"
                .to_string(),
            "plain output\n".to_string(),
            String::new(),
        ] {
            assert_eq!(split_envelope(&content), None, "{content:?}");
        }
    }

    #[test]
    fn nested_envelopes_are_peeled_down_to_the_payload() {
        let inner = shell("deep output\n", "");
        let content = format!(
            "<sub_agent_result>\n<response>{inner}</response>\n<finished>True</finished>\n</sub_agent_result>"
        );
        let envelope = split_envelope(&content).unwrap();
        assert_eq!(envelope.payload, "deep output\n");
        assert_eq!(
            envelope.head,
            "<sub_agent_result>\n<response><shell_results>\n<exit_code>0</exit_code>\n<stdout>"
        );
        assert_eq!(envelope.wrap(envelope.payload), content);
    }

    #[test]
    fn a_shell_result_printed_inside_stdout_is_peeled_rather_than_cut() {
        let inner = shell("deep output\n", "inner stderr\n");
        let content = shell(&inner, "outer stderr\n");
        let envelope = split_envelope(&content).unwrap();
        assert_eq!(envelope.payload, "deep output\n");
        assert!(
            envelope
                .head
                .ends_with("<stdout><shell_results>\n<exit_code>0</exit_code>\n<stdout>")
        );
        assert!(
            envelope.tail.starts_with(
                "</stdout>\n<stderr>inner stderr\n</stderr>\n</shell_results></stdout>"
            )
        );
        assert_eq!(envelope.wrap(envelope.payload), content);
    }

    #[test]
    fn a_nest_deeper_than_the_bound_is_left_whole() {
        let nest = |layers: usize| {
            let mut content = "payload\n".to_string();
            for _ in 0..layers {
                content = format!("<page_snapshot>\n{content}\n</page_snapshot>");
            }
            content
        };
        assert_eq!(
            split_envelope(&nest(MAX_LAYERS)).unwrap().payload,
            "payload\n"
        );
        let too_deep = nest(MAX_LAYERS + 1);
        let envelope = split_envelope(&too_deep).unwrap();
        assert_eq!(envelope.head, too_deep);
        assert_eq!(envelope.payload, "");
    }
}
