//! Running a corpus through the engine and scoring what came back.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use kompressor_engine::code::crush_code;
use kompressor_engine::detect::detect_result;
use kompressor_engine::envelope::{payload_of, split_envelope};
use kompressor_engine::markers::has_marker;
use kompressor_engine::options::EngineOptions;
use kompressor_engine::pipeline::compress_content;
use kompressor_engine::refs::{RefCollector, refs_in};
use kompressor_engine::text::chars_len;
use kompressor_engine::types::{ContentType, Context, Kind, Language};

use crate::corpus::Payload;
use crate::metrics::{
    Billed, BucketStats, CACHE_READ_RATE, Coverage, Timing, Violation, percentile,
};

/// What one payload did.
#[derive(Clone, Debug)]
pub struct PayloadResult {
    pub content_id: String,
    /// The compressor that claimed the payload, or `untouched:<detected type>`.
    ///
    /// Two payloads land in the same bucket when the same code decided their
    /// fate, which is what makes a bucket a thing you can go and fix.
    pub bucket: String,
    /// What the detector said, whether or not anything acted on it.
    pub detected_as: ContentType,
    /// The type the engine reported, or `None` when the payload came back
    /// untouched.
    pub kind: Option<Kind>,
    /// Whether the payload was at or above `min_compress_chars`.
    ///
    /// Everything headline is measured over eligible payloads only: a payload
    /// under the gate was never offered to a compressor, so scoring it would
    /// measure the gate rather than the compressors.
    pub eligible: bool,
    pub chars_in: usize,
    pub chars_out: usize,
    pub elapsed_ms: f64,
    /// Fragments retained for this payload.
    pub refs: usize,
    pub ref_bytes: u64,
    pub violations: Vec<Violation>,
    /// Requests that carried this payload, from the corpus.
    pub requests: u32,
}

impl PayloadResult {
    /// Whether some compressor took this payload, at any ratio.
    pub fn claimed(&self) -> bool {
        self.kind.is_some()
    }

    /// What one send of this payload costs relative to a payload seen once.
    ///
    /// The first send is fresh input; every later request re-sends it as part
    /// of the history, which the provider bills as a cache read at
    /// [`CACHE_READ_RATE`].
    pub fn weight(&self) -> f64 {
        1.0 + (self.requests.max(1) - 1) as f64 * CACHE_READ_RATE
    }
}

/// A corpus replay: every payload's result, and the options that produced them.
#[derive(Clone, Debug)]
pub struct ReplayReport {
    pub results: Vec<PayloadResult>,
    pub options: EngineOptions,
    /// Whether the determinism and query-stability passes ran. Recorded because
    /// "no violations" means something different when they did not.
    pub determinism_verified: bool,
}

impl ReplayReport {
    /// Payloads at or above the size gate.
    fn eligible(&self) -> impl Iterator<Item = &PayloadResult> {
        self.results.iter().filter(|r| r.eligible)
    }

    /// The headline: the share of eligible bytes removed.
    pub fn coverage(&self) -> Coverage {
        let mut chars_in = 0u64;
        let mut chars_out = 0u64;
        for result in self.eligible() {
            chars_in += result.chars_in as u64;
            chars_out += result.chars_out as u64;
        }
        Coverage::new(chars_in, chars_out)
    }

    /// The share of eligible bytes some compressor claimed, at any ratio.
    ///
    /// Read it against [`ReplayReport::coverage`]: the difference between them
    /// is corpus that is being looked at and handed back, which is a different
    /// problem from corpus nothing recognises at all.
    pub fn claimed_pct(&self) -> f64 {
        let mut claimed = 0u64;
        let mut total = 0u64;
        for result in self.eligible() {
            total += result.chars_in as u64;
            if result.claimed() {
                claimed += result.chars_in as u64;
            }
        }
        if total == 0 {
            0.0
        } else {
            claimed as f64 * 100.0 / total as f64
        }
    }

    /// Coverage weighted by how often each payload was really sent.
    ///
    /// `None` when the corpus records no request counts at all. That case must
    /// not read as zero and must not read as a measured 0%: an unmeasured run
    /// reported as a measured one is how a corpus without request counts gets
    /// compared against one that has them, and the second run then looks like a
    /// catastrophic regression in a metric nobody measured.
    pub fn billed(&self) -> Option<Billed> {
        if !self.results.iter().any(|r| r.requests > 0) {
            return None;
        }
        let mut chars_in = 0.0;
        let mut chars_out = 0.0;
        for result in self.eligible() {
            let weight = result.weight();
            chars_in += weight * result.chars_in as f64;
            chars_out += weight * result.chars_out as f64;
        }
        let pct = if chars_in <= 0.0 {
            0.0
        } else {
            (chars_in - chars_out) * 100.0 / chars_in
        };
        Some(Billed {
            chars_in,
            chars_out,
            pct,
            cache_read_rate: CACHE_READ_RATE,
        })
    }

    /// Per-bucket totals, over eligible payloads.
    ///
    /// Comparing one run's buckets against another's directly is a mistake this
    /// harness makes hard to make: see [`crate::compare_baseline`], which
    /// restricts each bucket to the payloads that were in it both before and
    /// after. Adding a grammar moves payloads out of `untouched:source_code`
    /// into `source_code` at a lower average rate than the payloads already
    /// there, so the bucket's average falls while the corpus improves and no
    /// individual payload got worse.
    pub fn buckets(&self) -> BTreeMap<String, BucketStats> {
        let mut buckets: BTreeMap<String, BucketStats> = BTreeMap::new();
        for result in self.eligible() {
            let entry = buckets.entry(result.bucket.clone()).or_default();
            entry.payloads += 1;
            entry.chars_in += result.chars_in as u64;
            entry.chars_out += result.chars_out as u64;
            entry.billed_out += result.weight() * result.chars_out as f64;
        }
        for stats in buckets.values_mut() {
            stats.saved_pct = Coverage::new(stats.chars_in, stats.chars_out).pct;
        }
        buckets
    }

    /// The same payloads grouped by size rather than by what claimed them.
    ///
    /// Added because the corpus grew a tail and the headline stopped meaning
    /// what it used to: ten payloads over 128 KB carry 80% of its bytes, so a
    /// single figure now reports how the large ones went and says nothing about
    /// the small majority. A regression confined to one band is invisible in
    /// the aggregate and obvious here.
    ///
    /// The boundaries are the ones the production distribution actually has.
    /// Measured across seven Harbor runs, once base64 image blocks are excluded
    /// — the agent passes screenshots as `input_image`, the wire walker never
    /// offers them to the engine, and counting them made a 51 MB corpus look
    /// like 176 MB — the median payload is 2.3 KB and the **largest is 40 KB**.
    /// The harness truncates tool output there, so there is no tail above it and
    /// a band for one would always be empty.
    pub fn size_bands(&self) -> BTreeMap<String, BucketStats> {
        let mut bands: BTreeMap<String, BucketStats> = BTreeMap::new();
        for result in self.eligible() {
            let label = match result.chars_in {
                0..=8_191 => "1 <8k",
                8_192..=32_767 => "2 8k-32k",
                _ => "3 >=32k",
            };
            let entry = bands.entry(label.to_string()).or_default();
            entry.payloads += 1;
            entry.chars_in += result.chars_in as u64;
            entry.chars_out += result.chars_out as u64;
            entry.billed_out += result.weight() * result.chars_out as f64;
        }
        for stats in bands.values_mut() {
            stats.saved_pct = Coverage::new(stats.chars_in, stats.chars_out).pct;
        }
        bands
    }

    /// Throughput and the latency tail.
    pub fn timing(&self) -> Timing {
        if self.results.is_empty() {
            return Timing::default();
        }
        let mut times: Vec<f64> = self.results.iter().map(|r| r.elapsed_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a measured duration"));
        let total_ms: f64 = times.iter().sum();
        let kb = self.results.iter().map(|r| r.chars_in).sum::<usize>() as f64 / 1024.0;
        Timing {
            ms_per_kb: if kb > 0.0 { total_ms / kb } else { 0.0 },
            p50_ms: percentile(&times, 50.0),
            p95_ms: percentile(&times, 95.0),
            max_ms: times.last().copied().unwrap_or(0.0),
            mean_ms: total_ms / times.len() as f64,
        }
    }

    /// Violations by name, with how many payloads each hit.
    pub fn violations(&self) -> BTreeMap<String, u64> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for result in &self.results {
            for violation in &result.violations {
                *counts.entry(violation.name().to_string()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Payloads whose output depended on the user turn that preceded them.
    pub fn query_dependent(&self) -> Vec<&PayloadResult> {
        self.results
            .iter()
            .filter(|r| r.violations.contains(&Violation::QueryDependent))
            .collect()
    }

    /// Whether every safety invariant held.
    pub fn is_clean(&self) -> bool {
        self.results.iter().all(|r| r.violations.is_empty())
    }
}

/// Replay `payloads` through the engine.
///
/// `verify_determinism` re-compresses every payload twice more: once with
/// identical inputs, which must give identical bytes, and once with the
/// preceding user turn blanked, which must also give identical bytes. The
/// second pass costs the same as the first and catches a different bug — a
/// compressor that ranks content against the question that prompted it — so it
/// is worth the doubling whenever the answer is going to be recorded.
///
/// Grammars are warmed before the clock starts; see [`warm_grammars`].
pub fn replay(
    payloads: &[Payload],
    options: &EngineOptions,
    verify_determinism: bool,
) -> ReplayReport {
    warm_grammars(options);

    let results = payloads
        .iter()
        .map(|payload| replay_one(payload, options, verify_determinism))
        .collect();

    ReplayReport {
        results,
        options: options.clone(),
        determinism_verified: verify_determinism,
    }
}

/// One payload, compressed and scored.
fn replay_one(payload: &Payload, options: &EngineOptions, verify: bool) -> PayloadResult {
    let chars_in = chars_len(&payload.content);
    let inside = payload_of(&payload.content);
    let detection = detect_result(inside, &payload.role, options);

    let mut refs = options.collector();
    let mut applied = Vec::new();
    let ctx = context_for(payload, &payload.preceding_user_text);

    let started = Instant::now();
    let (output, kind) = compress_content(
        &payload.content,
        &payload.role,
        options,
        None,
        &mut refs,
        &ctx,
        &mut applied,
    );
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let mut violations = check_invariants(&payload.content, &output, kind, &refs);
    if verify {
        violations.extend(check_stability(payload, options, &output));
    }

    PayloadResult {
        content_id: payload.content_id.clone(),
        bucket: bucket_name(kind, detection.content_type),
        detected_as: detection.content_type,
        kind,
        eligible: chars_len(inside) >= options.min_compress_chars,
        chars_in,
        chars_out: chars_len(&output),
        elapsed_ms,
        refs: refs.len(),
        ref_bytes: refs.bytes(),
        violations,
        requests: payload.requests,
    }
}

/// The conversation context a payload arrived with.
fn context_for<'a>(payload: &'a Payload, preceding_user_text: &'a str) -> Context<'a> {
    Context {
        role: &payload.role,
        message_index: None,
        preceding_user_text,
        source_tool: Some(&payload.source_tool),
        file_path: None,
    }
}

/// `<compressor>` when something claimed the payload, `untouched:<detected>`
/// when nothing did.
fn bucket_name(kind: Option<Kind>, detected: ContentType) -> String {
    match kind {
        Some(kind) => kind.as_str().to_string(),
        None => format!("untouched:{}", detected.as_str()),
    }
}

/// The checks that do not depend on running the payload a second time.
fn check_invariants(
    content: &str,
    output: &str,
    kind: Option<Kind>,
    refs: &RefCollector,
) -> Vec<Violation> {
    let mut violations = Vec::new();

    let chars_in = chars_len(content);
    let chars_out = chars_len(output);
    if chars_out > chars_in {
        violations.push(Violation::LongerThanInput {
            chars_in,
            chars_out,
        });
    }
    if kind.is_none() && output != content {
        violations.push(Violation::MutatedWhileDeclining);
    }

    if !refs.is_empty() && !has_marker(output) {
        violations.push(Violation::RefsWithoutMarker { refs: refs.len() });
    }
    let named = refs_in(output);
    for (id, fragment) in refs.pairs() {
        if !named.contains(id) {
            violations.push(Violation::OrphanedRef {
                handle: id.to_string(),
            });
        }
        if fragment.is_empty() {
            violations.push(Violation::EmptyRefFragment {
                handle: id.to_string(),
            });
        }
    }

    // Only where the mode promised a way back. With the collector off —
    // aggressive — a bare `[×N]` counts template matches rather than byte
    // matches and keeps no fragment by design; that is the mode's own
    // contract, and 126 corpus payloads exercise it.
    if refs.is_enabled() {
        // An envelope's payload is what was folded, and `<stdout>` is glued to
        // its first line on the wire.
        let payloads = split_envelope(content).and_then(|envelope| {
            let inside = output
                .strip_prefix(envelope.head)?
                .strip_suffix(envelope.tail)?;
            Some((envelope.payload, inside))
        });
        let (content, output) = payloads.unwrap_or((content, output));
        violations.extend(check_bare_folds(content, output));
    }

    violations
}

/// Every handle-free `[×N]` must be a claim the input supports.
///
/// The bare fold is the one marker that asserts what the removed bytes *were*
/// rather than merely that some went, so it is the one that can be a lie. The
/// way it becomes one is a fold whose fragment the store refused — over
/// `max_ref_bytes` — degrading to a marker instead of declining. That was a
/// real defect: 4,000 lines differing only in a number came back as one line
/// and `[×3980]`, against an input holding that line exactly once.
///
/// Checked only when the collector is on. Aggressive mode keeps no fragments
/// at all, so its bare folds count template matches and promise nothing — the
/// same branch `fold_suffix` takes first.
///
/// A line is counted four ways because the fold does not always see the bytes
/// this function is handed: `lift_gutter` strips a `grep -n` column before
/// anything folds and re-attaches it only to lines that survive, so a fold over
/// numbered output counts stripped bodies. Leading/trailing whitespace is
/// allowed to differ for the same reason.
///
/// The fourth view is terminal redraw. A progress line arrives as `\r`-separated
/// frames — `(Reading database ... \r... 5%\r... 10%` — and the compressors fold
/// over the FRAMES, so a claim of ten is answered by ten frames even though the
/// `\n`-delimited block holding them appears once. Counting only `\n` lines
/// reported every one of those honest folds as a violation: 19 of 19 on a
/// corpus extracted from real runs, all of it dpkg output. The failure mode is
/// the one `the_checkers_gutter_matches_the_engines` already names — a
/// disagreement about what a line is, reported as a fault in the compressor.
fn check_bare_folds(content: &str, output: &str) -> Vec<Violation> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut stripped: HashMap<&str, usize> = HashMap::new();
    let mut trimmed: HashMap<&str, usize> = HashMap::new();
    let mut resolved: HashMap<String, usize> = HashMap::new();
    for raw in content.split('\n') {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        *counts.entry(line).or_default() += 1;
        *stripped.entry(without_gutter(line)).or_default() += 1;
        *trimmed.entry(line.trim()).or_default() += 1;
        // A terminal redraw separates frames with `\r`, and the compressors
        // treat each frame as a line -- so a fold counts frames, not the
        // `\n`-delimited blocks this loop walks.
        if line.contains('\r') {
            for frame in line.split('\r') {
                *resolved.entry(frame.trim().to_string()).or_default() += 1;
            }
        }
    }

    let mut violations = Vec::new();
    for raw in output.split('\n') {
        let out_line = raw.strip_suffix('\r').unwrap_or(raw);
        let Some((line, claimed)) = bare_fold(out_line) else {
            continue;
        };
        let found = counts
            .get(line)
            .copied()
            .unwrap_or(0)
            .max(stripped.get(line).copied().unwrap_or(0))
            .max(trimmed.get(line.trim()).copied().unwrap_or(0))
            .max(resolved.get(line.trim()).copied().unwrap_or(0));
        if found < claimed {
            violations.push(Violation::FoldCountUnsupported {
                line: line.to_string(),
                claimed,
                found,
            });
        }
    }
    violations
}

/// `("the line", 47)` for `the line [×47]`, and `None` for anything else —
/// including `the line [×47 ref:k_…]`, which carries its own way back.
fn bare_fold(line: &str) -> Option<(&str, usize)> {
    let body = line.strip_suffix(']')?;
    let at = body.rfind(" [×")?;
    let count: usize = body[at + " [×".len()..].parse().ok()?;
    Some((&body[..at], count))
}

/// `line` with a leading `grep -n` column removed, if it has one.
fn without_gutter(line: &str) -> &str {
    let trimmed = line.trim_start_matches(' ');
    let digits = trimmed.len()
        - trimmed
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .len();
    if digits == 0 {
        return line;
    }
    let rest = &trimmed[digits..];
    // The separator set has to match the engine's (`gutter::split_prefix`) or
    // this check disagrees with the thing it is checking. `|` and the arrow
    // were missing: a boxed ASCII render numbered ` 358|` reported
    // `claimed x5, found 0` — not because the fold was wrong about five
    // occurrences, but because this function could not take the column off to
    // look. A checker that cannot reproduce the compressor's view of a line
    // reports the difference as the compressor's fault.
    match rest.chars().next() {
        Some('-' | ':' | '\t' | ' ' | '|') => &rest[1..],
        Some('\u{2192}') => &rest['\u{2192}'.len_utf8()..],
        _ => line,
    }
}

/// Re-compress twice: same input, then a different preceding user turn.
fn check_stability(payload: &Payload, options: &EngineOptions, output: &str) -> Vec<Violation> {
    let mut violations = Vec::new();

    let again = compress_once(payload, options, &payload.preceding_user_text);
    if again != output {
        violations.push(Violation::Nondeterministic);
    }

    // Blanked rather than perturbed: a compressor that reads the question at
    // all has to behave differently for *some* question, and the empty one is
    // the only value guaranteed not to coincide with the real one.
    let without_query = compress_once(payload, options, "");
    if without_query != output {
        violations.push(Violation::QueryDependent);
    }

    violations
}

/// Compress `payload` with `preceding_user_text` substituted in.
fn compress_once(payload: &Payload, options: &EngineOptions, preceding_user_text: &str) -> String {
    let mut refs = options.collector();
    let mut applied = Vec::new();
    let ctx = context_for(payload, preceding_user_text);
    let (output, _) = compress_content(
        &payload.content,
        &payload.role,
        options,
        None,
        &mut refs,
        &ctx,
        &mut applied,
    );
    output.into_owned()
}

/// A snippet every grammar will happily build a tree for, error nodes included.
const WARMUP_SOURCE: &str = "int main(void) { return 0; }\ndef f():\n    return 0\n";

/// Parse one payload per grammar before anything is timed.
///
/// Grammars load lazily and the first payload of a family pays for it: the
/// first C payload measured 2.5 s against 62 ms warm. Left cold that single
/// cost lands on whichever payload happened to come first, which puts a spike
/// in `max`, inflates one arbitrary bucket's average, and makes every timing in
/// the report depend on the order the corpus was written in.
///
/// Called by [`replay`]; public because a caller timing the engine some other
/// way needs the same guarantee.
pub fn warm_grammars(options: &EngineOptions) {
    let ctx = Context::for_role("tool");
    for language in [
        Language::Python,
        Language::JavaScript,
        Language::TypeScript,
        Language::Go,
        Language::Rust,
        Language::Java,
        Language::CSharp,
        Language::C,
        Language::Cpp,
    ] {
        let mut refs = RefCollector::disabled();
        let _ = crush_code(WARMUP_SOURCE, options, &ctx, &mut refs, Some(language));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kompressor_engine::types::Mode;

    /// A payload with the given content and request count.
    fn payload(id: &str, content: &str, requests: u32) -> Payload {
        Payload {
            content_id: id.to_string(),
            role: "tool".to_string(),
            content: content.to_string(),
            preceding_user_text: String::new(),
            source_tool: "exec_command".to_string(),
            occurrences: 1,
            requests,
            task: "t".to_string(),
        }
    }

    /// Identical log lines, comfortably more than `keep_tail_lines`: the log
    /// compressor folds the head and keeps the tail verbatim, so a payload of
    /// 40 lines would be half tail and score like a mediocre compression.
    /// Lines that share a template but differ in their values, so the count
    /// cannot give them back and the fold has to keep a handle.
    fn varying_log() -> String {
        (0..200)
            .map(|i| format!("2026-01-01T00:00:00Z INFO worker heartbeat ok, queue depth {i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn foldable_log() -> String {
        (0..200)
            .map(|_| "2026-01-01T00:00:00Z INFO worker heartbeat ok, queue depth 0")
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn coverage_and_claimed_are_computed_over_eligible_payloads_only() {
        let options = EngineOptions::default();
        let log = foldable_log();
        // 1000 characters of prose with nothing to remove: eligible, unclaimed.
        let prose = "the quick brown fox jumps over the lazy dog. "
            .repeat(23)
            .to_string();
        // Under the 512-character gate: must not count either way.
        let tiny = "short".to_string();

        let report = replay(
            &[
                payload("a", &log, 1),
                payload("b", &prose, 1),
                payload("c", &tiny, 1),
            ],
            &options,
            false,
        );

        let coverage = report.coverage();
        let expected_in = (chars_len(&log) + chars_len(&prose)) as u64;
        assert_eq!(
            coverage.chars_in, expected_in,
            "the payload under the gate must not be in the denominator"
        );
        assert!(coverage.pct > 50.0, "a folded log dominates: {coverage:?}");

        // The log is claimed and the prose is not, so claimed is the log's share.
        let expected_claimed = chars_len(&log) as f64 * 100.0 / expected_in as f64;
        assert!((report.claimed_pct() - expected_claimed).abs() < 0.001);
        assert!(
            report.claimed_pct() > coverage.pct,
            "claimed is an upper bound on coverage"
        );
    }

    #[test]
    fn a_stirrup_result_is_sized_and_classified_by_what_it_holds() {
        let options = EngineOptions::default();
        let shell = |stdout: &str, stderr: &str| {
            format!(
                "<shell_results>\n<exit_code>0</exit_code>\n<stdout>{stdout}</stdout>\n<stderr>{stderr}</stderr>\n</shell_results>"
            )
        };
        let table: String = std::iter::once("id,name,score\n".to_string())
            .chain((0..40).map(|i| format!("{i},user_{i},{}\n", i * 3)))
            .collect();
        let report = replay(
            &[
                payload("a", &shell("ok\n", &foldable_log()), 1),
                payload("b", &shell(&table, ""), 1),
            ],
            &options,
            false,
        );
        assert!(
            !report.results[0].eligible,
            "its stdout alone is under the gate"
        );
        assert_eq!(report.results[1].detected_as, ContentType::Tabular);
    }

    #[test]
    fn billed_weights_a_payload_by_the_requests_that_carried_it() {
        let options = EngineOptions::default();
        let log = foldable_log();
        let prose = "the quick brown fox jumps over the lazy dog. ".repeat(23);

        // The uncompressible payload is the one carried 261 times, so weighting
        // must pull billed coverage *below* flat coverage.
        let report = replay(
            &[payload("a", &log, 1), payload("b", &prose, 261)],
            &options,
            false,
        );
        let billed = report.billed().expect("request counts were recorded");
        assert_eq!(billed.cache_read_rate, CACHE_READ_RATE);

        let weight = 1.0 + 260.0 * CACHE_READ_RATE;
        let expected_in = chars_len(&log) as f64 + weight * chars_len(&prose) as f64;
        assert!((billed.chars_in - expected_in).abs() < 0.001);
        assert!(
            billed.pct < report.coverage().pct,
            "the expensive payload is the one that does not compress"
        );
    }

    #[test]
    fn billed_is_absent_rather_than_zero_when_the_corpus_counts_no_requests() {
        let options = EngineOptions::default();
        let report = replay(&[payload("a", &foldable_log(), 0)], &options, false);
        assert!(
            report.billed().is_none(),
            "an unmeasured run must never read as a measured one"
        );
        // and a single send weighs one, so the metric is still defined
        assert_eq!(report.results[0].weight(), 1.0);
    }

    #[test]
    fn an_untouched_payload_is_bucketed_by_what_the_detector_called_it() {
        let options = EngineOptions::default();
        let prose = "the quick brown fox jumps over the lazy dog. ".repeat(23);
        let report = replay(&[payload("a", &prose, 1)], &options, false);
        assert_eq!(report.results[0].bucket, "untouched:text");
        assert!(!report.results[0].claimed());
    }

    #[test]
    fn a_claimed_payload_is_bucketed_by_the_compressor_that_took_it() {
        let options = EngineOptions::default();
        let report = replay(&[payload("a", &foldable_log(), 1)], &options, false);
        assert_eq!(report.results[0].bucket, "log");
    }

    #[test]
    fn every_retained_fragment_is_named_by_the_output_that_replaced_it() {
        let options = EngineOptions::default();
        let report = replay(&[payload("a", &varying_log(), 1)], &options, false);
        let result = &report.results[0];
        assert!(result.refs > 0, "recoverable mode retains what it folds");
        assert!(result.violations.is_empty(), "{:?}", result.violations);
    }

    #[test]
    fn a_fold_of_identical_lines_retains_nothing_to_name() {
        // `[×200]` reconstructs to two hundred copies of the line above it, so
        // there is nothing for a handle to add. Storing one anyway is what put
        // 241 of 275 fragments in a measured run behind markers no model ever
        // fetched.
        let options = EngineOptions::default();
        let report = replay(&[payload("a", &foldable_log(), 1)], &options, false);
        let result = &report.results[0];
        assert!(
            result.chars_out < result.chars_in,
            "the fold still happened"
        );
        assert_eq!(result.refs, 0, "an identical repeat needs no handle");
        assert!(result.violations.is_empty(), "{:?}", result.violations);
    }

    #[test]
    fn the_checkers_gutter_matches_the_engines() {
        // `|` and the arrow are gutter separators to `gutter::split_prefix` and
        // were not to this checker, so a boxed ASCII render numbered ` 358|`
        // reported `claimed x5, found 0` — a disagreement about what a line is,
        // reported as a fault in the compressor.
        let boxed = " 358|....\n 359|....\n 360|....\n";
        assert!(
            check_bare_folds(boxed, ".... [\u{d7}3]\n").is_empty(),
            "the column comes off before the count is checked"
        );
        let arrowed = "1\u{2192}same\n2\u{2192}same\n3\u{2192}same\n";
        assert!(
            check_bare_folds(arrowed, "same [\u{d7}3]\n").is_empty(),
            "and so does an arrow column"
        );
        // Still a real check: three claimed, two in the input.
        assert_eq!(
            check_bare_folds(" 1|a\n 2|a\n", "a [\u{d7}3]\n"),
            vec![Violation::FoldCountUnsupported {
                line: "a".to_string(),
                claimed: 3,
                found: 2,
            }]
        );
    }

    #[test]
    fn the_checker_reads_an_envelope_the_way_the_engine_does() {
        // The fold runs on what the envelope holds, so the first line it
        // counts is not the one `<stdout>` is glued to on the wire.
        let content = format!(
            "<shell_results>\n<exit_code>0</exit_code>\n<stdout>{}</stdout>\n<stderr></stderr>\n</shell_results>",
            "same line from a noisy build\n".repeat(300)
        );
        let opts = EngineOptions {
            mode: Mode::Recoverable,
            ..Default::default()
        };
        let mut refs = opts.collector();
        let (output, kind) = compress_content(
            &content,
            "tool",
            &opts,
            None,
            &mut refs,
            &Context::for_role("tool"),
            &mut Vec::new(),
        );
        assert!(
            output.contains("<stdout>same line from a noisy build [\u{d7}"),
            "the first line carries a bare count: {output}"
        );
        assert_eq!(check_invariants(&content, &output, kind, &refs), vec![]);
    }

    #[test]
    fn the_checker_counts_redraw_frames_the_way_the_compressors_do() {
        // dpkg writes progress as `\r`-separated frames inside one `\n` line.
        // The compressors fold over the frames; a checker that splits only on
        // `\n` sees one line, finds zero, and calls an honest fold a safety
        // violation. Measured on a corpus extracted from real runs: 6 of 19
        // reported violations were exactly this, and all of them were dpkg.
        let content = "(Reading database ... \r(Reading database ... \r                       (Reading database ... \rdone\n";
        assert!(
            check_bare_folds(content, "(Reading database ...  [×3]\n").is_empty(),
            "three frames support a claim of three"
        );
        // Still a real check: the frames do not invent repeats that are absent.
        assert!(
            !check_bare_folds(content, "(Reading database ...  [×9]\n").is_empty(),
            "nine were never there"
        );
    }

    #[test]
    fn a_bare_fold_that_claims_a_repeat_the_input_lacks_is_a_violation() {
        // The shape of the real defect: `max_ref_bytes` refused the fragment,
        // the fold went ahead anyway, and the marker asserts three copies of a
        // line the input holds once.
        let content = "queue depth 0\nqueue depth 1\nqueue depth 2\n";
        assert_eq!(
            check_bare_folds(content, "queue depth 0 [\u{d7}3]\n"),
            vec![Violation::FoldCountUnsupported {
                line: "queue depth 0".to_string(),
                claimed: 3,
                found: 1,
            }]
        );
    }

    #[test]
    fn a_bare_fold_the_input_supports_is_not_a_violation() {
        let same = "same line\nsame line\nsame line\n";
        assert!(check_bare_folds(same, "same line [\u{d7}3]\n").is_empty());

        // A `grep -n` gutter comes off before anything folds, so the count is
        // over the stripped bodies and the numbered original still supports it.
        let numbered = "291-  }\n292-  }\n293-  }\n";
        assert!(check_bare_folds(numbered, "  } [\u{d7}3]\n").is_empty());

        // A fold that kept its handle makes no such claim, so it is not checked
        // however little the input supports the bare reading.
        let differing = "took 1ms\ntook 2ms\ntook 3ms\n";
        assert!(check_bare_folds(differing, "took 1ms [\u{d7}3 ref:k_AbC123dEf456]\n").is_empty());
    }

    #[test]
    fn the_engine_never_emits_an_unsupported_fold_count_for_the_whole_corpus() {
        // The regression guard for the fix: a fold whose fragment the store
        // will not take must decline, not degrade to a bare marker. A tiny
        // `max_ref_bytes` makes every fragment too large.
        let options = EngineOptions {
            max_ref_bytes: 8,
            ..EngineOptions::default()
        };
        let report = replay(&[payload("a", &varying_log(), 1)], &options, false);
        let result = &report.results[0];
        assert!(result.violations.is_empty(), "{:?}", result.violations);
        assert_eq!(
            result.chars_out, result.chars_in,
            "with nowhere to put the fragment the fold is refused, not faked"
        );
    }

    #[test]
    fn the_orphaned_ref_check_fires_when_a_handle_goes_nowhere() {
        // A collector holding a fragment the output never names is exactly the
        // write-nothing-can-read case, so the checker must report it rather
        // than relying on the pipeline never producing one.
        let mut refs = RefCollector::new(65536);
        let handle = refs
            .stash("the fragment that was removed")
            .expect("stashed");
        let violations = check_invariants(
            "a payload with some content in it",
            "a payload [... 1 line omitted]",
            Some(Kind::Log),
            &refs,
        );
        assert_eq!(
            violations,
            vec![Violation::OrphanedRef {
                handle: handle.to_string()
            }]
        );
    }

    #[test]
    fn the_declining_payload_check_fires_on_any_edit() {
        let violations = check_invariants("original", "originaL", None, &RefCollector::disabled());
        assert_eq!(violations, vec![Violation::MutatedWhileDeclining]);
    }

    #[test]
    fn the_longer_than_input_check_fires_when_output_grows() {
        let violations = check_invariants("ab", "abc", Some(Kind::Text), &RefCollector::disabled());
        assert_eq!(
            violations,
            vec![Violation::LongerThanInput {
                chars_in: 2,
                chars_out: 3
            }]
        );
    }

    #[test]
    fn retention_that_announces_nothing_is_reported() {
        let mut refs = RefCollector::new(65536);
        refs.stash("removed bytes");
        let violations = check_invariants("aaaa", "aa", Some(Kind::Text), &refs);
        assert!(violations.iter().any(|v| v.name() == "refs_without_marker"));
        assert!(violations.iter().any(|v| v.name() == "orphaned_ref"));
    }

    #[test]
    fn an_empty_retained_fragment_is_reported() {
        let mut refs = RefCollector::new(65536);
        let handle = refs.stash("").expect("an empty fragment is under the cap");
        let output = format!("kept [... 0 lines omitted [ref:{handle}]]");
        let content = "kept, and then a great deal more that was removed from it";
        let violations = check_invariants(content, &output, Some(Kind::Log), &refs);
        assert_eq!(
            violations,
            vec![Violation::EmptyRefFragment {
                handle: handle.to_string()
            }]
        );
    }

    #[test]
    fn the_engine_is_deterministic_and_query_independent_on_a_folded_log() {
        let options = EngineOptions {
            mode: Mode::Recoverable,
            ..EngineOptions::default()
        };
        let mut p = payload("a", &foldable_log(), 3);
        p.preceding_user_text = "why is the worker stuck?".to_string();

        let report = replay(&[p], &options, true);
        assert!(report.determinism_verified);
        assert!(report.is_clean(), "{:?}", report.results[0].violations);
        assert!(report.query_dependent().is_empty());
    }

    #[test]
    fn timing_reports_a_tail_and_a_throughput() {
        let options = EngineOptions::default();
        let report = replay(&[payload("a", &foldable_log(), 1)], &options, false);
        let timing = report.timing();
        assert!(timing.max_ms >= timing.p50_ms);
        assert!(timing.ms_per_kb > 0.0);
    }
}
