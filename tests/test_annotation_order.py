"""
Tests for annotation presentation order and session control:
`--workload` / `--workload-order` / `--order-seed` / `--session-limit`, the
resolved order persisted in the progress file, resume-in-the-same-order, and the
fingerprint that stops a re-sampled workload from inheriting the old labels.

The defect these cover is that the previous session presented pairs in file
order — one solid block per workload, in whatever sequence the sampler emitted,
with no way to stop partway. All offline; no terminal, no network.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from levy.dataset.annotation import (
    DEFAULT_WORKLOAD_ORDER,
    AnnotationOrderError,
    BlindAnnotationSession,
    parse_workload_order,
    prune_progress,
    resolve_presentation_order,
    unordered_workloads,
    validate_workloads,
)
from levy.dataset.io import load_dataset, save_dataset, save_json
from levy.dataset.schema import WORKLOADS, QueryPair

REPO_ROOT = Path(__file__).resolve().parent.parent

_PER_WORKLOAD = 8


def _pairs(per_workload: int = _PER_WORKLOAD, author_label=None):
    pairs = []
    for workload in WORKLOADS:
        for index in range(per_workload):
            pairs.append(
                QueryPair(
                    pair_id=f"{workload}-{index:04d}",
                    workload=workload,
                    source_corpus=f"corpus-{workload}",
                    source_pair_id=f"{workload}-src-{index}",
                    query_1=f"{workload} question {index} A",
                    query_2=f"{workload} question {index} B",
                    original_label=index % 2,
                    author_label=author_label,
                )
            )
    return pairs


def _session(pairs, tmp, answers=(), **kwargs):
    """A session whose input is a scripted answer sequence."""
    stream = iter(answers)
    shown = []
    session = BlindAnnotationSession(
        pairs,
        progress_path=Path(tmp) / "progress.json",
        input_fn=lambda prompt: next(stream),
        output_fn=shown.append,
        **kwargs,
    )
    return session, shown


def _blocks(order):
    """The workload sequence of an order, collapsed: ['faq','chat','code']."""
    collapsed = []
    for pair_id in order:
        workload = pair_id.split("-")[0]
        if not collapsed or collapsed[-1] != workload:
            collapsed.append(workload)
    return collapsed


# ---------------------------------------------------------------------------
# Order parsing and derivation
# ---------------------------------------------------------------------------

class TestWorkloadOrderParsing(unittest.TestCase):

    def test_default_is_faq_chat_code(self):
        self.assertEqual(DEFAULT_WORKLOAD_ORDER, ("faq", "chat", "code"))

    def test_parses_a_comma_separated_sequence(self):
        self.assertEqual(parse_workload_order("code,faq,chat"), ("code", "faq", "chat"))
        self.assertEqual(parse_workload_order(" chat , faq "), ("chat", "faq"))

    def test_unknown_workload_rejected(self):
        with self.assertRaises(AnnotationOrderError) as ctx:
            parse_workload_order("faq,rag,code")
        self.assertIn("rag", str(ctx.exception))

    def test_repeated_workload_rejected(self):
        with self.assertRaises(AnnotationOrderError) as ctx:
            parse_workload_order("faq,chat,faq")
        self.assertIn("more than once", str(ctx.exception))

    def test_empty_order_rejected(self):
        with self.assertRaises(AnnotationOrderError):
            parse_workload_order(" , ")

    def test_workload_restriction_is_deduplicated_and_validated(self):
        self.assertEqual(validate_workloads(["chat", "chat"]), ("chat",))
        self.assertIsNone(validate_workloads(None))
        self.assertIsNone(validate_workloads([]))  # no --workload flags given
        with self.assertRaises(AnnotationOrderError):
            validate_workloads(["nope"])

    def test_unordered_workloads_reports_what_the_order_omits(self):
        self.assertEqual(unordered_workloads(None, ("faq", "chat")), ["code"])
        self.assertEqual(unordered_workloads(["chat"], ("chat",)), [])
        self.assertEqual(unordered_workloads(["chat", "code"], ("faq",)), ["chat", "code"])


class TestPresentationOrder(unittest.TestCase):

    def test_blocks_follow_the_requested_order(self):
        pairs = _pairs()
        order = resolve_presentation_order(pairs, workload_order=("code", "faq", "chat"))
        self.assertEqual(_blocks(order), ["code", "faq", "chat"])
        self.assertEqual(len(order), len(pairs))

    def test_default_order_puts_code_last(self):
        order = resolve_presentation_order(_pairs())
        self.assertEqual(_blocks(order), ["faq", "chat", "code"])

    def test_within_block_order_is_shuffled_not_file_order(self):
        pairs = _pairs()
        order = resolve_presentation_order(pairs, workloads=["faq"], order_seed=7)
        file_order = [p.pair_id for p in pairs if p.workload == "faq"]
        self.assertEqual(sorted(order), sorted(file_order))
        self.assertNotEqual(order, file_order)

    def test_same_seed_is_deterministic(self):
        first = resolve_presentation_order(_pairs(), order_seed=13)
        second = resolve_presentation_order(_pairs(), order_seed=13)
        self.assertEqual(first, second)

    def test_different_seed_changes_the_order(self):
        self.assertNotEqual(
            resolve_presentation_order(_pairs(), order_seed=1),
            resolve_presentation_order(_pairs(), order_seed=2),
        )

    def test_restricting_workloads_does_not_disturb_the_others(self):
        """Each block is shuffled with its own stream, so scope is not entangled."""
        full = resolve_presentation_order(_pairs(), order_seed=5)
        chat_only = resolve_presentation_order(_pairs(), workloads=["chat"], order_seed=5)
        self.assertEqual([p for p in full if p.startswith("chat")], chat_only)

    def test_workload_missing_from_the_order_is_appended_not_dropped(self):
        order = resolve_presentation_order(_pairs(), workload_order=("chat",))
        self.assertEqual(_blocks(order), ["chat", "faq", "code"])
        self.assertEqual(len(order), 3 * _PER_WORKLOAD)


# ---------------------------------------------------------------------------
# Session behaviour
# ---------------------------------------------------------------------------

class TestSessionOrdering(unittest.TestCase):

    def test_pairs_are_presented_in_the_resolved_order(self):
        pairs = _pairs(per_workload=3)
        with TemporaryDirectory() as tmp:
            session, shown = _session(
                pairs, tmp, answers=["1"] * 9, workload_order=("code", "chat", "faq")
            )
            expected = list(session.order)
            session.run()
        presented = [
            line.split("(")[1].rstrip(") -")
            for line in shown
            if line.startswith("\n--- Pair ")
        ]
        self.assertEqual(presented, expected)
        self.assertEqual(_blocks(presented), ["code", "chat", "faq"])

    def test_workload_restriction_presents_only_that_workload(self):
        pairs = _pairs(per_workload=4)
        with TemporaryDirectory() as tmp:
            session, shown = _session(pairs, tmp, answers=["1"] * 4, workloads=["chat"])
            summary = session.run()
        self.assertEqual(summary.newly_labeled, 4)
        self.assertEqual(summary.selected_pairs, 4)
        self.assertTrue(all(p.author_label == 1 for p in pairs if p.workload == "chat"))
        self.assertTrue(all(p.author_label is None for p in pairs if p.workload != "chat"))
        self.assertNotIn("faq", "".join(line for line in shown if line.startswith("\n--- Pair")))

    def test_resolved_order_and_seed_are_persisted(self):
        pairs = _pairs(per_workload=3)
        with TemporaryDirectory() as tmp:
            session, _ = _session(pairs, tmp, answers=[], order_seed=99)
            session.run()
            saved = json.loads((Path(tmp) / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["order"], session.order)
        self.assertEqual(saved["order_seed"], 99)
        self.assertEqual(saved["workload_order"], list(DEFAULT_WORKLOAD_ORDER))

    def test_resume_after_interrupt_keeps_the_original_order(self):
        pairs = _pairs(per_workload=5)
        with TemporaryDirectory() as tmp:
            first, _ = _session(pairs, tmp, answers=["1", "1", "q"], order_seed=3)
            original_order = list(first.order)
            first.run()

            # Resume with a *different* seed: an in-progress session's order is
            # fixed once it starts, so the recorded one must win.
            reloaded = _pairs(per_workload=5)
            for pair in reloaded:
                pair.author_label = None
            second, shown = _session(
                reloaded, tmp, answers=["0"] * 20, order_seed=4242
            )
            self.assertEqual(second.order, original_order)
            second.run()

        presented = [
            line.split("(")[1].rstrip(") -")
            for line in shown
            if line.startswith("\n--- Pair ")
        ]
        # The two already answered are not re-asked, and the rest arrive in the
        # positions the original order gave them.
        self.assertEqual(presented, original_order[2:])
        self.assertTrue(any("order recorded at seed 3" in line for line in shown))

    def test_newly_sampled_pairs_are_appended_to_a_recorded_order(self):
        pairs = _pairs(per_workload=3)
        with TemporaryDirectory() as tmp:
            first, _ = _session(pairs, tmp, answers=["q"])
            first.run()

            grown = _pairs(per_workload=3) + [
                QueryPair(
                    pair_id="chat-0009",
                    workload="chat",
                    source_corpus="corpus-chat",
                    source_pair_id="chat-src-9",
                    query_1="new q1",
                    query_2="new q2",
                    original_label=1,
                )
            ]
            second, shown = _session(grown, tmp, answers=["1"] * 10)
            self.assertEqual(second.order[-1], "chat-0009")
        self.assertTrue(any("not in the recorded order" in line for line in shown))


class TestSessionLimit(unittest.TestCase):

    def test_limit_stops_cleanly_and_the_rest_resumes(self):
        pairs = _pairs(per_workload=4)
        with TemporaryDirectory() as tmp:
            first, shown = _session(pairs, tmp, answers=["1"] * 12, session_limit=3)
            summary = first.run()
            self.assertEqual(summary.newly_labeled, 3)
            self.assertTrue(summary.session_limit_reached)
            self.assertFalse(summary.quit_early)
            self.assertTrue(any("session limit of 3 reached" in line for line in shown))
            answered_first = [p.pair_id for p in pairs if p.author_label is not None]
            # Exactly the first three in presentation order, nothing else.
            self.assertEqual(set(answered_first), set(first.order[:3]))

            second, _ = _session(pairs, tmp, answers=["0"] * 12, session_limit=3)
            second_summary = second.run()
            self.assertEqual(second_summary.newly_labeled, 3)
            self.assertEqual(set(p.pair_id for p in pairs if p.author_label is not None),
                             set(first.order[:6]))

    def test_skips_do_not_consume_the_limit(self):
        pairs = _pairs(per_workload=4)
        with TemporaryDirectory() as tmp:
            session, _ = _session(pairs, tmp, answers=["s", "s", "1", "1"], session_limit=2)
            summary = session.run()
        self.assertEqual(summary.skipped, 2)
        self.assertEqual(summary.newly_labeled, 2)
        self.assertTrue(summary.session_limit_reached)

    def test_zero_limit_rejected(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(AnnotationOrderError):
                _session(_pairs(per_workload=1), tmp, session_limit=0)


class TestStaleProgress(unittest.TestCase):

    def test_a_resampled_pair_does_not_inherit_the_old_label(self):
        """
        Re-sampling a workload reuses its `pair_id`s for different pairs. The
        fingerprint is what stops the previous session's answers from being
        re-applied to pairs that are supposed to come back unannotated.
        """
        pairs = _pairs(per_workload=3)
        with TemporaryDirectory() as tmp:
            session, _ = _session(pairs, tmp, answers=["1"] * 9)
            session.run()
            self.assertTrue(all(p.author_label == 1 for p in pairs))

            # Chat is re-sampled: same pair_ids, new source_pair_ids, no labels.
            resampled = [p for p in _pairs(per_workload=3) if p.workload != "chat"]
            for pair in resampled:
                pair.author_label = 1
            for index in range(3):
                resampled.append(
                    QueryPair(
                        pair_id=f"chat-{index:04d}",
                        workload="chat",
                        source_corpus="corpus-chat",
                        source_pair_id=f"chat-src-{index + 100}",  # different pair
                        query_1=f"fresh chat {index} A",
                        query_2=f"fresh chat {index} B",
                        original_label=index % 2,
                    )
                )

            second, shown = _session(resampled, tmp, answers=["0"] * 3, workloads=["chat"])
            self.assertEqual(second.stale_progress_dropped, 3)
            self.assertTrue(all(
                p.author_label is None for p in resampled if p.workload == "chat"
            ))
            summary = second.run()
        self.assertEqual(summary.newly_labeled, 3)
        self.assertEqual(summary.stale_progress_dropped, 3)
        self.assertTrue(any("re-sampled" in line for line in shown))

    def test_matching_fingerprint_still_resumes(self):
        pairs = _pairs(per_workload=2)
        with TemporaryDirectory() as tmp:
            first, _ = _session(pairs, tmp, answers=["1", "1", "q"])
            first.run()
            reloaded = _pairs(per_workload=2)
            second, _ = _session(reloaded, tmp, answers=["0"] * 6)
            self.assertEqual(second.stale_progress_dropped, 0)
            self.assertEqual(
                sum(1 for p in reloaded if p.author_label is not None), 2
            )

    def test_legacy_v1_progress_file_is_still_applied(self):
        pairs = _pairs(per_workload=2)
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            progress.write_text(json.dumps({"faq-0000": 1}), encoding="utf-8")
            session, _ = _session(pairs, tmp, answers=["0"] * 6)
            by_id = {p.pair_id: p for p in pairs}
            self.assertEqual(by_id["faq-0000"].author_label, 1)
            # ...and rewritten in the v2 shape, so the next run has fingerprints.
            saved = json.loads(progress.read_text(encoding="utf-8"))
            self.assertEqual(saved["version"], 2)
            self.assertEqual(saved["labels"]["faq-0000"]["label"], 1)

    def test_bare_label_inside_a_v2_file_is_tolerated(self):
        """A hand-edited progress file keeps working, just without a fingerprint."""
        pairs = _pairs(per_workload=2)
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            progress.write_text(
                json.dumps({"version": 2, "labels": {"faq-0000": 1}}), encoding="utf-8"
            )
            session, _ = _session(pairs, tmp, answers=[])
            by_id = {p.pair_id: p for p in pairs}
            self.assertEqual(by_id["faq-0000"].author_label, 1)
            self.assertEqual(session.stale_progress_dropped, 0)

    def test_progress_file_that_is_not_an_object_is_rejected(self):
        pairs = _pairs(per_workload=1)
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "progress.json").write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(AnnotationOrderError) as ctx:
                _session(pairs, tmp, answers=[])
            self.assertIn("expected a JSON object", str(ctx.exception))

    def test_progress_file_is_backed_up_before_the_session_rewrites_it(self):
        pairs = _pairs(per_workload=2)
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            progress.write_text(json.dumps({"faq-0000": 1}), encoding="utf-8")
            original = progress.read_bytes()
            _session(pairs, tmp, answers=[])
            backups = list((Path(tmp) / "backups").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)


class TestPruneProgress(unittest.TestCase):
    """
    `prune_progress` is the primary defence against a re-sampled workload
    inheriting the previous sample's answers — the fingerprint is only the second,
    and it cannot help a v1 progress file, which is exactly what a file written
    before fingerprints existed is.
    """

    def test_named_pairs_are_dropped_and_the_rest_fingerprinted(self):
        pairs = _pairs(per_workload=2)
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            recorded = {pair.pair_id: 1 for pair in pairs}
            # An entry for a pair this dataset does not contain — another slice,
            # or a leftover. Kept, since there is nothing to contradict it.
            recorded["faq-9999"] = 0
            progress.write_text(json.dumps(recorded), encoding="utf-8")
            dropped = prune_progress(
                progress, [p.pair_id for p in pairs if p.workload == "chat"], pairs
            )
            saved = json.loads(progress.read_text(encoding="utf-8"))

        self.assertEqual(dropped, 2)
        self.assertEqual(saved["version"], 2)
        self.assertEqual(len(saved["labels"]), 5)
        self.assertNotIn("chat-0000", saved["labels"])
        self.assertIsNone(saved["labels"]["faq-9999"]["source_pair_id"])
        # Survivors gain the fingerprint they had no way to carry as v1.
        by_id = {p.pair_id: p for p in pairs}
        for pair_id, entry in saved["labels"].items():
            if pair_id in by_id:
                self.assertEqual(entry["source_pair_id"], by_id[pair_id].source_pair_id)

    def test_recorded_order_is_dropped_with_the_pairs_it_ordered(self):
        pairs = _pairs(per_workload=2)
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            session, _ = _session(pairs, tmp, answers=["1"] * 6)
            session.run()
            self.assertTrue(json.loads(progress.read_text())["order"])

            prune_progress(progress, ["chat-0000"], pairs)
            saved = json.loads(progress.read_text(encoding="utf-8"))
        self.assertEqual(saved["order"], [])

    def test_a_v1_file_is_pruned_even_though_it_has_no_fingerprints(self):
        """The case the real dataset is in: 900 flat entries, no fingerprints."""
        pairs = _pairs(per_workload=2)
        resampled = [p for p in pairs if p.workload != "chat"] + [
            QueryPair(
                pair_id=f"chat-{i:04d}",
                workload="chat",
                source_corpus="corpus-chat",
                source_pair_id=f"chat-src-{i + 100}",
                query_1=f"fresh {i} A",
                query_2=f"fresh {i} B",
                original_label=i % 2,
            )
            for i in range(2)
        ]
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            progress.write_text(
                json.dumps({pair.pair_id: 1 for pair in pairs}), encoding="utf-8"
            )
            prune_progress(progress, ["chat-0000", "chat-0001"], resampled)

            # Opening a session over the re-sampled dataset must not resurrect them.
            session, _ = _session(resampled, tmp, answers=[], workloads=["chat"])
        self.assertTrue(all(
            p.author_label is None for p in resampled if p.workload == "chat"
        ))
        self.assertTrue(all(
            p.author_label == 1 for p in resampled if p.workload != "chat"
        ))

    def test_missing_progress_file_is_a_no_op(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(
                prune_progress(Path(tmp) / "absent.json", ["faq-0000"], []), 0
            )

    def test_progress_file_that_is_not_an_object_is_rejected(self):
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.json"
            progress.write_text("[]", encoding="utf-8")
            with self.assertRaises(AnnotationOrderError):
                prune_progress(progress, ["faq-0000"], [])


class TestBlindnessStillHolds(unittest.TestCase):

    def test_shuffled_session_never_shows_the_original_label_or_corpus(self):
        pairs = _pairs(per_workload=4)
        with TemporaryDirectory() as tmp:
            session, shown = _session(
                pairs, tmp, answers=["1"] * 12, order_seed=11, workload_order=("chat", "faq")
            )
            session.run()
        joined = "\n".join(shown)
        self.assertNotIn("original_label", joined)
        for pair in pairs:
            self.assertNotIn(pair.source_pair_id, joined)
            self.assertNotIn(pair.source_corpus, joined)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestAnnotateCli(unittest.TestCase):

    def _run(self, args, input_text=None):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "annotate_dataset.py"), *[str(a) for a in args]],
            cwd=REPO_ROOT,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _dataset(self, tmp: Path):
        pairs = _pairs(per_workload=4)
        dataset = tmp / "ground_truth.full.json"
        save_json(pairs, dataset)
        save_dataset(pairs, tmp / "ground_truth.full.csv", dataset)
        return dataset

    def test_invalid_workload_order_is_rejected(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = self._dataset(tmp)
            result = self._run(
                ["--dataset", dataset, "--progress", tmp / "p.json",
                 "--workload-order", "faq,faq,code"],
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("more than once", result.stdout)

    def test_unknown_workload_order_entry_is_rejected(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = self._dataset(tmp)
            result = self._run(
                ["--dataset", dataset, "--progress", tmp / "p.json",
                 "--workload-order", "faq,rag"],
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("rag", result.stdout)

    def test_single_workload_session_with_a_limit(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = self._dataset(tmp)
            result = self._run(
                ["--dataset", dataset, "--progress", tmp / "p.json",
                 "--workload", "chat", "--session-limit", "2",
                 "--out-csv", tmp / "ground_truth.full.csv",
                 "--out-json", tmp / "ground_truth.full.json",
                 "--out-ids", tmp / "ground_truth.ids.csv"],
                input_text="1\n1\n1\n1\n",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            self.assertIn("session_limit_reached=True", result.stdout)

            pairs = load_dataset(tmp / "ground_truth.full.csv")
            labeled = [p for p in pairs if p.author_label is not None]
            self.assertEqual(len(labeled), 2)
            self.assertTrue(all(p.workload == "chat" for p in labeled))
            # The published projection was refreshed alongside the dataset...
            self.assertTrue((tmp / "ground_truth.ids.csv").is_file())
            # ...and the pre-existing files were backed up first.
            self.assertTrue(any((tmp / "backups").iterdir()))

    def test_default_flags_still_annotate_everything(self):
        """Backward compatibility: no new flags, whole dataset, one pass."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = self._dataset(tmp)
            result = self._run(
                ["--dataset", dataset, "--progress", tmp / "p.json",
                 "--out-csv", tmp / "out.csv", "--out-json", tmp / "out.json"],
                input_text="1\n" * 12,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            pairs = load_dataset(tmp / "out.csv")
            self.assertEqual(len(pairs), 12)
            self.assertTrue(all(p.author_label == 1 for p in pairs))


if __name__ == "__main__":
    unittest.main()
