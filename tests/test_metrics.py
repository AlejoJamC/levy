"""Tests for levy.metrics.LevyMetrics and levy.models.CacheEntry.is_expired."""

import time
import unittest

from levy.metrics import LevyMetrics
from levy.models import CacheEntry


class TestLevyMetrics(unittest.TestCase):

    def test_record_hit_exact(self):
        metrics = LevyMetrics()
        metrics.record_hit("exact", saved_tokens=5)
        self.assertEqual(metrics.exact_hits, 1)
        self.assertEqual(metrics.semantic_hits, 0)
        self.assertEqual(metrics.tokens_saved, 5)

    def test_record_hit_semantic(self):
        metrics = LevyMetrics()
        metrics.record_hit("semantic", saved_tokens=3)
        self.assertEqual(metrics.semantic_hits, 1)
        self.assertEqual(metrics.tokens_saved, 3)

    def test_record_hit_unknown_type_still_counts_tokens(self):
        """Neither 'exact' nor 'semantic' matches, but saved_tokens is still recorded."""
        metrics = LevyMetrics()
        metrics.record_hit("unknown", saved_tokens=7)
        self.assertEqual(metrics.exact_hits, 0)
        self.assertEqual(metrics.semantic_hits, 0)
        self.assertEqual(metrics.tokens_saved, 7)

    def test_record_miss(self):
        metrics = LevyMetrics()
        metrics.record_miss()
        self.assertEqual(metrics.misses, 1)

    def test_record_request_tracks_latency(self):
        metrics = LevyMetrics()
        metrics.record_request(12.5)
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.latencies, [12.5])

    def test_snapshot_with_no_requests_has_zero_avg_latency(self):
        metrics = LevyMetrics()
        snap = metrics.get_snapshot()
        self.assertEqual(snap.avg_latency_ms, 0.0)
        self.assertEqual(snap.total_requests, 0)

    def test_snapshot_with_requests_computes_avg_latency(self):
        metrics = LevyMetrics()
        metrics.record_request(10.0)
        metrics.record_request(20.0)
        snap = metrics.get_snapshot()
        self.assertEqual(snap.avg_latency_ms, 15.0)

    def test_str_with_no_requests_reports_zero_hit_rate(self):
        metrics = LevyMetrics()
        text = str(metrics)
        self.assertIn("Requests=0", text)
        self.assertIn("0.0%", text)

    def test_str_with_requests_reports_hit_rate(self):
        metrics = LevyMetrics()
        metrics.record_hit("exact", saved_tokens=2)
        metrics.record_request(5.0)
        metrics.record_miss()
        metrics.record_request(5.0)
        text = str(metrics)
        self.assertIn("Requests=2", text)
        self.assertIn("Hits=1", text)


class TestCacheEntryExpiry(unittest.TestCase):

    def test_no_expiry_set_is_never_expired(self):
        entry = CacheEntry(key_hash="k", prompt="p", response_text="r")
        self.assertFalse(entry.is_expired())

    def test_past_expiry_is_expired(self):
        entry = CacheEntry(key_hash="k", prompt="p", response_text="r", expires_at=time.time() - 1)
        self.assertTrue(entry.is_expired())

    def test_future_expiry_is_not_expired(self):
        entry = CacheEntry(key_hash="k", prompt="p", response_text="r", expires_at=time.time() + 100)
        self.assertFalse(entry.is_expired())


if __name__ == "__main__":
    unittest.main()
