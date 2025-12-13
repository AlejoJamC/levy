import time
from dataclasses import dataclass, field
from typing import List
import statistics
from levy.models import MetricsSnapshot

@dataclass
class LevyMetrics:
    total_requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    latencies: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_hit(self, hit_type: str, saved_tokens: int = 0):
        if hit_type == "exact":
            self.exact_hits += 1
        elif hit_type == "semantic":
            self.semantic_hits += 1
        self.tokens_saved += saved_tokens

    def record_miss(self):
        self.misses += 1

    def record_request(self, latency_ms: float):
        self.total_requests += 1
        self.latencies.append(latency_ms)

    def get_snapshot(self) -> MetricsSnapshot:
        avg_lat = 0.0
        if self.latencies:
            avg_lat = statistics.mean(self.latencies)
        
        return MetricsSnapshot(
            total_requests=self.total_requests,
            exact_hits=self.exact_hits,
            semantic_hits=self.semantic_hits,
            misses=self.misses,
            tokens_saved=self.tokens_saved,
            avg_latency_ms=avg_lat
        )

    def __str__(self) -> str:
        snap = self.get_snapshot()
        hit_rate = 0.0
        if snap.total_requests > 0:
            hit_rate = (snap.exact_hits + snap.semantic_hits) / snap.total_requests * 100
        
        return (
            f"LevyMetrics(Requests={snap.total_requests}, "
            f"Hits={snap.exact_hits+snap.semantic_hits} ({hit_rate:.1f}%), "
            f"TokensSaved={snap.tokens_saved}, AvgLat={snap.avg_latency_ms:.2f}ms)"
        )
