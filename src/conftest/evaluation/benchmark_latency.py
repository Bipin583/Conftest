"""
ConfTest Latency Benchmark & Performance Profiling Engine.

Times the scoring path -- array sanitisation, ensemble inference, calibration, and the
selective policy decision -- on the artifacts the study actually ships and on rows the
harvest actually produced.

What this does NOT time, and the reason the number is not an end-to-end figure: feature
mining. Building the 32 features for a commit means checking the repository out, walking
a git diff, parsing test files into ASTs, and querying a dependency graph. That happens
once per commit upstream of anything measured here, it is dominated by disk and git, and
folding it into a per-batch scoring latency would flatter the scoring path and mislead
about the pipeline. Any SLA claim from this module is a claim about scoring only.

Until 2026-09-03 this module trained three 20-estimator models on `rng.randn(100, 32)`
with Bernoulli labels and timed them on more Gaussian noise. Inference cost in LightGBM
follows the trees, and trees grown on noise are not the trees the study serves: the
published latency described a model nobody had trained. The suite now takes the loaded
ensemble, calibrator and policy as arguments, so there is nothing left for it to invent.
"""

import time
from typing import Any, Dict, List

import numpy as np

from conftest.logging_config import get_logger

logger = get_logger(__name__)


class LatencyBenchmarkSuite:
    """Profiles per-batch scoring latency for a loaded model bundle on real rows."""

    def __init__(self, ensemble, calibrator, policy, feature_rows: np.ndarray):
        if feature_rows.ndim != 2 or len(feature_rows) == 0:
            raise ValueError(
                "feature_rows must be a non-empty 2-D array of real feature rows. "
                "The benchmark scores rows it is given; it does not generate them."
            )
        self.ensemble = ensemble
        self.calibrator = calibrator
        self.policy = policy
        self.feature_rows = feature_rows.astype(np.float32, copy=False)

    def _batch(self, index: int, batch_size: int) -> np.ndarray:
        """
        Consecutive window over the real rows, wrapping at the end.

        Deterministic on purpose: a benchmark that samples randomly reports a different
        number every run and invites re-rolling until the SLA passes.
        """
        n = len(self.feature_rows)
        start = (index * batch_size) % n
        idx = [(start + offset) % n for offset in range(batch_size)]
        return self.feature_rows[idx]

    def benchmark_pipeline(
        self, num_iterations: int = 150, batch_size: int = 50, warmup: int = 5
    ) -> Dict[str, Any]:
        """
        Time each scoring stage over `num_iterations` batches of `batch_size` rows.

        One batch stands for one commit's candidate test set.

        The first `warmup` batches are timed separately rather than thrown away. The
        first call into LightGBM costs two orders of magnitude more than the rest --
        allocation and first-touch, not prediction -- so leaving it in the pooled mean
        reports a scoring cost no batch after the first one pays, while deleting it
        quietly hides what the first commit of a session actually waits for. Both are
        published.
        """
        feat_ms: List[float] = []
        infer_ms: List[float] = []
        cal_ms: List[float] = []
        dec_ms: List[float] = []
        total_ms: List[float] = []

        logger.info(
            f"Timing {num_iterations} batches of {batch_size} real rows "
            f"drawn from {len(self.feature_rows)} available rows "
            f"({warmup} warm-up batches reported separately)"
        )

        warmup_totals: List[float] = []
        for iteration in range(num_iterations + warmup):
            X_batch = self._batch(iteration, batch_size)

            t_start = time.perf_counter()

            t0 = time.perf_counter()
            X_clean = np.nan_to_num(X_batch, nan=0.0)
            t1 = time.perf_counter()

            mean_probs, epistemic_unc = self._infer(X_clean)
            t2 = time.perf_counter()

            cal_probs = self.calibrator.calibrate(mean_probs)
            t3 = time.perf_counter()

            self.policy.evaluate_commit(
                commit_sha=f"bench_batch_{iteration}",
                candidate_test_ids=[f"test_case_{i}" for i in range(batch_size)],
                calibrated_confidences=cal_probs,
                epistemic_uncertainties=epistemic_unc,
                num_changed_files=1,
                total_churn_lines=10,
            )
            t4 = time.perf_counter()

            if iteration < warmup:
                warmup_totals.append((t4 - t_start) * 1000.0)
                continue

            feat_ms.append((t1 - t0) * 1000.0)
            infer_ms.append((t2 - t1) * 1000.0)
            cal_ms.append((t3 - t2) * 1000.0)
            dec_ms.append((t4 - t3) * 1000.0)
            total_ms.append((t4 - t_start) * 1000.0)

        sla = float(np.mean(np.array(total_ms) < 100.0) * 100.0)
        return {
            "num_iterations": num_iterations,
            "batch_size_tests": batch_size,
            "rows_available": int(len(self.feature_rows)),
            "warmup_batches": warmup,
            "cold_start_first_batch_ms": round(warmup_totals[0], 3) if warmup_totals else None,
            "warmup_batches_ms": [round(v, 3) for v in warmup_totals],
            "timed_statistics_exclude_warmup": True,
            "measures": "scoring only: sanitisation, inference, calibration, policy",
            "excludes": "feature mining (git diff, AST parse, dependency graph), which "
            "runs once per commit upstream and is not part of this figure",
            "sla_under_100ms_compliance_pct": round(sla, 2),
            "stages": {
                "array_sanitisation": _stats(feat_ms),
                "ensemble_inference": _stats(infer_ms),
                "calibration": _stats(cal_ms),
                "policy_decision": _stats(dec_ms),
                "scoring_total": _stats(total_ms),
            },
        }

    def _infer(self, X: np.ndarray):
        """
        Mean probability and epistemic spread from the ensemble.

        Uses the ensemble's own uncertainty API when it has one, so the timing covers the
        code path that serves predictions rather than a reimplementation of it.
        """
        if hasattr(self.ensemble, "predict_with_uncertainty"):
            out = self.ensemble.predict_with_uncertainty(X)
            if isinstance(out, dict):
                return out["mean_prob"], out["epistemic_std"]
            return out[0], out[1]
        member_preds = [m.predict_proba(X) for m in self.ensemble.models]
        return np.mean(member_preds, axis=0), np.std(member_preds, axis=0)


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }
