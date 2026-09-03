"""
ConfTest Post-Hoc Confidence Calibration Module.

Implements Isotonic Regression, Temperature Scaling (Platt Scaling),
Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Reliability Diagram binning for Regression Test Selection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from conftest.logging_config import get_logger

logger = get_logger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary ground-truth labels {0, 1}.
        y_prob: Predicted confidence probabilities in range [0, 1].
        n_bins: Number of equal-width probability bins.

    Returns:
        Tuple of (ece, mce, reliability_diagram_bins).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    n_samples = len(y_true)
    bins_data = []

    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Calculate sample indices in this probability bin
        if bin_idx == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_accuracy = float(np.mean(y_true[in_bin]))
            bin_confidence = float(np.mean(y_prob[in_bin]))
            bin_error = abs(bin_accuracy - bin_confidence)

            ece += (bin_count / n_samples) * bin_error
            mce = max(mce, bin_error)

            bins_data.append({
                "bin_idx": bin_idx,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "sample_count": bin_count,
                "confidence": round(bin_confidence, 4),
                "accuracy": round(bin_accuracy, 4),
                "calibration_gap": round(bin_error, 4),
            })
        else:
            bins_data.append({
                "bin_idx": bin_idx,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "sample_count": 0,
                "confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return float(ece), float(mce), bins_data


class TemperatureScalingCalibrator:
    """Parametric Temperature Scaling calibrator optimizing scalar T > 0 on validation logits."""

    def __init__(self):
        self.temperature: float = 1.0
        # Populated by fit(). A temperature of exactly 1.0 is the identity, so a fit that
        # never moved and a fit that found no useful temperature look the same from the
        # outside unless the search itself is reported.
        self.fit_diagnostics: Dict[str, Any] = {}

    def _logit(self, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Overflow-free logistic. The bounded search evaluates T down to 0.01, where
        logits / T reaches several hundred and the naive exp(-z) overflows."""
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "TemperatureScalingCalibrator":
        """
        Fit optimal temperature T by minimizing negative log-likelihood on validation split.

        Args:
            val_probs: Raw uncalibrated probabilities in [0, 1].
            y_val: Binary ground-truth labels {0, 1}.
        """
        logits = self._logit(val_probs)
        y = np.asarray(y_val).astype(float)

        def nll(temp: float) -> float:
            # log(1 + exp(x)) via logaddexp, so no probability is ever clipped and no
            # intermediate overflows. Clipping to 1e-9 and taking logs used to hand the
            # optimiser a flat region at small T built out of the clip bound rather than
            # out of the data.
            z = logits / max(1e-3, temp)
            loss = np.mean(y * np.logaddexp(0.0, -z) + (1.0 - y) * np.logaddexp(0.0, z))
            return float(loss)

        # The search is derivative-free and runs in log-temperature.
        #
        # This used to be L-BFGS-B started at T = 1.0 with a finite-difference gradient.
        # On the ablation's validation probabilities that gradient came out below pgtol at
        # the starting point, so the optimiser returned T = 1.0 after zero iterations and
        # reported success -- while T = 0.75 scored NLL 0.1901 against 0.2094 for the
        # identity it returned. An unmoved optimiser is indistinguishable from a model that
        # needs no calibration, and the column was still published as "calibrated".
        res = minimize_scalar(
            lambda u: nll(float(np.exp(u))),
            bounds=(np.log(0.01), np.log(10.0)),
            method="bounded",
            options={"xatol": 1e-6},
        )
        candidate = float(np.clip(np.exp(res.x), 0.01, 10.0))
        nll_identity = nll(1.0)
        nll_candidate = nll(candidate)

        if nll_candidate <= nll_identity:
            self.temperature = candidate
        else:
            # A search that cannot beat the identity has not found a temperature.
            self.temperature = 1.0

        self.fit_diagnostics = {
            "optimiser": "scipy.optimize.minimize_scalar(method='bounded') on log T",
            "converged": bool(res.success),
            "iterations": int(getattr(res, "nit", 0)),
            "temperature": self.temperature,
            "nll_at_identity": nll_identity,
            "nll_at_fitted": min(nll_candidate, nll_identity),
            "nll_improvement": float(nll_identity - min(nll_candidate, nll_identity)),
            "moved_from_identity": bool(abs(self.temperature - 1.0) > 1e-6),
            "n_validation_rows": int(len(y)),
            "validation_positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        }
        if not self.fit_diagnostics["moved_from_identity"]:
            logger.warning(
                "Temperature scaling stayed at T = 1.0: no temperature in [0.01, 10] "
                f"beat the identity on {len(y)} validation rows (NLL {nll_identity:.6f}). "
                "The calibrated column is the uncalibrated model."
            )
        else:
            logger.info(
                f"Fitted Temperature Scaling calibrator: T = {self.temperature:.4f} "
                f"(NLL {nll_identity:.6f} -> {nll_candidate:.6f})"
            )
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling transformation."""
        logits = self._logit(probs)
        scaled = logits / max(1e-3, self.temperature)
        return self._sigmoid(scaled).astype(np.float32)


class IsotonicCalibrator:
    """Non-parametric piecewise monotonic calibration using Isotonic Regression."""

    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic step regression on validation predictions."""
        self.regressor.fit(val_probs, y_val)
        logger.info("Fitted Isotonic Regression calibrator.")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated empirical probabilities."""
        calibrated = self.regressor.predict(probs)
        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


class ConfidenceCalibrator:
    """Unified Confidence Calibration Manager supporting Isotonic and Temperature Scaling."""

    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.

        Args:
            method: 'isotonic' or 'temperature_scaling'.
        """
        self.method = method.lower()
        if self.method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif self.method in ("temperature", "temperature_scaling"):
            self.calibrator = TemperatureScalingCalibrator()
        elif self.method == "platt":
            # Platt scaling fits a slope and an intercept; temperature scaling fits one
            # scalar. Aliasing them would let a report claim it compared two methods
            # when it had fitted the same model twice.
            raise ValueError(
                "Platt scaling is not implemented. It fits sigmoid(a * logit + b), which "
                "is not temperature scaling's sigmoid(logit / T); use "
                "'temperature_scaling' or 'isotonic'."
            )
        else:
            raise ValueError(
                f"Unknown calibration method: {method}. "
                "Choose 'isotonic' or 'temperature_scaling'."
            )

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ConfidenceCalibrator":
        """Fit calibration model on validation predictions."""
        self.calibrator.fit(val_probs, y_val)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate input probability array."""
        return self.calibrator.calibrate(probs)

    def save(self, filepath: str) -> str:
        """Serialize calibrator to disk."""
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(out_path))
        logger.info(f"Calibrator saved to {out_path}")
        return str(out_path)

    @classmethod
    def load(cls, filepath: str) -> "ConfidenceCalibrator":
        """Load serialized calibrator from disk."""
        in_path = Path(filepath).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {in_path}")
        instance = joblib.load(str(in_path))
        logger.info(f"Loaded {instance.method} calibrator from {in_path}")
        return instance
