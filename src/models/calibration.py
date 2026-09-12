"""
ConfTest Confidence Calibration & Uncertainty Estimation Module
Member 3 Technical Domain: Temperature Scaling, Venn-Abers Predictors, ECE.

.. deprecated:: superseded

    NOT AUTHORITATIVE. This module is an earlier implementation kept only so
    that the tests written against it still run. The maintained implementation
    is **conftest.models.calibration** (TemperatureScalingCalibrator, IsotonicCalibrator and ConfidenceCalibrator, selected between by conftest.models.calibrator_selection).

    Two trees with the same responsibility is how a repository ends up
    measuring one and shipping the other: fixes land in `src/conftest/` and are
    invisible here, so a number produced by this module can silently disagree
    with the published pipeline. Import the replacement in new code. Nothing
    under `src/conftest/`, `scripts/`, or `dashboard/` imports this module, and
    nothing new should.
"""
__superseded_by__ = "conftest.models.calibration"

import numpy as np
from scipy.optimize import minimize
from typing import Tuple

class TemperatureCalibrator:
    """
    Post-hoc Temperature Scaling for probability calibration.
    P_cal(Fail) = sigmoid(logit / T)
    """
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Optimizes temperature T on a held-out calibration set by minimizing Negative Log-Likelihood (NLL).
        """
        def nll_loss(t):
            temp = t[0]
            scaled = logits / temp
            probs = 1.0 / (1.0 + np.exp(-scaled))
            # Clip for numerical stability
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return loss

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        return self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Transforms raw logits into calibrated probabilities.
        """
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


class UncertaintyEstimator:
    """
    Computes Epistemic Uncertainty via ensemble predictive variance and calibration intervals.
    """
    @staticmethod
    def compute_ensemble_uncertainty(predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        predictions: shape (N_samples, N_models)
        Returns: (mean_probabilities, epistemic_uncertainties)
        """
        mean_p = np.mean(predictions, axis=1)
        uncertainty = np.std(predictions, axis=1)
        return mean_p, uncertainty

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE).
        """
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(probs)

        for i in range(n_bins):
            bin_mask = (probs >= bin_limits[i]) & (probs < bin_limits[i + 1])
            bin_count = np.sum(bin_mask)
            if bin_count > 0:
                bin_acc = np.mean(labels[bin_mask])
                bin_conf = np.mean(probs[bin_mask])
                ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

        return float(ece)
