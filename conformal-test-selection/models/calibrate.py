"""Layer 2: probability calibration.

A gradient-boosted classifier trained with ``scale_pos_weight`` produces scores
that rank well but are not probabilities -- they are systematically pushed away
from the base rate. That is fatal here, because layer 3 turns
``1 - P(fail)`` into a non-conformity score and quantiles it. Conformal
prediction stays *valid* under miscalibration, but its prediction sets become
needlessly large: a badly-scaled score spreads the calibration distribution out,
so the quantile lands further into the tail and the selected subset balloons.
Calibration is what makes the guarantee cheap.

This module fits Platt scaling (a sigmoid) on the validation split -- data the
base model never saw -- and reports Expected Calibration Error before and after.

**ECE definition.** Predictions are placed in ``calibration.ece_bins`` equal-width
confidence bins; ECE is the sample-weighted mean absolute gap between each bin's
mean predicted probability and its observed failure frequency. Maximum
Calibration Error (the worst single bin) is reported alongside it, because a
model can post a flattering ECE while being badly wrong in the high-confidence
bin that actually drives selection.

Example:
    >>> from models.calibrate import calibrate
    >>> report = calibrate()
    >>> report["ece_after"] < report["ece_before"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ArtifactError, get_logger, load_artifact, load_config, resolve_path, save_artifact, set_seed  # noqa: E402
from models.train import classification_metrics, load_split  # noqa: E402

LOGGER = get_logger(__name__)


class CalibrationError(RuntimeError):
    """Raised when calibration cannot be fitted or evaluated."""


class ScoreCalibrator:
    """Wraps a fitted 1-D calibration map over a base model's scores.

    Fitting a sigmoid or isotonic map directly on held-out *scores* keeps the
    expensive base model frozen, which is what split conformal prediction
    assumes: the calibration split must be exchangeable with the test split
    *given* a fixed predictor.

    Args:
        method: ``"platt"`` for a sigmoid fit, ``"isotonic"`` for a monotone
            step fit.

    Attributes:
        method: The configured method name.
        mapper: The fitted transform, available after :meth:`fit`.
    """

    def __init__(self, method: str = "platt") -> None:
        normalised = str(method).lower()
        if normalised in {"platt", "sigmoid"}:
            self.method = "platt"
        elif normalised == "isotonic":
            self.method = "isotonic"
        else:
            raise CalibrationError(f"Unknown calibration method {method!r}; use 'platt' or 'isotonic'.")
        self.mapper: Optional[Any] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        """Fit the calibration map on held-out scores.

        Args:
            scores: Uncalibrated positive-class scores.
            labels: Ground-truth labels in ``{0, 1}``.

        Returns:
            ``self``, fitted.

        Raises:
            CalibrationError: If the labels contain a single class.
        """
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels).ravel()
        if len(np.unique(labels)) < 2:
            raise CalibrationError("Calibration split contains a single class.")

        if self.method == "platt":
            # A sigmoid on the logit of the score, which is the standard Platt
            # form and behaves better than fitting on raw probabilities.
            self.mapper = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self.mapper.fit(_safe_logit(scores), labels)
        else:
            self.mapper = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.mapper.fit(scores.ravel(), labels)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities.

        Args:
            scores: Uncalibrated positive-class scores.

        Returns:
            Calibrated probabilities in ``[0, 1]``.

        Raises:
            CalibrationError: If called before :meth:`fit`.
        """
        if self.mapper is None:
            raise CalibrationError("ScoreCalibrator.transform called before fit().")
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self.method == "platt":
            return self.mapper.predict_proba(_safe_logit(scores))[:, 1]
        return np.asarray(self.mapper.predict(scores.ravel()), dtype=float)

    # Alias so the object is drop-in wherever a probability source is expected.
    predict_proba_positive = transform


def _safe_logit(scores: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Logit transform with clipping so 0 and 1 do not produce infinities.

    Args:
        scores: Probabilities or scores in ``[0, 1]``.
        epsilon: Clipping margin.

    Returns:
        The logit-transformed array, same shape as the input.
    """
    clipped = np.clip(np.asarray(scores, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute ECE, MCE and the reliability-diagram bins.

    Args:
        labels: Ground-truth labels in ``{0, 1}``.
        probabilities: Predicted probability of the positive class.
        n_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        A dictionary with ``ece``, ``mce``, ``n_bins`` and a ``bins`` list of
        per-bin ``{lower, upper, count, mean_predicted, observed, gap}``.

    Raises:
        CalibrationError: If the arrays differ in length or are empty.
    """
    labels = np.asarray(labels, dtype=float).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    if labels.size == 0 or labels.size != probabilities.size:
        raise CalibrationError(
            f"ECE needs equal-length non-empty arrays; got {labels.size} labels and {probabilities.size} probabilities."
        )

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # Bin index in [0, n_bins-1]; the rightmost edge belongs to the last bin.
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)

    total = labels.size
    ece = 0.0
    mce = 0.0
    bins: List[Dict[str, Any]] = []

    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "lower": round(float(edges[b]), 4),
                    "upper": round(float(edges[b + 1]), 4),
                    "count": 0,
                    "mean_predicted": None,
                    "observed": None,
                    "gap": None,
                }
            )
            continue

        mean_predicted = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        gap = abs(mean_predicted - observed)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "lower": round(float(edges[b]), 4),
                "upper": round(float(edges[b + 1]), 4),
                "count": count,
                "mean_predicted": round(mean_predicted, 6),
                "observed": round(observed, 6),
                "gap": round(gap, 6),
            }
        )

    return {"ece": float(ece), "mce": float(mce), "n_bins": int(n_bins), "bins": bins}


def _fit_cv_calibrator(
    base_model: Any,
    features: Any,
    labels: np.ndarray,
    method: str,
    cv_folds: int,
) -> Optional[Any]:
    """Fit scikit-learn's cross-validated calibrator as a cross-check.

    ``CalibratedClassifierCV`` refits clones of the base model across folds,
    which is a different estimator from the frozen model layer 3 relies on. It
    is fitted here for comparison only; the shipped calibrator is the frozen
    :class:`ScoreCalibrator`.

    Args:
        base_model: The fitted base classifier.
        features: Calibration-split features.
        labels: Calibration-split labels.
        method: ``"platt"`` or ``"isotonic"``.
        cv_folds: Number of folds.

    Returns:
        The fitted calibrator, or ``None`` if the fit failed.
    """
    sklearn_method = "sigmoid" if method == "platt" else "isotonic"
    try:
        calibrator = CalibratedClassifierCV(
            estimator=base_model,
            method=sklearn_method,
            cv=int(cv_folds),
        )
        calibrator.fit(features, labels)
        return calibrator
    except Exception as exc:
        LOGGER.warning("CalibratedClassifierCV cross-check failed (%s); continuing without it.", exc)
        return None


def calibrate(
    model_path: Optional[Any] = None,
    calibrator_path: Optional[Any] = None,
    report_path: Optional[Any] = None,
    processed_dir: Optional[Any] = None,
    fit_cv_crosscheck: bool = False,
) -> Dict[str, Any]:
    """Fit calibration on the validation split and report ECE before/after.

    Args:
        model_path: Fitted base model artefact. Defaults to config.
        calibrator_path: Destination for the calibrator artefact.
        report_path: Destination for the JSON report.
        processed_dir: Directory of split CSVs.
        fit_cv_crosscheck: Also fit ``CalibratedClassifierCV`` for comparison.
            Off by default: it refits the base model ``cv_folds`` times.

    Returns:
        The calibration report, also written to ``reports/calibration_report.json``.

    Raises:
        CalibrationError: If the base model artefact is unusable.
    """
    cfg = load_config()
    calibration_cfg = cfg["calibration"]
    set_seed()

    try:
        bundle = load_artifact(model_path or cfg["artifacts"]["model_path"])
    except ArtifactError as exc:
        raise CalibrationError(f"{exc} Run 'python cli.py train' first.") from exc

    base_model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    X_val, y_val, _ = load_split("val", processed_dir)
    X_test, y_test, _ = load_split("test", processed_dir)
    X_val = X_val.loc[:, feature_columns]
    X_test = X_test.loc[:, feature_columns]

    raw_val = base_model.predict_proba(X_val)[:, 1]
    raw_test = base_model.predict_proba(X_test)[:, 1]

    method = str(calibration_cfg.get("method", "platt"))
    n_bins = int(calibration_cfg.get("ece_bins", 15))

    calibrator = ScoreCalibrator(method=method).fit(raw_val, y_val)
    cal_val = calibrator.transform(raw_val)
    cal_test = calibrator.transform(raw_test)

    report: Dict[str, Any] = {
        "method": calibrator.method,
        "cv_folds": int(calibration_cfg.get("cv_folds", 5)),
        "fitted_on": "val",
        "n_calibration_rows": int(len(y_val)),
        "ece_threshold": float(calibration_cfg.get("ece_threshold", 0.08)),
        "val": {
            "before": expected_calibration_error(y_val, raw_val, n_bins),
            "after": expected_calibration_error(y_val, cal_val, n_bins),
        },
        "test": {
            "before": expected_calibration_error(y_test, raw_test, n_bins),
            "after": expected_calibration_error(y_test, cal_test, n_bins),
        },
        "test_metrics": {
            "before": classification_metrics(y_test, raw_test),
            "after": classification_metrics(y_test, cal_test),
        },
    }

    # Headline numbers are the held-out (test) figures: the validation ECE is
    # optimistic because the map was fitted on it.
    report["ece_before"] = report["test"]["before"]["ece"]
    report["ece_after"] = report["test"]["after"]["ece"]
    report["ece_gate_passed"] = bool(report["ece_after"] <= report["ece_threshold"])

    LOGGER.info(
        "ECE on val  : %.4f -> %.4f (MCE %.4f -> %.4f)",
        report["val"]["before"]["ece"], report["val"]["after"]["ece"],
        report["val"]["before"]["mce"], report["val"]["after"]["mce"],
    )
    LOGGER.info(
        "ECE on test : %.4f -> %.4f (MCE %.4f -> %.4f)",
        report["test"]["before"]["ece"], report["test"]["after"]["ece"],
        report["test"]["before"]["mce"], report["test"]["after"]["mce"],
    )
    LOGGER.info(
        "ECE gate (<= %.3f): %s",
        report["ece_threshold"], "PASS" if report["ece_gate_passed"] else "FAIL",
    )

    if fit_cv_crosscheck:
        cv_model = _fit_cv_calibrator(
            base_model, X_val, y_val, calibrator.method, calibration_cfg.get("cv_folds", 5)
        )
        if cv_model is not None:
            cv_test = cv_model.predict_proba(X_test)[:, 1]
            report["cv_crosscheck"] = {
                "estimator": "CalibratedClassifierCV",
                "ece": expected_calibration_error(y_test, cv_test, n_bins)["ece"],
            }
            LOGGER.info("CalibratedClassifierCV cross-check test ECE: %.4f", report["cv_crosscheck"]["ece"])

    destination = save_artifact(
        {"calibrator": calibrator, "feature_columns": feature_columns, "report": report},
        calibrator_path or cfg["artifacts"]["calibrator_path"],
    )
    LOGGER.info("Saved calibrator to %s", destination)

    target = resolve_path(report_path or Path(cfg["artifacts"]["reports_dir"]) / "calibration_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", target)
    return report


def load_calibrated_scorer(
    model_path: Optional[Any] = None,
    calibrator_path: Optional[Any] = None,
) -> Tuple[Any, Any, List[str]]:
    """Load the base model and calibrator as a ready-to-use pair.

    Args:
        model_path: Base model artefact path.
        calibrator_path: Calibrator artefact path.

    Returns:
        A ``(base_model, calibrator, feature_columns)`` triple.

    Raises:
        CalibrationError: If either artefact is missing.
    """
    cfg = load_config()
    try:
        model_bundle = load_artifact(model_path or cfg["artifacts"]["model_path"])
        calibrator_bundle = load_artifact(calibrator_path or cfg["artifacts"]["calibrator_path"])
    except ArtifactError as exc:
        raise CalibrationError(
            f"{exc} Run 'python cli.py train' and 'python cli.py calibrate' first."
        ) from exc
    return model_bundle["model"], calibrator_bundle["calibrator"], model_bundle["feature_columns"]


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m models.calibrate``.

    Returns:
        ``0`` if the ECE gate passed, ``1`` otherwise or on error.
    """
    parser = argparse.ArgumentParser(description="Calibrate the failure-risk model.")
    parser.add_argument("--model-path", default=None, help="Base model artefact.")
    parser.add_argument("--calibrator-path", default=None, help="Destination calibrator path.")
    parser.add_argument("--report-path", default=None, help="Destination report path.")
    parser.add_argument("--processed-dir", default=None, help="Directory of split CSVs.")
    parser.add_argument(
        "--cv-crosscheck",
        action="store_true",
        help="Also fit CalibratedClassifierCV for comparison (slow: refits the base model).",
    )
    args = parser.parse_args(argv)

    try:
        report = calibrate(
            model_path=args.model_path,
            calibrator_path=args.calibrator_path,
            report_path=args.report_path,
            processed_dir=args.processed_dir,
            fit_cv_crosscheck=args.cv_crosscheck,
        )
    except CalibrationError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0 if report["ece_gate_passed"] else 1


if __name__ == "__main__":
    # Re-import under the canonical module name before running. Executing this
    # file as ``python -m models.calibrate`` would otherwise bind ScoreCalibrator to
    # ``__main__``, and joblib records that name inside the artefact -- so the
    # pickle would only load back in a process whose ``__main__`` happens to be
    # this same file. Importing the canonical copy pins ScoreCalibrator.__module__
    # to "models.calibrate" and makes the artefact loadable from anywhere.
    from models.calibrate import main as _main

    raise SystemExit(_main())
