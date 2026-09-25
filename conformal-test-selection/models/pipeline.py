"""End-to-end inference: raw change description in, selected tests out.

The three fitted artefacts -- preprocessor, gradient-boosted model, calibrator --
plus the conformal threshold only mean something when applied in the same order,
to the same columns, as during training. Getting that order wrong is the classic
way a model that scored well offline quietly degrades in production, so the
sequence lives here once and both :mod:`cli` and :mod:`api.server` call it.

    raw records -> feature frame -> preprocessor -> model -> calibrator -> selector

**Partial input is accepted, and reported.** A caller that supplies only a
handful of fields still gets a prediction: the preprocessor's median imputer
fills the rest. That is a genuine convenience and a genuine hazard, because an
imputed feature vector is a prediction about the *average* test, not about the
one that was asked about. Every response therefore carries
``feature_completeness``, and a request below
``selection.min_feature_completeness`` is flagged as degraded rather than
silently answered.

Example:
    >>> from models.pipeline import SelectionPipeline
    >>> pipeline = SelectionPipeline.load()
    >>> result = pipeline.predict([{"test_id": "tests/test_api.py::test_ok",
    ...                             "test_path": "tests/test_api.py",
    ...                             "changed_file_path": "src/api.py"}])
    >>> result["predictions"][0]["selected"]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ArtifactError, get_logger, load_artifact, load_config  # noqa: E402
from data.preprocess import _sanitize_feature_names  # noqa: E402
from models.conformal import ConformalSelector  # noqa: E402

LOGGER = get_logger(__name__)


class PipelineError(RuntimeError):
    """Raised when artefacts are missing or a request cannot be scored."""


@dataclass
class SelectionPipeline:
    """The four fitted artefacts, wired in training order.

    Attributes:
        preprocessor: Fitted column transformer.
        modelled_columns: Raw columns the preprocessor consumes, in order.
        model: Fitted gradient-boosted classifier.
        feature_columns: Transformed column names the model expects, in order.
        calibrator: Fitted probability calibration map.
        selector: Fitted conformal selection rule.
    """

    preprocessor: Any
    modelled_columns: List[str]
    model: Any
    feature_columns: List[str]
    calibrator: Any
    selector: ConformalSelector

    @classmethod
    def load(cls, config: Optional[Dict[str, Any]] = None) -> "SelectionPipeline":
        """Load every artefact named in the configuration.

        Args:
            config: Parsed configuration. Defaults to ``config.yaml``.

        Returns:
            A ready-to-serve pipeline.

        Raises:
            PipelineError: If any artefact is missing, naming the command that
                produces it.
        """
        cfg = config or load_config()
        artefacts = cfg["artifacts"]
        stages = (
            ("preprocessor_path", "python cli.py preprocess"),
            ("model_path", "python cli.py train"),
            ("calibrator_path", "python cli.py calibrate"),
            ("conformal_path", "python cli.py conformal"),
        )

        loaded: Dict[str, Any] = {}
        for key, command in stages:
            try:
                loaded[key] = load_artifact(artefacts[key])
            except ArtifactError as exc:
                raise PipelineError(f"{exc} Run '{command}' first.") from exc

        return cls(
            preprocessor=loaded["preprocessor_path"]["preprocessor"],
            modelled_columns=list(loaded["preprocessor_path"]["modelled_columns"]),
            model=loaded["model_path"]["model"],
            feature_columns=list(loaded["model_path"]["feature_columns"]),
            calibrator=loaded["calibrator_path"]["calibrator"],
            selector=loaded["conformal_path"]["selector"],
        )

    # ---- scoring ---------------------------------------------------------

    def _frame_from_records(self, records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        """Build a model-ready frame, filling absent columns with NaN.

        NaN is the correct filler rather than zero: the preprocessor's imputer
        was fitted to replace missing values with the training median, whereas a
        literal zero would be read as a real measurement.

        Args:
            records: Raw request dictionaries.

        Returns:
            A frame containing exactly :attr:`modelled_columns`, in order.

        Raises:
            PipelineError: If no records were supplied.
        """
        if not records:
            raise PipelineError("No records supplied.")

        frame = pd.DataFrame(list(records))
        for column in self.modelled_columns:
            if column not in frame:
                frame[column] = np.nan
        return frame.loc[:, self.modelled_columns]

    def _completeness(self, records: Sequence[Dict[str, Any]]) -> float:
        """Fraction of modelled columns actually supplied by the caller.

        Args:
            records: Raw request dictionaries.

        Returns:
            A value in ``[0, 1]``; ``1.0`` means nothing had to be imputed.
        """
        if not records:
            return 0.0
        supplied = set()
        for record in records:
            supplied.update(k for k, v in record.items() if v is not None)
        return len(supplied & set(self.modelled_columns)) / max(len(self.modelled_columns), 1)

    def probabilities(self, records: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Compute calibrated failure probabilities.

        Args:
            records: Raw request dictionaries.

        Returns:
            Calibrated ``P(fail)`` per record.

        Raises:
            PipelineError: If transformation or scoring fails.
        """
        frame = self._frame_from_records(records)
        try:
            matrix = self.preprocessor.transform(frame)
            names = _sanitize_feature_names(self.preprocessor.get_feature_names_out())
            transformed = pd.DataFrame(matrix, columns=names).astype(np.float32)
            # Reindex rather than trust column order: a preprocessor refit with
            # a new category emits columns in a different order, and silently
            # feeding those to the model would scramble every feature.
            transformed = transformed.reindex(columns=self.feature_columns, fill_value=0.0)
            raw = self.model.predict_proba(transformed)[:, 1]
            return np.asarray(self.calibrator.transform(raw), dtype=float)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Scoring failed: {exc}") from exc

    def predict(
        self,
        records: Sequence[Dict[str, Any]],
        min_tests: Optional[int] = None,
        max_tests: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Score candidate tests and apply the conformal selection rule.

        Args:
            records: One dictionary per candidate test. Recognised identifier
                keys (``test_id``, ``test_path``) are echoed back.
            min_tests: Minimum tests to return, topping up by descending risk.
                Defaults to ``selection.min_tests``.
            max_tests: Maximum tests to return. Defaults to
                ``selection.max_tests``.

        Returns:
            A dictionary with ``predictions``, ``summary`` and ``guarantee``.

        Raises:
            PipelineError: If scoring fails.
        """
        cfg = load_config()
        selection_cfg = cfg.get("selection", {})
        floor = int(selection_cfg.get("min_tests", 0) or 0) if min_tests is None else int(min_tests)
        cap = selection_cfg.get("max_tests") if max_tests is None else max_tests

        probabilities = self.probabilities(records)
        selected = self.selector.select(probabilities)
        conformal_only = int(selected.sum())

        # The budget is an operational override applied *after* the guarantee.
        # Topping up can only add tests, so it never hurts coverage; a cap can
        # remove a test the rule wanted, which is why the response says whether
        # the returned set is still the certified one.
        n = len(probabilities)
        if floor and conformal_only < min(floor, n):
            spare = np.flatnonzero(~selected)
            deficit = min(floor, n) - conformal_only
            selected[spare[np.argsort(probabilities[spare])[::-1][:deficit]]] = True
        capped = False
        if cap is not None and int(selected.sum()) > int(cap):
            kept = np.flatnonzero(selected)
            drop = kept[np.argsort(probabilities[kept])[: int(selected.sum()) - int(cap)]]
            selected[drop] = False
            capped = True

        completeness = self._completeness(records)
        minimum_completeness = float(selection_cfg.get("min_feature_completeness", 0.0) or 0.0)

        predictions: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            predictions.append(
                {
                    "test_id": record.get("test_id") or record.get("test_path") or f"row_{index}",
                    "failure_probability": round(float(probabilities[index]), 6),
                    "selected": bool(selected[index]),
                    "margin": round(float(probabilities[index] - self.selector.probability_floor), 6),
                }
            )

        summary = {
            "n_candidates": int(n),
            "n_selected": int(selected.sum()),
            "selection_rate": round(float(selected.sum() / n), 6) if n else 0.0,
            "cost_reduction": round(float(1.0 - selected.sum() / n), 6) if n else 0.0,
            "feature_completeness": round(completeness, 4),
            "degraded": bool(completeness < minimum_completeness),
            "budget_capped": capped,
        }
        if summary["degraded"]:
            LOGGER.warning(
                "Request supplied %.0f%% of modelled features (minimum %.0f%%); "
                "the remainder were imputed and the prediction is degraded.",
                100 * completeness, 100 * minimum_completeness,
            )

        return {
            "predictions": predictions,
            "summary": summary,
            "guarantee": self.guarantee_statement(capped=capped),
        }

    def guarantee_statement(self, capped: bool = False) -> Dict[str, Any]:
        """Describe the coverage guarantee attached to this rule.

        Args:
            capped: Whether ``max_tests`` removed a test the rule selected.

        Returns:
            A dictionary stating the guarantee and whether it still holds.
        """
        selector = self.selector
        statement: Dict[str, Any] = {
            "type": selector.guarantee,
            "target_coverage": selector.coverage,
            "confidence": selector.confidence,
            "certified_coverage": selector.certified_coverage,
            "probability_floor": round(float(selector.probability_floor), 6),
            "class_conditional": selector.class_conditional,
            "calibration_size": selector.n_calibration,
        }
        if selector.guarantee == "pac":
            statement["claim"] = (
                f"With probability {selector.confidence:.0%} over the calibration draw, "
                f"at least {selector.coverage:.0%} of failing tests are selected."
            )
        else:
            statement["claim"] = (
                f"Averaged over calibration draws, at least {selector.coverage:.0%} "
                "of failing tests are selected."
            )
        if capped:
            statement["holds"] = False
            statement["note"] = "max_tests removed selected tests; the coverage guarantee no longer applies."
        else:
            statement["holds"] = True
        return statement


def load_pipeline(config: Optional[Dict[str, Any]] = None) -> SelectionPipeline:
    """Convenience wrapper around :meth:`SelectionPipeline.load`.

    Args:
        config: Parsed configuration.

    Returns:
        A loaded pipeline.
    """
    return SelectionPipeline.load(config)


def records_from_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a dataframe into the record format :meth:`predict` expects.

    Args:
        frame: Rows of candidate tests.

    Returns:
        A list of dictionaries with NaN replaced by ``None``.
    """
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def iter_batches(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield fixed-size slices of a sequence.

    Args:
        items: The sequence to chunk.
        size: Maximum slice length.

    Yields:
        Consecutive slices.
    """
    step = max(int(size), 1)
    for start in range(0, len(items), step):
        yield items[start : start + step]
