"""
Unit tests for the front-page numbers.

This module exists because both dashboards hand-typed their headline metrics and
the typed values contradicted the repository's own measured table: the portal
claimed 100.0% failure recall with "0 Escaped Bugs" while
reports/baseline_comparison.csv recorded 40.0% recall and 3 escaped commits.

So the tests are mostly about the absence path. An artifact that has not been
produced has to surface as "not measured", naming the script that would produce
it, and never as a number that looks like a measurement.
"""

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from conftest.evaluation.headline import (
    BASELINE_CSV,
    CALIBRATION_JSON,
    REAL_DATASET,
    SPLIT_METADATA,
    UNCERTAINTY_JSON,
    Headline,
    MissingArtifact,
    as_number,
    calibration_block,
    calibration_headline,
    conftest_row,
    describe_difference,
    failure_recall_headline,
    headline_metrics,
    provenance,
    read_baseline_rows,
    read_json,
    time_reduction_headline,
    uncertainty_headline,
)

HEADERS = [
    "Strategy / Baseline",
    "Test Reduction (TRR %)",
    "Time Reduction (ETR %)",
    "Failure Recall (FR %)",
    "Missed-Failure (MFR %)",
    "Abstention Rate (AR %)",
    "Escaped Commits",
]


def _write_baseline(root: Path, rows: List[List[str]]) -> None:
    """Write reports/baseline_comparison.csv exactly as the benchmark does."""
    path = root / BASELINE_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def _conftest_row_cells(
    trr: str = "68.6%",
    etr: str = "45.2%",
    fr: str = "40.0%",
    mfr: str = "60.0%",
    ar: str = "12.5%",
    escaped: str = "3",
) -> List[str]:
    return ["4. ConfTest (Calibrated + Selective)", trr, etr, fr, mfr, ar, escaped]


def _write_json(root: Path, rel: Path, payload: Dict[str, Any]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------
# Absence: the only behaviour that matters more than the numbers themselves
# --------------------------------------------------------------------------

def test_read_json_names_the_script_that_would_produce_the_file(tmp_path):
    with pytest.raises(MissingArtifact) as excinfo:
        read_json(CALIBRATION_JSON, "python scripts/calibrate_model.py", tmp_path)

    message = str(excinfo.value)
    assert "reports/calibration_report.json" in message
    assert "python scripts/calibrate_model.py" in message
    assert excinfo.value.produced_by == "python scripts/calibrate_model.py"
    # A FileNotFoundError, so callers that already handle a missing report keep
    # working instead of crashing on an unfamiliar exception type.
    assert isinstance(excinfo.value, FileNotFoundError)


def test_read_json_returns_the_payload_when_it_exists(tmp_path):
    _write_json(tmp_path, CALIBRATION_JSON, {"best_method": "isotonic"})

    assert read_json(CALIBRATION_JSON, "irrelevant", tmp_path) == {"best_method": "isotonic"}


def test_read_baseline_rows_preserves_the_cells_verbatim(tmp_path):
    _write_baseline(tmp_path, [_conftest_row_cells()])

    rows = read_baseline_rows(tmp_path)

    assert len(rows) == 1
    # Not coerced here: the '%' is what the file says, and every reader that
    # needs a float goes through as_number.
    assert rows[0]["Failure Recall (FR %)"] == "40.0%"


def test_read_baseline_rows_refuses_to_invent_a_table(tmp_path):
    with pytest.raises(MissingArtifact) as excinfo:
        read_baseline_rows(tmp_path)

    assert "scripts/train_baseline.py" in str(excinfo.value)


# --------------------------------------------------------------------------
# Cell parsing: 'n/a' is a measurement outcome, not a zero
# --------------------------------------------------------------------------

def test_as_number_strips_the_percent_sign():
    assert as_number("68.6%") == 68.6
    assert as_number("  45.2 % ") == 45.2
    assert as_number("3") == 3.0
    assert as_number(12.5) == 12.5


def test_as_number_keeps_undefined_undefined():
    # A recall of 'n/a' means no commit had a failure to recall. Reading that as
    # 0.0 would report a perfect miss where nothing was measurable.
    assert as_number("n/a") is None
    assert as_number("") is None
    assert as_number(None) is None


def test_conftest_row_finds_the_selector_row(tmp_path):
    _write_baseline(tmp_path, [
        ["1. Retest-All", "0.0%", "0.0%", "100.0%", "0.0%", "0.0%", "0"],
        _conftest_row_cells(),
    ])

    row = conftest_row(read_baseline_rows(tmp_path))

    assert row["Strategy / Baseline"] == "4. ConfTest (Calibrated + Selective)"


def test_conftest_row_lists_what_it_did_find(tmp_path):
    # The failure mode this guards against is a renamed strategy silently
    # falling back to another row's numbers.
    _write_baseline(tmp_path, [["1. Retest-All", "0.0%", "0.0%", "100.0%", "0.0%", "0.0%", "0"]])

    with pytest.raises(ValueError) as excinfo:
        conftest_row(read_baseline_rows(tmp_path))

    assert "1. Retest-All" in str(excinfo.value)


def test_calibration_block_accepts_both_key_spellings():
    suffixed = {"test_metrics": {"isotonic_calibration": {"ece": 0.019}}}
    bare = {"test_metrics": {"uncalibrated": {"ece": 0.0258}}}

    assert calibration_block(suffixed, "isotonic")["ece"] == 0.019
    assert calibration_block(bare, "uncalibrated")["ece"] == 0.0258
    assert calibration_block(bare, "platt") == {}
    assert calibration_block({}, "isotonic") == {}


# --------------------------------------------------------------------------
# Paired differences: an interval that spans zero is not a gain
# --------------------------------------------------------------------------

def test_describe_difference_refuses_a_gain_when_the_interval_spans_zero():
    # The real numbers: a 25.47% ECE reduction whose paired difference is
    # -0.00657 [-0.01119, +0.01102]. The reduction is not a finding.
    sentence = describe_difference({
        "point": -0.00657, "ci_lower": -0.01119, "ci_upper": 0.01102,
        "excludes_zero": False,
    })

    assert sentence == "ECE difference -0.00657 [-0.01119, +0.01102] spans zero, so not a gain"


def test_describe_difference_reports_an_interval_that_excludes_zero():
    sentence = describe_difference({
        "point": -0.0142, "ci_lower": -0.0231, "ci_upper": -0.0038,
        "excludes_zero": True,
    })

    assert sentence.endswith("excludes zero")
    assert "-0.01420 [-0.02310, -0.00380]" in sentence


def test_describe_difference_says_so_when_there_is_no_interval():
    assert describe_difference(None) == "no paired interval on the ECE difference"
    assert describe_difference({}) == "no paired interval on the ECE difference"
    # A point estimate with no bounds is the case worth naming: it looks like a
    # result and decides nothing.
    assert describe_difference({"point": -0.006}) == "no paired interval on the ECE difference"
    assert describe_difference(None, metric="Brier") == "no paired interval on the Brier difference"


# --------------------------------------------------------------------------
# Failure recall
# --------------------------------------------------------------------------

def test_failure_recall_reports_the_measured_row(tmp_path):
    _write_baseline(tmp_path, [_conftest_row_cells(fr="40.0%", mfr="60.0%", escaped="3")])

    headline = failure_recall_headline(tmp_path)

    assert headline.measured
    assert headline.value == "40.0%"
    # The portal used to claim 100.0% recall and "0 Escaped Bugs" beside this
    # very row, so the escapes travel with the number that omits them.
    assert headline.note == "3 commit(s) let a failure escape, missed-failure rate 60.0%"
    assert headline.source == "reports/baseline_comparison.csv"


def test_failure_recall_is_unmeasured_without_the_table(tmp_path):
    headline = failure_recall_headline(tmp_path)

    assert not headline.measured
    assert headline.value is None
    assert headline.note == "not measured"
    assert headline.source == "python scripts/train_baseline.py"


def test_failure_recall_undefined_is_not_zero(tmp_path):
    # 'n/a' happens when no commit in the split had a failing test. Rendering
    # that as 0.0% would report a total miss.
    _write_baseline(tmp_path, [_conftest_row_cells(fr="n/a", mfr="n/a", escaped="0")])

    headline = failure_recall_headline(tmp_path)

    assert headline.value is None
    assert "undefined" in headline.note
    assert "no commit in the split had a failing test" in headline.note


# --------------------------------------------------------------------------
# Time reduction: ETR is the number, TRR is never allowed to stand in for it
# --------------------------------------------------------------------------

def test_time_reduction_reports_etr_and_keeps_trr_in_the_note(tmp_path):
    # Conflating these is the single most common overstatement in the RTS
    # literature: skipping 68.6% of the tests does not save 68.6% of the time.
    _write_baseline(tmp_path, [_conftest_row_cells(trr="68.6%", etr="45.2%", ar="12.5%")])

    headline = time_reduction_headline(tmp_path)

    assert headline.value == "45.2%"
    assert "test-count reduction 68.6%" in headline.note
    assert "abstained on 12.5% of commits" in headline.note
    assert "68.6%" != headline.value


def test_time_reduction_is_undefined_without_durations(tmp_path):
    _write_baseline(tmp_path, [_conftest_row_cells(etr="n/a")])

    headline = time_reduction_headline(tmp_path)

    assert headline.value is None
    assert "no per-test durations" in headline.note
    # The count reduction is real and still not a time saving, so it does not
    # get promoted into the empty slot.
    assert headline.source == "reports/baseline_comparison.csv"


def test_time_reduction_is_unmeasured_without_the_table(tmp_path):
    headline = time_reduction_headline(tmp_path)

    assert not headline.measured
    assert headline.note == "not measured"


# --------------------------------------------------------------------------
# Calibration: the served model's ECE, not the best candidate's
# --------------------------------------------------------------------------

def test_calibration_headline_reports_the_served_uncalibrated_model(tmp_path):
    # These are this project's real numbers: the selection declined to
    # calibrate, so the served ECE is the uncalibrated 0.0258 -- not the
    # isotonic candidate's lower value, which was measured and then rejected.
    _write_json(tmp_path, CALIBRATION_JSON, {
        "best_method": "uncalibrated",
        "selection": {"reason": "no candidate's paired ECE interval excluded zero"},
        "test_metrics": {
            "uncalibrated": {"ece": 0.0258, "mce": 0.2222, "brier_score": 0.0449},
            "isotonic_calibration": {"ece": 0.0192},
        },
    })

    headline = calibration_headline(tmp_path)

    assert headline.value == "0.0258"
    assert headline.note == (
        "served uncalibrated: no candidate's paired ECE interval excluded zero"
    )


def test_calibration_headline_carries_the_paired_interval_when_calibrated(tmp_path):
    _write_json(tmp_path, CALIBRATION_JSON, {
        "best_method": "isotonic",
        "test_metrics": {
            "isotonic_calibration": {
                "ece": 0.0192,
                "ece_vs_uncalibrated": {
                    "point": -0.0066, "ci_lower": -0.0112, "ci_upper": 0.0110,
                    "excludes_zero": False,
                },
            },
        },
    })

    headline = calibration_headline(tmp_path)

    assert headline.value == "0.0192"
    assert headline.note.startswith("served isotonic: ECE difference")
    assert "spans zero, so not a gain" in headline.note


def test_calibration_headline_says_when_the_named_method_has_no_ece(tmp_path):
    # A report that names a winner it carries no metrics for is corrupt, and the
    # honest headline is the absence rather than another method's number.
    _write_json(tmp_path, CALIBRATION_JSON, {
        "best_method": "platt",
        "test_metrics": {"uncalibrated": {"ece": 0.0258}},
    })

    headline = calibration_headline(tmp_path)

    assert headline.value is None
    assert headline.note == "report names 'platt' but carries no ECE for it"


def test_calibration_headline_defaults_to_uncalibrated_when_unnamed(tmp_path):
    _write_json(tmp_path, CALIBRATION_JSON, {
        "test_metrics": {"uncalibrated": {"ece": 0.0258}},
    })

    headline = calibration_headline(tmp_path)

    assert headline.value == "0.0258"
    assert headline.note == "served uncalibrated"


def test_calibration_headline_is_unmeasured_without_the_report(tmp_path):
    headline = calibration_headline(tmp_path)

    assert not headline.measured
    assert headline.source == "python scripts/calibrate_model.py"


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def test_uncertainty_headline_reports_the_mean_with_its_tail(tmp_path):
    _write_json(tmp_path, UNCERTAINTY_JSON, {
        "num_samples": 100,
        "mean_epistemic_uncertainty": 0.0083,
        "p95_epistemic_uncertainty": 0.0183,
    })

    headline = uncertainty_headline(tmp_path)

    assert headline.value == "0.0083"
    assert headline.note == "p95 0.0183 over 100 samples"


def test_uncertainty_headline_survives_a_report_without_the_tail(tmp_path):
    _write_json(tmp_path, UNCERTAINTY_JSON, {"mean_epistemic_uncertainty": 0.0083})

    headline = uncertainty_headline(tmp_path)

    assert headline.value == "0.0083"
    assert headline.note == "mean ensemble standard deviation"


def test_uncertainty_headline_says_when_the_report_lacks_the_metric(tmp_path):
    _write_json(tmp_path, UNCERTAINTY_JSON, {"num_samples": 100})

    headline = uncertainty_headline(tmp_path)

    assert headline.value is None
    assert headline.note == "report carries no mean_epistemic_uncertainty"


def test_uncertainty_headline_is_unmeasured_without_the_report(tmp_path):
    headline = uncertainty_headline(tmp_path)

    assert not headline.measured
    assert headline.source == "python scripts/uncertainty_eval.py"


# --------------------------------------------------------------------------
# The front page as a whole
# --------------------------------------------------------------------------

def test_headline_metrics_are_four_in_display_order(tmp_path):
    _write_baseline(tmp_path, [_conftest_row_cells()])
    _write_json(tmp_path, CALIBRATION_JSON, {
        "best_method": "uncalibrated",
        "test_metrics": {"uncalibrated": {"ece": 0.0258}},
    })
    _write_json(tmp_path, UNCERTAINTY_JSON, {"mean_epistemic_uncertainty": 0.0083})

    metrics = headline_metrics(tmp_path)

    assert [h.label for h in metrics] == [
        "Failure Recall",
        "Test Time Reduction",
        "Expected Calibration Error",
        "Epistemic Disagreement",
    ]
    assert all(h.measured for h in metrics)


def test_headline_metrics_against_an_empty_root_are_all_unmeasured(tmp_path):
    # The point of the module. Four "not measured" cells naming four scripts is
    # a usable state; four plausible numbers from nowhere is not.
    metrics = headline_metrics(tmp_path)

    assert len(metrics) == 4
    assert not any(h.measured for h in metrics)
    assert all(h.value is None for h in metrics)
    assert all(h.source.startswith("python scripts/") for h in metrics)


# --------------------------------------------------------------------------
# Provenance: whether a test suite was ever executed
# --------------------------------------------------------------------------

def test_provenance_is_false_without_the_real_dataset(tmp_path):
    result = provenance(tmp_path)

    assert result.real_labels is False
    assert "no test suite was executed" in result.detail
    assert "scripts/build_real_dataset.py --all" in result.detail


def test_provenance_quantifies_what_the_sampled_split_actually_holds(tmp_path):
    _write_json(tmp_path, SPLIT_METADATA, {
        "test": {"num_commits": 100, "positive_failure_samples": 5},
    })

    result = provenance(tmp_path)

    assert result.real_labels is False
    assert "current test split: 100 commits, 5 failing samples" in result.detail


def test_provenance_flips_once_the_measured_dataset_exists(tmp_path):
    dataset = tmp_path / REAL_DATASET
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("commit_sha,test_id,label_failed" + chr(10), encoding="utf-8")

    result = provenance(tmp_path)

    assert result.real_labels is True
    assert "labels measured by executing test suites" in result.detail
    assert "data/processed/real_features.csv" in result.detail


def test_provenance_tolerates_unreadable_split_metadata(tmp_path):
    # A truncated metadata file must not take the whole page down; the claim
    # that matters -- labels are not measured -- does not depend on it.
    path = tmp_path / SPLIT_METADATA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{truncated", encoding="utf-8")

    result = provenance(tmp_path)

    assert result.real_labels is False
    assert "no test suite was executed" in result.detail
    assert "current test split" not in result.detail


def test_headline_dataclass_is_frozen():
    # These get passed straight into Streamlit renderers; a page that could
    # rewrite a value in place would defeat every check above.
    headline = Headline("Failure Recall", "40.0%", "3 commit(s)", "reports/x.csv")

    with pytest.raises(dataclasses.FrozenInstanceError):
        headline.value = "100.0%"
