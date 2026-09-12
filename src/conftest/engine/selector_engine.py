"""
ConfTest Unified Regression Test Selection Engine.

End-to-end orchestrator combining repository discovery, 32-feature extraction,
5-seed ensemble uncertainty estimation, post-hoc confidence calibration,
selective policy abstention, test subprocess execution, and database logging.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session

from conftest.db import crud
from conftest.features.pipeline import FeatureExtractionPipeline, FEATURE_NAMES
from conftest.models.ensemble import EnsembleUncertaintyPredictor
from conftest.models.calibration import ConfidenceCalibrator
from conftest.models.policy import SelectivePredictionPolicy, PolicyDecision
from conftest.tests.discovery import PytestDiscovery
from conftest.tests.executor import SafeTestExecutor
from conftest.logging_config import get_logger

logger = get_logger(__name__)


class ConfTestEngine:
    """Core orchestrator executing end-to-end regression test selection."""

    __test__ = False  # Prevent pytest from treating engine as a test suite

    def __init__(
        self,
        repo_root: str,
        ensemble_path: Optional[str] = None,
        calibrator_path: Optional[str] = None,
        policy_config_path: Optional[str] = None,
        default_budget: float = 0.25,
    ):
        """
        Initialize the ConfTest engine with models and tools.

        Args:
            repo_root: Root path of the target code repository.
            ensemble_path: Path to serialized 5-seed ensemble directory.
            calibrator_path: Path to fitted calibrator joblib file.
            policy_config_path: Path to policy JSON config.
            default_budget: Default test budget fraction (e.g. 0.25 = top 25%).
        """
        self.repo_root = str(Path(repo_root).resolve())
        self.default_budget = default_budget

        # 1. Feature extraction pipeline & discovery
        self.discovery = PytestDiscovery(self.repo_root)
        self.feature_pipeline = FeatureExtractionPipeline(self.repo_root)
        self.executor = SafeTestExecutor(self.repo_root)

        # 2. Load or initialize ensemble model
        if ensemble_path and Path(ensemble_path).exists():
            self.ensemble = EnsembleUncertaintyPredictor.load_ensemble(ensemble_path)
        else:
            self.ensemble = None
            logger.warning(f"Ensemble model not found at {ensemble_path}. Using heuristic fallback.")

        # 3. Load or initialize calibrator
        if calibrator_path and Path(calibrator_path).exists():
            self.calibrator = ConfidenceCalibrator.load(calibrator_path)
        else:
            self.calibrator = None
            logger.warning(f"Calibrator not found at {calibrator_path}. Using identity calibration.")

        # 4. Load or initialize selective policy.
        #
        # Without a tuned artifact this used to fall back to tau_abstain=0.030,
        # tau_conf=0.10 -- a permissive operating point that lets fast mode run
        # on thresholds nobody measured, and that differed from every other
        # threshold in the codebase. A missing artifact now yields a policy that
        # always abstains, so a deployment with no tuning runs the full suite
        # instead of quietly selecting a subset at an unvalidated threshold.
        if policy_config_path and Path(policy_config_path).exists():
            self.policy = SelectivePredictionPolicy.load(policy_config_path)
            logger.info(
                f"Loaded tuned policy from {policy_config_path} "
                f"(tau_abstain={self.policy.tau_abstain}, tau_conf={self.policy.tau_conf})"
            )
        else:
            self.policy = SelectivePredictionPolicy.untuned_always_abstain(
                budget_ratio=default_budget,
            )
            logger.warning(
                "No tuned policy at %s. Falling back to unconditional abstention "
                "(full suite every commit): subset selection needs a threshold "
                "whose failure recall has been measured. Run "
                "scripts/tune_policy.py to produce one.",
                policy_config_path,
            )

    def analyze_and_select(
        self,
        commit_sha: str,
        changed_files: List[Dict[str, Any]],
        commit_message: str = "",
        commit_timestamp: Optional[datetime] = None,
        budget_ratio: Optional[float] = None,
        db: Optional[Session] = None,
        repository_id: Optional[int] = None,
        execute: bool = False,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform end-to-end regression test selection for a commit diff.

        Args:
            commit_sha: Commit identifier.
            changed_files: List of file diff dictionaries.
            commit_message: Raw commit message.
            commit_timestamp: Timestamp of the commit.
            budget_ratio: Optional override for test selection budget.
            db: Optional SQLAlchemy DB session for persistence.
            repository_id: Associated DB repository ID.
            execute: If True, executes the selected tests via SafeTestExecutor.
            timeout: Subprocess timeout in seconds.

        Returns:
            Dictionary containing selection decisions, candidate test rankings, and execution outcomes.
        """
        c_time = commit_timestamp or datetime.utcnow()
        active_budget = budget_ratio if budget_ratio is not None else self.default_budget
        self.policy.budget_ratio = active_budget

        logger.info(f"Analyzing commit {commit_sha[:8]} ({len(changed_files)} changed files, budget: {active_budget*100:.0f}%)...")

        # 1. Discover all candidate regression tests in the repository
        raw_tests = self.discovery.discover_via_pytest()
        if not raw_tests:
            logger.warning("No regression tests discovered in repository.")
            return {
                "commit_sha": commit_sha,
                "decision_mode": "SAFE_FULL_SUITE",
                "abstained": False,
                "selected_tests": [],
                "total_tests": 0,
                "reasons": ["No test files found in repository."],
            }

        candidate_test_ids = [t["test_id"] for t in raw_tests]
        total_tests = len(candidate_test_ids)

        # 2. Extract 32-dimensional feature matrix for every (commit, test) pair
        feature_dicts: List[Dict[str, float]] = []
        feature_vectors: List[np.ndarray] = []

        tc_id_map = {}
        if db and repository_id:
            db_tcs = crud.list_test_cases_for_repo(db, repository_id)
            tc_id_map = {tc.test_id: tc.id for tc in db_tcs}

        for t in raw_tests:
            t_id = t["test_id"]
            db_tc_id = tc_id_map.get(t_id)
            f_dict = self.feature_pipeline.extract_features_for_pair(
                test_path=t["test_path"],
                test_function=t["test_function"],
                changed_files=changed_files,
                commit_message=commit_message,
                commit_timestamp=c_time,
                test_case_id=db_tc_id,
                db=db,
            )
            f_vec = self.feature_pipeline.to_feature_vector(f_dict)
            feature_dicts.append(f_dict)
            feature_vectors.append(f_vec)

        X = np.array(feature_vectors, dtype=np.float32)  # Shape: (K, 32)

        # 3. Model Inference: 5-Seed Ensemble Uncertainty Quantification
        if self.ensemble is not None:
            ens_res = self.ensemble.predict_with_uncertainty(X)
            raw_probs = ens_res["mean_prob"]
            uncertainties = ens_res["epistemic_std"]
        else:
            # Fallback heuristic: coupling + historical failure
            raw_probs = np.array([
                float(fd.get("dep_is_direct_import", 0.0) * 0.6 + fd.get("hist_lifetime_failure_rate", 0.0) * 0.4)
                for fd in feature_dicts
            ], dtype=np.float32)
            uncertainties = np.array([0.01] * total_tests, dtype=np.float32)

        # 4. Post-Hoc Confidence Calibration
        if self.calibrator is not None:
            cal_confidences = self.calibrator.calibrate(raw_probs)
        else:
            cal_confidences = raw_probs

        # 5. Evaluate Selective Prediction Policy
        num_changed_files = len(changed_files)
        total_churn = sum(f.get("lines_added", 0) + f.get("lines_deleted", 0) for f in changed_files)

        decision: PolicyDecision = self.policy.evaluate_commit(
            commit_sha=commit_sha,
            candidate_test_ids=candidate_test_ids,
            calibrated_confidences=cal_confidences,
            epistemic_uncertainties=uncertainties,
            num_changed_files=num_changed_files,
            total_churn_lines=total_churn,
        )

        # 6. Build Candidate Test Rankings Data
        ranked_test_details = []
        for i in range(total_tests):
            ranked_test_details.append({
                "test_id": candidate_test_ids[i],
                "raw_score": round(float(raw_probs[i]), 4),
                "calibrated_confidence": round(float(cal_confidences[i]), 4),
                "epistemic_uncertainty": round(float(uncertainties[i]), 4),
                "is_selected": candidate_test_ids[i] in decision.selected_test_ids,
            })
        ranked_test_details.sort(key=lambda x: x["calibrated_confidence"], reverse=True)

        # 7. Persist to Database if session provided
        commit_db_id = None
        if db and repository_id:
            # Get or create commit record
            db_commit = crud.get_commit_by_sha(db, commit_sha)
            if not db_commit:
                db_commit = crud.create_commit(
                    db=db,
                    repository_id=repository_id,
                    sha=commit_sha,
                    message=commit_message,
                    timestamp=c_time,
                )
            commit_db_id = db_commit.id

            # Save Predictions
            pred_payload = []
            for i, item in enumerate(ranked_test_details):
                t_id = item["test_id"]
                tc_id = tc_id_map.get(t_id)
                if not tc_id:
                    parts = t_id.split("::")
                    new_tc = crud.get_or_create_test_case(
                        db=db,
                        repository_id=repository_id,
                        test_id=t_id,
                        test_path=parts[0],
                        test_function=parts[-1] if len(parts) > 1 else "test",
                    )
                    tc_id = new_tc.id
                    tc_id_map[t_id] = tc_id

                pred_payload.append({
                    "test_case_id": tc_id,
                    "model_version": getattr(self.ensemble, "ensemble_version", "lgbm_v1.0.0"),
                    "raw_score": item["raw_score"],
                    "calibrated_confidence": item["calibrated_confidence"],
                    "uncertainty": item["epistemic_uncertainty"],
                    "is_selected": item["is_selected"],
                })

            if pred_payload:
                crud.save_predictions(db, commit_db_id, pred_payload)

            # Save SelectionDecision
            crud.save_selection_decision(
                db=db,
                commit_id=commit_db_id,
                mode=decision.decision_mode,
                abstained=decision.abstained,
                uncertainty_score=decision.epistemic_uncertainty,
                threshold_used=self.policy.tau_abstain,
                selected_count=len(decision.selected_test_ids),
                total_count=total_tests,
                estimated_saving=decision.estimated_time_saved_pct,
                reasons={
                    "reasons": decision.reasons,
                    "top_confidence": decision.top_confidence,
                    # Persist the full operating point and where it came from, so
                    # a stored decision can be reproduced. threshold_used alone
                    # records tau_abstain and silently omits tau_conf, the budget,
                    # and whether any of them were ever tuned.
                    "policy": {
                        "tau_abstain": self.policy.tau_abstain,
                        "tau_conf": self.policy.tau_conf,
                        "budget_ratio": self.policy.budget_ratio,
                        "source": self.policy.source,
                        "is_tuned": self.policy.is_tuned,
                    },
                },
            )

        # 8. Optional Live Test Execution
        execution_outcome = None
        if execute:
            logger.info(f"Executing {len(decision.selected_test_ids)} tests ({decision.decision_mode})...")
            run_res = self.executor.run_tests(test_node_ids=decision.selected_test_ids, timeout=timeout)
            execution_outcome = run_res.to_dict()

            if db and commit_db_id:
                # Log TestRuns to DB
                test_run_payload = []
                for r in run_res.test_runs:
                    t_id = r["test_id"].lstrip("./")
                    tc_id = tc_id_map.get(t_id) or tc_id_map.get(r["test_id"])
                    if not tc_id and repository_id:
                        parts = t_id.split("::")
                        new_tc = crud.get_or_create_test_case(
                            db=db,
                            repository_id=repository_id,
                            test_id=t_id,
                            test_path=parts[0],
                            test_function=parts[-1] if len(parts) > 1 else "test",
                        )
                        tc_id = new_tc.id
                        tc_id_map[t_id] = tc_id

                    if tc_id:
                        test_run_payload.append({
                            "test_case_id": tc_id,
                            "status": r["status"],
                            "duration": r["duration"],
                            "source": "selective_run" if not decision.abstained else "full_suite",
                        })
                if test_run_payload:
                    crud.record_test_runs(db, commit_db_id, test_run_payload)

                # Save Outcome.
                #
                # This engine runs exactly one pass: the selected tests. When
                # abstaining, "selected" is the whole suite, so the full-suite
                # figures are observed and the outcome is verifiable. When it
                # selects a strict subset, the unselected tests never ran, and
                # three quantities are therefore unknown: how many failures the
                # full suite would have shown, how many of them escaped, and how
                # long it would have taken. Only a second, full run can settle
                # them -- see tests/runner_service.py, which does exactly that.
                #
                # The previous version filled all three in anyway, recording
                # missed_failures=0, full_duration=selected_duration, and the
                # policy's *count*-based estimate in the time_reduction column.
                # That turned every selective run into evidence of a perfect,
                # free safety record. save_outcome now rejects those arguments
                # unless ground_truth_complete says they were measured.
                ran_full_suite = len(decision.selected_test_ids) >= total_tests > 0

                if ran_full_suite:
                    crud.save_outcome(
                        db=db,
                        commit_id=commit_db_id,
                        detected_failures=run_res.failed_count,
                        selected_duration=run_res.total_duration,
                        ground_truth_complete=True,
                        # The full suite ran, so it is its own ground truth:
                        # nothing was withheld, hence no escapes and no saving.
                        actual_failures=run_res.failed_count,
                        missed_failures=0,
                        full_duration=run_res.total_duration,
                        time_reduction_ratio=0.0,
                    )
                else:
                    crud.save_outcome(
                        db=db,
                        commit_id=commit_db_id,
                        detected_failures=run_res.failed_count,
                        selected_duration=run_res.total_duration,
                        ground_truth_complete=False,
                    )
                    logger.warning(
                        "Outcome for %s recorded as UNVERIFIED: ran %d of %d tests, so "
                        "escaped failures and full-suite duration are unmeasured. Use "
                        "RunnerService (which also runs the full suite) to obtain a "
                        "verifiable safety outcome.",
                        commit_sha[:8],
                        len(decision.selected_test_ids),
                        total_tests,
                    )

        return {
            "commit_sha": commit_sha,
            "decision_mode": decision.decision_mode,
            "abstained": decision.abstained,
            "selected_count": len(decision.selected_test_ids),
            "total_count": total_tests,
            "test_reduction_pct": decision.estimated_time_saved_pct,
            "top_confidence": decision.top_confidence,
            "epistemic_uncertainty": decision.epistemic_uncertainty,
            "reasons": decision.reasons,
            "selected_test_ids": decision.selected_test_ids,
            "ranked_tests": ranked_test_details,
            "execution_outcome": execution_outcome,
        }
