"""
ConfTest SYNTHETIC repository generator -- PIPELINE SMOKE TESTS ONLY.

WARNING: every test outcome this module emits is a coin flip, not an observation.

    fail_prob = failure_rate if is_affected else 0.005
    is_fail = self.rng.random() < fail_prob

Nothing here executes a test suite. A test is marked FAILED because a uniform
draw came in under a threshold that the caller chose, so the "coupling" between
a changed module and its failing tests is an assumption written into the
sampler, not a fact recovered from a repository. A model scored on this data is
being measured against the generator's own prior. Those numbers MUST NOT appear
in the report.

Legitimate uses: exercising the database and collector plumbing, shape-checking
what the API and dashboard consume, and unit-test fixtures.

For real ground truth use conftest.groundtruth.mutation_harness, which injects a
fault into source, RUNS pytest, and records which tests actually failed. See
MASTER_PLAN.md section 2.
"""

from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from conftest.db import crud
from conftest.logging_config import get_logger

logger = get_logger(__name__)

# Same tag as src/benchmark/dataset_generator.py, so one grep finds every row in
# the system whose label was sampled rather than measured.
SYNTHETIC_ORIGIN_TAG = "SYNTHETIC_FABRICATED_LABELS"


class SyntheticRepositoryGenerator:
    """
    Fabricates commit histories with SAMPLED (not measured) test outcomes.

    Construction requires `acknowledge_synthetic=True` so this generator cannot
    be wired into an evaluation by accident. Every emitted commit, test run and
    dataset carries `data_origin=SYNTHETIC_FABRICATED_LABELS`.
    """

    def __init__(self, random_seed: int = 42, acknowledge_synthetic: bool = False):
        """
        Args:
            random_seed: fixed seed, so the fabrication is at least reproducible.
            acknowledge_synthetic: must be True. The keyword exists to make the
                caller state, at the call site, that it knows these labels are
                drawn. There is no default that quietly says yes.
        """
        if not acknowledge_synthetic:
            raise ValueError(
                "SyntheticRepositoryGenerator fabricates test outcomes with "
                "rng.random() < fail_prob and must not be used for evaluation. Pass "
                "acknowledge_synthetic=True if you are building a fixture or smoke "
                "test. For real labels use conftest.groundtruth.mutation_harness."
            )
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

    def generate_repository_suite(
        self,
        repo_name: str = "synthetic/demo-app",
        n_commits: int = 50,
        n_tests: int = 30,
        failure_rate: float = 0.12,
        flaky_rate: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic repository dataset.

        Every returned test-run status is a Bernoulli draw against `failure_rate`,
        so the returned dataset measures this sampler and nothing else.

        Args:
            repo_name: Synthetic repository identifier.
            n_commits: Number of commits to synthesize.
            n_tests: Number of regression test cases.
            failure_rate: Base failure probability on modified code.
            flaky_rate: Probability of flaky non-deterministic outcomes.

        Returns:
            Dictionary containing repository metadata, commits, test cases, and test runs.
        """
        logger.info(f"Generating synthetic dataset for {repo_name} (Seed: {self.random_seed})...")

        # 1. Define synthetic modules and corresponding tests
        modules = [
            {"module": "app/auth.py", "tests": ["tests/test_auth.py::test_login", "tests/test_auth.py::test_jwt", "tests/test_auth.py::test_logout"]},
            {"module": "app/database.py", "tests": ["tests/test_db.py::test_pool", "tests/test_db.py::test_migration", "tests/test_db.py::test_session"]},
            {"module": "app/payment.py", "tests": ["tests/test_payment.py::test_stripe", "tests/test_payment.py::test_fees", "tests/test_payment.py::test_refund"]},
            {"module": "app/router.py", "tests": ["tests/test_router.py::test_routes", "tests/test_router.py::test_cors", "tests/test_router.py::test_middleware"]},
            {"module": "app/utils.py", "tests": ["tests/test_utils.py::test_format", "tests/test_utils.py::test_crypto", "tests/test_utils.py::test_parser"]},
        ]

        # Expand or slice test cases to match n_tests
        all_test_cases: List[Dict[str, Any]] = []
        for mod in modules:
            for t_id in mod["tests"]:
                all_test_cases.append({
                    "test_id": t_id,
                    "test_path": t_id.split("::")[0],
                    "test_function": t_id.split("::")[1],
                    "framework": "pytest",
                    "module": mod["module"],
                    "average_duration": round(self.rng.uniform(0.05, 1.8), 3),
                    "flaky_indicator": round(self.rng.uniform(0.0, 0.08) if self.rng.random() < flaky_rate else 0.0, 3),
                })

        if len(all_test_cases) > n_tests:
            all_test_cases = all_test_cases[:n_tests]
        else:
            while len(all_test_cases) < n_tests:
                idx = len(all_test_cases) + 1
                mod = self.rng.choice(modules)
                t_id = f"tests/test_generated_{idx}.py::test_case_{idx}"
                all_test_cases.append({
                    "test_id": t_id,
                    "test_path": f"tests/test_generated_{idx}.py",
                    "test_function": f"test_case_{idx}",
                    "framework": "pytest",
                    "module": mod["module"],
                    "average_duration": round(self.rng.uniform(0.02, 0.8), 3),
                    "flaky_indicator": 0.0,
                })

        # 2. Generate chronological commits
        start_time = datetime.utcnow() - timedelta(days=n_commits * 2)
        commits: List[Dict[str, Any]] = []

        for c_idx in range(n_commits):
            c_time = start_time + timedelta(hours=c_idx * 12 + self.rng.randint(1, 6))
            sha = f"synth_{c_idx:04d}_{self.rng.getrandbits(32):08x}"
            parent_sha = commits[-1]["sha"] if commits else None

            # Choose 1 to 3 modified modules
            changed_modules = self.rng.sample(modules, k=self.rng.randint(1, min(3, len(modules))))
            changed_files = []
            for ch in changed_modules:
                added = self.rng.randint(2, 80)
                deleted = self.rng.randint(0, 30)
                changed_files.append({
                    "file_path": ch["module"],
                    "change_type": "MODIFIED",
                    "lines_added": added,
                    "lines_deleted": deleted,
                    "cyclomatic_complexity": round(self.rng.uniform(1.5, 8.0), 1),
                })

            # Simulate test outcomes: tests coupled with modified modules have higher failure probability
            test_runs = []
            commit_has_failure = False
            total_duration = 0.0

            for tc in all_test_cases:
                is_affected = any(ch["module"] == tc.get("module") for ch in changed_modules)
                fail_prob = failure_rate if is_affected else 0.005

                if tc["flaky_indicator"] > 0.0:
                    fail_prob += tc["flaky_indicator"]

                is_fail = self.rng.random() < fail_prob
                status = "FAILED" if is_fail else "PASSED"
                if is_fail:
                    commit_has_failure = True

                duration = max(0.01, tc["average_duration"] + self.rng.normalvariate(0, 0.02))
                total_duration += duration

                test_runs.append({
                    "test_id": tc["test_id"],
                    "status": status,
                    "data_origin": SYNTHETIC_ORIGIN_TAG,
                    "duration": round(duration, 3),
                    "retry_count": 1 if (is_fail and tc["flaky_indicator"] > 0) else 0,
                    "is_affected": is_affected,
                })

            ci_status = "failed" if commit_has_failure else "passed"

            commits.append({
                "sha": sha,
                "parent_sha": parent_sha,
                "data_origin": SYNTHETIC_ORIGIN_TAG,
                "timestamp": c_time,
                "message": f"feat/fix: synthetic modification to {[ch['module'] for ch in changed_modules]}",
                "ci_status": ci_status,
                "total_duration": round(total_duration, 2),
                "changed_files": changed_files,
                "test_runs": test_runs,
            })

        return {
            "metadata": {
                "data_origin": SYNTHETIC_ORIGIN_TAG,
                "labels_are": "sampled from rng.random(), never measured by running a test",
                "generated_at": datetime.utcnow().isoformat(),
                "random_seed": self.random_seed,
                "repository": repo_name,
                "total_commits": len(commits),
                "total_test_cases": len(all_test_cases),
                "intended_use": "smoke tests and fixtures only; never the report",
            },
            "test_cases": all_test_cases,
            "commits": commits,
        }

    def persist_to_database(self, db: Session, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Save synthesized dataset directly into ConfTest relational database."""
        meta = dataset["metadata"]
        repo = crud.get_repository_by_name(db, meta["repository"])
        if not repo:
            repo = crud.create_repository(
                db=db,
                full_name=meta["repository"],
                url="https://github.com/example/synthetic-demo",
                local_path="./data/raw/synthetic_demo",
                language="python",
            )

        # Map test cases
        tc_map: Dict[str, int] = {}
        for tc_data in dataset["test_cases"]:
            tc = crud.get_or_create_test_case(
                db=db,
                repository_id=repo.id,
                test_id=tc_data["test_id"],
                test_path=tc_data["test_path"],
                test_function=tc_data["test_function"],
                framework=tc_data["framework"],
            )
            tc.average_duration = tc_data["average_duration"]
            tc.flaky_indicator = tc_data["flaky_indicator"]
            tc_map[tc_data["test_id"]] = tc.id
        db.commit()

        # Persist commits, diffs, and runs
        commits_saved = 0
        runs_saved = 0
        for c_data in dataset["commits"]:
            commit = crud.get_commit_by_sha(db, c_data["sha"])
            if not commit:
                ts = c_data["timestamp"]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)

                commit = crud.create_commit(
                    db=db,
                    repository_id=repo.id,
                    sha=c_data["sha"],
                    parent_sha=c_data["parent_sha"],
                    timestamp=ts,
                    message=c_data["message"],
                    ci_status=c_data["ci_status"],
                    total_duration=c_data["total_duration"],
                )
                crud.add_changed_files(db, commit.id, c_data["changed_files"])

                runs_payload = [
                    {
                        "test_case_id": tc_map[tr["test_id"]],
                        "status": tr["status"],
                        "duration": tr["duration"],
                        "retry_count": tr["retry_count"],
                        "source": "synthetic_generator",
                    }
                    for tr in c_data["test_runs"]
                    if tr["test_id"] in tc_map
                ]
                crud.record_test_runs(db, commit.id, runs_payload)
                commits_saved += 1
                runs_saved += len(runs_payload)

        return {
            "repository_id": repo.id,
            "commits_saved": commits_saved,
            "test_cases": len(tc_map),
            "test_runs_saved": runs_saved,
        }
