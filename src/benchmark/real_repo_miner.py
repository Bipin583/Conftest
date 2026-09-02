"""
ConfTest Real-World Repository & Commit History Miner
Mines real git commit diffs, modified files, and test suites from open-source Python/Java repositories.
Formats extracted features into strict 70/15/15 chronological temporal train/cal/test CSV datasets.
"""
import os
import sys
import subprocess
import ast
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.ast_parser import ASTDiffAnalyzer
from src.features.dependency_graph import StaticDependencyGraph
from src.features.history_miner import HistoryMiner

class RealRepoMiner:
    """
    Extracts commit diffs, file churn, AST node changes, and maps test files from real git repositories.
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.ast_analyzer = ASTDiffAnalyzer(language="python")
        self.dep_graph = StaticDependencyGraph()
        self.history_miner = HistoryMiner(decay_rate=0.05)

    def get_commit_hashes(self, max_commits: int = 200) -> List[str]:
        """Returns list of chronological commit SHAs from oldest to newest."""
        try:
            cmd = ["git", "log", "--reverse", "--pretty=format:%H", f"-n {max_commits}"]
            res = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
            hashes = [h.strip() for h in res.stdout.splitlines() if h.strip()]
            return hashes
        except Exception as e:
            raise RuntimeError(
                f"git log failed in {self.repo_path!r}: {e}. Mining cannot "
                "proceed without real commit history. Fabricating commit names "
                "would stamp synthetic data as REAL_GIT_MINED."
            ) from e

    def extract_commit_diff(self, commit_sha: str) -> str:
        """Extracts unified diff patch for a given commit."""
        try:
            cmd = ["git", "show", "--pretty=format:", commit_sha]
            res = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
            return res.stdout
        except Exception:
            return ""

    def discover_test_files(self) -> List[str]:
        """Discovers all test files in the repository."""
        test_files = []
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root or ".venv" in root:
                continue
            for f in files:
                if (f.startswith("test_") or f.endswith("_test.py")) and f.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, f), self.repo_path)
                    test_files.append(rel_path)
        if not test_files:
            raise RuntimeError(
                f"No test files found under {self.repo_path!r}. Mining cannot "
                "proceed against an invented test suite."
            )
        return sorted(test_files)

    def mine_and_build_dataset(self, max_commits: int = 250, output_csv: str = "data/real_repo_benchmark.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extracts features for all (Commit, Test) pairs and produces temporal train/cal/test splits.
        """
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        commit_shas = self.get_commit_hashes(max_commits)
        test_files = self.discover_test_files()
        print(f"[RealRepoMiner] Mining {len(commit_shas)} commits across {len(test_files)} test files in '{self.repo_path}'...")

        records: List[Dict[str, Any]] = []
        skipped_empty = 0

        for idx, sha in enumerate(commit_shas):
            patch = self.extract_commit_diff(sha)
            if not patch:
                # An empty diff is a real observation (merge commit, or a commit
                # touching only binary files). Skip it rather than inventing
                # plausible-looking churn numbers for it.
                print(f"[RealRepoMiner] Skipping {sha}: empty diff (merge or binary-only commit)")
                skipped_empty += 1
                continue
            patch_metrics = self.ast_analyzer.parse_patch(patch)

            is_ood = (patch_metrics["total_churn"] > 800 or patch_metrics["modified_files_count"] > 15)
            churn = patch_metrics["total_churn"]
            mod_files = patch_metrics.get("modified_files", ["src/core.py"])
            n_mod_files = patch_metrics["modified_files_count"]

            for t_file in test_files:
                dep_score = self.dep_graph.compute_dependency_score(t_file, mod_files)
                direct_match = 1 if dep_score >= 0.8 else 0
                hist_fail = self.history_miner.compute_time_decayed_failure_rate(t_file)
                duration = self.history_miner.compute_average_duration(t_file)
                flakiness = self.history_miner.compute_flakiness_score(t_file)

                # NO LABEL IS PRODUCED HERE, BY DESIGN.
                #
                # This class mines *features* from git history. It never executes
                # the test suite, so it has no way to observe whether a test
                # passes or fails. The previous implementation drew a coin flip
                # (p=0.70 when the change touched a dependency, else p=0.40 or
                # p=hist_fail*0.1) and wrote it to a `label` column while
                # stamping the output "REAL_GIT_MINED".
                #
                # That made the whole evaluation vacuous: a p=0.70 Bernoulli
                # draw is ~30% irreducibly unpredictable, so no feature could
                # ever beat that ceiling -- which is why six structurally
                # different baselines all scored exactly 20% recall.
                #
                # Worse, the fabricated label was fed straight back into
                # history_miner.record_run(), so `historical_failure_rate`
                # became a smoothed function of past coin flips: the feature
                # was contaminated, not merely the target.
                #
                # Real labels come from conftest.groundtruth.mutation_harness,
                # which injects a fault, RUNS the suite, and records which tests
                # actually failed. See MASTER_PLAN.md section 2.

                records.append({
                    "commit_id": idx,
                    "commit_sha": sha,
                    "test_name": t_file,
                    "lines_added": patch_metrics["total_added_lines"],
                    "lines_deleted": patch_metrics["total_deleted_lines"],
                    "total_churn": churn,
                    "modified_files_count": n_mod_files,
                    "ast_node_delta": int(churn * 0.4),
                    "has_interface_change": int("class " in patch),
                    "has_import_change": int("import " in patch or "from " in patch),
                    "direct_dependency_match": direct_match,
                    "dependency_overlap_score": dep_score,
                    "historical_failure_rate": hist_fail,
                    "avg_test_duration": duration,
                    "flakiness_score": flakiness,
                    "is_ood_refactoring": int(is_ood),
                    "is_doc_only": int(patch_metrics["is_doc_only"])
                })

        if not records:
            raise RuntimeError(
                f"Mining produced 0 records from {len(commit_shas)} commits "
                f"({skipped_empty} had empty diffs). Check that {self.repo_path!r} "
                "is a real checkout with history and discoverable tests."
            )

        df = pd.DataFrame(records)

        # Structural guard: this miner must never emit a target column. If one
        # reappears here it means label fabrication was reintroduced.
        forbidden = {"label", "label_failed", "is_fail", "failed", "outcome"}
        present = forbidden & set(df.columns)
        if present:
            raise AssertionError(
                f"real_repo_miner emitted target column(s) {sorted(present)}. "
                "This miner never executes tests and therefore cannot know "
                "outcomes. Labels come from conftest.groundtruth.mutation_harness."
            )

        df.to_csv(output_csv, index=False)
        print(
            f"[RealRepoMiner] Exported {len(df)} FEATURE-ONLY records "
            f"(no labels) to '{output_csv}'. "
            f"Mined {len(commit_shas) - skipped_empty} commits, "
            f"skipped {skipped_empty} with empty diffs."
        )

        # Chronological 70% Train / 15% Calibration / 15% Test Split
        n_c = len(commit_shas)
        split_train = int(n_c * 0.70)
        split_cal = int(n_c * 0.85)

        train_df = df[df["commit_id"] < split_train].copy()
        cal_df = df[(df["commit_id"] >= split_train) & (df["commit_id"] < split_cal)].copy()
        test_df = df[df["commit_id"] >= split_cal].copy()

        return train_df, cal_df, test_df

if __name__ == "__main__":
    miner = RealRepoMiner(repo_path=".")
    train_df, cal_df, test_df = miner.mine_and_build_dataset(max_commits=200, output_csv="data/real_repo_benchmark.csv")
    print(f"Split distribution: Train={len(train_df)} | Cal={len(cal_df)} | Test={len(test_df)}")
