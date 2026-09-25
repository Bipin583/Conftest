"""Harvest test-execution history from GitHub Actions.

Walks the workflow runs of a list of repositories, downloads the JUnit-style
test reports attached to each run, and flattens them into one row per
(commit, test) observation:

===================  ========================================================
Column               Meaning
===================  ========================================================
``repo``             ``owner/name`` of the source repository
``commit_sha``       Head SHA of the workflow run
``run_id``           GitHub Actions run identifier
``test_id``          ``path::class::function`` node identifier
``test_path``        File the test lives in
``test_function``    Test function name
``label_failed``     1 if the test failed or errored, else 0
``duration``         Reported wall-clock seconds for the test
``timestamp``        Run creation time, UTC, ISO-8601
===================  ========================================================

Output lands in ``data/raw/<owner>__<name>.csv`` plus a combined
``data/raw/test_history.csv``.

The REST API allows 5,000 authenticated requests per hour. This module sleeps
between requests and blocks until the quota resets when the remaining budget
drops below ``collection.rate_limit_buffer``, so a long harvest degrades into
slowness rather than a wall of 403s.

Example:
    $ export GITHUB_TOKEN=ghp_...
    $ python -m data.collect --repos data/repos.txt --max-runs 200
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from xml.etree import ElementTree

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, load_config, resolve_path  # noqa: E402

LOGGER = get_logger(__name__)

#: Artefact names that usually hold a JUnit XML report.
_JUNIT_ARTIFACT_HINTS = ("junit", "test-result", "test_results", "pytest", "report")

#: pytest short-summary lines, e.g. "FAILED tests/test_x.py::test_y - AssertionError"
_PYTEST_SUMMARY = re.compile(
    r"^(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+"
    r"(?P<node>[\w./\\-]+\.py(?:::[\w\[\]<>., -]+)+)",
    re.MULTILINE,
)


class CollectionError(RuntimeError):
    """Raised when the harvest cannot proceed (bad credentials, no repos)."""


@dataclass(frozen=True)
class TestObservation:
    """A single observed test execution.

    Attributes:
        repo: ``owner/name`` slug.
        commit_sha: Head commit of the workflow run.
        run_id: GitHub Actions run identifier.
        test_id: Fully-qualified node id.
        test_path: Source file containing the test.
        test_class: Enclosing class, empty for module-level tests.
        test_function: Test function name.
        label_failed: 1 for failure/error, 0 for pass.
        duration: Reported seconds, 0.0 when unknown.
        timestamp: ISO-8601 UTC run creation time.
    """

    repo: str
    commit_sha: str
    run_id: int
    test_id: str
    test_path: str
    test_class: str
    test_function: str
    label_failed: int
    duration: float
    timestamp: str


class RateLimiter:
    """Politeness delay plus a hard block when the API quota runs low.

    Args:
        github: An authenticated ``github.Github`` client.
        delay_seconds: Minimum seconds between successive API calls.
        buffer: Sleep until the quota resets when fewer than this many
            requests remain in the hour.
    """

    def __init__(self, github: Any, delay_seconds: float = 1.0, buffer: int = 100) -> None:
        self._github = github
        self._delay = max(0.0, float(delay_seconds))
        self._buffer = max(0, int(buffer))
        self._last_call = 0.0

    def wait(self) -> None:
        """Throttle before the next API call, blocking if the quota is low."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.monotonic()

        try:
            core = self._github.get_rate_limit().core
        except Exception as exc:  # network hiccup, not fatal
            LOGGER.debug("Could not read rate limit (%s); continuing.", exc)
            return

        if core.remaining > self._buffer:
            return

        reset_at = core.reset
        if reset_at.tzinfo is None:
            reset_at = reset_at.replace(tzinfo=timezone.utc)
        sleep_for = max(0.0, (reset_at - datetime.now(timezone.utc)).total_seconds()) + 5.0
        LOGGER.warning(
            "Rate limit nearly exhausted (%d left). Sleeping %.0fs until %s.",
            core.remaining,
            sleep_for,
            reset_at.isoformat(),
        )
        time.sleep(sleep_for)


def _split_node_id(node_id: str, classname: str = "") -> Tuple[str, str, str]:
    """Break a test identifier into (path, class, function).

    Handles both pytest node ids (``tests/test_a.py::TestB::test_c``) and JUnit
    ``classname``/``name`` pairs (``tests.test_a.TestB`` / ``test_c``).

    Args:
        node_id: Node id or bare test name.
        classname: JUnit ``classname`` attribute, if available.

    Returns:
        A ``(test_path, test_class, test_function)`` triple.
    """
    if "::" in node_id:
        parts = node_id.split("::")
        path = parts[0]
        function = parts[-1]
        klass = parts[1] if len(parts) > 2 else ""
        return path, klass, function

    if classname:
        segments = classname.split(".")
        # A trailing CamelCase segment is conventionally the class.
        if len(segments) > 1 and segments[-1][:1].isupper():
            klass = segments[-1]
            path = "/".join(segments[:-1]) + ".py"
        else:
            klass = ""
            path = "/".join(segments) + ".py"
        return path, klass, node_id

    return "", "", node_id


def parse_junit_xml(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract per-test outcomes from a JUnit XML document.

    Skipped cases are dropped: they were never executed, so they carry no
    pass/fail signal and would dilute the failure rate.

    Args:
        xml_bytes: Raw XML content.

    Returns:
        One dictionary per ``<testcase>``, with ``test_id``, ``test_path``,
        ``test_class``, ``test_function``, ``label_failed`` and ``duration``.
        Returns an empty list if the document does not parse.
    """
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        LOGGER.debug("Skipping unparseable JUnit XML: %s", exc)
        return []

    rows: List[Dict[str, Any]] = []
    for case in root.iter("testcase"):
        name = case.get("name", "")
        if not name:
            continue
        if case.find("skipped") is not None:
            continue

        classname = case.get("classname", "")
        failed = int(case.find("failure") is not None or case.find("error") is not None)
        try:
            duration = float(case.get("time", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = 0.0

        path, klass, function = _split_node_id(name, classname)
        node = case.get("file") or path
        test_id = f"{node}::{klass}::{function}" if klass else f"{node}::{function}"
        rows.append(
            {
                "test_id": test_id,
                "test_path": node,
                "test_class": klass,
                "test_function": function,
                "label_failed": failed,
                "duration": duration,
            }
        )
    return rows


def parse_pytest_log(text: str) -> List[Dict[str, Any]]:
    """Recover test outcomes from a raw pytest console log.

    Fallback for repositories that do not upload a JUnit artefact. Only the
    short-summary lines are trusted; durations are unavailable from this source
    and are reported as ``0.0``.

    Args:
        text: Decoded job log.

    Returns:
        One dictionary per recognised test line, deduplicated by node id.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for match in _PYTEST_SUMMARY.finditer(text):
        outcome = match.group("outcome")
        if outcome in {"SKIPPED", "XFAIL", "XPASS"}:
            continue
        node = match.group("node")
        path, klass, function = _split_node_id(node)
        seen[node] = {
            "test_id": node,
            "test_path": path,
            "test_class": klass,
            "test_function": function,
            "label_failed": int(outcome in {"FAILED", "ERROR"}),
            "duration": 0.0,
        }
    return list(seen.values())


def _iter_run_reports(run: Any, limiter: RateLimiter) -> Iterator[List[Dict[str, Any]]]:
    """Yield parsed test rows for one workflow run.

    Args:
        run: A ``github.WorkflowRun.WorkflowRun``.
        limiter: Shared rate limiter.

    Yields:
        Lists of test-row dictionaries, one list per report file found.
    """
    limiter.wait()
    try:
        artifacts = list(run.get_artifacts())
    except Exception as exc:
        LOGGER.debug("Run %s: no artifacts listing (%s).", run.id, exc)
        return

    for artifact in artifacts:
        if not any(hint in artifact.name.lower() for hint in _JUNIT_ARTIFACT_HINTS):
            continue
        limiter.wait()
        try:
            status, _, raw = artifact._requester.requestBlob(
                "GET", artifact.archive_download_url
            )
            if status != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for member in archive.namelist():
                    if member.lower().endswith(".xml"):
                        rows = parse_junit_xml(archive.read(member))
                        if rows:
                            yield rows
        except (zipfile.BadZipFile, KeyError, AttributeError, OSError) as exc:
            LOGGER.debug("Run %s artefact %s unusable: %s", run.id, artifact.name, exc)
        except Exception as exc:
            LOGGER.debug("Run %s artefact %s failed: %s", run.id, artifact.name, exc)


def collect_repo(
    github: Any,
    repo_slug: str,
    limiter: RateLimiter,
    max_runs: int = 200,
) -> pd.DataFrame:
    """Harvest test-execution history for one repository.

    Args:
        github: Authenticated ``github.Github`` client.
        repo_slug: ``owner/name``.
        limiter: Shared rate limiter.
        max_runs: Upper bound on workflow runs to inspect.

    Returns:
        A DataFrame of :class:`TestObservation` rows; empty if nothing parsed.
    """
    observations: List[TestObservation] = []
    try:
        limiter.wait()
        repo = github.get_repo(repo_slug)
        limiter.wait()
        runs = repo.get_workflow_runs(status="completed")
    except Exception as exc:
        LOGGER.warning("Skipping %s: %s", repo_slug, exc)
        return pd.DataFrame()

    inspected = 0
    for run in runs:
        if inspected >= max_runs:
            break
        inspected += 1

        created = run.created_at or datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        timestamp = created.astimezone(timezone.utc).isoformat()

        for rows in _iter_run_reports(run, limiter):
            for row in rows:
                observations.append(
                    TestObservation(
                        repo=repo_slug,
                        commit_sha=run.head_sha,
                        run_id=run.id,
                        timestamp=timestamp,
                        **row,
                    )
                )

    if not observations:
        LOGGER.warning("%s: inspected %d runs, extracted 0 test rows.", repo_slug, inspected)
        return pd.DataFrame()

    LOGGER.info("%s: %d test rows from %d runs.", repo_slug, len(observations), inspected)
    return pd.DataFrame([asdict(obs) for obs in observations])


def read_repo_list(path: Any) -> List[str]:
    """Read newline-delimited ``owner/name`` slugs, ignoring blanks and comments.

    Args:
        path: Path to the repository list.

    Returns:
        A list of repository slugs.

    Raises:
        CollectionError: If the file is missing or contains no usable entry.
    """
    source = resolve_path(path)
    if not source.is_file():
        raise CollectionError(f"Repository list not found: {source}")

    slugs = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not slugs:
        raise CollectionError(f"{source} lists no repositories.")
    return slugs


def collect(
    repos: Optional[Iterable[str]] = None,
    token: Optional[str] = None,
    max_runs: Optional[int] = None,
    max_repos: Optional[int] = None,
    output_dir: Optional[Any] = None,
) -> pd.DataFrame:
    """Run the full harvest and write CSVs into the raw data directory.

    Args:
        repos: Repository slugs. Defaults to ``collection.repos_file``.
        token: GitHub personal access token. Defaults to ``$GITHUB_TOKEN``.
        max_runs: Workflow runs per repository. Defaults to config.
        max_repos: Repository cap. Defaults to config.
        output_dir: Destination directory. Defaults to ``data.raw_dir``.

    Returns:
        The combined DataFrame across all repositories; empty on total failure.

    Raises:
        CollectionError: If PyGitHub is unavailable or no token is supplied.
    """
    try:
        from github import Github
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise CollectionError("PyGitHub is required: pip install pygithub==1.59.0") from exc

    cfg = load_config()
    collection_cfg = cfg.get("collection", {})

    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise CollectionError(
            "A GitHub token is required. Set $GITHUB_TOKEN or pass --token. "
            "Unauthenticated access is capped at 60 requests/hour, which cannot "
            "complete a harvest of this size."
        )

    if repos is not None:
        slugs = list(repos)
    else:
        slugs = read_repo_list(collection_cfg.get("repos_file", "data/repos.txt"))
    slugs = slugs[: int(max_repos or collection_cfg.get("max_repos", 55))]
    runs_cap = int(max_runs or collection_cfg.get("max_runs_per_repo", 200))

    destination = resolve_path(output_dir or cfg["data"]["raw_dir"])
    destination.mkdir(parents=True, exist_ok=True)

    github = Github(token, per_page=100)
    limiter = RateLimiter(
        github,
        delay_seconds=float(collection_cfg.get("request_delay_seconds", 1.0)),
        buffer=int(collection_cfg.get("rate_limit_buffer", 100)),
    )

    frames: List[pd.DataFrame] = []
    for slug in tqdm(slugs, desc="Harvesting repositories", unit="repo"):
        try:
            frame = collect_repo(github, slug, limiter, max_runs=runs_cap)
        except Exception as exc:
            LOGGER.warning("Unhandled error on %s: %s", slug, exc)
            continue
        if frame.empty:
            continue
        frame.to_csv(destination / (slug.replace("/", "__") + ".csv"), index=False)
        frames.append(frame)

    if not frames:
        LOGGER.error("Harvest produced no rows across %d repositories.", len(slugs))
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined_path = destination / "test_history.csv"
    combined.to_csv(combined_path, index=False)
    LOGGER.info(
        "Wrote %d rows (%d repos, %d commits, %.2f%% failing) to %s",
        len(combined),
        combined["repo"].nunique(),
        combined["commit_sha"].nunique(),
        100.0 * combined["label_failed"].mean(),
        combined_path,
    )
    return combined


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m data.collect``."""
    parser = argparse.ArgumentParser(description="Harvest GitHub Actions test history.")
    parser.add_argument("--repos", default=None, help="File of owner/name slugs.")
    parser.add_argument("--token", default=None, help="GitHub token (default: $GITHUB_TOKEN).")
    parser.add_argument("--max-runs", type=int, default=None, help="Workflow runs per repository.")
    parser.add_argument("--max-repos", type=int, default=None, help="Repository cap.")
    parser.add_argument("--output-dir", default=None, help="Destination directory.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` on a handled collection error or empty harvest.
    """
    args = build_parser().parse_args(argv)
    try:
        frame = collect(
            repos=read_repo_list(args.repos) if args.repos else None,
            token=args.token,
            max_runs=args.max_runs,
            max_repos=args.max_repos,
            output_dir=args.output_dir,
        )
    except CollectionError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0 if not frame.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
