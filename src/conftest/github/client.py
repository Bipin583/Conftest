"""
ConfTest GitHub REST API Client.

Provides safe, authenticated/unauthenticated API queries with:
- Automatic token masking in logs.
- Exponential backoff and retry handling.
- Rate-limit detection and proactive backoff.
- Local JSON response caching.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from conftest.config import settings
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def mask_token(token: Optional[str]) -> str:
    """Mask sensitive GitHub API tokens for secure logging."""
    if not token or len(token) < 8:
        return "None"
    return f"{token[:4]}...{token[-4:]}"


class GitHubClient:
    """Safe GitHub API client with rate-limiting, retries, and disk caching."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 15,
    ):
        """
        Initialize the GitHub client.

        Args:
            token: Optional GitHub Personal Access Token.
            cache_dir: Directory to cache JSON responses.
            max_retries: Number of exponential backoff retry attempts.
            timeout: Request timeout in seconds.
        """
        self.token = token or settings.github_token
        self.max_retries = max_retries
        self.timeout = timeout

        # Cache directory setup
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = settings.data_dir / "interim" / "github_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure session headers
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ConfTest-RTS-Research-Tool/0.1.0",
        })
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            logger.info(f"GitHub client initialized with token: {mask_token(self.token)}")
        else:
            logger.info("GitHub client initialized in unauthenticated mode (60 req/hr rate limit).")

    def _get_cache_path(self, endpoint: str, params: Optional[Dict[str, Any]]) -> Path:
        """Generate a safe local cache path for an API endpoint."""
        key = f"{endpoint}_{json.dumps(params, sort_keys=True)}" if params else endpoint
        safe_name = key.replace("/", "_").replace("?", "_").replace("=", "_").replace("&", "_")
        return self.cache_dir / f"{safe_name[:120]}.json"

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True
    ) -> Optional[Any]:
        """
        Send a GET request to the GitHub API with caching, rate-limit check, and retries.

        Args:
            endpoint: API endpoint (e.g. '/repos/pallets/flask/commits').
            params: Optional query parameters.
            use_cache: If True, check and return local cached response first.

        Returns:
            Parsed JSON response or None on failure.
        """
        cache_path = self._get_cache_path(endpoint, params)
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{self.BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                # Check rate limit headers
                remaining = response.headers.get("X-RateLimit-Remaining")
                if remaining and int(remaining) == 0:
                    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                    sleep_seconds = max(1, reset_time - int(time.time()))
                    logger.warning(f"GitHub rate limit reached. Waiting {sleep_seconds}s until reset...")
                    if sleep_seconds < 120:  # Only wait if under 2 minutes
                        time.sleep(sleep_seconds + 1)
                        continue
                    else:
                        logger.error("GitHub rate limit reset too far in future. Aborting request.")
                        return None

                if response.status_code == 200:
                    data = response.json()
                    # Cache successful response
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    except Exception as exc:
                        logger.warning(f"Failed to cache response: {exc}")
                    return data
                elif response.status_code == 404:
                    logger.warning(f"GitHub resource not found: {endpoint}")
                    return None
                elif response.status_code in (401, 403):
                    logger.error(f"GitHub authentication error ({response.status_code}): {response.text}")
                    return None
                else:
                    logger.warning(f"GitHub request {endpoint} returned status {response.status_code}, retrying...")

            except requests.RequestException as exc:
                logger.warning(f"GitHub request attempt {attempt}/{self.max_retries} failed: {exc}")

            time.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to fetch {endpoint} after {self.max_retries} attempts.")
        return None

    def get_repo_metadata(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Fetch general repository metadata (stars, default branch, language)."""
        return self.get(f"/repos/{owner}/{repo}")

    def get_pull_request_files(
        self, owner: str, repo: str, pull_number: int, per_page: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch the changed files of a pull request in ConfTest's feature format.

        A `pull_request` webhook payload does not carry the file list, so callers
        that need the diff must ask for it. Returning None means the diff could
        not be established -- callers must treat that as unknown and fall back to
        the full suite, never as an empty or assumed diff.

        Args:
            owner: Repository owner login.
            repo: Repository name.
            pull_number: Pull request number.
            per_page: Page size; GitHub caps this endpoint at 100 files per page
                and 3000 files total.

        Returns:
            List of {file_path, change_type, lines_added, lines_deleted}, or None
            if the request failed.
        """
        collected: List[Dict[str, Any]] = []
        page = 1
        while True:
            batch = self.get(
                f"/repos/{owner}/{repo}/pulls/{pull_number}/files",
                params={"per_page": per_page, "page": page},
            )
            if batch is None:
                # Distinguish a failed request from a genuinely empty diff: a
                # partial list would silently understate the change.
                logger.error(
                    f"Could not fetch changed files for {owner}/{repo}#{pull_number} "
                    f"(page {page}); returning None so the caller can fall back."
                )
                return None
            if not isinstance(batch, list):
                logger.error(
                    f"Unexpected payload shape for PR files: {type(batch).__name__}"
                )
                return None

            for entry in batch:
                # GitHub statuses: added / removed / modified / renamed /
                # copied / changed / unchanged. ConfTest's extractor expects the
                # single-letter git codes.
                status_map = {"added": "A", "removed": "D", "renamed": "R"}
                collected.append({
                    "file_path": entry.get("filename", ""),
                    "change_type": status_map.get(entry.get("status", "modified"), "M"),
                    "lines_added": int(entry.get("additions", 0) or 0),
                    "lines_deleted": int(entry.get("deletions", 0) or 0),
                })

            if len(batch) < per_page:
                break
            page += 1

        logger.info(
            f"Fetched {len(collected)} changed files for {owner}/{repo}#{pull_number}"
        )
        return collected

    def get_commit_ci_status(self, owner: str, repo: str, commit_sha: str) -> str:
        """
        Fetch the CI status for a commit from GitHub Statuses and Check Runs.

        Returns:
            One of: 'passed', 'failed', 'pending', 'unknown'.
        """
        data = self.get(f"/repos/{owner}/{repo}/commits/{commit_sha}/status")
        if data and "state" in data:
            state = data["state"].lower()
            if state == "success":
                return "passed"
            elif state in ("failure", "error"):
                return "failed"
            elif state == "pending":
                return "pending"
        return "unknown"
