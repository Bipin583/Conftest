"""Security and behavior tests for the read-only CI failure analyzer."""

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import requests

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_ci_failure.py"
SPEC = importlib.util.spec_from_file_location("analyze_ci_failure", MODULE_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def event(conclusion="failure", pulls=None):
    return {
        "repository": {"full_name": "owner/repo"},
        "workflow_run": {
            "id": 123,
            "html_url": "https://github.com/owner/repo/actions/runs/123",
            "head_sha": "abcdef1234567890",
            "conclusion": conclusion,
            "pull_requests": [{"number": 42}] if pulls is None else pulls,
        },
    }


def archive(entries):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return target.getvalue()


def test_failed_pr_event_is_parsed_and_other_events_are_skipped():
    context = analyzer.parse_failed_pr_run(event())
    assert context.repository == "owner/repo"
    assert context.run_id == 123
    assert context.run_url.endswith("/actions/runs/123")
    assert context.pull_number == 42
    assert context.head_sha == "abcdef1234567890"
    assert analyzer.parse_failed_pr_run(event("success")) is None
    assert analyzer.parse_failed_pr_run(event("cancelled")) is None
    assert analyzer.parse_failed_pr_run(event(pulls=[])) is None


def test_redaction_removes_credentials_and_preserves_diagnostics():
    exact = "exact-secret-value"
    text = """Authorization: Bearer github_pat_abcdefghijklmnopqrstuvwxyz
ANTHROPIC_API_KEY=sk-ant-abcdefghijklmnopqrstuvwxyz
SERVICE_TOKEN=exact-secret-value
fetch https://user:password@example.com/a?token=visible-secret
ERROR test failed
"""
    redacted = analyzer.redact_text(text, [exact])
    assert exact not in redacted
    assert "sk-ant-" not in redacted
    assert "github_pat_" not in redacted
    assert "password" not in redacted
    assert "visible-secret" not in redacted
    assert "ERROR test failed" in redacted
    assert redacted.count("[REDACTED]") >= 5


def test_log_archive_is_read_in_memory_and_reduced_to_error_context():
    logs = archive({"job/step.txt": "safe line\ncontext\nERROR missing package\nafter\nignore me\n"})
    evidence = analyzer.extract_failure_evidence(logs)
    assert "step.txt" in evidence
    assert "ERROR missing package" in evidence
    assert "context" in evidence


def test_archive_rejects_unsafe_member_paths():
    for name in ("../escape.txt", "/absolute.txt", "C:\\escape.txt"):
        logs = archive({name: "ERROR hostile archive"})
        with pytest.raises(analyzer.AnalyzerError, match="unsafe member"):
            analyzer.extract_failure_evidence(logs)


def test_archive_rejects_symlink_members():
    target = io.BytesIO()
    info = zipfile.ZipInfo("linked.txt")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(target, "w") as bundle:
        bundle.writestr(info, "target.txt")
    with pytest.raises(analyzer.AnalyzerError, match="unsafe member"):
        analyzer.extract_failure_evidence(target.getvalue())


def test_archive_rejects_oversized_member(monkeypatch):
    monkeypatch.setattr(analyzer, "MAX_MEMBER_BYTES", 4)
    logs = archive({"step.txt": "ERROR too large"})
    with pytest.raises(analyzer.AnalyzerError, match="per-file"):
        analyzer.extract_failure_evidence(logs)


def test_archive_enforces_file_total_and_prompt_limits(monkeypatch):
    monkeypatch.setattr(analyzer, "MAX_FILES", 1)
    with pytest.raises(analyzer.AnalyzerError, match="too many files"):
        analyzer.extract_failure_evidence(archive({"a": "ERROR a", "b": "ERROR b"}))

    monkeypatch.setattr(analyzer, "MAX_FILES", 100)
    monkeypatch.setattr(analyzer, "MAX_UNCOMPRESSED_BYTES", 8)
    with pytest.raises(analyzer.AnalyzerError, match="uncompressed"):
        analyzer.extract_failure_evidence(archive({"a": "ERROR one", "b": "ERROR two"}))

    monkeypatch.setattr(analyzer, "MAX_UNCOMPRESSED_BYTES", 100)
    monkeypatch.setattr(analyzer, "MAX_LOG_CHARS", 12)
    assert len(analyzer.extract_failure_evidence(archive({"a": "ERROR bounded"}))) == 12


class Response:
    def __init__(self, *, content=b"", payload=None, status=200, headers=None):
        self.content = content
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("request failed")

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content

    def json(self):
        return self._payload


def test_download_uses_run_endpoint_and_bearer_token(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response(content=b"zip")

    monkeypatch.setattr(analyzer.requests, "get", fake_get)
    result = analyzer.download_run_logs(analyzer.parse_failed_pr_run(event()), "github-secret")
    assert result == b"zip"
    assert captured["url"].endswith("/repos/owner/repo/actions/runs/123/logs")
    assert captured["headers"]["Authorization"] == "Bearer github-secret"
    assert captured["allow_redirects"] is True
    assert captured["stream"] is True
    assert captured["timeout"] == 30


def test_download_stops_when_stream_exceeds_archive_limit(monkeypatch):
    monkeypatch.setattr(analyzer, "MAX_ARCHIVE_BYTES", 4)
    monkeypatch.setattr(
        analyzer.requests,
        "get",
        lambda *args, **kwargs: Response(content=b"oversized"),
    )
    with pytest.raises(analyzer.AnalyzerError, match="compressed size"):
        analyzer.download_run_logs(analyzer.parse_failed_pr_run(event()), "github-secret")


def test_network_failures_are_safely_wrapped(monkeypatch):
    monkeypatch.setattr(
        analyzer.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("secret detail")),
    )
    with pytest.raises(analyzer.AnalyzerError, match="could not download") as github_error:
        analyzer.download_run_logs(analyzer.parse_failed_pr_run(event()), "github-secret")
    assert "secret detail" not in str(github_error.value)

    monkeypatch.setattr(analyzer.requests, "post", lambda *args, **kwargs: Response(status=500))
    with pytest.raises(analyzer.AnalyzerError, match="analysis request failed") as api_error:
        analyzer.call_anthropic("evidence", "anthropic-secret")
    assert "anthropic-secret" not in str(api_error.value)


def test_anthropic_request_is_official_and_bounded(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response(payload={
            "content": [{"type": "text", "text": "## Failure\nA test failed."}],
            "stop_reason": "end_turn",
        })

    monkeypatch.setattr(analyzer.requests, "post", fake_post)
    result = analyzer.call_anthropic("evidence", "anthropic-secret")
    assert "A test failed" in result
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "anthropic-secret"
    assert captured["json"]["model"] == "claude-sonnet-5"
    assert captured["json"]["max_tokens"] == 1200
    assert "untrusted" in captured["json"]["system"]


def test_anthropic_rejects_missing_text_and_marks_truncation(monkeypatch):
    monkeypatch.setattr(
        analyzer.requests,
        "post",
        lambda *args, **kwargs: Response(payload={"content": [], "stop_reason": "end_turn"}),
    )
    with pytest.raises(analyzer.AnalyzerError, match="no text"):
        analyzer.call_anthropic("evidence", "key")

    monkeypatch.setattr(
        analyzer.requests,
        "post",
        lambda *args, **kwargs: Response(payload=["not", "a", "message"]),
    )
    with pytest.raises(analyzer.AnalyzerError, match="invalid structure"):
        analyzer.call_anthropic("evidence", "key")

    monkeypatch.setattr(
        analyzer.requests,
        "post",
        lambda *args, **kwargs: Response(
            payload={"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"}
        ),
    )
    assert "truncated" in analyzer.call_anthropic("evidence", "key")


def test_run_skips_before_network_or_model_calls(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event("success")), encoding="utf-8")
    output = tmp_path / "report.md"
    monkeypatch.setattr(
        analyzer,
        "download_run_logs",
        lambda *args: pytest.fail("logs must not be downloaded"),
    )
    monkeypatch.setattr(
        analyzer,
        "call_anthropic",
        lambda *args: pytest.fail("model must not be called"),
    )
    assert analyzer.run(event_path, output, {}) is False
    assert not output.exists()


def test_run_writes_a_finally_redacted_advisory(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event()), encoding="utf-8")
    output = tmp_path / "report.md"
    monkeypatch.setattr(analyzer, "download_run_logs", lambda *args: archive({"x": "ERROR x"}))
    monkeypatch.setattr(
        analyzer,
        "call_anthropic",
        lambda *args: "## Failure\nLeaked exact-anthropic-secret must be fixed.",
    )
    env = {
        "GITHUB_TOKEN": "exact-github-secret",
        "ANTHROPIC_API_KEY": "exact-anthropic-secret",
    }
    assert analyzer.run(event_path, output, env) is True
    report = output.read_text(encoding="utf-8")
    assert report.startswith(analyzer.COMMENT_MARKER)
    assert "actions/runs/123" in report
    assert "read-only recommendation" in report
    assert "exact-anthropic-secret" not in report
    assert "[REDACTED]" in report
