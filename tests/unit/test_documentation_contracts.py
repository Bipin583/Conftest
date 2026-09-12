"""Contracts that keep public documentation aligned with maintained code."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

from conftest.api.main import app
from conftest.config import Settings
from conftest.features.pipeline import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _without_fenced_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _documented_operations() -> set[tuple[str, str]]:
    text = _without_fenced_code((DOCS / "api.md").read_text(encoding="utf-8")).replace("`", "")
    matches = re.findall(
        r"\b(GET|POST|PUT|PATCH|DELETE)(?:\s*\|\s*|\s+)(/)(?=\s|\||$)|"
        r"\b(GET|POST|PUT|PATCH|DELETE)(?:\s*\|\s*|\s+)(/[^\s|)]+)",
        text,
    )
    return {
        (method or alt_method, (path or alt_path).rstrip("|.,;:"))
        for method, path, alt_method, alt_path in matches
    }


def _openapi_operations() -> set[tuple[str, str]]:
    operations: set[tuple[str, str]] = set()
    for path, path_item in app.openapi()["paths"].items():
        for method in path_item:
            upper_method = method.upper()
            if upper_method in HTTP_METHODS:
                operations.add((upper_method, path))
    return operations


def test_feature_schema_identifiers_match_pipeline_order():
    text = (DOCS / "feature_schema.md").read_text(encoding="utf-8")
    documented = re.findall(r"^\|\s*\d+\s*\|\s*`([a-z0-9_]+)`", text, re.MULTILINE)

    assert documented == FEATURE_NAMES


def test_api_reference_matches_generated_openapi_methods_and_paths():
    assert _documented_operations() == _openapi_operations()


def test_workflow_passes_only_supported_selector_options(monkeypatch):
    import shlex

    workflow = (ROOT / ".github/workflows/conftest.yml").read_text(encoding="utf-8")
    selector_commands: list[str] = []
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        if "python scripts/select_tests.py" not in line:
            continue
        command = line.strip()
        while command.endswith("\\") and index + 1 < len(lines):
            command = command[:-1]
            index += 1
            command += " " + lines[index].strip()
        selector_commands.append(command)

    assert selector_commands, "Workflow does not invoke scripts/select_tests.py"

    from scripts import select_tests

    for command in selector_commands:
        arguments = shlex.split(command, posix=True)[2:]
        monkeypatch.setattr("sys.argv", ["select_tests.py", *arguments])
        select_tests.parse_args()


def test_compose_environment_keys_are_real_settings():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    configured = set(re.findall(r"\bCONFTEST_([A-Z][A-Z0-9_]*)\b", compose))
    supported = {name.upper() for name in Settings.model_fields}

    assert configured <= supported, f"Unknown Compose settings: {sorted(configured - supported)}"
    assert "DATABASE_URL" in configured


def test_dockerfile_environment_keys_are_real_settings():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    configured = set(re.findall(r"\bCONFTEST_([A-Z][A-Z0-9_]*)\b", dockerfile))
    supported = {name.upper() for name in Settings.model_fields}

    assert configured <= supported, f"Unknown Dockerfile settings: {sorted(configured - supported)}"
    assert "DATABASE_URL" in configured


@pytest.mark.parametrize(
    "document",
    sorted(DOCS.glob("*.md")),
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_internal_markdown_links_resolve(document: Path):
    text = _without_fenced_code(document.read_text(encoding="utf-8"))
    missing: list[str] = []
    for raw_target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        path_text = unquote(target.split("#", 1)[0])
        resolved = (ROOT / path_text.lstrip("/")) if path_text.startswith("/") else (document.parent / path_text)
        if not resolved.resolve().exists():
            missing.append(target)

    assert not missing, f"Broken internal links in {document.relative_to(ROOT)}: {missing}"


def test_runtime_config_surfaces_do_not_reintroduce_policy_thresholds():
    surfaces = [ROOT / ".env.example", ROOT / "configs/default.yaml", ROOT / "docker-compose.yml"]
    forbidden_assignments = re.compile(
        r"(?im)^\s*(?:CONFTEST_)?(?:ABSTENTION_THRESHOLD|DEFAULT_RISK_TOLERANCE|"
        r"TAU_ABSTAIN|TAU_CONF|MIN_CONFIDENCE_THRESHOLD)\s*[:=]"
    )
    offenders = [path.relative_to(ROOT).as_posix() for path in surfaces if forbidden_assignments.search(path.read_text(encoding="utf-8"))]

    assert not offenders, f"Policy thresholds must live only in models/policy_config.json: {offenders}"


def test_authoritative_docs_do_not_claim_a_formal_safety_guarantee():
    documents = [ROOT / "README.md", ROOT / "DOCUMENTATION.md", DOCS / "limitations.md"]
    claim = re.compile(
        r"(?i)\b(?:guarante(?:e|es|ed|eing)\s+(?:zero|100\s*%)|"
        r"zero[- ](?:escape|failure)\s+guarantee|formal\s+safety\s+guarantee)\b"
    )
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in documents
        if claim.search(_without_fenced_code(path.read_text(encoding="utf-8")))
    ]

    assert not offenders, f"Documentation must not claim a formal safety guarantee: {offenders}"


def test_authoritative_docs_do_not_call_truthful_mutation_constants_a_broken_extractor():
    documents = [
        ROOT / "AUDIT_REPORT.md",
        DOCS / "audit_report.html",
        DOCS / "dataset.md",
        DOCS / "feature_schema.md",
        DOCS / "limitations.md",
    ]
    false_diagnosis = re.compile(
        r"(?i)(?:defect|bug|broken|failure)\s+in\s+(?:the\s+)?(?:diff[- ]feature\s+)?(?:mining|extractor)"
    )
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in documents
        if false_diagnosis.search(_without_fenced_code(path.read_text(encoding="utf-8")))
    ]

    assert not offenders, (
        "One-file, one-line, no-message mutation inputs make the diff fields "
        f"truthfully constant; do not diagnose invented extraction failures: {offenders}"
    )
