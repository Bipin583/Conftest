"""
Unit tests for the ConfTest configuration loader and settings management.
"""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest.config import PROJECT_ROOT, Settings, get_settings


def test_default_settings_instantiation():
    """Verify that default settings instantiate with expected values."""
    settings = Settings()
    assert settings.app_name == "ConfTest"
    assert settings.version == "0.1.0"
    assert settings.api_port == 8000
    assert settings.default_budget_ratio == 0.25
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.models_dir, Path)

    # Artifact locations are settings, not literals buried in route modules.
    assert settings.ensemble_path.name == "5_seed_lgbm"
    assert settings.calibrator_path.name == "calibrator.joblib"
    assert settings.policy_config_path.name == "policy_config.json"


def test_settings_do_not_duplicate_policy_thresholds():
    """
    Decision thresholds must live only in the tuned policy artifact.

    `abstention_threshold` (0.15) and `default_risk_tolerance` (0.18) were config
    fields that nothing read, while the policy actually ran at tau_abstain=0.02.
    Re-adding either would recreate a second, contradictory source of truth.
    """
    settings = Settings()
    assert not hasattr(settings, "abstention_threshold")
    assert not hasattr(settings, "default_risk_tolerance")
    assert settings.policy_config_path.name.endswith(".json")


def test_production_rejects_placeholder_webhook_secret():
    """The public .env.example secret must not be usable in production."""
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(env="production")

    # A real secret is accepted.
    ok = Settings(env="production", github_webhook_secret="s" * 64)
    assert ok.is_production is True


def test_production_rejects_wildcard_cors_origin():
    """'*' plus credentials is both unsafe and rejected by browsers."""
    with pytest.raises(ValidationError, match="CORS"):
        Settings(
            env="production",
            github_webhook_secret="s" * 64,
            cors_allow_origins=["*"],
        )


def test_settings_environment_flags():
    """Verify environment detection helpers."""
    dev_settings = Settings(env="development")
    assert dev_settings.is_production is False
    assert dev_settings.is_testing is False

    # Production requires a real webhook secret, so supply one here; the refusal
    # itself is covered by test_production_rejects_placeholder_webhook_secret.
    prod_settings = Settings(env="production", github_webhook_secret="s" * 64)
    assert prod_settings.is_production is True
    assert prod_settings.is_testing is False

    test_settings = Settings(env="testing")
    assert test_settings.is_production is False
    assert test_settings.is_testing is True


def test_get_settings_singleton():
    """Verify that get_settings() returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# --------------------------------------------------------------------------
# .env.example must describe the settings that exist -- no more, no fewer
# --------------------------------------------------------------------------

ENV_KEY = re.compile(r"^(CONFTEST_[A-Z0-9_]+)=", re.M)


def _template_keys():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    return {key[len("CONFTEST_"):].lower() for key in ENV_KEY.findall(text)}


def test_env_template_documents_no_setting_that_does_not_exist():
    """
    A template key with no Settings field behind it is documentation of a knob
    that does nothing. Three such keys shipped: CONFTEST_MODEL_PATH,
    CONFTEST_ABSTENTION_THRESHOLD and CONFTEST_DEFAULT_RISK_TOLERANCE. The
    abstention one was the damaging kind -- it advertised 0.15 while the tuned
    policy ran at 0.02, so a reader configuring safety was editing a dead value.
    """
    unknown = sorted(_template_keys() - set(Settings.model_fields))

    assert not unknown, f".env.example documents settings that do not exist: {unknown}"


def test_env_template_documents_every_configurable_setting():
    """A setting absent from the template is one a deployer cannot discover."""
    # app_name and version identify the build; they are not deployment inputs.
    expected = set(Settings.model_fields) - {"app_name", "version"}
    missing = sorted(expected - _template_keys())

    assert not missing, f".env.example omits configurable settings: {missing}"


def _readme_env_block_keys():
    """CONFTEST_* keys documented in the README's configuration ini block."""
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("```ini\n")
    end = text.index("\n```", start)
    return {key[len("CONFTEST_"):].lower() for key in ENV_KEY.findall(text[start:end])}


def test_readme_documents_exactly_the_real_settings():
    """
    The README's env block is a third copy of the same list, so it drifts too.

    Restating the README against measured results, the block was hand-written and
    immediately invented `CONFTEST_ENVIRONMENT` and `CONFTEST_SECRET_KEY` (neither
    is a field) while dropping five real keys. A reader who pastes that block gets
    a `.env` that silently does nothing for two lines and omits the artifact paths
    the API needs. Pin it to `Settings` the same way `.env.example` is pinned.
    """
    documented = _readme_env_block_keys()
    assert documented, "README ini block found no CONFTEST_* keys -- did the block move?"

    invented = sorted(documented - set(Settings.model_fields))
    assert not invented, (
        f"README documents settings that do not exist: {invented}. "
        f"Remove them or add the fields."
    )

    template = _template_keys()
    assert documented == template, (
        f"README and .env.example disagree.\n"
        f"  only in README:        {sorted(documented - template)}\n"
        f"  only in .env.example:  {sorted(template - documented)}"
    )
