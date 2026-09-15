"""
ConfTest Global Configuration Module.

Provides robust, type-safe settings management using Pydantic Settings.
Environment variables can override defaults, using the 'CONFTEST_' prefix.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The webhook secret shipped in .env.example. Fine for local development, fatal
# in production, where it would let anyone who has read this public repository
# forge a signed webhook. See Settings._reject_insecure_production_defaults.
INSECURE_DEFAULT_WEBHOOK_SECRET = "development_secret_only_change_in_ci"


class Settings(BaseSettings):
    """Application configuration model with default fallbacks."""

    model_config = SettingsConfigDict(
        env_prefix="CONFTEST_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_path` was removed below, but keep this explicit so a future
        # `model_*` setting fails loudly rather than emitting a warning that
        # everyone learns to ignore.
        protected_namespaces=(),
    )

    # Application & Environment
    app_name: str = Field(default="ConfTest", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    env: str = Field(default="development", description="Runtime environment")
    # Set by the Docker build (ARG GIT_SHA); Render instead injects
    # RENDER_GIT_COMMIT at runtime, which /health falls back to. Empty by
    # default so that fallback actually fires -- a truthy default like
    # "unknown" here would shadow it.
    git_sha: str = Field(
        default="",
        description="Git commit SHA baked into the container image at build time.",
    )
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level")

    # API Server Settings
    api_host: str = Field(default="127.0.0.1", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    api_workers: int = Field(default=1, description="FastAPI worker count")

    # Database Configuration (Defaults to SQLite with WAL mode)
    database_url: str = Field(
        default="sqlite:///./data/conftest.db",
        description="Database connection URI",
    )
    db_echo: bool = Field(default=False, description="SQLAlchemy query echoing")

    # ML Artifact Locations
    #
    # These replace paths that were hardcoded at three call sites (the selection
    # route, the webhook route, and the engine's own defaults), which made the
    # deployed model impossible to change without editing source.
    ensemble_path: Path = Field(
        default=PROJECT_ROOT / "models" / "ensembles" / "5_seed_lgbm",
        description="Directory holding the serialized deep-ensemble members and metadata",
    )
    calibrator_path: Path = Field(
        default=PROJECT_ROOT / "models" / "calibrator.joblib",
        description="Serialized confidence calibrator fitted on the validation split",
    )
    policy_config_path: Path = Field(
        default=PROJECT_ROOT / "models" / "policy_config.json",
        description=(
            "Tuned selective-prediction thresholds. This file is the single source "
            "of truth for tau_abstain, tau_conf and budget_ratio; it is written by "
            "scripts/tune_policy.py and must not be duplicated as scalar settings."
        ),
    )
    default_repo_root: Path = Field(
        default=PROJECT_ROOT / "tests" / "sample_suite",
        description="Repository analysed when a request does not name one",
    )

    # Decision Policy
    #
    # There is deliberately no `abstention_threshold` or `default_risk_tolerance`
    # setting here. Both existed, neither was ever read, and both disagreed with
    # the value actually in force: this config advertised an abstention threshold
    # of 0.15 while the tuned policy ran at tau_abstain=0.02 -- a 7.5x difference
    # a reader had no way to detect. Thresholds now come from
    # `policy_config_path` alone, so there is nothing to fall out of sync.
    default_budget_ratio: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Fallback test budget fraction when a request omits one",
    )

    # Security & Integration
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",  # Streamlit dashboard
            "http://127.0.0.1:8501",
        ],
        description=(
            "Origins permitted to call the API with credentials. Set explicitly "
            "per deployment (CONFTEST_CORS_ALLOW_ORIGINS as a JSON list); '*' is "
            "rejected in production because it cannot be combined with cookies."
        ),
    )
    github_webhook_secret: str = Field(
        default=INSECURE_DEFAULT_WEBHOOK_SECRET,
        description=(
            "HMAC SHA-256 secret for GitHub webhook payload verification. The "
            "default is public and is rejected when env is production."
        ),
    )
    github_token: Optional[str] = Field(
        default=None,
        description="Optional GitHub personal access token for higher API rate limits",
    )

    # Directories
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Base directory for raw, processed, and split datasets",
    )
    models_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        description="Base directory for serialized ML models and calibration artifacts",
    )

    @model_validator(mode="after")
    def _reject_insecure_production_defaults(self) -> "Settings":
        """
        Refuse to run in production with the placeholder webhook secret.

        The default secret is committed to .env.example, so in production it is
        equivalent to no secret at all: anyone can compute a valid signature and
        drive the selector. Failing at startup is the only outcome that cannot be
        missed -- a warning in a log would be, and the webhook route would keep
        returning 200 to forged payloads.
        """
        if self.is_production and "*" in self.cors_allow_origins:
            raise ValueError(
                "CONFTEST_CORS_ALLOW_ORIGINS contains '*', which cannot be used "
                "with credentialed requests: any site a developer visits could "
                "then call this API as them. List the dashboard origins instead."
            )
        if self.is_production and self.github_webhook_secret == INSECURE_DEFAULT_WEBHOOK_SECRET:
            raise ValueError(
                "CONFTEST_GITHUB_WEBHOOK_SECRET is still the placeholder from "
                ".env.example, which is public. Set a real secret (e.g. "
                "`python -c \"import secrets; print(secrets.token_hex(32))\"`) "
                "before running with CONFTEST_ENV=production."
            )
        return self

    @property
    def is_production(self) -> bool:
        """Return True if running in production mode."""
        return self.env.lower() in ("production", "prod")

    @property
    def is_testing(self) -> bool:
        """Return True if running under a test harness."""
        return self.env.lower() in ("test", "testing")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retrieve cached global application settings singleton.
    Cached with LRU cache to avoid re-reading disk on every access.
    """
    settings = Settings()
    # Ensure standard directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings


# Global settings singleton instance
settings: Settings = get_settings()
