"""
The documented import paths must resolve, and must resolve to the real thing.

`conftest.features.extractor` and `conftest.engine.abstention` are the module
names the documentation uses; the implementations live in
`conftest.features.pipeline` and in `conftest.models.policy` +
`conftest.engine.selector_engine`. The compatibility modules exist so that a
reader following the docs gets working code instead of an ImportError.

A re-export shim that nothing imports is a shim that can silently rot: rename
`FeatureExtractionPipeline` and the shim keeps importing until someone actually
follows the docs. These tests are that someone. They also pin the property that
makes the shims safe -- that they re-export rather than reimplement, so the two
names cannot disagree about how a feature is computed or when to abstain.
"""

import pytest

from conftest.engine import abstention
from conftest.engine.selector_engine import ConfTestEngine
from conftest.features import extractor
from conftest.features.pipeline import FEATURE_NAMES, FeatureExtractionPipeline
from conftest.models.policy import PolicyDecision, SelectivePredictionPolicy


def test_extractor_re_exports_the_pipeline_itself():
    """Same object, not a subclass or a copy: one implementation, two names."""
    assert extractor.FeatureExtractionPipeline is FeatureExtractionPipeline
    assert extractor.FeatureExtractor is FeatureExtractionPipeline
    assert extractor.FEATURE_NAMES is FEATURE_NAMES


def test_abstention_re_exports_the_policy_itself():
    assert abstention.SelectivePredictionPolicy is SelectivePredictionPolicy
    assert abstention.AbstentionPolicy is SelectivePredictionPolicy
    assert abstention.PolicyDecision is PolicyDecision
    assert abstention.ConfTestEngine is ConfTestEngine


@pytest.mark.parametrize(
    "module, names",
    [
        (extractor, extractor.__all__),
        (abstention, abstention.__all__),
    ],
)
def test_everything_declared_in_all_is_importable(module, names):
    """An `__all__` naming something absent breaks `from module import *`."""
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"{module.__name__}.__all__ names absent attributes: {missing}"


@pytest.mark.parametrize("module", [extractor, abstention])
def test_the_shims_define_no_logic_of_their_own(module):
    """
    A shim that grows a function is no longer a shim -- it is a second
    implementation, and the two will drift. Everything these modules expose must
    have been defined somewhere else.
    """
    # Functions and classes carry __module__; plain data (a list of feature
    # names, a sentinel string) does not, and is not a drift risk -- re-exporting
    # a constant is the one thing a re-export shim is *for*.
    def _defines_here(name):
        obj = getattr(module, name)
        return getattr(obj, "__module__", None) == module.__name__

    home_grown = [name for name in module.__all__ if _defines_here(name)]
    assert not home_grown, (
        f"{module.__name__} defines {home_grown} itself instead of re-exporting. "
        "Move the implementation to the maintained module and re-export it."
    )


def test_the_abstention_docstring_does_not_invent_a_default_threshold():
    """
    The one number a reader will copy out of this module is tau. There is no
    single default -- the constructor's values are placeholders and the shipped
    operating point is models/policy_config.json -- so the docstring must send
    them to the artifact rather than to a constant.
    """
    doc = abstention.__doc__ or ""

    assert "models/policy_config.json" in doc
    assert "scripts/tune_policy.py" in doc
    # The untuned constructor default must not be presented as the operating point.
    assert "is_tuned" in doc
