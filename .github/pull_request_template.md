## Description
<!-- Briefly describe the rationale, scope, and implementation of this pull request. -->

## Selective Prediction & ML Considerations
- [ ] Any model, calibrator, or policy thresholds altered? (Must update models/policy_config.json if so)
- [ ] Were any synthetic/fabricated failure labels introduced? (Forbidden by scripts/check_no_fabricated_labels.py)
- [ ] Does any change affect the 32-feature extraction pipeline order? (Must preserve FEATURE_NAMES order)
- [ ] Are safe abstention defaults preserved? (SAFE_FULL_SUITE must remain the default under uncertainty)

## Verification & Testing
<!-- Detail the test commands executed and verified results -->
- [ ] Unit tests pass: python -m pytest tests/unit -q
- [ ] Documentation contracts pass: python -m pytest tests/unit/test_documentation_contracts.py -q
- [ ] Label integrity check passes: python scripts/check_no_fabricated_labels.py
- [ ] Linter & formatting check passes: python -m ruff check src tests dashboard

## Checklist
- [ ] No hardcoded model thresholds added outside policy_config.json.
- [ ] All internal markdown links resolve to existing files.
- [ ] No unverified safety guarantees claimed in documentation.
