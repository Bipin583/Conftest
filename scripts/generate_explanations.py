"""
ConfTest Batch Explainability & SHAP Rationale Generator CLI.

Computes exact SHAP TreeExplainer feature attributions and generates natural language
developer reason cards for high-risk test selections.

Usage:
    python scripts/generate_explanations.py \
        --dataset data/splits/test.csv \
        --model models/ensembles/5_seed_lgbm/member_1_seed_42.joblib \
        --output reports/explanations.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conftest.explainability.shap_explainer import ShapExplainer
from conftest.explainability.rules import RuleBasedExplainer
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.trainer import prepare_feature_arrays
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfTest Model Explainability CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/splits/test.csv",
        help="Path to evaluation dataset CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ensembles/5_seed_lgbm/member_1_seed_42.joblib",
        help="Path to trained LightGBM model joblib artifact.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/explanations.json",
        help="Destination path for generated explainability report.",
    )
    parser.add_argument(
        "--num-cards",
        type=int,
        default=15,
        help="Reason cards to emit, taken from the highest-scoring rows.",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default="./models/policy_config.json",
        help="Tuned policy, used to decide whether a card's test is actually selected.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.dataset)
    model_path = Path(args.model)

    if not data_path.exists() or not model_path.exists():
        logger.error(f"Dataset {data_path} or model {model_path} not found.")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    predictor = LightGBMTestPredictor.load(str(model_path))

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X, y = prepare_feature_arrays(df)

    logger.info("Initializing SHAP TreeExplainer...")
    shap_explainer = ShapExplainer(predictor)

    # 1. Global Feature Importance across entire test set
    logger.info("Computing global SHAP feature importance...")
    global_shap = shap_explainer.explain_dataset(X)

    # 2. Local explanations for the rows the model actually scores highest.
    # Taking the first N rows of the file and calling them high-risk describes the CSV
    # ordering, not the model, so the cards are drawn by predicted probability.
    logger.info("Scoring the dataset to find the highest-risk rows...")
    scores = predictor.predict_proba(X)
    budget_ratio = float(
        json.loads(Path(args.policy_config).read_text(encoding="utf-8")).get("budget_ratio", 0.25)
    ) if Path(args.policy_config).exists() else 0.25

    # A test is selected when it survives the budget cut inside its own commit, which is
    # the decision the policy makes at serving time. Ranking across the whole split would
    # answer a question nobody asks.
    selected_mask = np.zeros(len(df), dtype=bool)
    if "commit_sha" in df.columns:
        score_series = pd.Series(scores, index=df.index)
        for _, rows in df.groupby("commit_sha", sort=False):
            k = max(1, int(round(len(rows) * budget_ratio)))
            keep = score_series.loc[rows.index].nlargest(k).index
            selected_mask[df.index.get_indexer(keep)] = True

    logger.info("Generating local SHAP attributions and developer reason cards...")
    sample_cards = []
    rule_explainer = RuleBasedExplainer()

    ranked = np.argsort(-scores)[: max(0, args.num_cards)]
    for idx in ranked:
        idx = int(idx)
        x_row = X[idx]
        row = df.iloc[idx]
        feat_dict = {name: float(row[name]) for name in FEATURE_NAMES if name in df.columns}
        t_id = str(row.get("test_id", f"test_{idx}"))

        shap_breakdown = shap_explainer.explain_instance(x_row, top_k=3)
        top_drivers = shap_breakdown["top_risk_increasing_features"]

        card = rule_explainer.generate_test_reason_card(
            test_id=t_id,
            feature_dict=feat_dict,
            shap_drivers=top_drivers,
            is_selected=bool(selected_mask[idx]),
            confidence=shap_breakdown["predicted_probability"],
        )
        has_label = "label_failed" in df.columns
        card["observed_label_failed"] = int(row["label_failed"]) if has_label else None
        card["repo"] = str(row["repo"]) if "repo" in df.columns else None
        sample_cards.append(card)

    report = {
        "model_file": str(model_path),
        "dataset_file": str(data_path),
        "rows_scored": int(len(df)),
        "cards_drawn_from": "highest predicted failure probability, not file order",
        "selection_rule": f"top {budget_ratio:.0%} of each commit by score",
        "labels_measured": True,
        "global_shap_importance": global_shap["global_feature_importance_shap"][:10],
        "sample_developer_cards": sample_cards,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Explainability report exported to: {out_path}")
    logger.info("\n=== Top 5 Global SHAP Features ===")
    for item in report["global_shap_importance"][:5]:
        logger.info(f"  • {item['feature']:<35} (Mean |SHAP|: {item['mean_abs_shap']:.4f})")

    logger.info("\n=== Sample Developer Reason Card ===")
    sample = sample_cards[0]
    logger.info(f"Test ID:     {sample['test_id']}")
    logger.info(f"Risk Level:  {sample['risk_level']} (Conf: {sample['confidence']})")
    for r in sample["primary_reasons"]:
        logger.info(f"  - {r}")


if __name__ == "__main__":
    main()
