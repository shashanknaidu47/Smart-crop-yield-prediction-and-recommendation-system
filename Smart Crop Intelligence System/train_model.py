"""
train_model.py
--------------
Main training pipeline.

Usage:
    python train_model.py [--skip-hpo]

Workflow:
    1.  Load & generate dataset if needed
    2.  Full preprocessing pipeline
    3.  Feature engineering (MI-based selection)
    4.  Train all ML models
    5.  Optional: Hyperparameter tuning

    7.  Evaluate and compare all models
    8.  Save best models
    9.  Generate & save visualisations
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    load_dataset, save_model, logger, MODEL_DIR, VIZ_DIR, DATA_DIR,
)
from preprocessing.data_preprocessing  import run_full_pipeline
from preprocessing.feature_engineering import (
    mutual_information_ranking, shap_feature_importance, select_best_features,
)
from models.ml_models       import get_base_models, StackingEnsemble
from models.model_evaluation import (
    evaluate_all_models, plot_model_comparison,
    plot_actual_vs_predicted, plot_residuals, plot_feature_importance,
)
from visualization.plots import (
    plot_correlation_heatmap, plot_yield_distribution,
    plot_rf_feature_importance, plot_crop_season_heatmap,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dataset() -> pd.DataFrame:
    """Generate dataset if it doesn't exist."""
    from utils import DATASET_PATH, FALLBACK_PATH
    if not os.path.exists(DATASET_PATH):
        gen_script = os.path.join(DATA_DIR, "generate_dataset.py")
        if os.path.exists(gen_script):
            logger.info("Generating dataset via generate_dataset.py …")
            import subprocess
            subprocess.run([sys.executable, gen_script], check=True)
        else:
            logger.warning("generate_dataset.py not found; using fallback dataset.")
    return load_dataset()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Load Dataset")
    logger.info("=" * 60)
    df_raw = ensure_dataset()
    logger.info(f"Raw dataset shape: {df_raw.shape}")
    logger.info(f"Columns: {df_raw.columns.tolist()}")

    # ── EDA Plots ─────────────────────────────────────────────────────────────
    logger.info("Generating EDA visualisations …")
    plot_correlation_heatmap(df_raw,
        save_path=os.path.join(VIZ_DIR, "correlation_heatmap.png"))
    plot_yield_distribution(df_raw,
        save_path=os.path.join(VIZ_DIR, "yield_distribution.png"))
    if "Season" in df_raw.columns:
        plot_crop_season_heatmap(df_raw,
            save_path=os.path.join(VIZ_DIR, "crop_season_heatmap.png"))

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)
    X_train, X_test, y_train, y_test, encoder, scaler, feature_cols = \
        run_full_pipeline(df_raw, target="Yield", scale_method="standard")

    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Selection (Mutual Information)")
    logger.info("=" * 60)
    n_select = min(15, X_train.shape[1])
    best_features = select_best_features(X_train, y_train,
                                          method="mi", n_features=n_select)
    logger.info(f"Selected {len(best_features)} features: {best_features}")

    X_tr = X_train[best_features]
    X_te = X_test[best_features]

    # ── 4. Train ML Models ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Train Base ML Models")
    logger.info("=" * 60)
    base_models = get_base_models()
    fitted_models = {}

    for name, model in base_models.items():
        logger.info(f"  Training {name} …")
        try:
            model.fit(X_tr, y_train)
            fitted_models[name] = model
            logger.info(f"  {name} ✓")
        except Exception as e:
            logger.warning(f"  {name} FAILED: {e}")

    # ── 5. Stacking Ensemble ──────────────────────────────────────────────────
    logger.info("  Training Stacking Ensemble …")
    ensemble = StackingEnsemble(n_folds=5)
    try:
        ensemble.fit(X_tr.values, y_train.values)
        fitted_models["StackingEnsemble"] = ensemble
        logger.info("  StackingEnsemble ✓")
    except Exception as e:
        logger.warning(f"  StackingEnsemble FAILED: {e}")



    # ── 7. Hyperparameter Tuning (optional) ───────────────────────────────────
    if not args.skip_hpo:
        logger.info("=" * 60)
        logger.info("STEP 6: Hyperparameter Optimisation")
        logger.info("=" * 60)
        from models.hyperparameter_tuning import tune_all_models
        tuned = tune_all_models(X_tr.values, y_train.values,
                                run_grid=True, run_random=True,
                                run_bayesian=True, n_trials=25)
        for name, info in tuned.items():
            fitted_models[name] = info["model"]
            logger.info(f"  Tuned model added: {name}")

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Model Evaluation")
    logger.info("=" * 60)

    # Wrap to handle different .predict signatures
    eval_models = {}
    for name, m in fitted_models.items():
        try:
            _ = m.predict(X_te[:5].values if hasattr(X_te, "values") else X_te[:5])
            eval_models[name] = m
        except Exception:
            try:
                _ = m.predict(X_te[:5])
                eval_models[name] = m
            except Exception as e:
                logger.warning(f"  Could not evaluate {name}: {e}")

    # Normalise predict interface
    class _Wrap:
        def __init__(self, m): self._m = m
        def predict(self, X):
            try:
                return self._m.predict(X.values if hasattr(X, "values") else X)
            except Exception:
                return self._m.predict(X)

    wrapped = {n: _Wrap(m) for n, m in eval_models.items()}
    results_df = evaluate_all_models(wrapped, X_te, y_test)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # ── 9. Save artefacts ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Saving Models & Visualisations")
    logger.info("=" * 60)

    # Best model by R²
    if len(results_df) > 0:
        best_name = results_df.iloc[0]["Model"]
        if best_name in fitted_models:
            save_model(fitted_models[best_name], "best_model.pkl")
            logger.info(f"Best model: {best_name}  (R²={results_df.iloc[0]['R²']:.4f})")

    # Save encoder & scaler for inference
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(best_features, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    logger.info("Encoder, scaler, feature list saved.")

    # Comparison plot
    plot_model_comparison(results_df,
        save_path=os.path.join(VIZ_DIR, "model_comparison.png"))

    # Best model plots
    if best_name in fitted_models:
        bm = fitted_models[best_name]
        try:
            y_pred = bm.predict(X_te.values if hasattr(X_te, "values") else X_te)
        except Exception:
            y_pred = bm.predict(X_te)

        plot_actual_vs_predicted(y_test.values, y_pred, best_name,
            save_path=os.path.join(VIZ_DIR, "actual_vs_predicted.png"))
        plot_residuals(y_test.values, y_pred, best_name,
            save_path=os.path.join(VIZ_DIR, "residuals.png"))

        # Feature importance (for tree models)
        if hasattr(bm, "feature_importances_"):
            plot_rf_feature_importance(bm, best_features,
                save_path=os.path.join(VIZ_DIR, "feature_importance.png"))
            # SHAP
            try:
                from preprocessing.feature_engineering import shap_feature_importance
                shap_feature_importance(bm, X_tr,
                    save_path=os.path.join(VIZ_DIR, "shap_importance.png"))
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")

    logger.info("=" * 60)
    logger.info("Training pipeline complete!")
    logger.info(f"Visualisations saved → {VIZ_DIR}")
    logger.info(f"Models saved → {MODEL_DIR}")
    logger.info("=" * 60)

    return results_df, fitted_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Yield Prediction Training Pipeline")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip hyperparameter optimisation (faster run)")

    args = parser.parse_args()
    main(args)
