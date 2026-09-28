from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PREDICTIONS = ROOT / "results" / "predictions"

SOURCE_MODEL = "CNN-Attention-GRU-ExtraTrees-Residual-TrainVal"
FINAL_MODEL = "CNN-Attention-GRU-Residual"


def main() -> None:
    main_metrics = pd.read_csv(TABLES / "main_horizon_model_comparison.csv")
    tuned = pd.read_csv(TABLES / "proposed_tuning_test.csv")
    tuned_row = tuned[tuned["model"].eq(SOURCE_MODEL)].iloc[0].to_dict()

    base_row = main_metrics[main_metrics["model"].eq("CNN-Attention-GRU")].iloc[0].to_dict()
    final_row = base_row.copy()
    final_row.update(
        {
            "model": FINAL_MODEL,
            "MAE": tuned_row["MAE"],
            "RMSE": tuned_row["RMSE"],
            "MAPE": tuned_row["MAPE"],
            "R2": tuned_row["R2"],
            "NMAE": tuned_row["NMAE"],
            "NRMSE": tuned_row["NRMSE"],
            "calibration": "ExtraTrees residual correction fitted on train+validation residuals",
        }
    )

    main_metrics["calibration"] = ""
    final_metrics = pd.concat([pd.DataFrame([final_row]), main_metrics], ignore_index=True)
    final_metrics = final_metrics.sort_values(["RMSE", "MAE", "MAPE"])
    final_metrics.to_csv(TABLES / "main_horizon_model_comparison_final.csv", index=False)

    best = pd.read_csv(TABLES / "best_metrics_by_model_horizon.csv")
    best["calibration"] = ""
    final_best_row = final_row.copy()
    final_best_row["horizon_steps"] = 1
    final_best_row["horizon_minutes"] = 15
    best_final = pd.concat([pd.DataFrame([final_best_row]), best], ignore_index=True)
    best_final.to_csv(TABLES / "best_metrics_by_model_horizon_final.csv", index=False)

    pred_main = pd.read_csv(PREDICTIONS / "main_horizon_best_predictions.csv")
    tuned_preds = pd.read_csv(PREDICTIONS / "proposed_tuning_predictions.csv")
    tuned_preds = tuned_preds[tuned_preds["model"].eq(SOURCE_MODEL)].copy()
    tuned_preds["model"] = FINAL_MODEL
    for col in pred_main.columns:
        if col not in tuned_preds.columns:
            tuned_preds[col] = ""
    final_preds = pd.concat([tuned_preds[pred_main.columns], pred_main], ignore_index=True)
    final_preds.to_csv(PREDICTIONS / "main_horizon_best_predictions_final.csv", index=False)

    print(final_metrics[["model", "MAE", "RMSE", "MAPE", "R2"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
