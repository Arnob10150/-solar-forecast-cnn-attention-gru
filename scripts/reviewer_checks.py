import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "results" / "predictions" / "main_horizon_best_predictions_final.csv"
DATA_PATH = ROOT / "data" / "solar_processed.csv"
OUT_DIR = ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY = 50.0
DAY_GHI_THRESHOLD = 20.0
PROPOSED = "CNN-Attention-GRU-Residual"


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err**2))
    mape = 100.0 * np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1.0))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "NMAE": mae / CAPACITY,
        "NRMSE": rmse / CAPACITY,
    }


def dm_test(y_true, pred_a, pred_b, loss="squared"):
    if loss == "squared":
        d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    elif loss == "absolute":
        d = np.abs(y_true - pred_a) - np.abs(y_true - pred_b)
    else:
        raise ValueError(loss)
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if n < 3 or var_d <= 0:
        return mean_d, float("nan"), float("nan")
    stat = mean_d / math.sqrt(var_d / n)
    # Two-sided normal approximation. For h=1 this is the standard DM variance.
    p_value = math.erfc(abs(stat) / math.sqrt(2.0))
    return mean_d, stat, p_value


pred = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])
data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
data = data.sort_values("timestamp")
data["prev_power"] = data["power"].shift(1)
data["prev_ghi"] = data["global_horizontal_irradiance"].shift(1)

meta = data[
    [
        "timestamp",
        "power",
        "global_horizontal_irradiance",
        "prev_power",
        "prev_ghi",
    ]
].rename(columns={"power": "target_power", "global_horizontal_irradiance": "target_ghi"})

pred = pred.merge(meta, on="timestamp", how="left")
pred["daytime"] = pred["target_ghi"] > DAY_GHI_THRESHOLD

models_for_day = [
    PROPOSED,
    "Random Forest",
    "XGBoost",
    "CNN-GRU",
    "LSTM",
    "Persistence",
]
day_rows = []
for model in models_for_day:
    subset = pred[(pred["model"] == model) & pred["daytime"]].copy()
    row = {"Model": model, "Subset": f"GHI>{DAY_GHI_THRESHOLD:g} W/m2", "N": len(subset)}
    row.update(metrics(subset["y_true"], subset["y_pred"]))
    day_rows.append(row)

prop_day = pred[(pred["model"] == PROPOSED) & pred["daytime"]].copy()
smart = prop_day.copy()
ratio = smart["target_ghi"].to_numpy() / np.maximum(smart["prev_ghi"].to_numpy(), DAY_GHI_THRESHOLD)
smart_pred = np.clip(smart["prev_power"].to_numpy() * ratio, 0.0, CAPACITY)
smart_row = {"Model": "GHI-scaled smart persistence", "Subset": f"GHI>{DAY_GHI_THRESHOLD:g} W/m2", "N": len(smart)}
smart_row.update(metrics(smart["y_true"], smart_pred))
day_rows.append(smart_row)

day_df = pd.DataFrame(day_rows)
smart_rmse = float(day_df.loc[day_df["Model"] == "GHI-scaled smart persistence", "RMSE"].iloc[0])
day_df["Skill_vs_smart"] = 1.0 - day_df["RMSE"] / smart_rmse
day_df.to_csv(OUT_DIR / "reviewer_daytime_smart_persistence.csv", index=False)

wide = pred.pivot_table(index="timestamp", columns="model", values="y_pred", aggfunc="first")
y = pred.drop_duplicates("timestamp").set_index("timestamp")["y_true"]
comparators = ["Random Forest", "XGBoost", "CNN-GRU", "LSTM", "CNN-Attention-GRU"]
dm_rows = []
for comparator in comparators:
    aligned = pd.concat([y, wide[[PROPOSED, comparator]]], axis=1).dropna()
    aligned.columns = ["y_true", PROPOSED, comparator]
    for loss in ["squared", "absolute"]:
        mean_d, stat, p_value = dm_test(
            aligned["y_true"].to_numpy(),
            aligned[PROPOSED].to_numpy(),
            aligned[comparator].to_numpy(),
            loss=loss,
        )
        dm_rows.append(
            {
                "Comparator": comparator,
                "Loss": loss,
                "N": len(aligned),
                "Mean_loss_diff_proposed_minus_comparator": mean_d,
                "DM_stat": stat,
                "p_value": p_value,
            }
        )

pd.DataFrame(dm_rows).to_csv(OUT_DIR / "reviewer_diebold_mariano.csv", index=False)

all_metrics = pd.read_csv(ROOT / "results" / "tables" / "all_metrics.csv")
abl = all_metrics[
    (all_metrics["horizon_steps"] == 1)
    & (all_metrics["model"].isin(["CNN-GRU", "CNN-Attention-GRU"]))
    & (all_metrics["window"].isin([16, 48]))
].copy()
abl = (
    abl.sort_values(["model", "window", "RMSE"])
    .groupby(["model", "window"], as_index=False)
    .first()
)
abl[
    ["model", "window", "hidden_size", "learning_rate", "MAE", "RMSE", "MAPE", "R2"]
].to_csv(OUT_DIR / "reviewer_fixed_window_ablation.csv", index=False)

print("Wrote reviewer_daytime_smart_persistence.csv")
print(day_df.round(6).to_string(index=False))
print("Wrote reviewer_diebold_mariano.csv")
print(pd.DataFrame(dm_rows).round(6).to_string(index=False))
print("Wrote reviewer_fixed_window_ablation.csv")
print(abl[["model", "window", "hidden_size", "learning_rate", "MAE", "RMSE", "MAPE", "R2"]].round(6).to_string(index=False))
