from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from train_evaluate import (
    CNNAttentionGRU,
    FEATURES,
    ROOT,
    TARGET,
    inverse_target,
    make_sequences,
    split_sequence_indices,
)


DATA_PATH = ROOT / "data" / "solar_processed.csv"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
PREDICTIONS = RESULTS / "predictions"
MODELS = RESULTS / "models"


def metric_row(name: str, y: np.ndarray, yp: np.ndarray, note: str) -> dict:
    yp = np.clip(yp, 0.0, 50.0)
    mae = mean_absolute_error(y, yp)
    rmse = mean_squared_error(y, yp) ** 0.5
    mape = np.mean(np.abs(y - yp) / np.maximum(np.abs(y), 1.0)) * 100
    return {
        "model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2_score(y, yp)),
        "NMAE": float(mae / 50.0),
        "NRMSE": float(rmse / 50.0),
        "note": note,
    }


def predict_model(model: torch.nn.Module, x: np.ndarray, target_scaler: StandardScaler, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xb = torch.from_numpy(x[start : start + 512]).to(device)
            pred, _ = model(xb)
            preds.append(pred.detach().cpu().numpy())
    return inverse_target(np.concatenate(preds), target_scaler)


def meta_features(seq_scaled: np.ndarray, pred: np.ndarray, target_times: pd.Series) -> np.ndarray:
    idx_power = FEATURES.index("power")
    idx_tsi = FEATURES.index("total_irradiance")
    idx_dni = FEATURES.index("direct_normal_irradiance")
    idx_ghi = FEATURES.index("global_horizontal_irradiance")
    idx_temp = FEATURES.index("air_temperature")
    idx_pressure = FEATURES.index("pressure")
    idx_humidity = FEATURES.index("relative_humidity")
    last = seq_scaled[:, -1, :]
    lag_block = seq_scaled[:, -8:, [idx_power, idx_tsi, idx_dni, idx_ghi]].reshape(len(seq_scaled), -1)
    hour = target_times.dt.hour.to_numpy() + target_times.dt.minute.to_numpy() / 60.0
    calendar = np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * target_times.dt.dayofyear.to_numpy() / 366),
            np.cos(2 * np.pi * target_times.dt.dayofyear.to_numpy() / 366),
        ]
    )
    summary = np.column_stack(
        [
            pred,
            np.clip(pred, 0, 50),
            last[:, idx_power],
            last[:, idx_tsi],
            last[:, idx_dni],
            last[:, idx_ghi],
            last[:, idx_temp],
            last[:, idx_pressure],
            last[:, idx_humidity],
            seq_scaled[:, -4:, idx_power].mean(axis=1),
            seq_scaled[:, -8:, idx_power].mean(axis=1),
            seq_scaled[:, -4:, idx_ghi].mean(axis=1),
            seq_scaled[:, -8:, idx_ghi].mean(axis=1),
        ]
    )
    return np.column_stack([summary, calendar, lag_block])


def apply_zero_threshold(pred: np.ndarray, threshold: float) -> np.ndarray:
    out = np.clip(pred.copy(), 0.0, 50.0)
    out[out < threshold] = 0.0
    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    raw_features = df[FEATURES].to_numpy(dtype=np.float32)
    raw_target = df[TARGET].to_numpy(dtype=np.float32)
    raw_train_end = int(len(df) * 0.70)
    feature_scaler = StandardScaler().fit(raw_features[:raw_train_end])
    target_scaler = StandardScaler().fit(raw_target[:raw_train_end].reshape(-1, 1))
    values = feature_scaler.transform(raw_features).astype(np.float32)
    target = target_scaler.transform(raw_target.reshape(-1, 1)).reshape(-1).astype(np.float32)

    window, horizon, hidden, lr = 48, 1, 64, 0.0005
    x, y, target_idx = make_sequences(values, target, window, horizon)
    train_i, val_i, test_i = split_sequence_indices(len(x))
    x_train, y_train = x[train_i], y[train_i]
    x_val, y_val = x[val_i], y[val_i]
    x_test, y_test = x[test_i], y[test_i]
    y_train_true = inverse_target(y_train, target_scaler)
    y_val_true = inverse_target(y_val, target_scaler)
    y_test_true = inverse_target(y_test, target_scaler)
    train_times = df["timestamp"].iloc[target_idx[train_i]].reset_index(drop=True)
    val_times = df["timestamp"].iloc[target_idx[val_i]].reset_index(drop=True)
    test_times = df["timestamp"].iloc[target_idx[test_i]].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNAttentionGRU(len(FEATURES), hidden).to(device)
    checkpoint = torch.load(
        MODELS / f"CNN_Attention_GRU_v2_h{horizon}_w{window}_hs{hidden}_lr{lr}.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    train_pred = predict_model(model, x_train, target_scaler, device)
    val_pred = predict_model(model, x_val, target_scaler, device)
    test_pred = predict_model(model, x_test, target_scaler, device)

    candidates: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    candidates.append(("CNN-Attention-GRU", "raw", val_pred, test_pred))
    candidates.append(("CNN-Attention-GRU-Clip", "physical clip to [0,50] MW", np.clip(val_pred, 0, 50), np.clip(test_pred, 0, 50)))

    best_threshold = None
    best_val_score = float("inf")
    for threshold in np.linspace(0.0, 1.2, 25):
        vp = apply_zero_threshold(val_pred, threshold)
        row = metric_row("threshold", y_val_true, vp, "")
        score = row["MAE"] + 0.02 * row["RMSE"] + 0.01 * row["MAPE"]
        if score < best_val_score:
            best_val_score = score
            best_threshold = float(threshold)
    candidates.append(
        (
            "CNN-Attention-GRU-Physical",
            f"clip plus validation-selected zero threshold {best_threshold:.2f} MW",
            apply_zero_threshold(val_pred, best_threshold),
            apply_zero_threshold(test_pred, best_threshold),
        )
    )

    z_train = meta_features(x_train, train_pred, train_times)
    z_val = meta_features(x_val, val_pred, val_times)
    z_test = meta_features(x_test, test_pred, test_times)
    meta_scaler = StandardScaler().fit(z_val)
    z_val_s = meta_scaler.transform(z_val)
    z_test_s = meta_scaler.transform(z_test)
    z_train_val = np.vstack([z_train, z_val])
    y_train_val = np.concatenate([y_train_true, y_val_true])
    train_val_scaler = StandardScaler().fit(z_train_val)
    z_train_val_s = train_val_scaler.transform(z_train_val)
    z_test_train_val_s = train_val_scaler.transform(z_test)

    calibrators = [
        ("Ridge-Calibrated", Ridge(alpha=0.1)),
        ("Huber-Calibrated", HuberRegressor(alpha=1e-4, max_iter=1000)),
        (
            "RF-Calibrated",
            RandomForestRegressor(
                n_estimators=350,
                max_depth=18,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees-Calibrated",
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=22,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
    if XGBRegressor is not None:
        calibrators.append(
            (
                "XGB-Calibrated",
                XGBRegressor(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            )
        )

    iso = IsotonicRegression(y_min=0, y_max=50, out_of_bounds="clip")
    iso.fit(np.clip(val_pred, 0, 50), y_val_true)
    candidates.append(
        (
            "CNN-Attention-GRU-Isotonic",
            "isotonic calibration on validation predictions",
            iso.predict(np.clip(val_pred, 0, 50)),
            iso.predict(np.clip(test_pred, 0, 50)),
        )
    )

    for suffix, calibrator in calibrators:
        calibrator.fit(z_val_s, y_val_true)
        candidates.append(
            (
                f"CNN-Attention-GRU-{suffix}",
                f"{suffix} fitted on validation meta-features",
                calibrator.predict(z_val_s),
                calibrator.predict(z_test_s),
            )
        )

    train_val_calibrators = [
        (
            "RF-Residual-TrainVal",
            RandomForestRegressor(
                n_estimators=450,
                max_depth=22,
                min_samples_leaf=1,
                random_state=123,
                n_jobs=-1,
            ),
            "residual",
        ),
        (
            "ExtraTrees-Residual-TrainVal",
            ExtraTreesRegressor(
                n_estimators=650,
                max_depth=None,
                min_samples_leaf=1,
                random_state=123,
                n_jobs=-1,
            ),
            "residual",
        ),
    ]
    if XGBRegressor is not None:
        train_val_calibrators.extend(
            [
                (
                    "XGB-Residual-TrainVal",
                    XGBRegressor(
                        n_estimators=900,
                        max_depth=5,
                        learning_rate=0.025,
                        subsample=0.95,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=123,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    "residual",
                ),
                (
                    "XGB-Direct-TrainVal",
                    XGBRegressor(
                        n_estimators=900,
                        max_depth=5,
                        learning_rate=0.025,
                        subsample=0.95,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=321,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    "direct",
                ),
            ]
        )

    train_val_pred = np.concatenate([train_pred, val_pred])
    for suffix, calibrator, mode in train_val_calibrators:
        if mode == "residual":
            calibrator.fit(z_train_val_s, y_train_val - np.clip(train_val_pred, 0, 50))
            test_out = np.clip(test_pred, 0, 50) + calibrator.predict(z_test_train_val_s)
            val_out = np.clip(val_pred, 0, 50)
            note = f"{suffix} fitted on train+validation residuals"
        else:
            calibrator.fit(z_train_val_s, y_train_val)
            test_out = calibrator.predict(z_test_train_val_s)
            val_out = np.clip(val_pred, 0, 50)
            note = f"{suffix} fitted directly on train+validation meta-features"
        candidates.append((f"CNN-Attention-GRU-{suffix}", note, val_out, test_out))

    val_rows = []
    test_rows = []
    pred_frames = []
    for name, note, vp, tp in candidates:
        val_row = metric_row(name, y_val_true, vp, note)
        test_row = metric_row(name, y_test_true, tp, note)
        val_rows.append(val_row)
        test_rows.append(test_row)
        pred_frames.append(
            pd.DataFrame(
                {
                    "timestamp": test_times,
                    "y_true": y_test_true,
                    "y_pred": np.clip(tp, 0, 50),
                    "model": name,
                    "horizon_steps": horizon,
                    "window": window,
                    "hidden_size": hidden,
                    "learning_rate": lr,
                    "note": note,
                }
            )
        )

    val_df = pd.DataFrame(val_rows).sort_values(["MAE", "RMSE", "MAPE"])
    test_df = pd.DataFrame(test_rows).sort_values(["MAE", "RMSE", "MAPE"])
    selected_name = val_df.iloc[0]["model"]
    selected_test = test_df[test_df["model"].eq(selected_name)].iloc[0].to_dict()
    TABLES.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(TABLES / "proposed_tuning_validation.csv", index=False)
    test_df.to_csv(TABLES / "proposed_tuning_test.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(PREDICTIONS / "proposed_tuning_predictions.csv", index=False)
    (RESULTS / "proposed_tuning_selection.json").write_text(
        json.dumps({"selected_by_validation": selected_name, "selected_test": selected_test}, indent=2),
        encoding="utf-8",
    )
    print("Validation ranking:")
    print(val_df[["model", "MAE", "RMSE", "MAPE", "R2", "note"]].to_string(index=False))
    print("\nTest ranking:")
    print(test_df[["model", "MAE", "RMSE", "MAPE", "R2", "note"]].to_string(index=False))
    print("\nSelected by validation:", selected_name)


if __name__ == "__main__":
    main()
