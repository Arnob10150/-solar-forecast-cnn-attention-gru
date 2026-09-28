from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "solar_processed.csv"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
PREDICTIONS = RESULTS / "predictions"
MODELS = RESULTS / "models"
LOGS = RESULTS / "logs"

FEATURES = [
    "total_irradiance",
    "direct_normal_irradiance",
    "global_horizontal_irradiance",
    "air_temperature",
    "pressure",
    "relative_humidity",
    "power",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]
TARGET = "power"


@dataclass
class RunConfig:
    seed: int = 42
    windows: tuple[int, ...] = (16, 24, 48)
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    hidden_sizes: tuple[int, ...] = (64, 128)
    learning_rates: tuple[float, ...] = (1e-3, 5e-4)
    max_epochs: int = 80
    patience: int = 10
    batch_size: int = 128
    classical_max_train: int = 30000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_sequences(values: np.ndarray, target: np.ndarray, window: int, horizon: int):
    x, y, target_idx = [], [], []
    for end in range(window - 1, len(values) - horizon):
        start = end - window + 1
        idx = end + horizon
        x.append(values[start : end + 1])
        y.append(target[idx])
        target_idx.append(idx)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(target_idx)


def split_sequence_indices(n: int):
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, n)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: float = 50.0) -> dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2_score(y_true, y_pred)),
        "NMAE": float(mae / capacity),
        "NRMSE": float(rmse / capacity),
    }


class RecurrentModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kind: str):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1), None


class CNNRNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kind: str):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 1))

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(z)
        return self.head(out[:, -1]).squeeze(-1), None


class CNNAttentionGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.10),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(z)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        fused = torch.cat([context, out[:, -1]], dim=1)
        return self.head(fused).squeeze(-1), weights


def make_model(name: str, input_size: int, hidden_size: int) -> nn.Module:
    if name == "LSTM":
        return RecurrentModel(input_size, hidden_size, "lstm")
    if name == "GRU":
        return RecurrentModel(input_size, hidden_size, "gru")
    if name == "CNN-LSTM":
        return CNNRNNModel(input_size, hidden_size, "lstm")
    if name == "CNN-GRU":
        return CNNRNNModel(input_size, hidden_size, "gru")
    if name == "CNN-Attention-GRU":
        return CNNAttentionGRU(input_size, hidden_size)
    raise ValueError(name)


def inverse_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def train_neural(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: StandardScaler,
    timestamps: np.ndarray,
    config: RunConfig,
    horizon: int,
    window: int,
    hidden_size: int,
    lr: float,
    device: torch.device,
):
    model = make_model(name, x_train.shape[-1], hidden_size).to(device)
    loss_fn = nn.MSELoss()
    version = "_v2" if name == "CNN-Attention-GRU" else ""
    model_path = MODELS / f"{name.replace('-', '_')}{version}_h{horizon}_w{window}_hs{hidden_size}_lr{lr}.pt"
    history_path = LOGS / f"{name.replace('-', '_')}{version}_h{horizon}_w{window}_hs{hidden_size}_lr{lr}.json"
    cached_metrics = None
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        cached_metrics = checkpoint.get("metrics")
        print(f"Reusing checkpoint: {model_path.name}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    history = []
    fit_seconds = 0.0
    if cached_metrics is None:
        best_state = None
        best_val = float("inf")
        stale = 0
        start = time.time()
        print(
            f"Training {name}: horizon={horizon}, window={window}, hidden={hidden_size}, lr={lr}",
            flush=True,
        )
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred, _ = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.item()) * len(xb)
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(val_x)
                val_loss = float(loss_fn(val_pred, val_y).item())
            train_loss = total / len(x_train)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        fit_seconds = time.time() - start
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    preds, attn_rows = [], []
    start_pred = time.time()
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            pred, attn = model(xb)
            preds.append(pred.detach().cpu().numpy())
            if attn is not None:
                attn_rows.append(attn.detach().cpu().numpy())
    predict_seconds = time.time() - start_pred
    pred_scaled = np.concatenate(preds)
    y_true = inverse_target(y_test, target_scaler)
    y_pred = inverse_target(pred_scaled, target_scaler)
    row_metrics = metrics(y_true, y_pred)
    if cached_metrics is not None:
        row_metrics.update(cached_metrics)
        row_metrics["predict_seconds"] = predict_seconds
    else:
        row_metrics.update(
            {
                "model": name,
                "horizon_steps": horizon,
                "horizon_minutes": horizon * 15,
                "window": window,
                "hidden_size": hidden_size,
                "learning_rate": lr,
                "epochs": len(history),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        torch.save({"state_dict": model.state_dict(), "metrics": row_metrics}, model_path)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    pred_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "y_pred": y_pred,
            "model": name,
            "horizon_steps": horizon,
            "window": window,
            "hidden_size": hidden_size,
            "learning_rate": lr,
        }
    )
    attn_arr = np.concatenate(attn_rows) if attn_rows else None
    return row_metrics, pred_df, attn_arr


def train_classical(name: str, x_train, y_train, x_test, y_test, target_scaler, timestamps, horizon, window):
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    if name == "Ridge":
        model = Ridge(alpha=1.0)
    elif name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "XGBoost" and XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        return None, None
    start = time.time()
    model.fit(x_train_flat, y_train)
    fit_seconds = time.time() - start
    start_pred = time.time()
    pred_scaled = model.predict(x_test_flat)
    predict_seconds = time.time() - start_pred
    y_true = inverse_target(y_test, target_scaler)
    y_pred = inverse_target(np.asarray(pred_scaled), target_scaler)
    row = metrics(y_true, y_pred)
    row.update(
        {
            "model": name,
            "horizon_steps": horizon,
            "horizon_minutes": horizon * 15,
            "window": window,
            "hidden_size": "",
            "learning_rate": "",
            "epochs": "",
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
    )
    pred_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "y_pred": y_pred,
            "model": name,
            "horizon_steps": horizon,
            "window": window,
            "hidden_size": "",
            "learning_rate": "",
        }
    )
    return row, pred_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    for path in [TABLES, PREDICTIONS, MODELS, LOGS]:
        path.mkdir(parents=True, exist_ok=True)
    config = RunConfig(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    raw_features = df[FEATURES].to_numpy(dtype=np.float32)
    raw_target = df[TARGET].to_numpy(dtype=np.float32)
    n_raw = len(df)
    raw_train_end = int(n_raw * 0.70)
    feature_scaler = StandardScaler().fit(raw_features[:raw_train_end])
    target_scaler = StandardScaler().fit(raw_target[:raw_train_end].reshape(-1, 1))
    values = feature_scaler.transform(raw_features).astype(np.float32)
    target = target_scaler.transform(raw_target.reshape(-1, 1)).reshape(-1).astype(np.float32)

    all_rows: list[dict] = []
    all_preds: list[pd.DataFrame] = []
    attention_payload = {}
    neural_names = ["LSTM", "GRU", "CNN-LSTM", "CNN-GRU", "CNN-Attention-GRU"]

    for horizon in config.horizons:
        horizon_rows = []
        for window in config.windows:
            x, y, target_idx = make_sequences(values, target, window, horizon)
            train_i, val_i, test_i = split_sequence_indices(len(x))
            x_train, y_train = x[train_i], y[train_i]
            x_val, y_val = x[val_i], y[val_i]
            x_test, y_test = x[test_i], y[test_i]
            test_times = df["timestamp"].iloc[target_idx[test_i]].to_numpy()

            y_true = inverse_target(y_test, target_scaler)
            last_power_scaled = x_test[:, -1, FEATURES.index("power")]
            persistence_pred = inverse_target(last_power_scaled, target_scaler)
            row = metrics(y_true, persistence_pred)
            row.update(
                {
                    "model": "Persistence",
                    "horizon_steps": horizon,
                    "horizon_minutes": horizon * 15,
                    "window": window,
                    "hidden_size": "",
                    "learning_rate": "",
                    "epochs": "",
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                }
            )
            horizon_rows.append(row)
            all_preds.append(
                pd.DataFrame(
                    {
                        "timestamp": test_times,
                        "y_true": y_true,
                        "y_pred": persistence_pred,
                        "model": "Persistence",
                        "horizon_steps": horizon,
                        "window": window,
                        "hidden_size": "",
                        "learning_rate": "",
                    }
                )
            )

            x_train_classical = np.concatenate([x_train, x_val], axis=0)
            y_train_classical = np.concatenate([y_train, y_val], axis=0)
            if len(x_train_classical) > config.classical_max_train:
                keep = np.linspace(0, len(x_train_classical) - 1, config.classical_max_train).astype(int)
                x_train_classical = x_train_classical[keep]
                y_train_classical = y_train_classical[keep]
            for classical in ["Ridge", "Random Forest", "XGBoost"]:
                row, pred_df = train_classical(
                    classical,
                    x_train_classical,
                    y_train_classical,
                    x_test,
                    y_test,
                    target_scaler,
                    test_times,
                    horizon,
                    window,
                )
                if row:
                    horizon_rows.append(row)
                    all_preds.append(pred_df)

            neural_to_train = neural_names if horizon == 1 else ["CNN-Attention-GRU"]
            if neural_to_train:
                for neural_name in neural_to_train:
                    for hidden in config.hidden_sizes:
                        for lr in config.learning_rates:
                            row, pred_df, attn = train_neural(
                                neural_name,
                                x_train,
                                y_train,
                                x_val,
                                y_val,
                                x_test,
                                y_test,
                                target_scaler,
                                test_times,
                                config,
                                horizon,
                                window,
                                hidden,
                                lr,
                                device,
                            )
                            horizon_rows.append(row)
                            all_preds.append(pred_df)
                            if neural_name == "CNN-Attention-GRU":
                                key = f"h{horizon}_w{window}_hs{hidden}_lr{lr}"
                                if attn is not None:
                                    attention_payload[key] = attn[:512].mean(axis=0).tolist()

        horizon_df = pd.DataFrame(horizon_rows)
        horizon_df.to_csv(TABLES / f"all_metrics_h{horizon}.csv", index=False)
        all_rows.extend(horizon_rows)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(TABLES / "all_metrics.csv", index=False)
    best = (
        metrics_df.sort_values(["horizon_steps", "RMSE", "MAE"])
        .groupby(["horizon_steps", "model"], as_index=False)
        .first()
    )
    best.to_csv(TABLES / "best_metrics_by_model_horizon.csv", index=False)
    main_best = best[best["horizon_steps"] == 1].sort_values("RMSE")
    main_best.to_csv(TABLES / "main_horizon_model_comparison.csv", index=False)

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all.to_csv(PREDICTIONS / "all_predictions.csv", index=False)
    best_keys = main_best[["model", "window", "hidden_size", "learning_rate"]].copy()
    pred_for_join = pred_all[pred_all["horizon_steps"].eq(1)].copy()
    for col in ["hidden_size", "learning_rate"]:
        pred_for_join[col] = pred_for_join[col].fillna("").astype(str)
        best_keys[col] = best_keys[col].fillna("").astype(str)
    pred_main = pred_for_join.merge(
        best_keys,
        on=["model", "window", "hidden_size", "learning_rate"],
    )
    pred_main.to_csv(PREDICTIONS / "main_horizon_best_predictions.csv", index=False)

    (RESULTS / "experiment_config.json").write_text(
        json.dumps({**asdict(config), "device": str(device), "features": FEATURES}, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "attention_weights.json").write_text(json.dumps(attention_payload, indent=2), encoding="utf-8")
    print("Device:", device)
    print(main_best.to_string(index=False))


if __name__ == "__main__":
    main()
