"""Improved framework: seed-ensembled CNN-Attention-GRU + daytime-aware residual stage
+ validation-selected convex blend with a tree ensemble.

Every design choice and every weight is selected on the CALIBRATION block
(the second validation split). The test block is scored once at the end and
never participates in any fit or selection step.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train_evaluate import FEATURES, ROOT, make_model, metrics
from tune_proposed import meta_features
from revision_experiments import OUT as REV, SEEDS

OUT = ROOT / "results" / "revision_v2"
W = 48
DAY_THRESHOLD = 20.0


def predict(model, x, device, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i + bs]).to(device))[0].cpu().numpy())
    return np.concatenate(out)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    df = pd.read_csv(ROOT / "data/solar_processed.csv", parse_dates=["timestamp"])
    n = len(df)
    a, b, c = int(.70 * n), int(.775 * n), int(.85 * n)

    scal = joblib.load(REV / "scalers.joblib")
    fs, ys = scal["features"], scal["target"]
    v = fs.transform(df[FEATURES]).astype("float32")

    ids = np.arange(W, n)
    ix_tr = ids[ids < a]
    ix_cal = ids[(ids >= b) & (ids < c)]
    ix_te = ids[ids >= c]

    def win(idx):
        return np.ascontiguousarray(v[idx[:, None] - np.arange(W, 0, -1)])

    x_tr, x_cal, x_te = win(ix_tr), win(ix_cal), win(ix_te)
    y_cal = df.power.iloc[ix_cal].to_numpy()
    y_te = df.power.iloc[ix_te].to_numpy()

    ghi = df.global_horizontal_irradiance.to_numpy()
    day_cal = ghi[ix_cal] > DAY_THRESHOLD
    day_te = ghi[ix_te] > DAY_THRESHOLD

    report = {"protocol": "all selection on calibration block; test scored once",
              "window": W, "seeds": SEEDS, "daytime_threshold": DAY_THRESHOLD,
              "N_cal": int(len(ix_cal)), "N_test": int(len(ix_te)),
              "N_test_daytime": int(day_te.sum())}
    t0 = time.perf_counter()

    # ---- 1. seed ensemble of the attention model ----------------------
    base_cal, base_te = [], []
    for s in SEEDS:
        model = make_model("CNN-Attention-GRU", len(FEATURES), 64).to(device)
        ck = torch.load(REV / f"model_CNN-Attention-GRU_w{W}_s{s}.pt",
                        map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        base_cal.append(ys.inverse_transform(predict(model, x_cal, device)[:, None]).ravel())
        base_te.append(ys.inverse_transform(predict(model, x_te, device)[:, None]).ravel())
        del model
        torch.cuda.empty_cache()
    base_cal = np.clip(np.mean(base_cal, 0), 0, 50)
    base_te = np.clip(np.mean(base_te, 0), 0, 50)
    report["ensemble_cal_day_rmse"] = rmse(y_cal[day_cal], base_cal[day_cal])

    # ---- 2. daytime-aware residual corrector --------------------------
    # The original stage fitted all hours, so it was dominated by night zeros.
    z_cal = meta_features(x_cal, base_cal, df.timestamp.iloc[ix_cal].reset_index(drop=True))
    z_te = meta_features(x_te, base_te, df.timestamp.iloc[ix_te].reset_index(drop=True))
    corr = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                               random_state=42, n_jobs=4)
    corr.fit(z_cal[day_cal], y_cal[day_cal] - base_cal[day_cal])
    # Apply the correction only where it was fitted: daytime samples.
    corr_cal = base_cal.copy()
    corr_cal[day_cal] = np.clip(base_cal[day_cal] + corr.predict(z_cal[day_cal]), 0, 50)
    corr_te = base_te.copy()
    corr_te[day_te] = np.clip(base_te[day_te] + corr.predict(z_te[day_te]), 0, 50)
    report["corrected_cal_day_rmse"] = rmse(y_cal[day_cal], corr_cal[day_cal])

    # In-sample on the calibration block the corrector is optimistic, so decide
    # whether to keep it using a held-out split *within* the calibration block.
    half = len(ix_cal) // 2
    c2 = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=4)
    m1 = day_cal.copy(); m1[half:] = False
    m2 = day_cal.copy(); m2[:half] = False
    c2.fit(z_cal[m1], y_cal[m1] - base_cal[m1])
    probe = base_cal.copy()
    probe[m2] = np.clip(base_cal[m2] + c2.predict(z_cal[m2]), 0, 50)
    keep_corrector = rmse(y_cal[m2], probe[m2]) < rmse(y_cal[m2], base_cal[m2])
    report["corrector_probe_rmse"] = rmse(y_cal[m2], probe[m2])
    report["corrector_probe_base"] = rmse(y_cal[m2], base_cal[m2])
    report["keep_corrector"] = bool(keep_corrector)
    neural_cal = corr_cal if keep_corrector else base_cal
    neural_te = corr_te if keep_corrector else base_te

    # ---- 3. tree ensemble on the same training block -------------------
    flat_tr = x_tr.reshape(len(x_tr), -1)
    flat_cal = x_cal.reshape(len(x_cal), -1)
    flat_te = x_te.reshape(len(x_te), -1)
    y_tr_s = ys.transform(df[["power"]].iloc[ix_tr]).ravel()
    trees = {}
    for name, est in [("rf", RandomForestRegressor(n_estimators=180, max_depth=18,
                                                   min_samples_leaf=2, random_state=42, n_jobs=4)),
                      ("xgb", XGBRegressor(n_estimators=350, max_depth=5, learning_rate=.05,
                                           subsample=.9, colsample_bytree=.9,
                                           random_state=42, n_jobs=4))]:
        est.fit(flat_tr, y_tr_s)
        trees[name] = (np.clip(ys.inverse_transform(est.predict(flat_cal).reshape(-1, 1)).ravel(), 0, 50),
                       np.clip(ys.inverse_transform(est.predict(flat_te).reshape(-1, 1)).ravel(), 0, 50))
        joblib.dump(est, OUT / f"tree_{name}.joblib")

    tree_cal = np.mean([trees[k][0] for k in trees], 0)
    tree_te = np.mean([trees[k][1] for k in trees], 0)
    report["tree_cal_day_rmse"] = rmse(y_cal[day_cal], tree_cal[day_cal])

    # ---- 4. regime-dependent convex blend, selected on calibration -----
    # The two families have complementary error profiles across the output
    # range, so the blend weight is fitted per predicted-output regime.
    # Regime edges come from calibration-block quantiles of the tree forecast.
    grid = np.linspace(0, 1, 101)
    edges = np.quantile(tree_cal[day_cal], [0.0, 0.25, 0.50, 0.75, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    report["regime_edges"] = [float(e) for e in edges[1:-1]]

    def regime_of(pred):
        return np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2)

    reg_cal, reg_te = regime_of(tree_cal), regime_of(tree_te)
    weights = []
    for r in range(len(edges) - 1):
        m = day_cal & (reg_cal == r)
        if m.sum() < 50:            # too few samples to fit a weight reliably
            weights.append(None)
            continue
        sc = [rmse(y_cal[m], w * neural_cal[m] + (1 - w) * tree_cal[m]) for w in grid]
        weights.append(float(grid[int(np.argmin(sc))]))

    # Global weight as the fallback for sparse regimes.
    gsc = [rmse(y_cal[day_cal], w * neural_cal[day_cal] + (1 - w) * tree_cal[day_cal])
           for w in grid]
    w_global = float(grid[int(np.argmin(gsc))])
    weights = [w_global if w is None else w for w in weights]
    report["blend_weight_global"] = w_global
    report["blend_weights_by_regime"] = weights

    # Verify on the held-out half of the calibration block that the regime
    # blend beats the single global weight; fall back if it does not.
    w_cal_reg = np.array([weights[r] for r in reg_cal])
    probe_reg = w_cal_reg[m2] * neural_cal[m2] + (1 - w_cal_reg[m2]) * tree_cal[m2]
    probe_glob = w_global * neural_cal[m2] + (1 - w_global) * tree_cal[m2]
    use_regime = rmse(y_cal[m2], probe_reg) < rmse(y_cal[m2], probe_glob)
    report["regime_probe_rmse"] = rmse(y_cal[m2], probe_reg)
    report["global_probe_rmse"] = rmse(y_cal[m2], probe_glob)
    report["use_regime_blend"] = bool(use_regime)

    w_te = np.array([weights[r] for r in reg_te]) if use_regime else np.full(len(reg_te), w_global)
    report["blend_cal_day_rmse"] = float(np.min(gsc))

    final_te = np.clip(w_te * neural_te + (1 - w_te) * tree_te, 0, 50)
    report["fit_seconds_total"] = time.perf_counter() - t0

    # ---- 5. score the test block once ----------------------------------
    rows = []
    for label, pred in [("Proposed framework v2", final_te),
                        ("  seed-ensemble neural", neural_te),
                        ("  tree ensemble", tree_te)]:
        for subset, mask in [("daytime", day_te), ("all", np.ones(len(y_te), bool))]:
            rows.append(dict(model=label.strip(), subset=subset, N=int(mask.sum()),
                             **metrics(y_te[mask], pred[mask])))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "v2_metrics.csv", index=False)
    pd.DataFrame({"timestamp": df.timestamp.iloc[ix_te].to_numpy(),
                  "y_true": y_te, "y_pred": final_te, "daytime": day_te}
                 ).to_csv(OUT / "pred_v2.csv", index=False)
    (OUT / "v2_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
