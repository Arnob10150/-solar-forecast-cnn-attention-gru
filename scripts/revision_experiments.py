"""Fixed-protocol reviewer experiments; no test-driven model selection."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from train_evaluate import FEATURES, ROOT, make_model, metrics, set_seed
from tune_proposed import meta_features

OUT = ROOT / "results" / "revision"
SEEDS = [11, 23, 42, 67, 101]


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(len(FEATURES), 64)
        pos = torch.arange(48)[:, None]
        freq = torch.exp(torch.arange(0, 64, 2) * (-np.log(10000.) / 64))
        pe = torch.zeros(48, 64)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(64, 4, 128, .1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        z = self.encoder(self.project(x) + self.pe[:x.shape[1]])
        return self.head(z[:, -1]).squeeze(-1), None


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+256]).to(device))[0].cpu().numpy()
                               for i in range(0, len(x), 256)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(ROOT / "data/solar_processed.csv", parse_dates=["timestamp"])
    assert df.timestamp.is_unique and df.timestamp.is_monotonic_increasing
    assert df.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    n = len(df)
    a, b, c = int(.70*n), int(.775*n), int(.85*n)
    # First half of validation selects epochs; second half calibrates corrections/intervals.
    fs = StandardScaler().fit(df[FEATURES].iloc[:a])
    ys = StandardScaler().fit(df[["power"]].iloc[:a])
    v = fs.transform(df[FEATURES]).astype("float32")
    y = ys.transform(df[["power"]]).ravel().astype("float32")
    ids = np.arange(48, n)
    ix = [ids[ids<a], ids[(ids>=a)&(ids<b)], ids[(ids>=b)&(ids<c)], ids[ids>=c]]
    joblib.dump({"features":fs,"target":ys}, OUT / "scalers.joblib")
    protocol = dict(seeds=SEEDS, train_end=a, early_stop_end=b, calibration_end=c,
                    max_epochs=80, patience=10, batch_size=128, hidden=64, lr=.0005,
                    common_first_target=48, device=str(device), torch=torch.__version__)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    day = df.global_horizontal_irradiance.iloc[ix[3]].to_numpy()>20
    truth = df.power.iloc[ix[3]].to_numpy()
    rows = []

    def record(name, seed, window, pred, seconds, params, cal_pred=None):
        pred = np.clip(pred, 0, 50)
        frame = pd.DataFrame({"timestamp":df.timestamp.iloc[ix[3]].to_numpy(),
                              "y_true":truth,"y_pred":pred,"daytime":day})
        if cal_pred is not None:
            errors = np.abs(df.power.iloc[ix[2]].to_numpy()-np.clip(cal_pred,0,50))
            cal_day = df.global_horizontal_irradiance.iloc[ix[2]].to_numpy()>20
            q = np.quantile(errors[cal_day], min(1,np.ceil((cal_day.sum()+1)*.9)/cal_day.sum()), method="higher")
            frame["lower90"], frame["upper90"] = np.clip(pred-q,0,50), np.clip(pred+q,0,50)
        frame.to_csv(OUT / f"pred_{name}_w{window}_s{seed}.csv",index=False)
        for subset, mask in [("all",np.ones(len(day),bool)),("daytime",day)]:
            row = dict(model=name,seed=seed,window=window,subset=subset,N=int(mask.sum()),
                       fit_seconds=seconds,parameters=params,**metrics(truth[mask],pred[mask]))
            if cal_pred is not None:
                row["coverage90"] = np.mean((truth[mask]>=frame.lower90.to_numpy()[mask]) & (truth[mask]<=frame.upper90.to_numpy()[mask]))
                row["width90"] = np.mean((frame.upper90-frame.lower90).to_numpy()[mask])
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv",index=False)
        print(name,seed,window,"day RMSE",round(rows[-1]["RMSE"],5),flush=True)

    configs = [(name,48) for name in ["LSTM","GRU","CNN-LSTM","CNN-GRU","CNN-Attention-GRU","Transformer"]]
    configs += [("CNN-GRU",16),("CNN-Attention-GRU",16)]
    for name, window in configs:
        x = [np.ascontiguousarray(v[t[:,None]-np.arange(window,0,-1)]) for t in ix]
        for seed in SEEDS:
            set_seed(seed)
            model = (Transformer() if name=="Transformer" else make_model(name,len(FEATURES),64)).to(device)
            path = OUT / f"model_{name}_w{window}_s{seed}.pt"
            if path.exists():
                saved = torch.load(path,map_location=device,weights_only=False)
                model.load_state_dict(saved["state_dict"])
                seconds = saved["fit_seconds"]
            else:
                optimizer = torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=1e-4)
                # Keep the training arrays on GPU to avoid repeated host transfer overhead.
                tx, ty = torch.from_numpy(x[0]).to(device), torch.from_numpy(y[ix[0]]).to(device)
                best, stale, history = float("inf"), 0, []
                start = time.perf_counter()
                for epoch in range(1,81):
                    model.train()
                    order = torch.randperm(len(tx),device=device)
                    total = 0.
                    for batch in order.split(128):
                        optimizer.zero_grad(set_to_none=True)
                        yp,_ = model(tx[batch])
                        loss = (yp-ty[batch]).square().mean()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step()
                        total += loss.detach().item()*len(batch)
                    vl = float(np.mean((predict(model,x[1],device)-y[ix[1]])**2))
                    history.append(dict(epoch=epoch,train_loss=total/len(tx),val_loss=vl))
                    if vl < best-1e-6:
                        best, stale = vl, 0
                        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    else:
                        stale += 1
                    if epoch%5==0:
                        print(name,window,seed,"epoch",epoch,"val",round(vl,6),flush=True)
                    if stale>=10:
                        break
                seconds = time.perf_counter()-start
                model.load_state_dict(state)
                torch.save(dict(state_dict=state,fit_seconds=seconds,history=history,seed=seed),path)
                path.with_suffix(".json").write_text(json.dumps(history,indent=2))
                del tx, ty
            pp = [ys.inverse_transform(predict(model,z,device)[:,None]).ravel() for z in x[2:]]
            params = sum(p.numel() for p in model.parameters())
            record(name,seed,window,pp[1],seconds,params,pp[0])
            if name=="CNN-Attention-GRU" and window==48:
                # Calibration labels never influence checkpoint selection.
                zc = meta_features(x[2],pp[0],df.timestamp.iloc[ix[2]].reset_index(drop=True))
                zt = meta_features(x[3],pp[1],df.timestamp.iloc[ix[3]].reset_index(drop=True))
                corrector = ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,random_state=seed,n_jobs=4)
                start = time.perf_counter()
                corrector.fit(zc,df.power.iloc[ix[2]].to_numpy()-np.clip(pp[0],0,50))
                seconds += time.perf_counter()-start
                joblib.dump(corrector,OUT / f"residual_s{seed}.joblib")
                record("CNN-Attention-GRU-Residual",seed,48,np.clip(pp[1],0,50)+corrector.predict(zt),seconds,params)
            del model
            torch.cuda.empty_cache()
    # Classical models receive the same training labels and common test timestamps.
    for window in [16,48]:
        train_x = np.ascontiguousarray(v[ix[0][:,None]-np.arange(window,0,-1)]).reshape(len(ix[0]),-1)
        test_x = np.ascontiguousarray(v[ix[3][:,None]-np.arange(window,0,-1)]).reshape(len(ix[3]),-1)
        for name, model in [("Ridge",Ridge(alpha=1.)),
                            ("Random Forest",RandomForestRegressor(n_estimators=180,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=4)),
                            ("XGBoost",XGBRegressor(n_estimators=350,max_depth=5,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=4))]:
            start = time.perf_counter()
            model.fit(train_x,y[ix[0]])
            seconds = time.perf_counter()-start
            joblib.dump(model,OUT / f"model_{name}_w{window}.joblib")
            pred = ys.inverse_transform(model.predict(test_x).reshape(-1,1)).ravel()
            record(name,42,window,pred,seconds,0)
    record("Persistence",0,1,df.power.iloc[ix[3]-1].to_numpy(),0,0)
    # Empirical monthly upper envelope is estimated from training observations only.
    # This is a causal reference, not a physical clear-sky model.
    slots = df.timestamp.dt.hour*4+df.timestamp.dt.minute//15
    envelope = df.iloc[:a].assign(slot=slots.iloc[:a],month=df.timestamp.iloc[:a].dt.month).groupby(["month","slot"]).power.quantile(.95)
    lookup = pd.MultiIndex.from_arrays([df.timestamp.dt.month,slots])
    env = envelope.reindex(lookup).to_numpy()
    assert np.isfinite(env).all()
    pred = df.power.iloc[ix[3]-1].to_numpy()*env[ix[3]]/np.maximum(env[ix[3]-1],1.)
    record("Envelope persistence",0,1,pred,0,0)
    pd.DataFrame(rows).groupby(["model","window","subset"]).agg(
        RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),MAE_mean=("MAE","mean"),
        MAE_std=("MAE","std"),fit_seconds_mean=("fit_seconds","mean")
    ).to_csv(OUT / "summary.csv")


if __name__=="__main__":
    main()
