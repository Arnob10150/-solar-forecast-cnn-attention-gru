# Attention-Enhanced CNN-GRU Framework for Short-Term Solar Power Forecasting

Research code for a 15-minute-ahead photovoltaic power forecasting framework
evaluated on a real 50 MW grid-tied solar station, using measured
meteorological time-series data (2019–2020, 70,176 observations).

The framework combines a CNN-GRU encoder with temporal attention and
final-state fusion, seed ensembling, a daytime-restricted residual
correction stage (gated by a held-out retention probe), and a
regime-dependent blend with a tree ensemble (Random Forest + XGBoost).

## Results

Evaluated under a strict chronological split (train / early-stopping /
calibration / test) with five random seeds, daytime-first scoring, and
paired Diebold–Mariano significance tests against every baseline.

| Model | Daytime MAE (MW) | Daytime RMSE (MW) | Skill vs. envelope persistence |
|---|---|---|---|
| **Proposed framework** | **1.318** | **2.403** | **+0.041** |
| Random Forest | 1.376 | 2.450 | +0.022 |
| GRU (5-seed mean) | 1.356 | 2.453 | +0.021 |
| LSTM (5-seed mean) | 1.366 | 2.460 | +0.018 |
| XGBoost | 1.444 | 2.487 | +0.008 |
| Transformer (5-seed mean) | 1.499 | 2.525 | −0.008 |

The proposed framework achieves the lowest error on every metric tested,
with a negative (favorable) loss differential against all seven baselines
under paired Diebold–Mariano tests (HAC variance, Holm-adjusted), significant
under absolute loss for all seven comparators.

## Repository structure

```
scripts/
  preprocess.py              # Raw data cleaning and feature engineering
  train_evaluate.py          # Core model definitions, training loop, baselines
  tune_proposed.py           # Hyperparameter search for the proposed model
  finalize_tuned_results.py  # Consolidates the tuning run into final metrics
  revision_experiments.py    # Fixed-protocol multi-seed baseline comparison
  revision_v2_framework.py   # Proposed framework: seed ensemble + residual
                              #   stage + regime-dependent tree blend
  reviewer_checks.py         # Statistical checks (Diebold-Mariano tests, etc.)
requirements.txt
```

## Data

Expects a preprocessed CSV at `data/solar_processed.csv` with columns:
timestamp, irradiance/weather features, cyclic time encodings, and measured
power output. `preprocess.py` builds this file from the raw station record.

## Usage

```bash
pip install -r requirements.txt

# 1. Preprocess raw station data
python scripts/preprocess.py

# 2. Train and evaluate baselines + proposed model
python scripts/train_evaluate.py

# 3. Hyperparameter tuning for the proposed model
python scripts/tune_proposed.py
python scripts/finalize_tuned_results.py

# 4. Fixed-protocol, five-seed reviewer-requested comparison
python scripts/revision_experiments.py

# 5. Final proposed framework (seed ensemble + residual + tree blend)
python scripts/revision_v2_framework.py

# 6. Statistical significance checks
python scripts/reviewer_checks.py
```

Scripts write results (metrics, predictions, checkpoints) under a local
`results/` directory, created automatically on first run.

## Method summary

1. **Chronological split** — train / early-stopping / calibration / test
   blocks in time order, with scalers fit on the training block only.
2. **Seed ensembling** — the CNN-Attention-GRU is trained under five seeds
   and forecasts are averaged, removing most of the seed-to-seed variance.
3. **Daytime-restricted residual correction** — an ExtraTrees corrector is
   fit only on daytime residuals, and is kept only if a held-out probe on
   the calibration block confirms it improves accuracy (it is rejected in
   the final configuration).
4. **Regime-dependent blending** — the neural ensemble is combined with a
   Random Forest + XGBoost tree ensemble using blend weights fit
   per output-level regime on the calibration block, then evaluated once
   on the held-out test block.

## License

Research code released for reproducibility of the associated paper.
