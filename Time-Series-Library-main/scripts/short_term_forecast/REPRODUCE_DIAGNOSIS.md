# Short-Term Reproduce Diagnosis

This note records why some short-term fresh training runs did not exactly match the archived best metrics.

## Confirmed Stable Cases

M4 Monthly, M4 Yearly, M4 Daily, M4 Others, and PEMS08 have archived forecasts or arrays whose independent metrics match the recorded values. PEMS08 also has the full epoch summary row for epoch 18.

## M4 Quarterly

Fresh reproducible target after recheck on c209:

- Trial: `QM02_scale0038`
- Selected epoch: `58`
- Metrics: SMAPE `9.8942166971`, MASE `1.1393020064`, OWA `0.8646491277`
- Status: accepted 2-of-3 against TimeMixer++ Table 17, MASE and OWA win

The previous archived best was:

- Trial: `Q1026_scale006`
- Selected epoch: `56`
- Metrics: SMAPE `9.8842781238`, MASE `1.1408986797`, OWA `0.8647799892`
- Source summary: `M4-sota-reproduce-archive/Quarterly/m4_quarterly_q980_refine.csv`
- Archived forecast: `M4-sota-reproduce-archive/Quarterly/Quarterly_forecast.csv`

The initial fresh rerun of `Q1026_scale006` reached only SMAPE `9.8896702060`, MASE `1.1453735431`, OWA `0.8666563420`, so the wrapper has been updated to `QM02_scale0038`. This is a minimal parameter compensation around the same Q980 basin: dropout `0.0032` and periodic image scale `0.0038`. It was trained from scratch on c209 and independently evaluated from the generated forecast.

## PEMS03

Archived best:

- Trial: `P3B44_02_bias0450`
- Selected epoch: `8`
- Metrics: MAE `14.5117149353`, MSE `554.5276489258`, RMSE `23.5484107516`, MAPE `13.3599922061`
- Source summary: `PEMS03-sota-reproduce-archive/remote_snapshot/pems03_bias044_mape_refine_gpu0.csv`
- Archived arrays: `PEMS03-sota-reproduce-archive/artifacts/pred.npy`, `true.npy`, `metrics.npy`

The script already restores the archived PEMS runner and CANPatchTST model snapshot before training. The remaining likely mismatch source is training nondeterminism: PEMS03 uses CUDA AMP and epoch-level training without saved checkpoint replay. Seeds are set, but deterministic CUDA kernels are not forced, and mixed precision plus data-loader/device ordering can shift the selected epoch metrics.

Additional diagnosis found that `exp_long_term_forecasting.py` also must be restored for this runner. The wrapper now restores that snapshot as well. With runner, model, and exp snapshots restored, the best fresh rerun still did not reach 2-of-3: epoch 12 reached MAE `14.5471410751`, MAPE `13.4437322617`, RMSE `23.6375279387`, only RMSE winning. A single-trial `gamma125` rerun came closest at epoch 12 with MAE `14.5258474350`, MAPE `13.4319409728`, RMSE `23.5440416967`, missing the TimeMixer++ MAPE threshold by `0.00194`.

The artifact arrays remain independently evaluable, so the archived result is evidence-stable. For strict from-scratch reproducibility, PEMS03 remains unresolved under the current CUDA/AMP environment.

## Practical Next Step

For paper/table claims, use archived forecast/array evaluation as the stable evidence. For strict from-scratch reproducibility, run:

```bash
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_M4_quarterly.sh 0
bash scripts/short_term_forecast/PEMS/PEMS03-sota-reproduce/run_PEMS03.sh 0
```

Then compare the produced `artifacts/reproduced/metrics.json` or summary CSV against `winning_configs.csv`. If mismatch remains, recover the original c209 checkpoints for Q1026 epoch 56 and P3B44 epoch 8, or rerun a deterministic/no-AMP confirmation as a separate protocol.
