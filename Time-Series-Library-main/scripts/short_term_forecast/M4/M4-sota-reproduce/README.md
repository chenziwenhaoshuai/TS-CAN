# M4 SOTA Reproduce

Clean entry points for the M4 accepted 2-of-3 CANPatchTST reproduction set.

Run from `Time-Series-Library-main`:

```bash
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_reproduce.sh 0
```

Single-subset entry points:

```bash
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_M4_monthly.sh 0
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_M4_yearly.sh 0
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_M4_quarterly.sh 0
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_M4_daily.sh 0
bash scripts/short_term_forecast/M4/M4-sota-reproduce/run_M4_others.sh 0
```

`winning_configs.csv` contains the compact result table. Full evidence, forecasts, logs, model snapshots, and detailed notes are kept in `../M4-sota-reproduce-archive/`.
