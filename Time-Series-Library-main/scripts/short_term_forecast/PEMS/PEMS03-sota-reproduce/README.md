# PEMS03 Reproduce

Clean entry point for the PEMS03 CANPatchTST reproduction. The archived arrays are accepted 2-of-3; the latest fresh retraining on c209 is not accepted yet.

Run from `Time-Series-Library-main`:

```bash
bash scripts/short_term_forecast/PEMS/PEMS03-sota-reproduce/run_PEMS03.sh 0
```

`winning_configs.csv` contains the compact result row. Full evidence, arrays, runner snapshots, and detailed notes are kept in `../PEMS03-sota-reproduce-archive/`.
