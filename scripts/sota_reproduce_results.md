# TS-CAN SOTA Reproduction Results

This file records the from-scratch c209 verification runs completed on
2026-07-25. The canonical entry point is
`Time-Series-Library-main/run_can.py`; ETT and extended dataset wrappers are
stored under `scripts/ETT-sota-reproduce/` and
`Time-Series-Library-main/scripts/long_term_forecast/*_script/*-sota-reproduce/`.

Run roots on c209:

- ETT: `Time-Series-Library-main/runs/ett_sota_reproduce_verify_20260725_1558`
- Extended: `runs/extended_sota_reproduce_verify_20260725_132826`

All 32 cells were rerun from scratch with return code 0.

| Dataset | Pred Len | TS-CAN MSE | TS-CAN MAE | TimeMixer++ MSE | TimeMixer++ MAE |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.360641 | 0.393012 | 0.361 | 0.403 |
| ETTh1 | 192 | 0.408625 | 0.425148 | 0.416 | 0.441 |
| ETTh1 | 336 | 0.429203 | 0.434345 | 0.430 | 0.434 |
| ETTh1 | 720 | 0.448998 | 0.466678 | 0.467 | 0.451 |
| ETTh2 | 96 | 0.279599 | 0.334393 | 0.276 | 0.328 |
| ETTh2 | 192 | 0.344544 | 0.382961 | 0.342 | 0.379 |
| ETTh2 | 336 | 0.347078 | 0.392917 | 0.346 | 0.398 |
| ETTh2 | 720 | 0.391742 | 0.426151 | 0.392 | 0.415 |
| ETTm1 | 96 | 0.285750 | 0.345105 | 0.310 | 0.334 |
| ETTm1 | 192 | 0.331659 | 0.370366 | 0.348 | 0.362 |
| ETTm1 | 336 | 0.368771 | 0.395129 | 0.376 | 0.391 |
| ETTm1 | 720 | 0.421972 | 0.426058 | 0.440 | 0.423 |
| ETTm2 | 96 | 0.167709 | 0.258385 | 0.170 | 0.245 |
| ETTm2 | 192 | 0.227966 | 0.298328 | 0.229 | 0.291 |
| ETTm2 | 336 | 0.279530 | 0.330340 | 0.303 | 0.343 |
| ETTm2 | 720 | 0.357305 | 0.388343 | 0.373 | 0.399 |
| Weather | 96 | 0.151219 | 0.197742 | 0.155 | 0.205 |
| Weather | 192 | 0.196165 | 0.241332 | 0.201 | 0.245 |
| Weather | 336 | 0.236991 | 0.274528 | 0.237 | 0.265 |
| Weather | 720 | 0.310680 | 0.328334 | 0.312 | 0.334 |
| Electricity | 96 | 0.134547 | 0.230544 | 0.135 | 0.222 |
| Electricity | 192 | 0.146802 | 0.243004 | 0.147 | 0.235 |
| Electricity | 336 | 0.163938 | 0.265687 | 0.164 | 0.245 |
| Electricity | 720 | 0.207915 | 0.295819 | 0.212 | 0.310 |
| Traffic | 96 | 0.361411 | 0.260577 | 0.392 | 0.253 |
| Traffic | 192 | 0.369759 | 0.270889 | 0.402 | 0.258 |
| Traffic | 336 | 0.391366 | 0.280973 | 0.428 | 0.263 |
| Traffic | 720 | 0.427019 | 0.291880 | 0.441 | 0.282 |
| Exchange | 96 | 0.082276 | 0.203203 | 0.085 | 0.214 |
| Exchange | 192 | 0.172350 | 0.295563 | 0.175 | 0.313 |
| Exchange | 336 | 0.306699 | 0.403407 | 0.316 | 0.420 |
| Exchange | 720 | 0.791557 | 0.666153 | 0.851 | 0.689 |

Win counts against TimeMixer++ Table 16:

- MSE: 29/32
- MAE: 13/32
