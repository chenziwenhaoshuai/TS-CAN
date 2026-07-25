# PowerShell reproduction of ETTh1/96 for TS-CAN in Time-Series-Library-main
# Expected result: MSE = 0.360279, MAE = 0.393127

param(
    [string]$PythonBin = "python",
    [int]$Seed = 2
)

& "$PythonBin" -u run_can.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path "./dataset/ETT-small/" `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_96 `
  --model CANPatchTST `
  --data ETTh1 `
  --features M `
  --seq_len 192 `
  --label_len 48 `
  --pred_len 96 `
  --e_layers 2 `
  --d_model 128 `
  --d_ff 192 `
  --patch_len 16 `
  --can_stride 8 `
  --can_shifts "1,2,4,8,16" `
  --can_cli_mode full `
  --can_temporal_cli_mode full `
  --can_ctx_mode diff `
  --can_drop_path 0.05 `
  --can_kernel_size 3 `
  --can_use_gffng 1 `
  --can_temporal_roll 1 `
  --can_use_orth 0 `
  --can_context_pyramid 1 `
  --dropout 0.05 `
  --batch_size 8 `
  --learning_rate 0.0005 `
  --lradj cosine `
  --train_epochs 2 `
  --patience 2 `
  --des CAN_TSLIB_BEST `
  --itr 1 `
  --num_workers 0 `
  --use_amp `
  --seed $Seed
