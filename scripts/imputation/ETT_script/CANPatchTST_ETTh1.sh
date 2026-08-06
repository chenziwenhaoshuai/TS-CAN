export CUDA_VISIBLE_DEVICES=0

model_name=CANPatchTST

for mask_rate in 0.125 0.25 0.375 0.5
do
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_mask_${mask_rate} \
  --mask_rate ${mask_rate} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 1024 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 8 \
  --d_model 128 \
  --d_ff 192 \
  --patch_len 16 \
  --can_stride 8 \
  --can_shifts 1,2,4,8,16 \
  --can_cli_mode full \
  --can_temporal_cli_mode full \
  --can_temporal_roll 1 \
  --can_context_pyramid 1 \
  --can_use_gffng 1 \
  --can_drop_path 0.0 \
  --dropout 0.0 \
  --des 'CAN_ETTh1_D128_MSEMAE' \
  --itr 1 \
  --learning_rate 0.001 \
  --lradj cosine \
  --train_epochs 12 \
  --patience 4 \
  --loss MSEMAE \
  --loss_mse_weight 0.8 \
  --use_amp \
  --seed 2 \
  --num_workers 0 \
  --freq h
done
