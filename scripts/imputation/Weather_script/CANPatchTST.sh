export CUDA_VISIBLE_DEVICES=0

model_name=CANPatchTST

for mask_rate in 0.125 0.25 0.375 0.5
do
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_${mask_rate} \
  --mask_rate ${mask_rate} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 1024 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 128 \
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
  --des 'CAN_Weather_1024' \
  --itr 1 \
  --learning_rate 0.001 \
  --lradj cosine \
  --train_epochs 5 \
  --patience 3 \
  --use_amp \
  --seed 2 \
  --num_workers 0 \
  --freq t
done
