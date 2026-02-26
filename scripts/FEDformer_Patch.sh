#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=FEDformer_Patch

# ETTh1 数据集
for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id ETTh1_96_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 64 \
      --d_ff 128 \
      --n_heads 8 \
      --moving_avg 25 \
      --patch_len 16 \
      --stride 8 \
      --modes 32 \
      --mode_select random \
      --dropout 0.1 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 10 \
      --batch_size 32 \
      --learning_rate 0.0001
done