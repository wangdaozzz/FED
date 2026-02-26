#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# 创建日志目录
mkdir -p ./logs/FEDformer_Patch_K_wo_RevIn

for model in FEDformer_Patch_K_wo_RevIn
do

for preLen in 96
do

# exchange (8变量)
python -u run_K.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --task_id Exchange \
  --model $model \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 5 > ./logs/${model}/Exchange_96_${preLen}.log 2>&1

echo "Finished Exchange pred_len=${preLen}"



done



done

echo "All experiments finished!"

