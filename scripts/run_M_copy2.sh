#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# 创建日志目录
mkdir -p ./logs/FEDformer_Patch_wo_Frequency_Attention

for model in FEDformer_Patch_wo_Frequency_Attention
do

for preLen in 96 192 336 720
do

### electricity (321变量，高维数据集)
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --task_id ECL \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --itr 3 > ./logs/${model}/ECL_96_${preLen}.log 2>&1
#
#echo "Finished ECL pred_len=${preLen}"

# exchange (8变量)
python -u run.py \
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
  --itr 1 > ./logs/${model}/Exchange_96_${preLen}.log 2>&1

echo "Finished Exchange pred_len=${preLen}"



## weather (21变量)
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --task_id weather \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --itr 1 > ./logs/${model}/Weather_96_${preLen}.log 2>&1
#
#echo "Finished Weather pred_len=${preLen}"

done



done

echo "All experiments finished!"

##!/bin/bash
#
#export CUDA_VISIBLE_DEVICES=1
#
## 创建日志目录
#mkdir -p ./logs/FEDformer_Patch
#
#for model in FEDformer_Patch
#do
#
#for preLen in 720
#do
#
#### electricity
##python -u run.py \
##  --is_training 1 \
##  --root_path ./dataset/electricity/ \
##  --data_path electricity.csv \
##  --task_id ECL \
##  --model $model \
##  --data custom \
##  --features M \
##  --seq_len 96 \
##  --label_len 48 \
##  --pred_len $preLen \
##  --e_layers 2 \
##  --d_layers 1 \
##  --factor 3 \
##  --enc_in 321 \
##  --dec_in 321 \
##  --c_out 321 \
##  --des 'Exp' \
##  --itr 3 > ./logs/${model}/ECL_96_${preLen}.log 2>&1
##
##echo "Finished ECL pred_len=${preLen}"
##
### exchange
##python -u run.py \
##  --is_training 1 \
##  --root_path ./dataset/exchange_rate/ \
##  --data_path exchange_rate.csv \
##  --task_id Exchange \
##  --model $model \
##  --data custom \
##  --features M \
##  --seq_len 96 \
##  --label_len 48 \
##  --pred_len $preLen \
##  --e_layers 2 \
##  --d_layers 1 \
##  --factor 3 \
##  --enc_in 8 \
##  --dec_in 8 \
##  --c_out 8 \
##  --des 'Exp' \
##  --itr 3 > ./logs/${model}/Exchange_96_${preLen}.log 2>&1
##
##echo "Finished Exchange pred_len=${preLen}"
#
#
#
## weather
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --task_id weather \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $preLen \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --itr 3 > ./logs/${model}/Weather_96_${preLen}.log 2>&1
#
#echo "Finished Weather pred_len=${preLen}"
#
#done
#
#
#
#done
#
#echo "All experiments finished!"