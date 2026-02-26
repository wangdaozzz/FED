export CUDA_VISIBLE_DEVICES=1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --task_id Exchange \
  --model FEDformer_Patch_Inv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --train_epochs 5 \
  --des 'Test' \
  --itr 1