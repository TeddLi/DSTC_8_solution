#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../evaluate_pretrain.py \
  --task_name eval_pre \
  --sample_num 2776510 \
  --mid_save_step 15000 \
  --restore_dir ../adapted_L-12_H-768_A-12/ubuntu/bert_model_2019-09-30-08:13:26_epoch_0\
  --input_file ../output/manual_external_REBED \
  --output_dir ../adapted_L-12_H-768_A-12 \
  --vocab_file ../../DSTC_finetune/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../../DSTC_finetune/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../../DSTC_finetune/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 512 \
  --max_predictions_per_seq 25 \
  --train_batch_size 7 \
  --eval_batch_size 7 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --warmup_proportion 0.1 > ../log/eval_pretrain.log 2>&1 &

