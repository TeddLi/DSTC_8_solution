#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../run_adapt_REBED.py  \
  --task_name REBED_adapt_manual_external_512 \
  --sample_num 2776510 \
  --mid_save_step 15000 \
  --input_file ../output/manual_external_REBED \
  --output_dir ../adapted_L-12_H-768_A-12/ubuntu \
  --vocab_file ../../DSTC_finetune/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../../DSTC_finetune/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../../DSTC_finetune/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 512 \
  --max_predictions_per_seq 25 \
  --train_batch_size 7 \
  --eval_batch_size 7 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --warmup_proportion 0.1 > ../log/ubuntu_REBED_adapted_extern_512_0d0.log 2>&1 &
