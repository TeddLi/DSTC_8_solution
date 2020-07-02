#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../task1_ubuntu_eval.py  \
  --task_name 'EVAL_DSTC_task1_ubuntu' \
  --valid_dir '../DSTC8_DATA/Task_1/ubuntu/data_HUB/Ubuntu_REB_dstc_valid.tfrecord' \
  --output_dir '../output/eval' \
  --restore_model_dir '../output/train_task1/DSTC_task1_ubuntu_HAE_finetune2019-09-30_11:08' \
  --do_lower_case True \
  --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 320 \
  --do_train True  \
  --do_eval True  \
  --train_batch_size 12 \
  --eval_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --warmup_proportion 0.1 > ../log_file/EVAL_DSTC_task1_ubuntu.log 2>&1 &
