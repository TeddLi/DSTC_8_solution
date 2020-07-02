#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../task2_eval.py  \
  --task_name 'EVAL_DSTC_task2' \
  --valid_dir '../DSTC8_DATA/Task_2/data_HUB/Ubuntu_REB_dstc_valid.tfrecord' \
  --restore_model_dir '../../DSTC_finetune/output/train_task1/HAE_task2_finetune2019-09-30_10:20' \
  --output_dir '../output/eval' \
  --do_lower_case True \
  --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 512 \
  --do_train True  \
  --do_eval True  \
  --train_batch_size 7 \
  --eval_batch_size 7 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --warmup_proportion 0.1 > ../log_file/EVAL_DSTC_task2.log 2>&1 &
