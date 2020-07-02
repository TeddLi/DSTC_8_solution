#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../task1_advisor_eval.py  \
  --task_name 'EVAL_DSTC_task1_advisor' \
  --valid_dir '../DSTC8_DATA/Task_1/advising/data/Advising_dstc_valid.tfrecord' \
  --output_dir '../output/eval' \
  --restore_model_dir '../output/train_task1/DSTC_adapt2019-09-30_09:25/' \
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
  --warmup_proportion 0.1 > ../log_file/EVAL_DSTC_task1_advisor.log 2>&1 &
