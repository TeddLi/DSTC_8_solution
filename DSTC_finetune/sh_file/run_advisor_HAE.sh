#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../run_task1_advisor_HAE_adapt.py \
  --task_name 'DSTC_advisor_finetune' \
  --train_dir '../DSTC8_DATA/Task_1/advising/data/Advising_REBED_dstc_train_' \
  --valid_dir '../DSTC8_DATA/Task_1/advising/data/Advising_dstc_valid.tfrecord' \
  --output_dir '../output/train_task1' \
  --adapt_model_dir '../../Pre_trainprocess/adapted_L-12_H-768_A-12/advisor/bert_model_2019-09-30-08:10:20_epoch_0' \
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
  --num_train_epochs 30 \
  --warmup_proportion 0.1 > ../log_file/Task1_HAE_advisor_epoch6_lr2d_len320.log 2>&1 &
