#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../run_task2_HAE_adapt.py \
  --task_name 'HAE_task2_finetune' \
  --train_dir '../DSTC8_DATA/Task_2/data_HUB/Ubuntu_REB_dstc_train_' \
  --valid_dir '../DSTC8_DATA/Task_2/data_HUB/Ubuntu_REB_dstc_valid.tfrecord' \
  --adapt_model_dir '../../Pre_trainprocess/adapted_L-12_H-768_A-12/ubuntu/bert_model_2019-09-30-08:13:26_epoch_0' \
  --output_dir '../output/train_task2' \
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
  --num_train_epochs 30 \
  --warmup_proportion 0.1 > ../log_file/REBED_1_HUB_adapt_task2_lr2d_epoch6_model.log 2>&1 &
