#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python -u ../create_pretraining_data_advisor_REBED.py   \
  --vocab_file ../../DSTC_finetune/uncased_L-12_H-768_A-12/vocab.txt \
  --max_seq_length 512 \
  --max_predictions_per_seq 25 \
  --input_file ../DSTC8_DATA/External_data/advising.course-info.json \
  --output_file ../output/manual_external_advisor_REBED > ../log/REBED_advisor_manual_title_512_extern_rb0d0.log 2>&1 &
