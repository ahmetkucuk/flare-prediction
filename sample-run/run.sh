#!/usr/bin/env bash

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/lenet-model

# Where the dataset is saved to.
DATASET_DIR=/home/ahmet/Documents/Research/Time_Series/ARDataLarge

python ./../main.py \
  --train_dir=${TRAIN_DIR} \
  --norm_type=min_max \
  --n_input=14 \
  --dataset_dir=${DATASET_DIR} \
  --n_hidden=128 \
  --n_steps=60 \
  --n_classes=2 \
  --learning_rate=0.0001 \
  --n_cells=1 \
  --learning_rate=0.01 \
  --batch_size=20 \
  --is_lstm=False \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=sgd