#!/usr/bin/env bash

# Where the checkpoint and logs will be saved to.
#TRAIN_DIR=/tmp/lenet-model
TRAIN_DIR=/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/FlarePrediction

# Where the dataset is saved to.
#DATASET_DIR=/home/ahmet/Documents/Research/Time_Series/ARDataLarge
DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARDataLarge

python ./../main.py \
  --train_dir=${TRAIN_DIR} \
  --norm_type=z_score \
  --n_input=14 \
  --dataset_dir=${DATASET_DIR} \
  --n_hidden=64 \
  --n_steps=120 \
  --n_classes=2 \
  --learning_rate=0.0001 \
  --n_cells=2 \
  --dropout=0.5 \
  --batch_size=20 \
  --is_lstm=False \
  --display_step=100 \
