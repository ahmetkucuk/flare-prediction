#!/usr/bin/env bash

# Where the checkpoint and logs will be saved to.
#TRAIN_DIR=/tmp/lenet-model
TRAIN_DIR=/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/FlarePrediction/embedding5

# Where the dataset is saved to.
#DATASET_DIR=/home/ahmet/Documents/Research/Time_Series/ARDataLarge
DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARFinal

python ./../ensemble_main.py \
  --train_dir=${TRAIN_DIR} \
  --norm_type=zero_center \
  --n_input=1 \
  --dataset_dir=${DATASET_DIR} \
  --n_classes=2 \
  --learning_rate=0.0001 \
  --training_iters=120000 \
  --n_cells=1 \
  --dropout=0.7 \
  --batch_size=20 \
  --cell_type=GRU \
  --augmentation_type=1 \
  --display_step=100
