#!/usr/bin/env bash

# Where the checkpoint and logs will be saved to.
#TRAIN_DIR=/tmp/lenet-model
TRAIN_DIR=/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/FlarePrediction/embedding4

# Where the dataset is saved to.
#DATASET_DIR=/home/ahmet/Documents/Research/Time_Series/ARDataLarge
DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARFinal

python ./../main.py \
  --train_dir=${TRAIN_DIR} \
  --norm_type=z_score \
  --n_input=1 \
  --dataset_dir=${DATASET_DIR} \
  --n_hidden=8 \
  --n_steps=60 \
  --n_classes=2 \
  --learning_rate=0.0005 \
  --training_iters=120000 \
  --n_cells=1 \
  --dropout=0.8 \
  --batch_size=15 \
  --cell_type=GRU \
  --augmentation_type=0 \
  --dataset_name=12_12 \
  --display_step=100 \
  --feature_indexes=1 \
