
from flare_dataset import get_prior12_span12
from basic_lstm import BasicLSTMModel
from train_basic_lstm import TrainLSTM
from configurations import get_configs

import tensorflow as tf


def run(args):

	if len(args) < 3:
		print("Provide args: args[0] = local/server, args[1] = min_max/z_score/zero_center, args[2] = experiment_name")
		exit()

	data_root, model_dir, norm_func = get_configs(args[0], args[1], args[2])

	dataset = get_prior12_span12(data_root=data_root, norm_func=norm_func)

	lstm = BasicLSTMModel()
	train_lstm = TrainLSTM(lstm, dataset, model_dir=model_dir)
	train_lstm.train()
