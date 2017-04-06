
from flare_dataset import get_prior12_span12
from basic_rnn import BasicRNNModel
from train_basic_lstm import TrainRNN
from configurations import get_configs

import tensorflow as tf


def run(args):

	if len(args) < 3:
		print("Provide args: args[0] = local/server, args[1] = min_max/z_score/zero_center, args[2] = experiment_name")
		exit()

	data_root, model_dir, norm_func = get_configs(args[0], args[1], args[2])

	dataset = get_prior12_span12(data_root=data_root, norm_func=norm_func)

	n_input = 14
	n_steps = 60
	n_hidden = 256
	n_classes = 2

	learning_rate = 0.0001
	training_iters = 300000
	batch_size = 20
	display_step = 100

	n_cells = 1

	is_lstm = False

	lstm = BasicRNNModel(n_input=n_input, n_steps=n_steps, n_hidden=n_hidden, n_classes=n_classes, n_cells=n_cells, is_lstm=is_lstm)
	train_lstm = TrainRNN(lstm, dataset, model_dir=model_dir, learning_rate=learning_rate, training_iters=training_iters, batch_size=batch_size, display_step=display_step)
	train_lstm.train()
