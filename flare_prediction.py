
from flare_dataset import get_prior12_span12
from basic_lstm import BasicLSTMModel
from train_basic_lstm import TrainLSTM
import sys


def get_configs(type):

	data_root, model_dir = "", ""
	if type == "local":
		data_root = "/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARData"
		model_dir = "/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/BasicLSTM"
	elif type == "server":
		data_root = "/home/ahmet/workspace/tensorflow/flare_prediction/ARData"
		model_dir = "/home/ahmet/workspace/tensorflow/tensorboard/flare_prediction/basic_lstm"
	else:
		print("invalid config type")
		exit()
	return data_root, model_dir


def run(args):

	if len(args) < 1:
		print("Provide args: args[0] = local/server")
		exit()

	data_root, model_dir = get_configs(args[0])

	dataset = get_prior12_span12(data_root=data_root)
	lstm = BasicLSTMModel()
	train_lstm = TrainLSTM(lstm, dataset, model_dir=model_dir)
	train_lstm.train()
