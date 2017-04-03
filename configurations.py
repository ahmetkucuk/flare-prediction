
import numpy as np


def _norm_min_max(data):
	data = (data - np.min(data)) / (np.max(data) - np.min(data))
	return data


def _norm_z_score(data):
	data = (data - np.mean(data)) / np.std(data)
	return data


def _norm_zero_center(data):
	data = (data - np.mean(data))
	return data


def norm_min_max(data):
	data = np.nan_to_num(data)
	data = np.apply_along_axis(_norm_min_max, 1, data)
	return data


def norm_z_score(data):
	data = np.nan_to_num(data)
	data = np.apply_along_axis(_norm_z_score, 1, data)
	return data


def norm_zero_center(data):
	data = np.nan_to_num(data)
	data = np.apply_along_axis(_norm_zero_center, 1, data)
	return data


def get_configs(type, norm_type, expriment_name):

	data_root, model_dir = "", ""
	if type == "local":
		data_root = "/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARData"
		model_dir = "/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Tensorboard/FlarePrediction/" + expriment_name
	elif type == "server":
		data_root = "/home/ahmet/workspace/tensorflow/flare_prediction/ARData"
		model_dir = "/home/ahmet/workspace/tensorflow/tensorboard/flare_prediction/" + expriment_name
	else:
		print("invalid config type")
		exit()

	if norm_type == 'min_max':
		norm_func = norm_min_max
	elif norm_type == 'z_score':
		norm_func = norm_z_score
	elif norm_type == 'zero_center':
		norm_func = norm_zero_center
	else:
		norm_func = norm_min_max

	return data_root, model_dir, norm_func