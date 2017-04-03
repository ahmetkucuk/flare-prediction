
import numpy as np
from sklearn.decomposition import PCA


def _norm_min_max(data):
	data = (data - np.min(data)) / (np.max(data) - np.min(data))
	return data


def _norm_z_score(data):
	data = (data - np.mean(data)) / np.std(data, axis=0)
	return data


def _norm_zero_center(data):
	data = (data - np.mean(data, axis=0))
	return data


def _svd_whiten(X):
	X -= np.mean(X, axis=0) # zero-center the data (important)
	cov = np.dot(X.T, X) / X.shape[0]
	cov = np.nan_to_num(cov)
	U,S,V = np.linalg.svd(cov)
	Xrot = np.dot(X, U)
	Xwhite = Xrot / np.sqrt(S + 1e-5)
	return Xwhite


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


def norm_pca_whiten(data):
	data = np.nan_to_num(data)
	_data = np.zeros_like(data)
	for i in xrange(data.shape[0]):
		_data[i,:] = _svd_whiten(data[i,:])
	#data = np.apply_over_axes(_svd_whiten, data, [1, 2])
	return _data


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
	elif norm_type == 'pca_whiten':
		norm_func = norm_pca_whiten
	else:
		norm_func = norm_min_max

	return data_root, model_dir, norm_func