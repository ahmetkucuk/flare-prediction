
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
	U,S,V = np.linalg.svd(cov)
	Xrot = np.dot(X, U)
	Xwhite = Xrot / np.sqrt(S + 1e-5)
	return Xwhite


def norm_min_max(data):
	data = np.apply_along_axis(_norm_min_max, 1, data)
	return data


def norm_z_score(data):
	data = np.apply_along_axis(_norm_z_score, 1, data)
	return data


def norm_zero_center(data):
	data = np.apply_along_axis(_norm_zero_center, 1, data)
	return data


def norm_pca_whiten(data):
	_data = np.zeros_like(data)
	for i in xrange(data.shape[0]):
		_data[i, :] = _svd_whiten(data[i,:])
	return _data


def no_norm(data):
	return data


def get_norm_func(norm_type):

	if norm_type == 'min_max':
		return norm_min_max
	elif norm_type == 'z_score':
		return norm_z_score
	elif norm_type == 'zero_center':
		return norm_zero_center
	elif norm_type == 'pca_whiten':
		return norm_pca_whiten
	elif norm_type == 'no_norm':
		return no_norm
	else:
		print("invalid config norm_func name")
		exit()

	return None


def get_feature_indexes(feature_str):

	if feature_str == "all":
		return range(14)

	return [int(a) for a in feature_str.split(",")]


def get_dataset_size(dataset_name):

	if dataset_name == "12_6":
		return 30
	if dataset_name == "12_12":
		return 60
	if dataset_name == "12_24":
		return 120
	if dataset_name == "24_6":
		return 30
	if dataset_name == "24_12":
		return 60
	if dataset_name == "24_24":
		return 120

	print("dataset name couldn't find given: " + dataset_name)

	return -1