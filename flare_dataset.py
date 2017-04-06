import re
import os
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from dataset_iterator import DatasetIterator


def get_files(data_root):
	data_path_by_name = {}
	for file in os.listdir(data_root):
		if file.endswith(".csv"):
			data_path_by_name[file] = os.path.join(data_root, file)
	return data_path_by_name


def get_two_char_after(filename, word):
	index = filename.index(word) + len(word)
	if filename[index+1].isdigit():
		return filename[index:index+2]
	else:
		return filename[index:index+1]


def find_labels(filename):

	label = [1, 0]
	prior = get_two_char_after(filename, "Prior")
	span = get_two_char_after(filename, "Span")
	if "noflare" in filename:
		label = [0, 1]
	return label, prior, span


def read_data(data_root):

	data_path_by_name = get_files(data_root=data_root)
	dataset_by_identifier = {}

	for filename in data_path_by_name.keys():

		label, prior, span = find_labels(filename)
		dataset_values = str(prior) + "_" + str(span)
		dataset_labels = str(prior) + "_" + str(span) + "_labels"
		# print(dataset_values)

		if not dataset_values in dataset_by_identifier:
			dataset_by_identifier[dataset_values] = []
			dataset_by_identifier[dataset_labels] = []

		with open(data_path_by_name[filename], "r") as f:
			skip_first = True
			ts_features_in_file = []
			for line in f:
				if skip_first:
					skip_first = False
					continue
				features = line.replace("\"", "").replace("\n", "").split(",")
				ts_features_in_file.append(features[1:])
			dataset_by_identifier[dataset_values].append(ts_features_in_file)
			dataset_by_identifier[dataset_labels].append(label)

	return dataset_by_identifier


def get_prior12_span12(data_root, norm_func):

	dataset_by_identifier = read_data(data_root=data_root)

	prior12_span12 = dataset_by_identifier["12_12"]
	prior12_span12_labels = dataset_by_identifier["12_12_labels"]

	return generate_test_train(prior12_span12, prior12_span12_labels, norm_func)


def get_prior12_span24(data_root, norm_func):

	dataset_by_identifier = read_data(data_root=data_root)

	prior12_span24 = dataset_by_identifier["12_24"]
	prior12_span24_labels = dataset_by_identifier["12_24_labels"]

	return generate_test_train(prior12_span24, prior12_span24_labels, norm_func)


def generate_test_train(data, labels, norm_func):

	n_of_records = len(data)
	split_at = int(n_of_records*0.8)

	data, labels = shuffle(data, labels)
	training_data = data[:split_at]
	training_labels = labels[:split_at]

	testing_data = data[split_at:]
	testing_labels = labels[split_at:]

	x, y = norm_func(np.array(training_data).astype("float32")), np.array(training_labels).astype("int8")
	test_x, test_y = norm_func(np.array(testing_data).astype("float32")), np.array(testing_labels).astype("int8")
	return DatasetIterator(x, y, test_x, test_y)
