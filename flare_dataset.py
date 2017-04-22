import re
import os
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from dataset_iterator import DatasetIterator
from dataset_iterator import MultiDatasetIterator
from collections import deque
from sklearn.preprocessing import Imputer


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
	timestamp = filename[:filename.index("Prior")]
	prior = get_two_char_after(filename, "Prior")
	span = get_two_char_after(filename, "Span")
	if "noflare" in filename:
		label = [0, 1]
	return timestamp, label, prior, span


def impute_by_mean(data):

	data = np.array(data, dtype=np.float32)

	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	imputed_data = imp.fit_transform(data)

	return imputed_data.tolist()


def read_data(data_root, feature_indexes):

	print("Reading data from: " + data_root)

	data_path_by_name = get_files(data_root=data_root)
	dataset_by_identifier = {}

	for filename in data_path_by_name.keys():

		timestamp, label, prior, span = find_labels(filename)
		dataset_values = str(prior) + "_" + str(span)
		dataset_labels = str(prior) + "_" + str(span) + "_labels"
		dataset_id = str(prior) + "_" + str(span) + "_ids"
		# print(dataset_values)

		if not dataset_values in dataset_by_identifier:
			dataset_by_identifier[dataset_values] = []
			dataset_by_identifier[dataset_labels] = []
			dataset_by_identifier[dataset_id] = []

		with open(data_path_by_name[filename], "r") as f:
			skip_first = True
			ts_features_in_file = []
			for line in f:
				if skip_first:
					skip_first = False
					continue
				features = line.replace("\"", "").replace("\n", "").split(",")
				features = features[1:]
				features = [features[i] for i in feature_indexes]
				ts_features_in_file.append(features)
			ts_features_in_file = impute_by_mean(ts_features_in_file)
			dataset_by_identifier[dataset_values].append(ts_features_in_file)
			dataset_by_identifier[dataset_labels].append(label)
			dataset_by_identifier[dataset_id].append(timestamp)

	return dataset_by_identifier


def get_prior12_span12(data_root, norm_func):

	dataset_by_identifier = read_data(data_root=data_root)

	prior12_span12 = dataset_by_identifier["12_12"]
	prior12_span12_labels = dataset_by_identifier["12_12_labels"]

	return generate_test_train(prior12_span12, prior12_span12_labels, norm_func)


def get_raw_prior12_span24(data_root):

	dataset_by_identifier = read_data(data_root=data_root)

	prior12_span24 = dataset_by_identifier["12_24"]
	prior12_span24_labels = dataset_by_identifier["12_24_labels"]
	return prior12_span24, prior12_span24_labels


def get_prior12_span24(data_root, norm_func, should_augment):

	dataset_by_identifier = read_data(data_root=data_root)

	prior12_span24 = dataset_by_identifier["12_24"]
	prior12_span24_labels = dataset_by_identifier["12_24_labels"]

	return generate_test_train(prior12_span24, prior12_span24_labels, norm_func)


def get_data(data_root, name, norm_func, augmentation_type, feature_indexes=range(14)):

	dataset_by_identifier = read_data(data_root=data_root, feature_indexes=feature_indexes)

	data = dataset_by_identifier[name]
	labels = dataset_by_identifier[name + "_labels"]

	return generate_test_train(data, labels, norm_func, augmentation_type)


def extract_data_and_sort(dataset_by_identifier, dataname):

	data = dataset_by_identifier[dataname]
	labels = dataset_by_identifier[dataname + "_labels"]
	ids = dataset_by_identifier[dataname + "_ids"]

	ids, labels, data = zip(*sorted(zip(ids, labels, data)))
	return ids, labels, data


def get_multi_data(data_root, norm_func, augmentation_type, feature_indexes):

	dataset_by_identifier = read_data(data_root=data_root, feature_indexes=feature_indexes)

	dataname1 = "12_24"
	dataname2 = "24_24"
	ids1, labels1, data1 = extract_data_and_sort(dataset_by_identifier, dataname1)
	ids2, labels2, data2 = extract_data_and_sort(dataset_by_identifier, dataname2)

	new_labels1, new_data1 = [], []
	new_labels2, new_data2 = [], []
	common_records = set(ids1).intersection(set(ids2))
	for id, l, d in zip(ids1, labels1, data1):
		if id in common_records:
			new_labels1.append(l)
			new_data1.append(d)

	for id, l, d in zip(ids2, labels2, data2):
		if id in common_records:
			new_labels2.append(l)
			new_data2.append(d)

	dataset1 = generate_test_train(new_data1, new_labels1, norm_func, augmentation_type)
	dataset2 = generate_test_train(new_data2, new_labels2, norm_func, augmentation_type)
	return MultiDatasetIterator(dataset1=dataset1, dataset2=dataset2)


def apply_augmentation(data, labels, augmentation_type):

	stretched_data, stretched_labels = [], []
	squeezed_data, squeezed_labels = [], []
	shifted_data, shifted_labels = [], []

	print(augmentation_type)
	if augmentation_type == 0 or augmentation_type == 1:
		print("Stretching Augmentation applied")
		stretched_data, stretched_labels = stretch_augmentation(data, labels)

	if augmentation_type == 0 or augmentation_type == 2:
		print("Squeezing Augmentation applied")
		squeezed_data, squeezed_labels = squeeze_augmentation(data, labels)

	if augmentation_type == 0 or augmentation_type == 3:
		print("Shifting Augmentation applied")
		shifted_data, shifted_labels = shift_augmentation(data, labels, 5)

	data = data + stretched_data
	labels = labels + stretched_labels

	data = data + squeezed_data
	labels = labels + squeezed_labels

	data = data + shifted_data
	labels = labels + shifted_labels

	return data, labels


def double_array(data):
	if len(data) == 0:
		return data
	new_data = []
	prev = np.array(data[0], dtype=float)
	for i in range(len(data)):
		d_array = np.array(data[i], dtype=float)
		new_data.append(((d_array + prev) / 2.0).tolist())
		new_data.append(data[i])
		prev = np.array(data[i], dtype=float)
	return new_data


def stretch_augmentation(data, labels):

	new_data = []
	new_labels = []

	for (single_record, label) in zip(data, labels):

		double_record = double_array(single_record)
		ts_length = len(double_record)

		new_single_record = []
		for i in range(ts_length / 2):
			new_single_record.append(double_record[i])

		new_data.append(new_single_record)
		new_labels.append(label)
		new_single_record = []

		for i in range(ts_length / 2, ts_length, 1):
			new_single_record.append(double_record[i])

		new_data.append(new_single_record)
		new_labels.append(label)
	return new_data, new_labels


def squeeze_augmentation(data, labels):

	new_data = []
	new_labels = []

	for (single_record, label) in zip(data, labels):

		ts_length = len(single_record)
		new_single_record = []

		for i in range(0, ts_length, 2):
			new_single_record.append(single_record[i])

		for i in range(1, ts_length, 2):
			new_single_record.append(single_record[i])

		new_data.append(new_single_record)
		new_labels.append(label)
	return new_data, new_labels


def shift_2d_list(list2d, rotate=1):

	d1_len = len(list2d)
	new_list2d = []
	for i in range(d1_len):
		data = list2d[i]
		items = deque(data)
		items.rotate(rotate)
		new_list2d.append(list(items))
	return new_list2d


def t_list(list):
	return np.asarray(list).T.tolist()


def shift_augmentation(data, labels, rotate):

	new_data = []
	new_labels = []

	for (single_record, label) in zip(data, labels):
		single_record = t_list(single_record)
		single_record = shift_2d_list(single_record, rotate=rotate)
		single_record = t_list(single_record)
		new_data.append(single_record)
		new_labels.append(label)
	return new_data, new_labels


def generate_test_train(data, labels, norm_func, augmentation_type):

	n_of_records = len(data)
	split_at = int(n_of_records*0.8)

	data, labels = shuffle(data, labels)
	training_data = data[:split_at]
	training_labels = labels[:split_at]

	if augmentation_type != -1:
		training_data, training_labels = apply_augmentation(training_data, training_labels, augmentation_type)
		training_data, training_labels = shuffle(training_data, training_labels)

	testing_data = data[split_at:]
	testing_labels = labels[split_at:]

	x, y = norm_func(np.array(training_data).astype("float32")), np.array(training_labels).astype("int8")
	test_x, test_y = norm_func(np.array(testing_data).astype("float32")), np.array(testing_labels).astype("int8")
	return DatasetIterator(x, y, test_x, test_y)