import re
import os
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from dataset_iterator import DatasetIterator
from dataset_iterator import MultiDatasetIterator
from sklearn.preprocessing import Imputer
from augmentation import *


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


def get_data(data_root, name, norm_func, augmentation_types, feature_indexes=range(14)):

	dataset_by_identifier = read_data(data_root=data_root, feature_indexes=feature_indexes)

	data = dataset_by_identifier[name]
	labels = dataset_by_identifier[name + "_labels"]
	ids = dataset_by_identifier[name + "_ids"]

	return generate_test_train(data, labels, ids, norm_func, augmentation_types)


def get_merged_data(data_root, span, norm_func, augmentation_types, feature_indexes=range(14)):

	dataset_by_identifier = read_data(data_root=data_root, feature_indexes=feature_indexes)
	d1 = "12_" + span
	d2 = "24_" + span

	data1 = dataset_by_identifier[d1]
	labels1 = dataset_by_identifier[d1 + "_labels"]
	ids1 = dataset_by_identifier[d1 + "_ids"]

	data2 = dataset_by_identifier[d2]
	labels2 = dataset_by_identifier[d2 + "_labels"]
	ids2 = dataset_by_identifier[d2 + "_ids"]

	data = data1 + data2
	labels = labels1 + labels2
	ids = ids1 + ids2

	return generate_test_train(data, labels, ids, norm_func, augmentation_types)


def extract_data_and_sort(dataset_by_identifier, dataname):

	data = dataset_by_identifier[dataname]
	labels = dataset_by_identifier[dataname + "_labels"]
	ids = dataset_by_identifier[dataname + "_ids"]

	ids, labels, data = zip(*sorted(zip(ids, labels, data), key=lambda t: t[0]))
	return ids, labels, data


def get_multi_data(data_root, norm_func, augmentation_types, feature_indexes):

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
			new_data1.append(d)
			new_labels1.append(l)

	for id, l, d in zip(ids2, labels2, data2):
		if id in common_records:
			new_data2.append(d)
			new_labels2.append(l)

	print(len(new_labels1))
	print(len(new_labels2))
	if not np.isclose(np.array(new_labels1).astype("int8"), np.array(new_labels2).astype("int8")).all():
		print("There is serios error in Multi dataset creation")
		exit()
	return generate_multi_test_train(data1=new_data1, data2=new_data2, labels=new_labels1, norm_func=norm_func, augmentation_types=augmentation_types)


def get_multi_feature(data_root, norm_func, augmentation_types, f1, f2, dataname):

	dataset_by_identifier = read_data(data_root=data_root, feature_indexes=f1)

	data1 = dataset_by_identifier[dataname]
	labels1 = dataset_by_identifier[dataname + "_labels"]

	dataset_by_identifier = read_data(data_root=data_root, feature_indexes=f2)

	data2 = dataset_by_identifier[dataname]
	labels2 = dataset_by_identifier[dataname + "_labels"]

	if not np.isclose(np.array(labels1).astype("int8"), np.array(labels2).astype("int8")).all():
		print("There is serios error in Multi dataset creation")
		exit()

	return generate_multi_test_train(data1=data1, data2=data2, labels=labels1, norm_func=norm_func, augmentation_types=augmentation_types)


def apply_augmentation(data, labels, augmentation_types):

	stretched_data, stretched_labels = [], []
	squeezed_data, squeezed_labels = [], []
	shifted_data, shifted_labels = [], []
	mirror_data, mirror_labels = [], []
	flip_data, flip_labels = [], []
	reverse_data, reverse_labels = [], []

	if STRETCH_AUGMENTATION in augmentation_types:
		stretched_data, stretched_labels = stretch_augmentation(list(data), list(labels))
	print("Stretching Augmentation applied")

	if SQUEEZE_AUGMENTATION in augmentation_types:
		squeezed_data, squeezed_labels = squeeze_augmentation(list(data), list(labels))
		print("Squeezing Augmentation applied")

	if SHIFT_AUGMENTATION in augmentation_types:
		shifted_data, shifted_labels = shift_augmentation(list(data), list(labels), 5)
		print("Shifting Augmentation applied")

	if MIRROR_AUGMENTATION in augmentation_types:
		mirror_data, mirror_labels = mirror_augmentation(list(data), list(labels))
		print("Mirror Augmentation applied")

	if FLIP_AUGMENTATION in augmentation_types:
		flip_data, flip_labels = flip_augmentation(list(data), list(labels))
		print("FLIP Augmentation applied")

	if REVERSE_AUGMENTATION in augmentation_types:
		reverse_data, reverse_labels = reverse_augmentation(list(data), list(labels))
		print("Reverse Augmentation applied")

	data = data + stretched_data
	labels = labels + stretched_labels

	data = data + squeezed_data
	labels = labels + squeezed_labels

	data = data + shifted_data
	labels = labels + shifted_labels

	data = data + mirror_data
	labels = labels + mirror_labels

	data = data + flip_data
	labels = labels + flip_labels

	data = data + reverse_data
	labels = labels + reverse_labels

	return data, labels


def generate_test_train(data, labels, ids, norm_func, augmentation_types):

	n_of_records = len(data)
	split_at = int(n_of_records*0.8)

	data, labels, ids = shuffle(data, labels, ids)
	training_data = data[:split_at]
	training_labels = labels[:split_at]

	testing_data = data[split_at:]
	testing_labels = labels[split_at:]
	testing_ids = ids[split_at:]

	if not NO_AUGMENTATION in augmentation_types:
		training_data, training_labels = apply_augmentation(training_data, training_labels, augmentation_types)
		training_data, training_labels = shuffle(training_data, training_labels)

	x, y = norm_func(np.array(training_data).astype("float32")), np.array(training_labels).astype("int8")
	test_x, test_y = norm_func(np.array(testing_data).astype("float32")), np.array(testing_labels).astype("int8")
	return DatasetIterator(x, y, test_ids=testing_ids, test_data=test_x, test_labels=test_y)


def generate_multi_test_train(data1, data2, labels, norm_func, augmentation_types):

	n_of_records = len(data1)

	split_at = int(n_of_records*0.8)

	data1, data2, labels = shuffle(data1, data2, labels)
	training_data1 = data1[:split_at]
	training_data2 = data2[:split_at]
	training_labels = labels[:split_at]

	testing_data1 = data1[split_at:]
	testing_data2 = data2[split_at:]
	testing_labels = labels[split_at:]

	if not NO_AUGMENTATION in augmentation_types:
		training_data1, training_labels = apply_augmentation(training_data1, training_labels, augmentation_types)
		training_data2, _ = apply_augmentation(training_data2, training_labels, augmentation_types)
		training_data1, training_data2, training_labels = shuffle(training_data1, training_data2, training_labels)

	x1, x2, y = norm_func(np.array(training_data1).astype("float32")), norm_func(np.array(training_data2).astype("float32")), np.array(training_labels).astype("int8")
	test_x1, test_x2, test_y = norm_func(np.array(testing_data1).astype("float32")), norm_func(np.array(testing_data2).astype("float32")), np.array(testing_labels).astype("int8")

	dataset1 = DatasetIterator(data=x1, labels=y)
	dataset2 = DatasetIterator(data=x2, labels=y)

	test_dataset1 = DatasetIterator(data=test_x1, labels=test_y)
	test_dataset2 = DatasetIterator(data=test_x2, labels=test_y)
	return MultiDatasetIterator(dataset1=dataset1, dataset2=dataset2, test_dataset1=test_dataset1, test_dataset2=test_dataset2)