
import numpy as np
from collections import deque

NO_AUGMENTATION = -1
STRETCH_AUGMENTATION = 1
SQUEEZE_AUGMENTATION = 2
SHIFT_AUGMENTATION = 3
MIRROR_AUGMENTATION = 4
FLIP_AUGMENTATION = 5
REVERSE_AUGMENTATION = 6


def mirror_augmentation(data, labels):
	new_labels = list(labels)
	new_data = (np.array(data).astype(np.float32)*-1).tolist()
	return new_data, new_labels


def flip_augmentation(data, labels):
	new_labels = list(labels)
	mid_point = int(len(data)/2)
	new_data = data[mid_point:] + data[:mid_point]
	return new_data, new_labels


def reverse_augmentation(data, labels):
	new_labels = list(labels)
	new_data = list(data)
	new_data.reverse()
	return new_data, new_labels


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