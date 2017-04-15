import numpy as np


class DatasetIterator(object):

	def __init__(self, data, labels, test_data, test_labels):
		self.data = data
		self.labels = labels
		self.batch_index = 0
		self.test_data = test_data
		self.test_labels = test_labels

	def next_batch(self, batch_size):
		if self.batch_index*batch_size + batch_size > len(self.data):
			self.batch_index = 0
		batched_data, batched_labels = self.data[self.batch_index*batch_size: self.batch_index*batch_size + batch_size], self.labels[self.batch_index*batch_size: self.batch_index*batch_size + batch_size]
		self.batch_index += 1
		return batched_data, batched_labels

	def get_all_data(self):
		return self.data

	def get_all_labels(self):
		return self.labels

	def size(self):
		return len(self.data)

	def get_test(self):
		return self.test_data, self.test_labels


class MultiDatasetIterator(object):

	def __init__(self, dataset1, dataset2):
		self.dataset1 = dataset1
		self.dataset2 = dataset2
		if dataset1.size() != dataset2.size():
			print("There is serios error in Multi dataset creation")
		if np.isclose(self.dataset1.get_all_labels(), self.dataset2.get_all_labels()).all():
			print("There is serios error in Multi dataset creation")

	def next_batch(self, batch_size):
		batched_data1, batched_labels1 = self.dataset1.next_batch(batch_size)
		batched_data2, batched_labels2 = self.dataset2.next_batch(batch_size)
		return batched_data1, batched_data2, batched_labels1

	def get_test(self):
		batched_data1, batched_labels1 = self.dataset1.get_test()
		batched_data2, batched_labels2 = self.dataset2.get_test()
		return batched_data1, batched_data2, batched_labels1