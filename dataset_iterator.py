import numpy as np


class DatasetIterator(object):

	def __init__(self, data, labels, validation_labels=None, validation_data=None, test_data=None, test_ids=None):

		self.batch_index = 0

		self.data = data
		self.labels = labels

		self.validation_data = validation_data
		self.validation_labels = validation_labels

		self.test_ids = test_ids
		self.test_data = test_data

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
		return self.test_ids, self.test_data

	def get_validation(self):
		return self.validation_data, self.validation_labels

	def get_validation_as_dataset_iterator(self):
		return DatasetIterator(data=self.validation_data, labels=self.validation_labels)


class MultiDatasetIterator(object):

	def __init__(self, dataset1, dataset2, test_dataset1, test_dataset2):
		self.dataset1 = dataset1
		self.dataset2 = dataset2
		self.test_dataset1 = test_dataset1
		self.test_dataset2 = test_dataset2
		if dataset1.size() != dataset2.size():
			print("There is serious error in Multi dataset creation")
			exit()
		if not np.isclose(self.dataset1.get_all_labels(), self.dataset2.get_all_labels()).all():
			print("There is serious error in Multi dataset creation")
			exit()
		if not np.isclose(self.test_dataset1.get_all_labels(), self.test_dataset2.get_all_labels()).all():
			print("There is serious error in Multi dataset creation for TEST")
			exit()

	def next_batch(self, batch_size):
		batched_data1, batched_labels1 = self.dataset1.next_batch(batch_size)
		batched_data2, batched_labels2 = self.dataset2.next_batch(batch_size)
		return batched_data1, batched_data2, batched_labels1

	def get_multi_test(self):
		return self.test_dataset1.get_all_data(), self.test_dataset2.get_all_data(), self.test_dataset1.get_all_labels()