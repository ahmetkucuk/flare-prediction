import numpy as np
import os


def read_prob_file(file):
	pred_by_key = {}
	with open(file, 'r') as f:
		for line in f.readlines():
			cols = line.replace("\n", "").split("\t")
			pred_by_key[cols[0]] = [float(i) for i in cols[1:3]]
	return pred_by_key


def create_filename(exp_name, dataset_name, epoch):
	return '/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Prediction_Probs/flare_prediction/' + exp_name + '_' + dataset_name + '/test/test_probabilities_epoch' + str(epoch) + '.txt'


def create_merged_probs_filename(exp_name, dataset_name, epoch):
	data_root = '/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Prediction_Probs/flare_prediction/merged/' + exp_name + '_' + dataset_name
	if not os.path.exists(data_root):
		os.makedirs(data_root)
	return data_root + '/merged_probs' + str(epoch) + '.txt'


def calculate_accuracy(pred_by_key):
	correct_count = 0
	for key in pred_by_key:
		cols = pred_by_key[key]
		#Index 1 is noflare
		true_label_index = 0
		if "noflare" in key:
			true_label_index = 1

		if cols[true_label_index] >= 0.5:
			correct_count = correct_count + 1

	return (float(correct_count) / len(pred_by_key)) * 100


def merge_probs(probs1, probs2):
	pred_by_key = {}
	for key in probs1:
		cols1 = probs1[key]
		cols2 = probs2[key]

		if max(cols1) > max(cols2):
			pred_by_key[key] = cols1
		else:
			pred_by_key[key] = cols2
	return pred_by_key


def write_merged_probs(merged_probs, output_file_path):
	with open(output_file_path, "w") as output_file:
		for key in merged_probs:
			cols = merged_probs[key]
			if np.argmax(cols) == 1:
				output_file.write(key + "\t" + "N\n")
			else:
				output_file.write(key + "\t" + "F\n")

epoch = 1000
#12_6 = 400
#12_12 = 400
#12_24 = 400
#24_6 = 400
#24_12 = 400
#24_24 = 400
for dataset_name in ['12_6', '12_12', '12_24', '24_6', '24_12', '24_24']:
	for i in range(10, epoch, 10):
		print("Start for epoch: " + str(i))
		probs1 = read_prob_file(create_filename('final0', dataset_name, i))
		probs2 = read_prob_file(create_filename('final1', dataset_name, i))
		output_file_path = create_merged_probs_filename('final0final1', dataset_name, i)

		merged_probs = merge_probs(probs1, probs2)
		# print(calculate_accuracy(probs1))
		# print(calculate_accuracy(probs2))
		# print(calculate_accuracy(merged_probs))
		write_merged_probs(merged_probs, output_file_path)