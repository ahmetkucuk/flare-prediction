

def read_prob_file(file):
	pred_by_key = {}
	with open(file, 'r') as f:
		for line in f.readlines():
			cols = line.replace("\n", "").split("\t")
			pred_by_key[cols[0]] = [float(i) for i in cols[1:3]]
	return pred_by_key


def create_filename(exp_name, dataset_name, epoch):
	return '/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Prediction_Probs/flare_prediction/' + exp_name + '_' + dataset_name + '/test/test_probabilities_epoch' + str(epoch) + '.txt'


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

epoch = 600
dataset_name = '12_6'
for i in range(10, epoch, 10):
	print("Start for epoch: " + str(i))
	probs1 = read_prob_file(create_filename('exp_f11', dataset_name, i))
	probs2 = read_prob_file(create_filename('exp_f12', dataset_name, i))

	acc1 = calculate_accuracy(probs1)
	acc2 = calculate_accuracy(probs2)
	print(acc1)
	print(acc2)

	merged_probs = merge_probs(probs1, probs2)
	merged_acc = calculate_accuracy(merged_probs)
	print(merged_acc)