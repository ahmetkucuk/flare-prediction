

base_path = '/Users/ahmetkucuk/Documents/Research/Flare_Prediction/Prediction_Probs/flare_prediction/'
datasets = ['12_6', '12_12', '12_24', '24_6', '24_12', '24_24']
features = ["Mean Gradient Total","Mean Gradient Horizontal","Mean Gradient Vertical","Twist (alpha)","Mean Inclination angle fro Radial","Mean Photospheric Magnetic Free Energy","Mean Shear Angle","Fraction Area Shear gt 45 deg","Mean Vertical Current Density","Unsigned Magnetic Flux in Gauss/Cm^2","Total Unsigned current Ampere","Total Photospheric Energy Density","Total Unsigned Current Helicity","Absolute Val of Current Helicity"]

def read_average_accuracy_from_file(filepath):

	sum = 0
	count = 0
	with open(filepath, "r") as f:
		for l in f.readlines():
			tuple = l.replace("\n", "").split("\t")
			sum = sum + float(tuple[1])
			count = count + 1
	return sum / count


def create_filename(exp_name, dataset_name):
	return base_path + exp_name + "_" + dataset_name + "/test/validation_accuracy.txt"


def print_feature_experiment_avg_acc():
	for feature in range(14):
		for dataset in datasets:
			avg_acc = read_average_accuracy_from_file(create_filename("exp_feature" + str(feature), dataset))
			print(features[feature] + "\t" + dataset + "\t" + "{0:.2f}".format(avg_acc))


print_feature_experiment_avg_acc()