import numpy as np
from flare_dataset import get_raw_prior12_span24
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Rescale data into [0,1]
def rescale(serie):
	maxval = max(serie)
	minval = min(serie)
	gap = float(maxval-minval)
	return [(each-minval)/gap for each in serie]


#PAA function
def paa(series, now, opw):
	if now == None:
		now = len(series) / opw
	if opw == None:
		opw = len(series) / now
	return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]


def create_image(each):

	std_data = rescale(each)

	paalistcos = paa(std_data, 2, None)

	datacos = np.array(std_data)
	datasin = np.sqrt(1-np.array(std_data)**2)

	paalistcos = np.array(paalistcos)
	paalistsin = np.sqrt(1-paalistcos**2)

	datacos = np.matrix(datacos)
	datasin = np.matrix(datasin)

	paalistcos = np.matrix(paalistcos)
	paalistsin = np.matrix(paalistsin)

	paamatrix = paalistsin.T*paalistcos-paalistcos.T*paalistsin
	matrix = np.array(datasin.T*datacos - datacos.T*datasin)

	return matrix


def no_norm(data):
	return data


def scale(I):
	return (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)


def plot(sample):
	fig = plt.figure()
	sample = scale(sample)
	plt.axis('off')
	plt.imshow(sample)

	return fig

data, label = get_raw_prior12_span24(data_root="/Users/ahmetkucuk/Documents/Research/Flare_Prediction/ARDataLarge/")

print(label[1])
print(label[6])

data = np.nan_to_num(np.array(data).astype(np.float32))

paper_data = [10.364243,10.569554,10.448875,10.100696,9.620371,8.882499,8.415529,8.210217,7.916430,7.806151,7.746734,7.739054,7.825141,7.725533,7.602954,7.581851,7.518337,7.436305,7.472943,7.561754,7.512576,7.438912,7.404399,7.229231,7.021992,6.951436,7.010480,7.030379,6.918896,6.771265,6.698433,6.788380,6.920205,6.981216,7.044445,7.176120,7.361769,7.554219,7.815139,8.169803,8.502066,8.759834,9.070481,9.444034,9.633329,9.672459,9.819816,10.049488,10.242047,10.444507,10.725154,11.181386,11.786936,12.370038,13.061426,13.688799,14.157277,15.025812,15.794914,16.341785,17.137178,17.105600,16.856855,17.452514,17.818658,17.731623,18.104903,18.622070,19.071488,19.356830,19.894282,20.412883,20.248427,20.332815,20.171856,19.693733,19.294176,18.576514,18.298972,18.293209,18.596757,18.681908,18.426792,18.107913,17.625184,17.654017,17.632459,17.648564,17.802928,17.955134,17.924891,17.644847,17.468598,17.441981,16.887715,16.452399,16.473880,16.334339,15.915413,15.646766,15.495058,15.090164,14.998315,14.912714,14.971236,15.134401,14.917223,14.788855,14.928515,15.129588,15.616064,16.053952,16.191802,16.409787,16.701072,17.002885,17.179578,17.441662,17.721085,17.678840,17.918111,18.203322,18.139260,18.276740,17.994029,17.626485,17.215113,16.869374,16.632141,16.364775,16.187834,15.775114,15.289302,15.043267,15.159046,15.152410,14.943802,14.978605,15.210032,15.252456,15.317919,15.774887,16.433198,17.044576,17.644100,18.105112,18.468331,19.153136,19.209367,18.976273,19.824706,20.242011,20.199990,20.181935,20.150939,20.153533,20.218712,20.104891,19.676247,19.834607,19.537585,18.955578,18.432279,18.185626,17.893510,17.524748,17.462834,17.401095,16.681007,15.684240,14.904768,14.272827,13.420076,12.629529,12.060267,11.620882,11.485278,11.380921,11.395782,11.477000,11.476917,11.826483,12.241685,12.450265,12.749899,13.095385,13.047783,12.885725,12.701516,12.785449,13.393270,13.869409,14.503924,15.304961,16.108104,16.805240,17.134942,17.578700,18.174890,19.332056,20.136107,20.667895,21.574551,22.459263,23.067630,23.648103,23.979652,24.369788,24.818125,25.008890,24.134428,23.405342,23.137598,22.520758,22.548832,22.896264,23.220230,23.312567,22.905799,24.130076,24.234114,23.256460,22.155179,21.052644,20.002074,19.366134,19.347928,19.606128,19.721116,20.101111,21.387463,21.935976,22.302522,22.183594,20.384239,18.372517,16.707258,15.138856,13.538920,12.072288,10.742817,9.803680,8.717828,7.903624,7.206921,6.794415,6.670088,6.692805,6.718133,6.739781,6.803307,6.733101,6.391111,5.826248,5.134848,4.286434,3.579532,2.989805,2.582228,2.312904,2.124389,2.003812,1.919437,1.857304,1.801970,1.760452,1.726789,1.695984,1.665493,1.642137,1.618473,1.597051,1.583376,1.565658,1.542886,1.531117,1.518927,1.505118,1.493133,1.475563,1.464682,1.456444,1.442087,1.433698,1.425288,1.418705]
fig3 = plot(create_image(paper_data))
fig3.savefig("fig3.png")

print(len(data))
transposed_data = data.T.tolist()
for i in range(10):
	single_feature1 = transposed_data[i][i][:60]
	single_feature2 = transposed_data[i][i][:60]

	fig1 = plot(create_image(single_feature1))
	fig2 = plot(create_image(single_feature2))


	fig1.savefig("fig1%d.png" % i)
	fig2.savefig("fig2%d.png" % i)






