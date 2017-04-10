import numpy as np


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


std_data = rescale(each[1:])

paalistcos = paa(std_data,s,None)

################raw###################
datacos = np.array(std_data)
datasin = np.sqrt(1-np.array(std_data)**2)

paalistcos = np.array(paalistcos)
paalistsin = np.sqrt(1-paalistcos**2)

datacos = np.matrix(datacos)
datasin = np.matrix(datasin)

paalistcos = np.matrix(paalistcos)
paalistsin = np.matrix(paalistsin)


paamatrix = paalistcos.T*paalistcos-paalistsin.T*paalistsin
matrix = np.array(datacos.T*datacos-datasin.T*datasin)