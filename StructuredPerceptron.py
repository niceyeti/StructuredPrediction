from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

#global labels
#global labelSet
XDIM = 128
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#labels = list("ACDIGOMN")
LABELSET = set(LABELS)
K = len(LABELS)
#build the lookup table of feature vector indices
g_unaryFeatureVectorIndices = {} #key=alpha (the class) val=tuple of (start,end+1) indices of the z/w vector components corresponding with each x-yi
for i in range(K):
	g_unaryFeatureVectorIndices[LABELS[i]] = (i*XDIM, (i+1)*XDIM)

#print(str(g_unaryFeatureVectorIndices))
#exit()
#k**2 pairwise indices come immediately after the k unary indices, 
g_pairwiseFeatureVectorIndices = {}
i = K * XDIM
for alpha1 in LABELS:
	for alpha2 in LABELS:
		g_pairwiseFeatureVectorIndices[alpha1+alpha2] = i
		i += 1

g_tripleFeatureVectorIndices = {}
for alpha1 in LABELS:
	for alpha2 in LABELS:
		for alpha3 in LABELS:
			g_tripleFeatureVectorIndices[alpha1+alpha2+alpha3] = i
			i += 1

g_quadFeatureVectorIndices = {}
for alpha1 in LABELS:
	for alpha2 in LABELS:
		for alpha3 in LABELS:
			for alpha4 in LABELS:
				g_quadFeatureVectorIndices[alpha1+alpha2+alpha3+alpha4] = i
				i += 1

"""
[print(str(item)) for item in g_unaryFeatureVectorIndices.items() if item[0] == "A" or item[0] == "Z" or item[0]=="Y"]
[print(str(item)) for item in g_pairwiseFeatureVectorIndices.items() if item[0] == "AA" or item[0] == "ZZ" or item[0]=="YZ"]
exit()
print(str(g_unaryFeatureVectorIndices))
print(str(g_pairwiseFeatureVectorIndices))
print(str(g_tripleFeatureVectorIndices))
exit()
"""

"""
Returns a random label sequence of the same length as x, as a list.
"""
def _getRandomY(yLabels, length):
	return [yLabels[random.randint(0,len(yLabels)-1)] for i in range(0,length)]

"""
Inner driver for feature function Phi1.

@x: The x sequence of inputs
@y: The y sequence of inputs
@i: The index into the sequence; NOTE the index may be less than the span of the features, for instance i==0, when we're considering
pairwise features (y_i-1,y_i). For these cases 0 is returned.
@d: The dimension of the weight vector

def _phi1(x,y,i,d):
	#init an all zero vector
	z = np.zeros((1,d))
	
	#print("y: "+str(y))
	
	#unary features
	unaryIndex = g_unaryFeatureVectorIndices[y[i]]
	z[0,unaryIndex] = x[i]
	
	#pairwise features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
	if i > 0:
		pairwiseIndex = g_pairwiseFeatureVectorIndices[y[i-1]+y[i]]
		z[0, pairwiseIndex] = 1.0

	return z
"""
	
	
"""
The structured feature function, mapping x and y to some score.
Given a structured input and output, returns a d-dimensional vector of the
combined features, eg, to be dotted with some w vector in a perceptron or other method.

This is phi_1 since it uses only up to pairwise features: phi(x,y) + phi(y_k,y_k-1) giving (d = m*k + k**2). 

@x: A list of m-dimensional binary strings; here, likely 128-bit binary vectors of optical character data
@y: The structured output for this input as a list of labels,
@d: The dimension of the Phi() and little phi() functions

returns: an R^d numpy vector representing the sum over all little phi features for the entire sequence
"""
def _Phi1(xseq,yseq,d):
	z = np.zeros((1,d))
	#print("z shape: "+str(z.shape[1]))
	#0th unary features are done first to avert if-checking index-bounds for pairwise features inside this high-frequency loop
	urange = g_unaryFeatureVectorIndices[yseq[0]]
	z[0,urange[0]:urange[1]] = xseq[0]
			
	#iterate pairwise and other y sequence features
	for i in range(1,len(yseq)):
		#unary features
		urange = g_unaryFeatureVectorIndices[yseq[i]]
		z[0,urange[0]:urange[1]] = xseq[i]

		#pairwise features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
		pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[i-1]+yseq[i]]
		z[0, pairwiseIndex] += 1.0
		#print("z: "+str(z))

	return z

def _Phi2(x,y,d):
	 pass
	 
def _Phi3(x,y,d):
	 pass

"""
Just implements score(x,y,w) = w dot _Phi(x,y). Returns float.

@x
"""
def _score(x,y,w,phi):
	"""
	if len(x) != len(y):
		print("ERROR |x| != |y| in _score()! exiting")
		exit()
	"""
	return w.dot(phi(x,y,w.shape[1]).T)[0,0]

"""
Randomized greedy search inference, as specified in the hw1 spec.

@xseq: Structured input
@w: the weights for the structured featured function as a 1xd numpy vector
@phi: the structured feature function
@R: the number of random restarts for the greedy search
"""
def InferRGS(xseq,w,phi,R):
	d = w.shape[1]
	#intialize y_hat structured output to random labels
	#y_max = _getRandomY(LABELS, len(xseq))
	#phi_y_max = np.zeros((1,d)) #the vector phi(x,y_hat,d), stored and returned so the caller need not recompute it
	#maxScore = _score(xseq, y_max, w, phi)
	yLen = len(xseq)
	maxScore = -1000000
	#print("y_max: "+str(y_max)+"  score: "+str(maxScore))
	for dummy in range(R):
		y_r = _getRandomY(LABELS, yLen)
		#there is an optimization here, since only changing one label at a time, only certain components of z change in the loop below; hence one can leverage this to avoid repeated calls to phi()
		z = phi(xseq, y_r, d)
		baseScore = w.dot(z.T)[0,0]
		#print("y_test: "+str(y_test))
		#evaluate all one label changes for y_test, a search space of size len(y_test)*k, where k is the number of labels/colors
		for i in range(yLen):
			#c_original = str(y_test[j])
			cOriginal = y_r[i]
			xi = xseq[i][0,:]
			#get the unary feature component
			urange = g_unaryFeatureVectorIndices[cOriginal]
			w_xy_c_original = w[0, urange[0]:urange[1]]
			#decrement the original unary component/feature
			tempScore = baseScore - w_xy_c_original.dot(xi)
			#get the pairwise component
			if i > 0:
				pairwiseIndex = g_pairwiseFeatureVectorIndices[y_r[i-1]+cOriginal]
				tempScore -= w[0, pairwiseIndex]
			#TODO: triple and quad components; ALSO in loop below
			#evaluate all k different modifications to this label
			for j in range(len(LABELS)):
				c = LABELS[j]
				urange = g_unaryFeatureVectorIndices[c]
				w_c = w[0, urange[0]:urange[1]]
				cScore = tempScore + w_c.dot(xi)
				#add the pairwise component
				if i > 0:
					pairwiseIndex = g_pairwiseFeatureVectorIndices[y_r[i-1]+c]
					cScore += w[0, pairwiseIndex]
				#TODO: triples and quad features

				if cScore > maxScore:
					maxScore = cScore
					#save y_hat, z_y_hat
					y_max = list(y_r)
					y_max[i] = c

	#print("ymax: "+str(y_max))
	#print("score: "+str(maxScore))

	return y_max, phi(xseq, y_max, d), maxScore

"""
The old core loop from InferRGS

#manipulate z cleverly to avoid full computation of score = w.dot(phi(x,y_test,d)) on every iteration;
#update the unary features f(x|y), decrementing the previous ones and incrementing the new ones
if j > 0:
	#decrement the previous f(x|y) features on successive iterations
	urange = g_unaryFeatureVectorIndices[prevC]
	z[0,urange[0]:urange[1]] -= x[0,:]

#increment current f(x,y) components by x vector
urange = g_unaryFeatureVectorIndices[c]
z[0,urange[0]:urange[1]] += x[0,:]

#update the pairwise features
if i > 0:
	#adjust the pairwise component of the z vector
	curYYIndex = g_pairwiseFeatureVectorIndices[y_test[i-1]+c]
	#add the new feature at its proper index
	z[0,curYYIndex] += 1.0
	if i > 1:
		#subtract the old feature 
		z[0,prevYYIndex] -= 1.0
	prevYYIndex = curYYIndex
	
prevC = c
#all components updated (without calling phi), so just get the score directly
score = w.dot(z.T)[0,0]
#print("score: "+str(score))
if score > maxScore:
	y_max = list(y_test)
	y_max[i] = c
	maxScore = score
	phi_y_max[0,:] = z[0,:]
	#print("new y_max: "+str(y_max)+"  score: "+str(maxScore))
"""
	
	
	
def InferRGS_Inefficient(x,w,phi,R):
	d = w.shape[1]
	#intialize y_hat structured output to random labels
	#y_max = _getRandomY(LABELS, len(x))
	yLen = len(x)
	phi_y_max = np.zeros((1,d))
	maxScore = -10000000
	#print("y_max: "+str(y_max)+"  score: "+str(maxScore))
	for _ in range(1,R):
		y_test = _getRandomY(LABELS, yLen)
		z = np.zeros(shape=(1,d))
		#print("y_test: "+str(y_test))
		#evaluate all one label changes for y_test, a search space of size len(x)*k, where k is the number of labels/colors
		for j in range(yLen):
			c_original = str(y_test[j])
			#evaluate all k different modifications to this label
			for c in LABELS:
				y_test[j] = c
				z = phi(x, y_test, d)
				score = w.dot(z.T)[0,0]
				#print("score: "+str(score)+"  maxscore: "+str(maxScore))
				if score > maxScore:
					y_max = list(y_test)
					phi_y_max[0,:] = z[0,:]
					maxScore = score
					#print("new y_max: "+str(y_max)+"  score: "+str(maxScore))
			y_test[j] = str(c_original)
		#reset z to all zeroes
		z[:,] = 0.0

	#print("ymax: "+str(y_max))
	#print("score: "+str(maxScore))
			
	return y_max, phi_y_max, maxScore	

"""
Checks whether or not this was a prediction error. In this case, just whether or not y* == y_hat

Returns: hamming loss
"""
def _getHammingError(y_star, y_hat):
	loss = 0
	for i in range(0,len(y_star)):
		if y_star[i] != y_hat[i]:
			loss += 1
	
	return loss
	
"""
Util for getting the correct phi function to pass around

@phiNum: The integer number of the phi funtion (1 for pairwise, 2 for triples, 3 for quadruples)
"""
def _getPhi(phiNum):
	if phiNum == 1: #first order features
		phi = _Phi1
	elif phiNum == 2: #second order features
		phi = _Phi2
	elif phiNum == 3: #third order features
		phi = _Phi3
	else:
		print("ERROR phi not found "+str(phiNum))
		exit()
		
	return phi

"""
@phiNum: phi number (1 for pairwise/bigram y features, 2 for triplets, etc)
@xdim: The dimensionality of each x; here, likely 128, for 128-bit inpu vectors
"""
def _getDim(phiNum, xdim):
	if phiNum == 1: #first order features
		# dim = number of components of the weight vector
		dim = xdim * K + K**2
	elif phiNum == 2: #second order features
		# dim = number of components of the weight vector
		dim = xdim * K + K**2 + K**3
	elif phiNum == 3: #third order features
		# dim = number of components of the weight vector
		dim = xdim * K + K**2 + K**3 + K**4
	else:
		print("ERROR phi not found "+str(phiNum))
		exit()

	return dim

"""
Utility for getting the ocr data as a dataset.

Returns: A list of training examples. Each training example is a pairwise x and y sequence: ([xsequence], [ysequence])
	xsequence: A list of real-valued numpy vectors of size m; each x vector is binary
	ysequence: A list of symbols/labels


"""
def _getOcrData(dataPath):
	#build the dataset as a list of x/y tuples
	D = []
	dataFile = open(dataPath,"r")
	records = [line.strip().replace("\t"," ") for line in dataFile.readlines()]
	xseq = []
	yseq = []
	#get dimension from the first x example
	XDIM = len(records[0].split(" ")[1].replace("im",""))
	print("XDIM in getOCRData: "+str(XDIM))
	#exit()
	for line in records:
		#print("line: "+line)
		if len(line) > 10:
			binaryString = line.split(" ")[1].replace("im","")
			#x = int(binaryString, 2)
			x = np.zeros((1,XDIM))
			for i in range(len(binaryString)):
				if binaryString[i] == "1":
					x[0,i] = 1
			xseq.append(x)
			yseq.append(line.split(" ")[2][0].upper())
		if len(line.strip()) < 10 and not high:
			D.append((xseq,yseq))
			high = True
			xseq = []
			yseq = []
		else:
			high = False

	return D

"""
_getOcrData returns a dataset as a list of x/y sequence pairs: [  (["0000101","00010110"],["a","t"]), etc. ]
This is inefficient for the structured perceptron construction, as the binary strings must be parsed during learning to map
each x_i to its component in an input vector. 

The solution is obviously to format the dataset in the proper manner before training, instead of doing so many thousands of more times
during training. Recall that each x_i binary string maps to some component of the unary components of the weight vector;


Returns: The new dataset as a list of x+y numpy vectors of the required dimension, under the construction required by the Phi function.

"""
def _preformatOcrData(D,phi,d):
	return [phi(example[0], example[1], d) for example in D]

"""
@D: A list of training examples in the pairwise form [ (xseq,yseq), (xseq, yseq) ... ].
@R: Number of restarts for RGS
@phiNum: The structured feature function phi(x,y)
@maxIt: max iterations
@eta: learning rate
"""
def OnlinePerceptronTraining(D, R, phiNum, maxIt, eta):
	phi = _getPhi(phiNum)
	xdim = XDIM
	print("xdim: "+str(xdim))
	dim = _getDim(phiNum, xdim)
	print("wdim: "+str(dim))
	#intialize weights of scoring function to 0
	w = np.zeros((1,dim), dtype=np.float32)
	d = w.shape[1]
	#get a preformatted version of the data, to help cut down on some computations
	preprocessedData = _preformatOcrData(D,phi,d)

	#a list of sum losses over an entire iteration
	losses = []
	print("num training examples: "+str(len(D)))
	for i in range(maxIt):
		sumItLoss = 0.0
		for j in range(len(D)):
			#sequential training
			xseq, y_star = D[j]
			#stochastic training: select a random training example (stochastic training)
			#x, y_star = D[random.randint(0,len(D)-1)] #IF USED, MAKE SURE TO USE CORRECT zStar BELOW IN preprocessedData[] index!
			#print("y_star: "+str(y_star))
			#get predicted structured output
			#print("j="+str(j)+" of "+str(len(D)))
			#y_hat = _getRandomY(labels, len(y_star))
			y_hat, phi_y_hat, score = InferRGS(xseq, w, phi, R)
			#y_hat, phi_y_hat, score = InferRGS_Inefficient(xseq, w, phi, R)
			#print("y_hat: "+str(y_hat)+"   score: "+str(score))
			#get the hamming loss
			#print("ystar: "+str(y_star))
			#print("yhat:  "+str(y_hat))
			loss = _getHammingError(y_star, y_hat)
			if loss > 0:
				#zStar = preprocessedData[j]  #effectively this is phi(x, y_star, d), but preprocessed beforehand to cut down on computations
				#zStar = phi(xseq, y_star, d)
				w = w + eta * (phi(xseq, y_star, d) - phi_y_hat)
				#w = w + eta * (zStar - phi(x, phi_y_hat, d))
				#w = w + eta * (zStar - phi_y_hat)
			sumItLoss += loss
		#append the total loss for this iteration, for plotting
		losses.append(sumItLoss)
		print("iter: "+str(i)+"  it-loss: "+str(losses[-1]))

	#plot the losses
	xs = [i for i in range(0,len(losses))]
	plt.ylim([0,max(losses)])
	plt.title("Total Hamming Loss per Iteration")
	plt.xlabel("Iteration")
	plt.ylabel("Sum Hamming Loss")
	plt.plot(xs, losses)
	plt.savefig("hammingLoss_Phi"+str(phiNum)+"_R"+str(R)+"_maxIt"+str(maxIt)+".png")
	plt.show()
	
	return w, losses

	
trainPath = None
testPath = None
R = 10
phi = 1
maxIt = 100
eta = 0.01
for arg in sys.argv:
	if "--trainPath=" in arg:
		trainPath = arg.split("=")[1]
	if "--testPath=" in arg:
		testPath = arg.split("=")[1]
	if "--eta=" in arg:
		eta = float(arg.split("=")[1])
	if "--phi=" in arg:
		phi = int(arg.split("=")[1])
	if "--maxIt=" in arg:
		maxIt = int(arg.split("=")[1])
	if "--R=" in arg:
		R = int(arg.split("=")[1])

if trainPath == None:
	print("ERROR no trainPath passed")
	exit()
if testPath == None:
	print("ERROR no testPath passed")
	exit()

trainData = _getOcrData(trainPath)
testData = _getOcrData(testPath)

print("Executing with  maxIt="+str(maxIt)+"   R="+str(R)+"   eta="+str(eta)+"   trainPath="+trainPath+"   testPath="+testPath)
#print(str(trainData[0]))

#print(str(trainData[0]))
#print("lenx: "+str(len(trainData[0][0]))+"  leny: "+str(len(trainData[0][1])))
w = OnlinePerceptronTraining(trainData, R, phi, maxIt, eta)
#TestPerceptron(w,testData)

