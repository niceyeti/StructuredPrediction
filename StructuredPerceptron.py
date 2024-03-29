from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

XDIM = 0
USE_TRIPLES = False
USE_QUADS = False
LABELS = ""
#prepare a set of pseudo-random numbers indices into labels in advance, of prime length, such that randint() doesn't need to be called at high frequency
RAND_LABEL_INDICES = []
RAND_RING_INDEX = 0
RAND_INTS_LEN = 0

#labels = list("ACDIGOMN")
K = len(LABELS)
#create the global vector index dicts
g_unaryFeatureVectorIndices = {} #key=alpha (the class) val=tuple of (start,end+1) indices of the z/w vector components corresponding with each x-yi
g_pairwiseFeatureVectorIndices = {}
g_tripleFeatureVectorIndices = {}
g_quadFeatureVectorIndices = {}

"""
Builds the set of dictionaries for looking up indices into the feature vector.

"""
def _buildVectorIndexDicts(xdim,labels):
	global g_unaryFeatureVectorIndices
	global g_pairwiseFeatureVectorIndices
	global g_tripleFeatureVectorIndices
	global g_quadFeatureVectorIndices

	k = len(labels)
	
	#build the lookup table of feature vector indices
	g_unaryFeatureVectorIndices = {} #key=alpha (the class) val=tuple of (start,end+1) indices of the z/w vector components corresponding with each x-yi
	for i in range(len(labels)):
		g_unaryFeatureVectorIndices[labels[i]] = (i*xdim, (i+1)*xdim)

	#k**2 pairwise indices come immediately after the k unary indices, 
	g_pairwiseFeatureVectorIndices = {}
	i = g_unaryFeatureVectorIndices[labels[-1]][1]
	for alpha1 in labels:
		for alpha2 in labels:
			g_pairwiseFeatureVectorIndices[alpha1+alpha2] = i
			i += 1

	g_tripleFeatureVectorIndices = {}
	for alpha1 in labels:
		for alpha2 in labels:
			for alpha3 in labels:
				g_tripleFeatureVectorIndices[alpha1+alpha2+alpha3] = i
				i += 1

	g_quadFeatureVectorIndices = {}
	for alpha1 in labels:
		for alpha2 in labels:
			for alpha3 in labels:
				for alpha4 in labels:
					g_quadFeatureVectorIndices[alpha1+alpha2+alpha3+alpha4] = i
					i += 1

"""
Gets a random label via a list of cached random numbers in the range of the length of the label set.
"""
def _getRandLabel():
	global RAND_RING_INDEX
	
	RAND_RING_INDEX += 1
	if RAND_RING_INDEX >= RAND_INTS_LEN:
		RAND_RING_INDEX = 0

	return LABELS[ RAND_LABEL_INDICES[RAND_RING_INDEX] ]

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
	#c = _getRandLabel()
	#return [c for i in range(0,length)]
	return [_getRandLabel() for i in range(0,length)]

	#opted to use a prepared list of random ints instead of randint(), which is likely to be slow
	#return [yLabels[random.randint(0,len(yLabels)-1)] for i in range(0,length)]

"""
Inner driver for feature function Phi1.

@x: The x sequence of inputs
@y: The y sequence of inputs
@i: The index into the sequence; NOTE the index may be less than the span of the features, for instance i==0, when we're considering
pairwise features (y_i-1,y_i). For these cases 0 is returned.
@d: The dimension of the weight vector

def _phi1(x,y,i,d):
	#init an all zero vector
	z = np.zeros((1,d), dtype=np.float32)
	
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
def _Phi1(xseq, yseq, d):
	z = np.zeros((1,d), dtype=np.float32)
	#print("z shape: "+str(z.shape[1]))
	#0th unary features are done first to avert if-checking index-bounds for pairwise features inside this high-frequency loop
	urange = g_unaryFeatureVectorIndices[yseq[0]]
	z[0,urange[0]:urange[1]] = xseq[0][0,:]

	#iterate pairwise and other y sequence features
	for i in range(1,len(yseq)):
		#unary features
		urange = g_unaryFeatureVectorIndices[yseq[i]]
		z[0,urange[0]:urange[1]] += xseq[i][0,:]
		
		#pairwise features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
		pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[i-1]+yseq[i]]
		z[0, pairwiseIndex] += 1.0

	return z

"""
Includes up to triple-gram features
"""
def _Phi2(xseq,yseq,d):
	z = np.zeros((1,d), dtype=np.float32)
	#print("z shape: "+str(z.shape[1]))
	#0th unary features are done first to avert if-checking index-bounds for pairwise features inside this high-frequency loop
	urange = g_unaryFeatureVectorIndices[yseq[0]]
	z[0,urange[0]:urange[1]] = xseq[0][0,:]

	#initialize the first unary, pairwise features at index 1
	urange = g_unaryFeatureVectorIndices[yseq[1]]
	z[0,urange[0]:urange[1]] += xseq[1][0,:]
	pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[0]+yseq[1]]
	z[0, pairwiseIndex] = 1.0	#assignment, since this is the first pairwise feature

	#iterate pairwise, and triples from index 2 forward
	for i in range(2,len(yseq)):
		#unary features
		urange = g_unaryFeatureVectorIndices[yseq[i]]
		z[0,urange[0]:urange[1]] += xseq[i][0,:]

		#pairwise features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
		pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[i-1]+yseq[i]]
		z[0, pairwiseIndex] += 1.0

		#triple features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
		tripleIndex = g_tripleFeatureVectorIndices[yseq[i-2]+yseq[i-1]+yseq[i]]
		z[0, tripleIndex] += 1.0

	return z

def _Phi3(xseq,yseq,d):
	z = np.zeros((1,d), dtype=np.float32)
	#print("z shape: "+str(z.shape[1]))
	#0th unary features are done first to avert if-checking index-bounds for pairwise features inside this high-frequency loop
	urange = g_unaryFeatureVectorIndices[yseq[0]]
	z[0,urange[0]:urange[1]] = xseq[0][0,:]
			
	#initialize the first unary and pairwise features at index 1
	urange = g_unaryFeatureVectorIndices[yseq[1]]
	z[0,urange[0]:urange[1]] += xseq[1][0,:]
	pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[0]+yseq[1]]
	z[0, pairwiseIndex] = 1.0 #assignment, since this is the first pairwise feature increment
	
	#initialize the first unary, pairwise, and triple at index 2
	urange = g_unaryFeatureVectorIndices[yseq[2]]
	z[0,urange[0]:urange[1]] += xseq[2][0,:]
	pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[1]+yseq[2]]
	z[0, pairwiseIndex] += 1.0
	tripleIndex = g_tripleFeatureVectorIndices[yseq[0]+yseq[1]+yseq[2]]
	z[0, tripleIndex] = 1.0 #assignment, since this is the first triple feature increment

	#iterate pairwise, triples, and quads from index 2 forward
	for i in range(3,len(yseq)):
		#unary features
		urange = g_unaryFeatureVectorIndices[yseq[i]]
		z[0,urange[0]:urange[1]] += xseq[i][0,:]

		#pairwise features
		pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[i-1]+yseq[i]]
		z[0, pairwiseIndex] += 1.0

		#triple features
		tripleIndex = g_tripleFeatureVectorIndices[yseq[i-2]+yseq[i-1]+yseq[i]]
		z[0, tripleIndex] += 1.0
		
		#quad features
		quadIndex = g_quadFeatureVectorIndices[yseq[i-3]+yseq[i-2]+yseq[i-1]+yseq[i]]
		z[0, quadIndex] += 1.0

	return z

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


"""
def SaveLosses(accuracies, finalAccuracy, beamWidth, dataPath, searchMethod, beamUpdateType, titleStr, show, isTraining):
	#write the losses to file, for repro
	if isTraining:
		fnamePrefix = "Results/"+dataPath[0:4]+"Train_b"+str(beamWidth)+"_"+searchMethod+"_"+beamUpdateType
	else:
		fnamePrefix = "Results/"+dataPath[0:4]+"Test_b"+str(beamWidth)+"_"+searchMethod+"_"+beamUpdateType

	ofile = open(fnamePrefix+".txt","w+")
	ofile.write(str(accuracies)+"\n")
	ofile.write(str(finalAccuracy)+"\n")
	ofile.close()

	if isTraining:
		#plot the training loss
		xs = [i for i in range(0,len(accuracies))]
		plt.ylim([0,max(accuracies)])
		plt.title(titleStr)
		plt.xlabel("Iteration")
		plt.ylabel("Hamming Accuracy")
		plt.plot(xs, accuracies)
		x1,x2,y1,y2 = plt.axis()
		plt.axis((x1,x2,0.0,1.0))
		plt.savefig(fnamePrefix+".png")
		if show:
			plt.show()

"""
#Old; this was for InferRGS methods, hw1
def eSaveLosses(losses, datapath, show=True):
def SaveLosses(losses, datapath, show=True):
	#plot the losses
	xs = [i for i in range(0,len(losses))]
	plt.ylim([0,max(losses)])
	plt.title("Total Hamming Loss per Iteration")
	plt.xlabel("Iteration")
	plt.ylabel("Sum Hamming Loss")
	plt.plot(xs, losses)
	plt.savefig(datapath[0:5]+"_HamLoss_Phi"+str(phiNum)+"_R"+str(R)+"_maxIt"+str(maxIt)+".png")
	if show:
		plt.show()
"""
		
"""
Randomized greedy search inference, as specified in the hw1 spec.

@xseq: Structured input
@w: the weights for the structured featured function as a 1xd numpy vector
@phi: the structured feature function
@R: the number of random restarts for the greedy search

def InferRGS(xseq, w, phi, R):
	d = w.shape[1]
	#intialize y_hat structured output to random labels
	#y_max = _getRandomY(LABELS, len(xseq))
	#phi_y_max = np.zeros((1,d), dtype=np.float32) #the vector phi(x,y_hat,d), stored and returned so the caller need not recompute it
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
			#decrement all the other relevant pairwise, triple, or quad components from score
			if i > 0:
				pairwiseIndex = g_pairwiseFeatureVectorIndices[y_r[i-1]+cOriginal]
				tempScore -= w[0, pairwiseIndex]
			if USE_TRIPLES and i > 1:
				tripleIndex = g_tripleFeatureVectorIndices[y_r[i-2]+y_r[i-1]+cOriginal]
				tempScore -= w[0, tripleIndex]
			if USE_QUADS and i > 2:
				quadIndex = g_quadFeatureVectorIndices[y_r[i-3]+y_r[i-2]+y_r[i-1]+cOriginal]
				tempScore -= w[0, quadIndex]
			
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
				if USE_TRIPLES and i > 1:
					tripleIndex = g_tripleFeatureVectorIndices[y_r[i-2]+y_r[i-1]+c]
					cScore += w[0, tripleIndex]
				if USE_QUADS and i > 2:
					quadIndex = g_quadFeatureVectorIndices[y_r[i-3]+y_r[i-2]+y_r[i-1]+c]
					cScore += w[0, quadIndex]

				if cScore > maxScore:
					maxScore = cScore
					#save y_hat, z_y_hat
					y_max = list(y_r)
					y_max[i] = str(c)

	#print("ymax: "+str(y_max))
	#print("score: "+str(maxScore))

	return y_max, phi(xseq, y_max, d), maxScore
"""
def InferRGS(xseq, w, phi, R):
	d = w.shape[1]
	#intialize y_hat structured output to random labels
	#y_max = _getRandomY(LABELS, len(xseq))
	#phi_y_max = np.zeros((1,d), dtype=np.float32) #the vector phi(x,y_hat,d), stored and returned so the caller need not recompute it
	#maxScore = _score(xseq, y_max, w, phi)
	yLen = len(xseq)
	maxIterations = 10000
	yLenSeq = [i for i in range(yLen)]
	maxScore = -10000000
	#print("y_max: "+str(y_max)+"  score: "+str(maxScore))
	for _ in range(R):
		y_test = _getRandomY(LABELS, yLen)
		#print("y_test: "+str(y_test))
		#there is an optimization here, since only changing one label at a time, only certain components of z change in the loop below; hence one can leverage this to avoid repeated calls to phi()
		#z = phi(xseq, y_test, d)
		#get the initial/base score for this 'good' instance; note the math below only modifies this base score based on the single label component that changes, instead of recalculating the complete score
		#baseScore = w.dot(z.T)[0,0]
		#y_local_max = y_test
		localMaxScore = -10000000
		iterations = 0
		convergence = False #convergence satisfied when there is no further improvemet to be made via one character changes, eg, the label sequence does not change
		while not convergence:
			#there is an optimization here, since only changing one label at a time, only certain components of z change in the loop below; hence one can leverage this to avoid repeated calls to phi()
			z = phi(xseq, y_test, d)
			#get the initial/base score for this 'good' instance; note the math below only modifies this base score based on the single label component that changes, instead of recalculating the complete score
			baseScore = w.dot(z.T)[0,0]

			#until convergence, evaluate all one label changes for y_test, a search space of size len(y_test)*k, where k is the number of labels
			for i in yLenSeq:
				######## begin by decrementing the score by the original ith-label's components ###################
				cOriginal = y_test[i]
				xi = xseq[i][0,:]
				#get the unary feature component
				urange = g_unaryFeatureVectorIndices[cOriginal]
				w_xy_c_original = w[0, urange[0]:urange[1]]
				#decrement the original unary component/feature
				tempScore = baseScore - w_xy_c_original.dot(xi)
				#decrement all the other relevant pairwise, triple, or quad components from score
				if i > 0:
					pairwiseIndex = g_pairwiseFeatureVectorIndices[y_test[i-1]+cOriginal]
					tempScore -= w[0, pairwiseIndex]
					if USE_TRIPLES and i > 1:
						tripleIndex = g_tripleFeatureVectorIndices[y_test[i-2]+y_test[i-1]+cOriginal]
						tempScore -= w[0, tripleIndex]
						if USE_QUADS and i > 2:
							quadIndex = g_quadFeatureVectorIndices[y_test[i-3]+y_test[i-2]+y_test[i-1]+cOriginal]
							tempScore -= w[0, quadIndex]
				######## end decrements; now we can add individual components for each label change, below #######
				
				###### evaluate all k different modifications to this label, incrementing the base score by each component ###
				for c in LABELS:
					if c != cOriginal:
						urange = g_unaryFeatureVectorIndices[c]
						w_c = w[0, urange[0]:urange[1]]
						cScore = tempScore + w_c.dot(xi)
						#add active ngram components to cScore
						if i > 0:
							pairwiseIndex = g_pairwiseFeatureVectorIndices[y_test[i-1]+c]
							cScore += w[0, pairwiseIndex]
							#add the triple components
							if USE_TRIPLES and i > 1:
								tripleIndex = g_tripleFeatureVectorIndices[y_test[i-2]+y_test[i-1]+c]
								cScore += w[0, tripleIndex]
								#add the quad components
								if USE_QUADS and i > 2:
									quadIndex = g_quadFeatureVectorIndices[y_test[i-3]+y_test[i-2]+y_test[i-1]+c]
									cScore += w[0, quadIndex]

						if cScore > localMaxScore:
							localMaxScore = cScore
							#save max character; y_local_max list can be updated outside this loop, to spare repeated list-construction
							y_local_max = list(y_test)
							y_local_max[i] = c
				### end-for: evaluate all k label changes for this position, and possibly obtained max as y_local_max and localMaxScore
			### end-for (over entire sequence), check for convergence
			if y_local_max == y_test or iterations > maxIterations:
				convergence = True
				#for debugging only; i just want to know if I ever bottom out in terms of iterations
				if iterations >= maxIterations:
					print("WARNING: ITERATIONS BOTTOMED OUT IN INFER_RGS()")
			else:
				y_test = y_local_max
				baseScore = localMaxScore
			iterations += 1
			#print("iterations: "+str(iterations))+"  y_test: "+str(y_test)+"     y_local_max: "+str(y_local_max), end="")
		### end while: converged to single label sequence, so update the global greedy max over R iterations as needed
		if localMaxScore > maxScore:
			maxScore = localMaxScore
			y_max = list(y_local_max)

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

def InferRGS_Inefficient(x, w, phi, R):
	d = w.shape[1]
	yLen = len(x)
	#phi_y_max = np.zeros((1,d), dtype=np.float32)
	maxScore = -10000000
	#print("y_max: "+str(y_max)+"  score: "+str(maxScore))
	for r in range(0,R):
		#get a random y as a starting point for greedy search
		y_test = _getRandomY(LABELS, yLen)
		#print("y_test: "+str(y_test))
		convergence = False
		while not convergence: #until y_test == y_local_max, keep updating single char changes
			#evaluate all one label changes for y_test, a search space of size len(x)*k, where k is the number of labels/colors
			localMaxScore = -10000000
			for i in range(yLen):
				c_original = str(y_test[i])
				#evaluate all k different modifications to this label
				for c in LABELS:
					y_test[i] = c
					z = phi(x, y_test, d)
					cScore = w.dot(z.T)[0,0]
					#print("score: "+str(score)+"  maxscore: "+str(maxScore))
					if cScore > localMaxScore:
						y_local_max = list(y_test)
						localMaxScore = cScore
						#print("new y_max: "+str(y_max)+"  score: "+str(maxScore))
				#replace original character and continue to next position
				y_test[i] = c_original
		
			#loop convergence check/update
			if y_test == y_local_max:
				convergence = True
			else:
				y_test = list(y_local_max)
		#end while
		
		#update local max score and sequence as needed
		if localMaxScore > maxScore:
			maxScore = localMaxScore
			y_max = list(y_local_max)

	#print("ymax: "+str(y_max))
	#print("score: "+str(maxScore))
			
	return y_max, phi(x, y_max, d), maxScore	

"""
Given some x sequence, returns the top scoring b nodes representing the first character
in a y sequence, where b is the beam width.
"""
def _beamSearchInitialization(xseq, w, beamWidth):
	beam = []
	for c in LABELS:
		urange = g_unaryFeatureVectorIndices[c]
		score = w[0,urange[0]:urange[1]].dot(xseq[0][0,:].T)
		beam.append(([c],score))

	#sort and beamify the beam
	beam.sort(key = lambda tup: tup[1], reverse=True)
	if beamWidth > 0:
		beam = beam[0:beamWidth]
	#print("BEAM: "+str(beam))
	return beam

"""
Performs best-first-search beam update: expand only the highest scoring node on the beam,
delete that node from beam, and append all of its children. Re-sort the beam, then truncate the beam
by beamWidth.

@sortedBeam: A sorted beam, with the highest scoring node at the 0th index.
@beamWidth: The beam width; if <= 0, beam is infinity
@xseq: The input x-sequence, required here just for scoring the nodes
@w: The current weight vector, again required just for scoring nodes
@phi: Feature function, also just for scoring nodes

Returns: new, sorted beam, with max at 0th index
"""
def _bestFirstBeamUpdate(sortedBeam,beamWidth,xseq,w,phi):
	d = w.shape[1]
	#get the max on the beam
	y_max = sortedBeam[0][0]
	cxSeqIndex = len(y_max)
	baseScore = sortedBeam[0][1]
	#remove y_max from beam, to prevent cycling
	sortedBeam = sortedBeam[1:]
	xvec = xseq[cxSeqIndex][0,:]
	prevC = y_max[-1]
	
	#expand the beam with children (candidate set) of highest scoring node
	for c in LABELS:
		candidate = y_max + [c]
		#adjust unary score component
		urange = g_unaryFeatureVectorIndices[c]
		candidateScore = baseScore + w[0,urange[0]:urange[1]].dot(xvec)
		#pairwise features
		pairwiseIndex = g_pairwiseFeatureVectorIndices[prevC+c]
		candidateScore += w[0, pairwiseIndex]
		#candidateScore = w.dot( phi(xseq,candidate,d).T ) #TODO: if necessary, optimize this score update; full-score computation is repetitive and unnecessary
		sortedBeam.append( (candidate,candidateScore) )

	#sort beam by node scores
	sortedBeam.sort(key = lambda tup: tup[1],reverse=True)
	#prune beam
	if len(sortedBeam) > beamWidth and beamWidth > 0:
		sortedBeam = sortedBeam[0:beamWidth]

	#print("BEAM: "+str(sortedBeam))
		
	return sortedBeam

"""
Performs breadth-first-search beam update: replace all nodes on the beam with their children, re-score, sort by score,
and truncate by @beamWidth.

@beam: The current beam
@beamWidth: The beam width; if <= 0, beam is infinity
@xseq: The input x-sequence, required here just for scoring the nodes
@w: The current weight vector, again required just for scoring nodes
@phi: Feature function, also just for scoring nodes

Returns: new, sorted beam, with max at 0th index


	urange = g_unaryFeatureVectorIndices[yseq[0]]
	z[0,urange[0]:urange[1]] = xseq[0][0,:]

	#iterate pairwise and other y sequence features
	for i in range(1,len(yseq)):
		#unary features
		urange = g_unaryFeatureVectorIndices[yseq[i]]
		z[0,urange[0]:urange[1]] += xseq[i][0,:]
		
		#pairwise features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
		pairwiseIndex = g_pairwiseFeatureVectorIndices[yseq[i-1]+yseq[i]]
		z[0, pairwiseIndex] += 1.0


"""
def _breadthFirstBeamUpdate(beam,beamWidth,xseq,w,phi):
	d = w.shape[1]
	newBeam = []
	
	#expand all nodes on the beam
	for node in beam:
		baseScore = node[1]
		#if int(baseScore) % 6 == 5: #periodically recalculate full score calculation, to mitigate accumulating error in code optimization
		#	baseScore = w.dot( phi(xseq, node[0], d).T )[0,0]
		cxSeqIndex = len(node[0])
		for c in LABELS:
			candidate = node[0] + [c]
			#adjust unary score component
			urange = g_unaryFeatureVectorIndices[c]
			candidateScore = baseScore + w[0, urange[0]:urange[1]].dot(xseq[cxSeqIndex][0,:])
			#pairwise features; z_yi_i == 1 iff y_i == alpha and y_i-1 == alpha
			pairwiseIndex = g_pairwiseFeatureVectorIndices[candidate[-2]+candidate[-1]]
			candidateScore += w[0, pairwiseIndex]
			#candidateScore = w.dot( phi(xseq,candidate,d).T ) #TODO: if necessary, optimize this score update; full-score computation is repetitive and unnecessary
			newBeam.append( (candidate,candidateScore) )

	#sort beam by node scores
	newBeam.sort(key = lambda tup: tup[1],reverse=True)
	
	#prune beam
	if len(newBeam) > beamWidth and beamWidth > 0:
		newBeam = newBeam[0:beamWidth]

	#print("BEAM: "+str(newBeam))
		
	return newBeam
	
"""
Verifies whether or not any node in the beam is a prefix of the correct output, @y_star
"""
def _beamHasGoodNode(beam,y_star):
	#early update check: if beam contains no y_good nodes, return highest scoring node to perform update
	for node in beam:
		if node[0] == y_star[0:len(node[0])]:
			return True
	return False

def _getFirstCompleteNode(sortedBeam,yLen):	
	for node in sortedBeam:
		if len(node[0]) >= yLen:
			return node
	return None

"""
Just checks if any node on the beam is a complete output
"""
def _beamHasCompleteOutput(beam,yLen):
		return _getFirstCompleteNode(beam,yLen) != None
"""
For two lists, check if the members of y_hat are a prefix of y_star.
This is degenerately true if y_star is a prefix of y_hat, and y_hat is longer.
"""
def _isPrefix(y_hat,y_star):
	for i in range(min(len(y_hat),len(y_star))):
		if y_hat[i] != y_star[i]:
			return False
	return True

"""
This is a utility of max-violation, which finds the first wrong node in a beam instance,
where "wrong" means node is not a prefix of y_star. The beam is assumed sorted,
such that the first non-matching node is the max wrong one.

Returns: first non-matching node, or None if no such node (which should happen only very rarely)
"""
def _getFirstWrongNode(sortedBeam,y_star):
	for node in sortedBeam:
		if not _isPrefix(node[0],y_star):
			return node
	return None
			
"""

Returns the top-scoring example of the lowest scoring beam.
"""
def _maxViolationUpdate(xseq, y_star, w, phi, beamUpdateMethod, beamWidth):
	d = w.shape[1]
	#beam initialization function, I() in the lit. The beam contains (sequence,score) tuples
	beam = _beamSearchInitialization(xseq, w, beamWidth)
	
	#max-score y_hat of the min-scoring beam instance
	#yMaxViolation = beam[0][0]
	#maxViolationScore = beam[0][1]
	yMaxViolation = beam[0][0]
	maxViolationScore = beam[0][1]
	maxViolationDelta = -10000000
	yLen = len(y_star)
	
	#until beam highest scoring node in beam is a complete structured output or terminal node
	t = 1
	while not _beamHasCompleteOutput(beam, yLen):
		beam = beamUpdateMethod(beam, beamWidth, xseq, w, phi)
		#update the max-violation delta param
		wrongNode = _getFirstWrongNode(beam,y_star)
		if wrongNode != None:
			correctPrefixScore = w.dot( phi(xseq, y_star[0:t], d).T )[0,0]
			delta = abs(wrongNode[1] - correctPrefixScore)
			if delta > maxViolationDelta:
				maxViolationDelta = delta
				yMaxViolation = wrongNode[0]
				maxViolationScore = wrongNode[1]
		#else:
			#in this case all nodes in the beam matched the y_star prefix, which should be extremely unlikely, except for very small beams
			#print("WARNING wrongNode == None in maxViolationUpdate()")
		t += 1
			
	return yMaxViolation, phi(xseq, yMaxViolation, d), maxViolationScore

def _standardUpdate(xseq, y_star, w, phi, beamUpdateMethod, beamWidth):
	d = w.shape[1]
	yLen = len(y_star)
	#beam initialization function, I() in the lit. The beam contains (sequence,score) tuples
	beam = _beamSearchInitialization(xseq, w, beamWidth)
	
	#until beam contains a complete structured output/terminal node
	while not _beamHasCompleteOutput(beam, yLen):
		beam = beamUpdateMethod(beam, beamWidth, xseq, w, phi)
	
	#beam is sorted on each iteration, so max is first node
	return beam[0][0], phi(xseq, beam[0][0], d), beam[0][1]

"""
def _maxViolationUpdate(xseq, y_star, w, phi, beamUpdateMethod, beamWidth):
	d = w.shape[1]
	#beam initialization function, I() in the lit. The beam contains (sequence,score) tuples
	beam = _beamSearchInitialization(xseq, w, beamWidth)
	
	#max-score y_hat of the min-scoring beam instance
	#yMaxViolation = beam[0][0]
	#maxViolationScore = beam[0][1]
	yMaxViolation = ""
	maxViolationScore = 10000000
	yLen = len(y_star)
	
	#until beam highest scoring node in beam is a complete structured output or terminal node
	t = 0
	while not _beamHasCompleteOutput(beam, yLen):
		beam = beamUpdateMethod(beam, beamWidth, xseq, w, phi)
		completeMaxNode = _getMaxWrongNodeInBeam(beam, yLen)
		if completeMaxNode != None and completeMaxNode[1] < maxViolationScore:
			#print("hit it")
			maxViolationScore = completeMaxNode[1]
			yMaxViolation = completeMaxNode[0]

	return yMaxViolation, phi(xseq, yMaxViolation, d), maxViolationScore
"""
	
"""
Returns the max-scoring complete node in beam; None if there is no complete node.

Returns: max-complete-Node (or None)
"""
def _getMaxCompleteNodeInBeam(beam,completeLen):
	maxCompleteScore = -100000000.0
	maxCompleteNode = None
	for node in beam:
		if len(node[0]) == completeLen and node[1] > maxCompleteScore:
			maxCompleteScore = node[1]
			maxCompleteNode = node
	
	return maxCompleteNode
"""
Runs early update: progress until beam contains no y_good node (no node that could lead to a solution),
and return the highest-scoring node in the beam at that point as the y_hat by which to make perceptron updates.

@beamUpdateMethod: A function for performing the beam update, in this case either best-first or breadth-first beam update.
"""	
def _earlyUpdate(xseq, y_star, w, phi, beamUpdateMethod, beamWidth):
	d = w.shape[1]
	#beam initialization function, I() in the lit. The beam contains (sequence,score) tuples
	beam = _beamSearchInitialization(xseq,w,beamWidth)
	yLen = len(y_star)
	
	#until beam highest scoring node in beam is a complete structured output or terminal node
	searchError = False #the condition by which we exit search and return the current highest scoring node in the beam when beam contains no good nodes
	while not _beamHasCompleteOutput(beam, yLen) and _beamHasGoodNode(beam,y_star):
		beam = beamUpdateMethod(beam,beamWidth,xseq,w,phi)

	return beam[0][0], phi(xseq, beam[0][0], d), beam[0][1]

"""
The test version of beam-inference. Currently this implements best-first search, but can be easily modified to do BFS instead.
"""
def _beamSearchInference(xseq, w, phi, outputLen, beamUpdateMethod, beamWidth):
	d = w.shape[1]
	#beam initialization function, I() in the lit. The beam contains (sequence,score) tuples
	beam = _beamSearchInitialization(xseq,w,beamWidth)
	
	#until beam highest scoring node in beam is a complete structured output or terminal node
	while not _beamHasCompleteOutput(beam, outputLen):
		beam = beamUpdateMethod(beam,beamWidth,xseq,w,phi)

	return beam[0][0], phi(xseq, beam[0][0], d), beam[0][1]

"""
Returns hamming loss for two strings. See wiki.

Returns: hamming loss, which is not the total number of incorrect characters, but rather the number
of incorrect characters weighed by length.
"""
def _getHammingDist(y_star, y_hat):
	loss = 0.0
	#count character by character losses
	for i in range(min(len(y_star), len(y_hat))):
		if y_star[i] != y_hat[i]:
			loss += 1.0
	#loss, accounting for differences in length
	loss += float(abs(len(y_star) - len(y_hat)))
	
	return loss
	#return loss / float(max(len(y_star), len(y_hat)))  #TODO: Div zero
	
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
Configures all global parameters. Note that xdim is determined beforehand by the dataset.
"""
def _configureGlobalParameters(xdim, phiNum, dataPath):
	global LABELS
	global K
	global XDIM
	global USE_TRIPLES
	global USE_QUADS
	global RAND_LABEL_INDICES
	global RAND_RING_INDEX
	global RAND_INTS_LEN

	XDIM = xdim
	
	#label set is simply hardcoded to nettalk or ocr datasets; this assumes nettalk dataset has been manually modified to map 01 to 'A', 02 to 'B' and so on, for simplicityss
	if "nettalk" in dataPath:
		LABELS = "ABCDE"
	else:
		LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	K = len(LABELS)

	#prepare a set of pseudo-random numbers indices into labels in advance, of prime length, such that randint() doesn't need to be called at high frequency
	RAND_LABEL_INDICES = [random.randint(0,len(LABELS)-1) for i in range(0,1000000)]
	RAND_RING_INDEX = 0
	RAND_INTS_LEN = len(RAND_LABEL_INDICES)
	
	#build the vector index dictionaries
	_buildVectorIndexDicts(XDIM,LABELS)
	
	USE_TRIPLES = False
	USE_QUADS = False
	if phiNum >= 2:
		USE_TRIPLES = True
	if phiNum >= 3:
		USE_QUADS = True
	
	
	
"""
Utility for getting the ocr data as a dataset.

Expects a dataset of tab delimited records, formatted as in the ocr and nettalk data:
	"3	im000000010000000000000000000000010000000000000000000000100000000000000000000000000	01	_"

Note that as a side-effect, this function configures the feature vector dimension component XDIM, which is determined by the dataset.

Returns: A list of training examples. Each training example is a pairwise x and y sequence: ([xsequence], [ysequence])
	xsequence: A list of real-valued numpy vectors of size m; each x vector is binary
	ysequence: A list of symbols/labels
	
	Also returns xdim, the dimension of the data.
	
"""
def _getData(dataPath):
	#build the dataset as a list of x/y tuples
	D = []
	dataFile = open(dataPath,"r")
	records = [line.strip().replace("\t"," ") for line in dataFile.readlines()]
	dataFile.close()
	xseq = []
	yseq = []
	
	#get dimension from the first x example
	xdim = len(records[0].split(" ")[1].replace("im","").strip())
	
	#exit()
	for line in records:
		#print("line: "+line)
		if len(line) > 10:
			binaryString = line.split(" ")[1].replace("im","")
			#x = int(binaryString, 2)
			x = np.zeros((1,xdim), dtype=np.float32)
			for i in range(len(binaryString)):
				if binaryString[i] == "1":
					x[0,i] = 1
			xseq.append(x)
			yseq.append(line.split(" ")[2][0].strip().upper())
		if len(line.strip()) < 10 and not high:
			D.append((xseq,yseq))
			high = True
			xseq = []
			yseq = []
		else:
			high = False

	return D, xdim

"""
_getData returns a dataset as a list of x/y sequence pairs: [  (["0000101","00010110"],["a","t"]), etc. ]
This is inefficient for the structured perceptron construction, as the binary strings must be parsed during learning to map
each x_i to its component in an input vector. 

The solution is obviously to format the dataset in the proper manner before training, instead of doing so many thousands of more times
during training. Recall that each x_i binary string maps to some component of the unary components of the weight vector;


Returns: The new dataset as a list of x+y numpy vectors of the required dimension, under the construction required by the Phi function.

"""
def _preformatData(D,phi,d):
	return [phi(example[0], example[1], d) for example in D]

"""
Given some test data, get a prediction from InferRGS()
"""
def TestPerceptron(w, phiNum, R, beamUpdateMethod, beamWidth, testData):
	losses = []
	phi = _getPhi(phiNum)
	
	#filter the data of too-short examples
	testData = _filterShortData(testData, phiNum)
		
	print("Testing weights over "+str(len(testData))+" examples. This may take a while.")

	i = 0
	for example in testData:
		xseq = example[0]
		y_star = example[1]
		#y_hat, phi_y_hat, score = InferRGS(xseq, w, phi, R)
		y_hat, phi_y_hat, score = _beamSearchInference(xseq, w, phi, len(y_star), beamUpdateMethod, beamWidth)
		#y_hat, phi_y_hat, score = InferRGS_Inefficient(xseq, w, phi, R)
		#print("hat: "+str(y_hat))
		#print("star: "+str(y_star))
		loss = _getHammingDist(y_star, y_hat)
		losses.append(loss)
		i += 1
		if i % 50 == 49:
			print("\rTest datum "+str(i)+" of "+str(len(testData))+"            ",end="")
	
	#get Hamming accuracy
	accuracy = 100.0 * (1.0 - sum(losses) / (float(K) * float(len(testData))))
	print("Accuracy: "+str(accuracy)+"%")
	
	return losses, accuracy
	
def _filterShortData(D, phiNum):
	requiredLength = 2
	if phiNum == 2:
		requiredLength = 3
	if phiNum == 3:
		requiredLength = 4
	return [d for d in D if len(d[1]) >= requiredLength]

"""
@D: A list of training examples in the pairwise form [ (xseq,yseq), (xseq, yseq) ... ].
@R: Number of restarts for RGS
@phiNum: The structured feature function phi(x,y)
@maxIt: max iterations
@eta: learning rate
@errorUpdateMethod: A function performing either bfs or early update
@beamUpdateMethod: A function for performing beam updates, either best-first or breadth-first beam update

Returns: trained weight vector @w, the losses, and the accuracy of the final iteration
"""
def OnlinePerceptronTraining(D, R, phiNum, maxIt, eta, errorUpdateMethod, beamUpdateMethod, beamWidth):
	phi = _getPhi(phiNum)
	xdim = XDIM
	print("xdim: "+str(xdim))
	dim = _getDim(phiNum, xdim)
	print("wdim: "+str(dim))
	#intialize weights of scoring function to 0
	w = np.zeros((1,dim), dtype=np.float32)
	#w[0,:] = 1.0
	
	d = w.shape[1]
	
	#filter out the training examples of length less than the n-gram features
	D = _filterShortData(D,phiNum)
	
	#get a preformatted cache of the data, to help cut down on constant phi re-computations
	preprocessedData = _preformatData(D,phi,d)

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
			#y_hat, phi_y_hat, score = InferBestFirstBeamSearch(xseq, w, phi, beamWidth)
			y_hat, phi_y_hat, score = errorUpdateMethod(xseq, y_star, w, phi, beamUpdateMethod, beamWidth)
			#y_hat, phi_y_hat, score = InferRGS(xseq, w, phi, R)
			#y_hat, phi_y_hat, score = InferRGS_Inefficient(xseq, w, phi, R)
			#print("y_hat: "+str(y_hat)+"   score: "+str(score))
			#get the hamming loss
			#print("ystar: "+str(y_star))
			#print("yhat:  "+str(y_hat))
			loss = _getHammingDist(y_star, y_hat)
			if loss > 0:
				#zStar = preprocessedData[j]  #effectively this is phi(x, y_star, d), but preprocessed beforehand to cut down on computations
				#w = w + eta * (phi(xseq, y_star, d) - phi_y_hat)
				#w = w + eta * (phi(xseq, y_star, d) - phi(xseq, y_hat, d))
				w = w + eta * (preprocessedData[j] - phi_y_hat)
			sumItLoss += loss
		#append the total loss for this iteration, for plotting
		losses.append(sumItLoss/float(K))
		print("iter: "+str(i)+"  it-loss: "+str(losses[-1])+"  it-accuracy: "+str(1.0 - losses[-1]/float(len(D))))
	
	return w, losses, 1.0 - losses[-1]/float(len(D))

def usage():
	print("python StructuredPerceptron --trainPath=[] --testPath=[]")
	print("Optional (prefixed '--'): eta, maxIt, R, beamWidth, showPlot, phi, searchError (mv, early, std), beamUpdate (bfs, best) ")

trainPath = None
testPath = None
showPlot = False
searchMethod = "std"
beamUpdateType = "best"
beamUpdateMethod = _bestFirstBeamUpdate
searchErrorUpdateMethod = _standardUpdate
R = 10
phiNum = 1
maxIt = 50
eta = 0.01
beamWidth = 10
for arg in sys.argv:
	if "--trainPath=" in arg:
		trainPath = arg.split("=")[1]
	if "--testPath=" in arg:
		testPath = arg.split("=")[1]
	if "--eta=" in arg:
		eta = float(arg.split("=")[1])
	if "--phi=" in arg:
		phiNum = int(arg.split("=")[1])
	if "--maxIt=" in arg:
		maxIt = int(arg.split("=")[1])
	if "--R=" in arg:
		R = int(arg.split("=")[1])
	if "--showPlot" in arg:
		showPlot = True
	if "--beamWidth=" in arg:
		beamWidth = int(arg.split("=")[1])
	if "--beamUpdate=" in arg:
		if arg.split("=")[1].lower() == "bfs":
			beamUpdateType = "bfs"
			beamUpdateMethod = _breadthFirstBeamUpdate
		elif arg.split("=")[1].lower() == "best":
			beamUpdateType = "best"
			beamUpdateMethod = _bestFirstBeamUpdate
		else:
			print("ERROR beam update option not found: "+arg.split("=")[1])
			usage()
			exit()
	if "--searchError=" in arg:
		if arg.split("=")[1].lower() == "mv":
			searchMethod = "mv"
			searchErrorUpdateMethod = _maxViolationUpdate
		elif arg.split("=")[1].lower() == "early":
			searchMethod = "early"
			searchErrorUpdateMethod = _earlyUpdate
		elif arg.split("=")[1].lower() == "std" or arg.split("=")[1].lower() == "standard":
			searchMethod = "std"
			searchErrorUpdateMethod = _standardUpdate
		else:
			print("ERROR search update option not found: "+arg.split("=")[1])
			usage()
			exit()

if phiNum == 2:
	USE_TRIPLES = True
elif phiNum == 3:
	USE_QUADS = True

if trainPath == None:
	print("ERROR no trainPath passed")
	usage()
	exit()
if testPath == None:
	print("ERROR no testPath passed")
	usage()
	exit()

trainData, xdim = _getData(trainPath)
testData, _ = _getData(testPath)
_configureGlobalParameters(xdim, phiNum, trainPath)

print("Global params configured")
print("\tLABELS: "+LABELS)
print("\tK: "+str(K))
print("\tXDIM: "+str(XDIM))
print("\tUSE_TRIPLES: "+str(USE_TRIPLES))
print("\tUSE_QUADS: "+str(USE_QUADS))
print("Executing with maxIt="+str(maxIt)+"  R="+str(R)+"  eta="+str(eta)+"  phiNum="+str(phiNum)+"  beam="+str(beamWidth)+"  trainPath="+trainPath+"  testPath="+testPath)
print("searchUpdate="+searchMethod+"    beamUpdate="+beamUpdateType+"    maxIt="+str(maxIt))
print("xdim="+str(xdim))
#print(str(trainData[0]))
#print("lenx: "+str(len(trainData[0][0]))+"  leny: "+str(len(trainData[0][1])))
w, trainingLosses, trainAccuracy = OnlinePerceptronTraining(trainData, R, phiNum, maxIt, eta, searchErrorUpdateMethod, beamUpdateMethod, beamWidth)
#normalize the summed-losses to make them accuracy measures
trainingLosses = [1.0 - loss / float(len(trainData)) for loss in trainingLosses]
SaveLosses(trainingLosses, trainAccuracy, beamWidth, trainPath, searchMethod, beamUpdateType, "Hamming Accuracy per Training Iteration",False,True)

#run testing on only 1/4th of test data, since for this assignment we have way more test data than training data
print("WARNING: Truncating test data from "+str(len(testData))+" data points to "+str(len(testData)/4))
testData = testData[0:int(len(testData)/4)]

testLosses, testAccuracy = TestPerceptron(w, phiNum, R, beamUpdateMethod, beamWidth, testData)
SaveLosses(testLosses, testAccuracy, beamWidth, trainPath, searchMethod, beamUpdateType, "Hamming Accuracy per Test Iteration",False,False)
