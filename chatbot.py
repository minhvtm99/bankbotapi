#################   DO NOT EDIT THESE IMPORTS #################
import math
import random
import numpy
from collections import *
import codecs


class HMM:
	"""
	Simple class to represent a Hidden Markov Model.
	"""
	def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
		self.initial_distribution = initial_distribution
		self.emission_matrix = emission_matrix
		self.transition_matrix = transition_matrix
		
def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns: 
	The file represented as a list of tuples, where each tuple 
	is of the form (word, POS-tag).
	A list of unique words found in the file.
	A list of unique POS tags found in the file.
	"""
	file_representation = []
	unique_words = set()
	unique_tags = set()
	# f = open(str(filename), "r")
	f = codecs.open(str(filename), encoding='utf-8', errors='ignore')

	for line in f:
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		file_representation.append( (word, tag) )
		unique_words.add(word)
		unique_tags.add(tag)
	f.close()
	return file_representation, unique_words, unique_tags

#####################  STUDENT CODE BELOW THIS LINE  #####################

def compute_counts(training_data, order):
	"""
	Given the training data, compute the counts that will be used to train the HMM

	Input:
	- training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
	- order: 2 or 3

	Output: a tuple that contains:
	- the number of tokens in training_data (an integer).
	- a dictionary that contains C(ti,wi) for every unique tag and unique word (keys correspond to tags).
	- a dictionary that contains C(ti) as above.
	- a dictionary that contains C(ti_1,ti) as above.
	- an additional dictionary that contains C(ti_2,ti_1,ti) as above IF order = 3
	"""
	#Raise exception if order is not 2 and 3
	if order != 2 and order != 3:
		raise Exception("Order must be 2 or 3")
	#Number of tokens:
	token = len(training_data)

	#Initialize counts dictionaries
	count_tag_word = defaultdict(lambda : defaultdict(int)) #C(ti,wi)
	count_tag = defaultdict(int) #C(ti)
	count_consec_tags2 = defaultdict(lambda : defaultdict(int)) #C(ti_1,ti)
	count_consec_tags3 = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) #C(ti_2,ti_1,ti) 

	#Loop through every pair (word, tag) in training data and update the counts
	for i in range(len(training_data)):
		pair = training_data[i]
		word, tag = pair[0], pair[1]
		#Update count_tag_word and count_tag
		count_tag_word[tag][word] += 1
		count_tag[tag] += 1
		#Update C(ti_1,ti) and #C(ti_2,ti_1,ti)
		if i > 0:
			prev_tag = training_data[i-1][1]
			count_consec_tags2[prev_tag][tag] += 1
		if i > 1:
			prev_tag1 = training_data[i-1][1]
			prev_tag2 = training_data[i-2][1]
			count_consec_tags3[prev_tag2][prev_tag1][tag] += 1

	#Return tokens, C(ti, wi), C(ti), C(ti_1, ti) if order = 2
	if order == 2:
		return token, count_tag_word, count_tag, count_consec_tags2
	#Return an additional C(ti_2, ti_1, ti) if order = 3
	if order == 3:
		return token, count_tag_word, count_tag, count_consec_tags2, count_consec_tags3

#Test cases
# train1 = [('He', 'pronounce'), ('plays', 'verb'), ('football', 'noun'), ('.', '.'),
#  ('I', 'noun'), ('love', 'verb'), ('cats', 'nounS'), ('.', '.'), 
#  ('I', 'noun'), ('cook', 'verb'), ('amazing', 'adjective'), ('food', 'noun'), ('.', '.')]
# unique_w1 = set(['plays', 'I', 'football', '.', 'food', 'amazing', 'cats', 'cook', 'love', 'He'])
# unique_t1 = set(['noun', 'pronounce', '.', 'adjective', 'verb', 'nounS'])

# tokens1, W_1, C_1, C_2, C_3 = compute_counts(train1, 3)
# print "token 1: ", tokens1, "Expected: 13"
# print "-------------------"
# print "C(t, w): ", W_1["noun"], "Expected noun: {'I': 2, 'food': 1, 'football': 1} "
# print "C(t, w): ", W_1["verb"], "Expected verb: {'cook': 1, 'love': 1, 'plays': 1} "
# print "-------------------"
# print "C(t): ", C_1["noun"], "Expected 4 nouns"
# print "C(t): ", C_1["verb"], "Expected 3 verbs"
# print "-------------------"
# print "C(ti_1, ti): ", C_2["noun"]["verb"], "Expected C(noun, verb) = 2"
# print "C(ti_1, ti): ", C_2["verb"]["adjective"], "C(verb, adjective) = 1"

# print "-------------------"
# print "C(ti_2, ti_1, ti): ", C_3["."]["noun"]["verb"], "Expected C(., noun, verb) = 2"
# print "C(ti_2, ti_1, ti): ", C_3["verb"]["noun"]["adjective"], "Expected C(verb, noun, adjective) = 0"
# print "-------------------"

def compute_initial_distribution(training_data, order):
	"""
	Given the training data, compute initial distribution

	Input:
	- training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
	- order: 2 or 3

	Output:
	- IF order = 2, return a dictionary whose keys are tags and values are probability that the tags appear at the beginning
	of a sentence.
	- IF order = 3, return a dictionary whose keys are tag1 and values are dictionaries: each key of the dictionary
	is another tag2 and values are probabilities that the tag1, tag2 appear at the beginning of a sentence
	of a sentence.

	"""
	#Raise exception if order is not 2 and 3
	if order != 2 and order != 3:
		raise Exception("Order must be 2 or 3")

	#Initialize number of sentences
	num_sentence = 0

	#Initialize 2 distribution dictionaries
	pi1 = defaultdict(int) #order 2
	pi2 = defaultdict(lambda:defaultdict(int)) #order 3

	#Loop through every pair (word, tag) in training data
	# and update the total number of times tags appear at the beginning of a sentence
	for i in range(len(training_data)):
		if i == 0:
			pi1[i] += 1
			pi2[i][i+1] += 1

		current_tag = training_data[i][1]
		if current_tag == ".":
			#Increase the number of sentence by 1 everytime a "." is found
			num_sentence += 1
			#Find current tag and next tag and update pi1, pi2
			if i < len(training_data) - 2:
				next_tag2 = training_data[i+2][1]
				next_tag = training_data[i+1][1]
				pi2[next_tag][next_tag2] += 1

			if i < len(training_data) - 1:
				next_tag = training_data[i+1][1]
				pi1[next_tag] += 1

	#Divide the number of time each tag appears by the total number of sentences to find the probability
	for tag in pi1:

		pi1[tag] /= float(num_sentence)
	for tag1 in pi2:
		for tag2 in pi2[tag1]:
			pi2[tag1][tag2] /= float(num_sentence)

	if order == 2:
		return pi1
	if order == 3:
		return pi2

#Test compute initial distribution

# print "Pi1 for train1: ", compute_initial_distribution(train1, 2), "Expected: {'noun': 0.6666666666666666, 'pronounce': 0.3333333333333333} "
# print "Pi2 for train1: ", compute_initial_distribution(train1, 3), "Expected: noun-verb : 0.666666666666666, pronounce-verb: 0.333333333333333"

def compute_emission_probabilities(unique_words, unique_tags, W, C):
	"""
	Inputs: 
	- unique_words: a set, returned by read_pos_file.
	- unique_tags: a set, returned by read_pos_file.
	- W: a dictionary, computed by compute_counts for C(ti,wi).
	- C: a dictionary, computed by compute_counts for C(ti).

	Output:
	- a dictionary represents the matrix of emission probabilities
	"""
	#Initialize the dictionary representing the matrix
	P = defaultdict(lambda:defaultdict(float))

	#Loop through every pair of unique word and unique tag
	for word in unique_words:
		for tag in unique_tags:
			#Update P(wi|ti)
			if word in W[tag]:
				P[tag][word] = float(W[tag][word])/C[tag]

	return P

#Test emission probability
# emission1 = compute_emission_probabilities(unique_w1, unique_t1, W_1, C_1)
# print "Emission matrix for train1: ", emission1
# print "Noun: ", emission1["noun"], "Expected: 'I': 0.5, 'food': 0.25, 'football': 0.25 "
# print "Adjective: ", emission1["adjective"], "Expected: 'amazing : 1' "

def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
	"""
	Given a training corpus, returns coefficients lambdas for the linear interpolation

	Inputs:
	- unique_tags: a set, returned by read_pos_file.
	- num_tokens: the number of tokens in training_data (an integer).
	- C1: a dictionary that represents C(ti) (returned by compute_counts)
	- C2: a dictionary that contains C(ti_1,ti) (returned by compute_counts).
	- C3: a dictionary that represents C(ti_2,ti_1,ti) (returned by compute_counts).
	- order: 2 or 3 

	Outputs:
	- A list that contains 3 numbers
	"""
	#Raise exception if order is not 2 and 3
	if order != 2 and order != 3:
		raise Exception("Order must be 2 or 3")

	#Initialize lambdas
	lambdas = [0.0, 0.0, 0.0]

	#Loop through all trigrams ti_2,ti_1,ti whose C(ti_2,ti_1,ti) > 0 if order = 3
	#Loop through all bigrams ti_1,ti whose C(ti_1,ti) > 0 if order = 2
	for tag1 in unique_tags:
		for tag2 in unique_tags:

			if order == 2: #Compute lambdas based on equation (8) if order is 2
				if (tag1 in C2) and (tag2 in C2[tag1]):
						alphas = [0, 0, 0]
						#Compute alpha0
						if num_tokens != 0:
							alphas[0] = (C1[tag2] - 1)/float(num_tokens)
						#Compute alpha1
						if C1[tag1] - 1 != 0:
							alphas[1] = (C2[tag1][tag2] - 1)/float(C1[tag1] - 1)
						#Compute i and lambda_i
						i = numpy.argmax(alphas)
						lambdas[i] += C2[tag1][tag2]

			if order == 3:	#Compute lambdas based on equation (9) if order is 3
				for tag3 in unique_tags:
					if (tag1 in C3) and (tag2 in C3[tag1]) and (tag3 in C3[tag1][tag2]): 
						alphas = [0, 0, 0]
						#Compute alpha0
						if num_tokens != 0:
							alphas[0] = (C1[tag3] - 1)/float(num_tokens)
						#Compute alpha1
						if C1[tag2] - 1 != 0:
							alphas[1] = (C2[tag2][tag3] - 1)/float(C1[tag2] - 1)
						#Compute alpha2
						if C2[tag1][tag2] - 1 != 0:
							alphas[2] = (C3[tag1][tag2][tag3] - 1)/float(C2[tag1][tag2] - 1)
						#Compute i and lambda_i
						i = numpy.argmax(alphas)
						lambdas[i] += C3[tag1][tag2][tag3]

	#Calculate lambda1, lambda2, lambda3	
	lambdas_sum = sum(lambdas)
	for i in range(len(lambdas)):
		lambdas[i] /= lambdas_sum
	return lambdas

#Test compute lambdas
# print "Lambdas for order 2: ", compute_lambdas(unique_t1, tokens1, C_1, C_2, C_3, 2), "Expected: [0.5, 0.5, 0] and lambdas' sum = 1"
# print "Lambdas for order 3: ", compute_lambdas(unique_t1, tokens1, C_1, C_2, C_3, 3), "Expected: [0.45, 0.36, 0.18] and lambdas' sum = 1"
# print "Lambdas for order 4, expected error raised: ", compute_lambdas(unique_t1, tokens1, C_1, C_2, C_3, 4)


def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	"""
	Build a hidden markov model on the training data

	Inputs:
	- training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
	- unique_words: a set, returned by read_pos_file.
	- unique_tags: a set, returned by read_pos_file.
	- order: 2 or 3
	- use_smoothing: a Boolean value

	Output:
	An instance of the class HMM

	"""
	#Compute intial distribution
	initial_dist = compute_initial_distribution(training_data, order)

	#Compute all the counts
	if order == 3:
		tokens, W, C1, C2, C3 = compute_counts(training_data, order)
	if order == 2:
		tokens, W, C1, C2 = compute_counts(training_data, order)
		C3 = None 

	#Compute emission probability
	emission_prob = compute_emission_probabilities(unique_words, unique_tags, W, C1)

	#Compute lambdas
	if use_smoothing:
		lambdas = compute_lambdas(unique_tags, tokens, C1, C2, C3, order)
	else:
		if order == 2:
			lambdas = [0.0, 1.0, 0.0]
		if order == 3:
			lambdas = [0.0, 0.0, 1.0]

	#Compute transition matrix
	bigram_transition = defaultdict(lambda : defaultdict(int)) #Initialize a 2D dictionary for bigram model
	trigram_transition = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) #Initialize a 3D dictionary for trigram model

	#Compute transition probabilities based on equation 8 if order = 2
	if order == 2:
		for tag1 in unique_tags:
			for tag2 in unique_tags:
				if C2[tag1][tag2] > 0:
					bigram_transition[tag1][tag2] += (lambdas[1]*float(C2[tag1][tag2])/C1[tag1]) 
				if C1[tag2] > 0:
					bigram_transition[tag1][tag2] += (lambdas[0]*float(C1[tag2])/tokens)

	#Compute transition probabilities based on equation 9 if order = 3
	if order == 3:
		for tag1 in unique_tags: 
			for tag2 in unique_tags:
				for tag3 in unique_tags:
					if C1[tag3] > 0:
						trigram_transition[tag1][tag2][tag3] += (lambdas[0]*float(C1[tag3])/tokens)
					if C2[tag2][tag3] > 0:
						trigram_transition[tag1][tag2][tag3] += (lambdas[1]*float(C2[tag2][tag3])/C1[tag2])
					if C3[tag1][tag2][tag3] > 0:
						trigram_transition[tag1][tag2][tag3] += (lambdas[2]*float(C3[tag1][tag2][tag3])/C2[tag1][tag2]) 

	#Build HMM
	if order == 2:
		hidden_model = HMM(order, initial_dist, emission_prob, bigram_transition)
	if order == 3:
		hidden_model = HMM(order, initial_dist, emission_prob, trigram_transition)
	return hidden_model


def bigram_viterbi(hmm, sentence):
	"""
	Given a hidden markov model and a sentence, return an optimal tag sequence

	Inputs:
	- hmm: an instance of the class HMM.
	- sentence: a list of strings.

	Output:
	- A list of tuple (word, tag)
	"""
	#get the properties of the hmm
	pi = hmm.initial_distribution
	E = hmm.emission_matrix
	A = hmm.transition_matrix
	X = sentence
	#Initialize matrix v and matrix bp, both as 2D dictionary
	v = defaultdict(lambda : defaultdict(int)) 
	bp = defaultdict(lambda: defaultdict(str))
	#Compute v[l,0] for all states l
	for l in A.keys():
		v[l][0] = numpy.log(pi[l]) + numpy.log(E[l][X[0]])
	#Compute v and bp
	for i in range(1, len(X)):
		for l in A.keys():
			v[l][i] = numpy.log(E[l][X[i]]) + max([(v[l_prime][i-1] + numpy.log(A[l_prime][l])) for l_prime in A.keys()])
			bp[l][i] =  max(A, key=lambda l_prime: v[l_prime][i-1] + numpy.log(A[l_prime][l]))

	#Initialize tag sequence Z
	Z = [None] * len(X)
	#Compute Z[L-1]
	Z[-1] = max(A, key=lambda l_prime: v[l_prime][len(X)-1])
	#Compute the rest of Z
	for i in range(len(X)-2, -1, -1):
		Z[i] = bp[Z[i+1]][i+1]
	
	#Zip X and Z
	result = []
	for i in range(len(X)):
		result.append((X[i],Z[i]))

	return result

def update_hmm(hmm, test_data, epsilon):
	"""
	Given a hidden markov model and a test data, update the hmm with
	words that did not appear in training data.

	Inputs:
	- hmm: an instance of the class HMM
	- test_data: a set of strings
	- epsilon: a small probability value to assign to words not appear in training data 

	Modify the hmm
	"""
	#Get the set of words in hmm
	E = hmm.emission_matrix
	word_set = set()
	for state in E:
		for word in E[state]:
			word_set.add(word)

	#Initialize a set of unseen words 
	new_words = set([])
	#Loop over words in test data to find new words 
	for word in test_data:
		if word not in word_set:
			new_words.add(word)
			#Loop over every state in the emission matrix and add epsilon to the probability of the new word appears
			for state in E:
				E[state][word] = epsilon

	#Add epsilon to the emission probability of every existing word in each state
	for state in E:
		for word in E[state]:
			if word not in new_words:
				E[state][word] += epsilon

	#Re-normalize the data
	for state in E:
		new_sum = float(sum(E[state].values()))
		for word in E[state]:
			E[state][word] /= new_sum

def trigram_viterbi(hmm, sentence):
	"""
	Given a hidden markov model and a sentence, return an optimal tag sequence

	Inputs:
	- hmm: an instance of the class HMM.
	- sentence: a list of strings.

	Output:
	- A list of tuple (word, tag)
	"""
	#get the properties of the hmm
	pi = hmm.initial_distribution
	E = hmm.emission_matrix
	A = hmm.transition_matrix
	X = sentence

	#Initialize matrix v and matrix bp, both as 3D dictionary
	v = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) 
	bp = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) 
	#Compute v[l,l',1] for all states l,l'
	for l in A.keys():
		for l_prime in A.keys():
			v[l][l_prime][1] = numpy.log(pi[l][l_prime]) + numpy.log(E[l][X[0]]) + numpy.log(E[l_prime][X[1]])

	#Compute v and bp
	for i in range(2, len(X)):
		for m in A.keys():
			for n in A.keys():
				v[m][n][i] = numpy.log(E[n][X[i]]) + max([(v[l][m][i-1] + numpy.log(A[l][m][n])) for l in A.keys()])
				bp[m][n][i] =  max(A, key=lambda l: v[l][m][i-1] + numpy.log(A[l][m][n]))

	# Initialize tag sequence Z
	Z = [None] * len(X)
	cur_max = -float("inf")
	#Compute Z[L-2], Z[L-1]
	for m in v:
		for n in v[m]:
			if v[m][n][len(X)-1] > cur_max:
				cur_max = v[m][n][len(X)-1]
				Z[-1] = n
				Z[-2] = m
	#Compute the rest of Z
	for i in range(len(X)-3, -1, -1):
		Z[i] = bp[Z[i+1]][Z[i+2]][i+2]
	
	#Zip X and Z:
	result = []
	for i in range(len(X)):
		result.append((X[i],Z[i]))

	return result


def tag_experiment(msg, order):
	"""
	Train the model on filename and test on msg
	"""
	training_data = read_pos_file('training.txt')
	words_tagged = training_data[0]
	# print (words_tagged)
	words = training_data[1]
	tags = training_data[2]

	hmm = build_hmm(words_tagged, tags, words, order, True)
	update_hmm(hmm, msg.split(), 0.0001)

	if order == 2:
		result = bigram_viterbi(hmm, msg.split())
	else:
		result = trigram_viterbi(hmm, msg.split())

	return result


print(tag_experiment('tim atm gan day', 2))

# import sys
# print(sys.version)