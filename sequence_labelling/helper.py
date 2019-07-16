'''
Created on Apr 27, 2018

@author: 703188429
'''
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import numpy as np
from nltk.data import load
from sklearn import metrics

class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   DEFAULT = '\033[99m'
   WHITE = '\033[97m'
   END = '\033[0m'

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
	 
	# looping till length l
	for i in range(0, len(l), n): 
		yield l[i:i + n]
		
def partial_ratio(s1, s2):
	""""Return the ratio of the most similar substring
	as a number between 0 and 100."""
	if not s1.strip():
		return 0
	if not s2.strip():
		return 0

	if len(s1) <= len(s2):
		shorter = s1
		longer = s2
	else:
		shorter = s2
		longer = s1

	m = SequenceMatcher(None, shorter, longer, autojunk=False)
	blocks = m.get_matching_blocks()
	scores = []
	for (short_start, long_start, _) in blocks:
		long_end = long_start + len(shorter)
		long_substr = longer[long_start:long_end]

		m2 = SequenceMatcher(None, shorter, long_substr, autojunk=False)
		r = m2.ratio()
		if r > .995:
			return 100
		else:
			scores.append(r)

	return max(scores) * 100.0

def print_metrics(labels, predictions):
	labels[labels==2] = 1
	predictions[predictions==2] = 1
	return ("F1:", round(metrics.f1_score(labels, predictions), 2), "P:", round(metrics.precision_score(labels, predictions), 2), "R:", round(metrics.recall_score(labels, predictions), 2))


def consecutive(data, stepsize=1):
	return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def getPhrasesFromIndex(tokens, indexes):
	phrases = []
	for index_ in consecutive(indexes):
		phrases.append(" ".join(tokens[index_]))
	return phrases

def getTokensFromIndex(tokens, indexes):
	phrases = []
	for index_ in indexes:
		phrases.append(tokens[index_])

	#for index_ in consecutive(indexes):
	#	phrases.append(" ".join(tokens[index_]))
	return phrases

def getPosTagDict():
	tagdict = load('help/tagsets/upenn_tagset.pickle')
	d = {}
	numtags = len(tagdict.keys())
	for cntr, tag in enumerate(tagdict.keys()):
		vectors = np.zeros(numtags)
		vectors[cntr] = 1
		#print(tag, vectors)
		d[tag] = vectors
	return d

def getWordFeatures(word):
	islower = word.islower()
	isupper = word.isupper()
	isalpha = word.isalpha()
	isalnum = word.isalnum()
	isdigit = word.isdigit()
	vector = [islower, isupper, isalpha, isalnum, isdigit]
	vector = list(map(lambda x: 1 if x else 0, vector))
	return vector 

if __name__ == "__main__":
	tokens = np.array(['Thus', ',', 'clinical', 'recognition' ,'of', 'sleep', 'disordered' ,'breathing',
	 'should', 'be', 'taken', 'into', 'account', 'when', 'rheumatoid', 'arthritis',
	 'patients', 'are', 'to', 'be', 'treated', 'with', 'infliximab', '.'])
	indexes = np.array([3, 5, 6, 7, 9, 10, 12, 14, 15])
	#print(getPhrasesFromIndex(tokens, indexes))
	#print(getPosTagDict())
	word = "ww131"
	print(getWordFeatures(word))
