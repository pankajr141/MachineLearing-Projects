import os
import argparse
from helper import Colour

def print_colored_result(datafileline, labelfileline, resultfileline, tags):
	tokens = datafileline.strip().split()
	results = resultfileline.strip().split()
	labels = []
	if labelfileline:
		labels = labelfileline.strip().split()
	print('Tokens:',len(tokens), 'Predictions:', len(results), 'Actual:', len(labels))
	token_colored = []
	for i, token in enumerate(tokens):
		if results[i] == tags[1]:
			color_incorrect_prediction =  Colour.UNDERLINE if labels and labels[i] != results[i] else ''
			token_colored.append(color_incorrect_prediction + Colour.PURPLE + token + Colour.END)
		elif results[i] == tags[2]:
			color_incorrect_prediction =  Colour.UNDERLINE if labels and labels[i] != results[i] else ''
			token_colored.append(color_incorrect_prediction + Colour.GREEN + token + Colour.END)
		elif labels and (labels[i] == tags[1] or labels[i] == tags[2]):
			token_colored.append(Colour.RED + token + Colour.END)
		else:
			token_colored.append(Colour.WHITE + token + Colour.END)
		
	sentence_p = " ".join(token_colored)
	print(sentence_p)


def print_results(datafile, labelfile, resultfile, tagfile, index):
	datafile_lines = open(datafile).readlines()
	resultfile_lines = open(resultfile).readlines()
	if labelfile:
		labelfile_lines = open(labelfile).readlines()
	else:
		labelfile_lines = [None] * len(datafile_lines)
	tagfile_lines = open(tagfile).readlines()
	tags = list(map(lambda x: x.strip(), tagfile_lines))
	if index:
		index = int(index)
		print_colored_result(datafile_lines[index], labelfile_lines[index], resultfile_lines[index], tags)
	else:
		for i in range(len(datafile_lines)):
			print_colored_result(datafile_lines[i], labelfile_lines[i], resultfile_lines[i], tags)

if __name__ == "__main__":
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument('--datafile', help='filepath', required=True)
		parser.add_argument('--labelfile', help='filepath', required=False)
		parser.add_argument('--resultfile', help='filepath', required=True)
		parser.add_argument('--tagfile', help='tagfile', required=True)
		parser.add_argument('--index', help='index', required=False)

		args = parser.parse_args()
		datafile = args.datafile
		labelfile = args.labelfile
		resultfile = args.resultfile
		tagfile = args.tagfile
		index = args.index

		print_results(datafile, labelfile, resultfile, tagfile, index)
	except Exception as err:
		print(err)
