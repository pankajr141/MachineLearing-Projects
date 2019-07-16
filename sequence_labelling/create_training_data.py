from helper import partial_ratio
import pandas as pd
import numpy as np
import argparse
import re
import os
import sys
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from multiprocessing.pool import Pool
from datetime import datetime
from helper import Colour

flatten = lambda l: list(filter(lambda x: x.strip(), [item for sublist in l for item in sublist]))
flatmap = lambda l: np.array([item for sublist in l for item in sublist])

debug = True

def pprint(*string):
	global debug
	if debug:
		print(string)

def process_row(row):
	try:
		labels_found = {}
		labels_notfound = {}
		
		index, row = row
		id_ = row[col_id]
		
		text = str(row[col_x])
		cols = list(map(lambda x: x.strip(), col_y.split(',')))

		text = text.replace('\n', '').replace('\r', '').strip()
		
		tokens_nltk = list(map(lambda words: word_tokenize(words), sent_tokenize(text)))
		tokens_nltk = [pos_tag(sent) for sent in tokens_nltk]
		tokens_nltk = flatmap(tokens_nltk)

		tokens = []
		postags = []
		
		for token, postag in tokens_nltk:
			words = token.replace("/", "-").split("-")
			tokens.extend(words)
			postags.extend([postag] * len(words))

		tokens = list(map(lambda x: re.sub(r'[^a-zA-Z0-9,.]', '', x), tokens))
		#tokens = list(map(lambda x: re.sub(r'\W+', '', x), tokens))

		# Filter postag and token for which token is not present
		tokens_zipped = [(token, postag) for token, postag in zip(tokens, postags) if token]
		tokens, postags = zip(*tokens_zipped)
		labels = [tags[0]] * len(tokens)

		#print("\n\n\n")
		for cntr, col in enumerate(cols):
			answers = str(row[col])
			answers = answers.split('|')
			answers = list(filter(lambda x: str(x).strip() and not str(x) == 'nan', answers))
			for answer in answers:
				#pprint('answer:', col, answer)
				ltoken = word_tokenize(answer)
				ltoken = flatten(list(map(lambda x: x.lower().replace("/", "-").split("-"), ltoken)))
				ltoken = list(map(lambda x: re.sub(r'[^a-zA-Z0-9,.]', '', x), ltoken))
				ltoken = list(filter(lambda x: x, ltoken))
				ltoken_string = " ".join(ltoken)
				cont = False
				''' Direct comparision for matching terms'''
				for i in range(len(tokens)):
					if (i + len(ltoken)) > len(tokens):
						break
					if [a.lower() for a in tokens[i: i + len(ltoken)]] != ltoken:
						continue

					labels[i: i + len(ltoken)] = [tags[cntr+1]] * len(ltoken)
					cont = True
				''' Fuzzy comparision for matching terms'''
				if not cont:
					'''Taking window of size ltoken from tokens'''
					for i in range(len(tokens)):
						if (i + len(ltoken)) > len(tokens):
							break
						token_to_match = " ".join(tokens[i: i + len(ltoken)])
						if not partial_ratio(ltoken_string.lower(), token_to_match.lower()) > 85:
							continue
						labels[i: i + len(ltoken)] = [tags[cntr+1]] * len(ltoken)
						cont = True
				if not cont:
					labels_notfound[col] = labels_notfound.get(col, 0) + 1
					print(">>>>>>>>>>> Not found(%s)" % id_, answer)
				else:
					labels_found[col] = labels_found.get(col, 0) + 1

		sentence = " ".join(tokens)
		sentence_postag = " ".join(postags)

		#pprint("sentence:",  sentence)
		token_colored = []
		for i, token in enumerate(tokens):
			if labels[i] == tags[1]:
				token_colored.append(Colour.RED + token + Colour.END)
			elif labels[i] == tags[2]: 
				token_colored.append(Colour.GREEN + token + Colour.END)
			else:
				token_colored.append(Colour.END + token + Colour.END)
		sentence_p = " ".join(token_colored)
		#print("sentence:", sentence_p)
		labels = " ".join(labels)
		return id_, sentence, sentence_postag, labels, labels_found, labels_notfound
	except Exception as err:
		print(id_, err)
		return id_, "Blank", "Blank", tags[0], labels_found, labels_notfound
		
def create_dataset(inputfile, output_dir, outputfile_id, outputfile_x, outputfile_y, num_processes=16):
	df = pd.read_csv(inputfile, encoding = 'utf8')
	#df = df.head(100)
	p = Pool(num_processes)
	outputs = p.map(process_row, df.iterrows())
	p.terminate()

	outputfile_postag = outputfile_x + "_postag"
	writer_id = open(outputfile_id, 'w')
	writer_x = open(outputfile_x, 'w')
	writer_x_lower = open(outputfile_x + "_lower", 'w')
	writer_postag = open(outputfile_postag, 'w')
	writer_y = open(outputfile_y, 'w')
	
	for output in outputs:
		id_, sentence, sentence_postag, labels, labels_found, labels_notfound = output
		writer_id.write(str(id_) + "\n")
		writer_x.write(str(sentence) + "\n")
		writer_x_lower.write(str(sentence).lower() + "\n")
		writer_postag.write(str(sentence_postag) + "\n")
		writer_y.write(str(labels) +"\n")

	df_tp = pd.DataFrame(list(map(lambda x: x[4], outputs)))
	df_tnp = pd.DataFrame(list(map(lambda x: x[5], outputs)))
	print("Term Present: ", df_tp.sum().tolist())
	print("Term Not Present: ", df_tnp.sum().tolist())

	writer_id.close()
	writer_x.close()
	writer_x_lower.close()
	writer_postag.close()
	writer_y.close()
	
if __name__ == "__main__":
	try:
		st = datetime.now()
		parser = argparse.ArgumentParser(description='Siamese')
		parser.add_argument('--inputfile', required=True)
		parser.add_argument('--output-dir', required=True)
		parser.add_argument('--output-prefix', required=True)
		parser.add_argument('--col-id', required=True)
		parser.add_argument('--col-x', required=True)
		parser.add_argument('--col-y', required=True)
		parser.add_argument('--tag-file', required=True)
		parser.add_argument('--jobs', required=False)

		args = parser.parse_args()
		inputfile = args.inputfile
		output_dir = args.output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		output_prefix = args.output_prefix
		global col_id, col_x, col_y, tags
		col_id = args.col_id
		col_x = args.col_x
		col_y = args.col_y
		num_processes = 16 if not args.jobs else args.jobs
		num_processes = int(num_processes)
		tag_file = args.tag_file
		tags = list(map(lambda x: x.strip(), open(tag_file).readlines()))
		outputfile_id = os.path.join(output_dir, output_prefix + "_id")
		outputfile_x = os.path.join(output_dir, output_prefix + "_x")
		outputfile_y = os.path.join(output_dir, output_prefix + "_y")
		print(Colour.RED + "================START================" + Colour.END)
		create_dataset(inputfile, output_dir, outputfile_id, outputfile_x, outputfile_y, num_processes)
		print("Time taken:", datetime.now() - st)
		print(Colour.RED + "================END================" + Colour.END)
	except Exception as err:
		print(err)
		import traceback
		traceback.print_exc()
