import os
import sys
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
from collections import Counter
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import argparse
import model
from sklearn import metrics
import time
import traceback
import helper
from collections import OrderedDict

'''
Function load w2v model and create a mapping object expected by tensorflow module
Returns: 
a. embeddings - Embeddings which contains array of vectors, with each index representing 1 word
c. table - Look up table to be used
'''

#dimentions = None
#tags_file = None
#w2v_file = None

#w2v_file = "/efsdata/data/embeddings/euclidean_vectors.bin"
#w2v_file = "/efsdata/data/embeddings/PubMed-w2v.bin"

def _load_initial_models(w2v_file, tags_file):
	model = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
	global dimentions
	dimentions = model.vector_size
	embeddings = np.zeros((len(model.wv.vocab), dimentions))
	#global
	lookup_index = OrderedDict()
	for i in range(len(model.wv.vocab)):
		# Remember to lower case it ? 
		embedding_vector = model.wv[model.wv.index2word[i]]
		''' Lower and uppercases added in index '''
		lookup_index[model.wv.index2word[i]] = i
		# Trick check if lower present if not add same value at end
		#lookup_index[model.wv.index2word[i].lower()] = i
		#lookup_index[model.wv.index2word[i].upper()] = i
		if embedding_vector is not None:
			embeddings[i] = embedding_vector
	#print(lookup_index)
	#print(lookup_index.keys())
	#exit()
	""" Dictionary which gives index of word in embeddings,
	Eg. lookup_index['word'] will return integer value, and at that index location in embeddings we can find its vector """
	lookup_index_tf = tf.constant(list(lookup_index.keys()))
	table = tf.contrib.lookup.index_table_from_tensor(mapping=lookup_index_tf, default_value=lookup_index['Unc5c(-/-)'])
	tags = tf.contrib.lookup.index_table_from_file(tags_file)
	return embeddings, table, tags

'''
w2v_file = "/efsdata/data/embeddings/wikipedia-pubmed-and-PMC-w2v.bin"
tags_file = 'tags.txt'
embeddings, table, tags = _load_initial_models(w2v_file, tags_file)
embeddings_tf = tf.placeholder(tf.float32, embeddings.shape, name='embeddings_tf') 
'''

def parse_function(row):
	#row  = tf.contrib.lookup.index_table_from_tensor()
	#row = tf.contrib.lookup.string_to_index(row, mapping=lookup_index_tf, default_value=lookup_index['defunct'])
	#row = tf.constant([1,2,3,4]) 
	row = table.lookup(row)
	#output = tf.nn.embedding_lookup(embeddings, row)
	output = tf.nn.embedding_lookup(embeddings_tf, row)
	""" See how can we add below features that were their in the original 
	pos_tag(sent)
	getWordFeatures(token)
	""" 
	return output

def getDatasetIterator(sentence_file, labels_file=None, mode="TRAIN", batchsize=2):

	sentences = tf.data.TextLineDataset(sentence_file)
	#sentences = sentences.map(lambda string: tf.string_lower(string))
	sentences = sentences.map(lambda string: tf.string_split([string]).values)
	sentences = sentences.map(parse_function, num_parallel_calls=4)

	if mode in ['TRAIN', 'EVAL']:
		labels = tf.data.TextLineDataset(labels_file)
		labels = labels.map(lambda string: tf.string_split([string]).values)
		# Converting labels into one hot encoding based on lookupfile
		labels = labels.map(lambda tokens: tags.lookup(tokens))
		dataset = tf.data.Dataset.zip((sentences, labels))
		if mode == "TRAIN":
			dataset = dataset.shuffle(buffer_size=batchsize*2)
		dataset = dataset.padded_batch(batchsize, padded_shapes=([None, dimentions], [None]))

	elif mode in ['PREDICT']:
		#print(sentences)
		#dataset = tf.data.Dataset.from_tensor_slices(sentences)
		dataset = sentences
		dataset = dataset.padded_batch(batchsize, padded_shapes=([None, dimentions]))

	'''
	padding_values = ("",   # sentence padded on the right with id_pad_word
					  "O")	# labels padded on the right with id_pad_tag
	#dataset = dataset.padded_batch(batchsize, padded_shapes=padded_shapes, padding_values=padding_values)
	'''

	dataset = dataset.prefetch(1)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	init_op = iterator.initializer
	return init_op, next_element

def execute_model(cls, mode, args):
	with cls.sess.as_default():

		st = time.time()
		x = args['x']
		y = args['y']
		m = x.shape[0]
		
		if mode == "TRAIN":
			writer = args['writer']
			'''Run Training step'''
			output = cls.sess.run([cls.train_op, cls.summary, cls.loss_all, cls.global_step,
									cls.precision, cls.recall, cls.f1,
									cls.precision_c, cls.recall_c, cls.f1_c, cls.predictions,
									cls.reset_vars_initializer], 
								feed_dict = {
											cls.x : x,
											cls.y : y,
											cls.fa_init_state: np.zeros([m, cls.n_y]),
											cls.keep_prob: 0.85,
											cls.learning_rate: args['learning_rate']
								})
			summary = output[1]
			loss_all, global_step = output[2:4]
			precision, recall, f1  = output[4:7]
			precision_c, recall_c, f1_c = output[7:10]
			precision	= round(precision, 2)
			recall 		= round(recall, 2)
			f1 			= round(f1, 2)	
			precision_c = list(map(lambda x: round(x, 2), precision_c))
			recall_c 	= list(map(lambda x: round(x, 2), recall_c))
			f1_c 		= list(map(lambda x: round(x, 2), f1_c))

			writer.add_summary(summary, global_step)
			
			'''
			labels = y.reshape(y.shape[0] * y.shape[1])
			predictions = output[10]
			predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1])
			print(helper.print_metrics(labels, predictions))
			'''
			print("Train, Step:", global_step, "LR:", args['learning_rate'], "Loss:", np.mean(loss_all), "Y_SIZE:", np.size(y), "Y:", str(Counter(y.reshape(-1))), 'p:', precision, str(precision_c), 'r:', recall, str(recall_c), 'f1:', f1, str(f1_c), x.shape, "Time:", time.time()-st)

			return global_step, loss_all

		elif mode == "EVAL":
			'''Run Eval step'''
			output = cls.sess.run([cls.summary, cls.loss_all,
									cls.precision, cls.recall, cls.f1,
									cls.precision_c, cls.recall_c, cls.f1_c, cls.predictions,
									cls.reset_vars_initializer],
								feed_dict = {
											cls.x : x,
											cls.y : y,
											cls.fa_init_state: np.zeros([m, cls.n_y]),
											cls.keep_prob: 1.0,
								})
			loss_all = output[1]
			precision, recall, f1  = output[2:5]
			precision_c, recall_c, f1_c = output[5:8]
			precision	= round(precision, 2)
			recall 		= round(recall, 2)
			f1 			= round(f1, 2)	
			precision_c = list(map(lambda x: round(x, 2), precision_c))
			recall_c 	= list(map(lambda x: round(x, 2), recall_c))
			f1_c 		= list(map(lambda x: round(x, 2), f1_c))


			if 'writer' in args.keys():
				writer = args['writer']
				summary = output[0]
				global_step = args['global_step']
				writer.add_summary(summary, global_step)
				print("Eval, Step:", global_step, "LR:", args['learning_rate'], "Loss:", np.mean(loss_all), "Y_SIZE:", np.size(y), "Y:", str(Counter(y.reshape(-1))),
					'p:', precision, str(precision_c), 'r:', recall, str(recall_c), 'f1:', f1, str(f1_c), x.shape, "Time:", time.time()-st)

			return output

		elif mode == "PREDICT":
			'''Run Predict step'''
			output = cls.sess.run(cls.predictions_normalized,
								feed_dict = {
											cls.x : x,
											cls.fa_init_state: np.zeros([m, cls.n_y]),
											cls.keep_prob: 1.0,
								})
		return output
'''
Function is used to perform training on training data, it also require Eval files in order to save better model to disk.
'''
def training(datafile, modelfile, batchsize=4):
	args = {}
	args['learning_rate'] = 0.001
	try:

		n_x = dimentions
		n_a = 300
		n_y = len(open(tags_file).readlines())
		num_layers = 1
		num_memoryunit = 1
		cls = model.SequenceLabelling(modelfile, n_a, n_x, n_y, num_layers, num_memoryunit)

		LOSS = 100000000000
		F1 = 0
		trainfile, evalfile = datafile.split(",")		
		sentence_file_train, label_file_train = trainfile.split('|')
		sentence_file_eval, label_file_eval = evalfile.split('|')

		init_op_train, next_element_train = getDatasetIterator(sentence_file_train, label_file_train, mode="TRAIN", batchsize=32)
		init_op_eval, next_element_eval = getDatasetIterator(sentence_file_eval, label_file_eval, mode="EVAL", batchsize=128)

		sess = tf.Session()
		sess.run(tf.tables_initializer())

		writer_train = tf.summary.FileWriter(cls.writer_dir_train, cls.sess.graph)
		writer_eval = tf.summary.FileWriter(cls.writer_dir_eval, cls.sess.graph)
		totalEpoch = 100

		for epoch in range(totalEpoch):
			print("Initializing the dataset iterator epoch:", epoch)
			sess.run(init_op_train, feed_dict={embeddings_tf: embeddings})
			while True:
				try:
					#batch_train = sess.run(next_element_train, feed_dict={embeddings_tf: embeddings})
					batch_train = sess.run(next_element_train)
				except tf.errors.OutOfRangeError:
					break
				args['x'] = batch_train[0]
				args['y'] = batch_train[1]
				args['writer'] = writer_train
				#print(args['x'])
				global_step, loss_ = execute_model(cls, mode="TRAIN", args=args)
				if global_step % 50 == 0:
					print("saving model", cls.modelfile)
					with cls.graph.as_default():
						cls.saver.save(cls.sess, cls.modelfile)
				if global_step > 2000 and global_step % 200 == 0:
					print("mode => EVAL")
					args['global_step'] = global_step
					sess.run(init_op_eval, feed_dict={embeddings_tf: embeddings})
					loss = []
					f1 = []
					args['writer'] = writer_eval

					while True:
						try:
							batch_eval = sess.run(next_element_eval)
						except tf.errors.OutOfRangeError:
							break
						args['x'] = batch_eval[0]
						args['y'] = batch_eval[1]

						output = execute_model(cls, mode="EVAL", args=args) 
						loss.extend(output[1])
						f1.append(output[4])
					loss = np.mean(loss)
					f1 = np.mean(f1)
					if LOSS > loss or F1 < f1:
						loss = round(loss, 2)
						LOSS = loss if LOSS > loss else LOSS
						f1 = round(f1, 2)
						F1 = f1 if F1 < f1 else F1
						dir_ = "%s_step=%s_loss=%s_f1=%s" % (cls.modeldir, str(global_step), str(loss), str(f1))
						os.system('cp -Rv %s %s' % (cls.modeldir, dir_))
						#with cls.graph.as_default():
						#	cls.saver.save(cls.sess, os.path.join(dir_, "model.ckpt"))
		writer_train.close()
		writer_eval.close()
	except Exception as err:
		traceback.print_exc()
		print("APPLICATION", err)

def evaluation(datafile, modelfile, batchsize=4):
	try:
		print("mode => EVAL")
		sentence_file_eval, label_file_eval = datafile.split('|')
		init_op_eval, next_element_eval = getDatasetIterator(sentence_file_eval, label_file_eval, mode="EVAL", batchsize=128)
		n_x = dimentions
		n_a = 300
		n_y = len(open(tags_file).readlines())
		num_layers = 2
		num_memoryunit = 1
		cls = model.SequenceLabelling(modelfile, n_a, n_x, n_y, num_layers, num_memoryunit)

		sess = tf.Session()
		sess.run(tf.tables_initializer())

		sess.run(init_op_eval, feed_dict={embeddings_tf: embeddings})
		args = {}
		loss = []
		cntr = 0
		labels = []
		preds = []
		while True: 
			try:
				batch_eval = sess.run(next_element_eval)
			except tf.errors.OutOfRangeError:
				break
			cntr += 1
			if cntr % 50 == 0:
				print("cntr =>", cntr, "Loss:", np.mean(loss), "F1:", metrics.f1_score(labels, preds), 
					  "P:", metrics.precision_score(labels, preds), "R:", metrics.recall_score(labels, preds))
			args['x'] = batch_eval[0]
			args['y'] = batch_eval[1]

			output = execute_model(cls, mode="EVAL", args=args)
			loss_all, precision, recall, f1, precision_c, recall_c, f1_c, predictions = output[1:9]
			#'''
			label = batch_eval[1].reshape(batch_eval[1].shape[0] * batch_eval[1].shape[1])
			predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1])
			
			label[label==2] = 1
			predictions[predictions==2] = 1
			labels.extend(label)
			preds.extend(predictions)
			#'''
			loss.extend(loss_all)
		print("EVAL Loss:", np.mean(loss), "F1:", metrics.f1_score(labels, preds), "P:", metrics.precision_score(labels, preds), "R:", metrics.recall_score(labels, preds))
	except Exception as err:
		traceback.print_exc()
		print("APPLICATION", err)

	
def predict(datafile, modelfile, resultfile, batchsize=4):
	try:
		sentence_file_predict = datafile
		sess = tf.Session()
		sess.run(tf.tables_initializer())

		n_x = dimentions
		n_a = 400
		n_y = len(open(tags_file).readlines())
		num_layers = 1
		num_memoryunit = 1
		tags_index_lookup = list(map(lambda x: x.strip(), open(tags_file).readlines()))
		cls = model.SequenceLabelling(modelfile, n_a, n_x, n_y, num_layers, num_memoryunit)
	
		init_op_predict, next_element_predict = getDatasetIterator(sentence_file_predict, labels_file=None, mode="PREDICT", batchsize=1)
		sess.run(init_op_predict, feed_dict={embeddings_tf: embeddings})
		args = {}
		ow = open(resultfile, 'a')
		while True:
			try:
				batch_predict = sess.run(next_element_predict)
			except tf.errors.OutOfRangeError:
				break
			args['x'] = batch_predict
			args['y'] = None
			print('batch:', batch_predict.shape)
			try:
				outputs = execute_model(cls, mode="PREDICT", args=args)
				outputs = np.argmax(outputs, axis=2)
				for output in outputs:
					output_mapping = list(map(lambda x: tags_index_lookup[x], output))
					ow.write(' '.join(output_mapping) + "\n")
			except Exception as err:
				print(err)
				ow.write(' ' + "\n")

		ow.close()
		sess.close()
	except Exception as err:
		traceback.print_exc()
		print("APPLICATION", err)

if __name__ == "__main__":
	try:
		parser = argparse.ArgumentParser(description='Sequence Labelling')
		parser.add_argument('--mode', help='TRAIN|EVAL|PREDICT', required=True)
		parser.add_argument('--datafile', help='train_x|train_y,test_x|test_y', required=True)
		parser.add_argument('--tagsfile', help='tags.txt', required=True)
		parser.add_argument('--w2vfile', help='wikipedia-pubmed-and-PMC-w2v.bin', required=True)
		parser.add_argument('--modeldir', required=False)
		parser.add_argument('--resultfile', required=False)

		args = parser.parse_args()
		datafile = args.datafile
		mode = args.mode
		resultfile = args.resultfile

		global tags_file, w2v_file
		tags_file = args.tagsfile
		w2v_file = args.w2vfile

		global embeddings, table, tags, embeddings_tf
		embeddings, table, tags = _load_initial_models(w2v_file, tags_file)
		embeddings_tf = tf.placeholder(tf.float32, embeddings.shape, name='embeddings_tf') 

		if not os.path.exists(tags_file):
			raise Exception('tags_file %s not exist' % (tags_file))

		if not os.path.exists(w2v_file):
			raise Exception('w2v_file %s not exist' % (w2v_file))

		modeldir = args.modeldir if args.modeldir else "models"
		modelfile = os.path.join(modeldir, "model.ckpt")
		print("modelfile", modelfile)

		if mode not in ['TRAIN', 'EVAL', 'PREDICT']:
			raise Exception("mode value not in ['TRAIN', 'EVAL', 'PREDICT']")

		if mode == "TRAIN":
			training(datafile, modelfile, batchsize=4)
		elif mode == "EVAL":
			evaluation(datafile, modelfile, batchsize=4)
		elif mode == "PREDICT":
			if not resultfile:
				raise(Exception("--resultfile Argument missing"))
			predict(datafile, modelfile, resultfile, batchsize=4)
	except Exception as err:
		print(err)
