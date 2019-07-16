import os
import sys
import tensorflow as tf
	
"""
Define the model function which include 2CNN Model + Features N/w feeded into a fully DNN model
"""

class SequenceLabelling():
	def __init__(self, modelfile, n_a, n_x, n_y, num_layers, num_memoryunit):
		self.n_a = n_a
		self.n_x = n_x
		self.n_y = n_y
		self.num_layers = num_layers
		self.num_memoryunit = num_memoryunit
		self.MINORFLOAT = 0.00000001
		self.modelfile = modelfile
		self.modeldir = os.path.dirname(self.modelfile)

		self.writer_dir_train = os.path.join(os.path.dirname(modelfile), "train")
		self.writer_dir_eval = os.path.join(os.path.dirname(modelfile), "eval")

		#variables_names = [v.name for v in tf.trainable_variables()] 
		#print(variables_names)
		#for op in self.model_graph.get_operations():
		#	print(str(op))
		#sys.exit()
		
		#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		#self.sess.run(init_op)
		#print([n.name for n in tf.get_default_graph().as_graph_def().node])

		if os.path.exists(os.path.join(self.modeldir, "checkpoint")):
			self.sess = tf.Session()
			self.saver = tf.train.import_meta_graph(os.path.join(self.modeldir, 'model.ckpt.meta'))
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.modeldir))
			graph = tf.get_default_graph()
			self.x = graph.get_tensor_by_name('x:0')
			self.y = graph.get_tensor_by_name('y:0')
			self.summary = graph.get_tensor_by_name('summary:0')
			self.global_step = graph.get_tensor_by_name('global_step:0')
			self.precision = graph.get_tensor_by_name('precision:0')
			self.recall = graph.get_tensor_by_name('recall:0')
			self.f1 = graph.get_tensor_by_name('f1:0')
			self.precision_c = graph.get_tensor_by_name('precision_c:0')
			self.recall_c = graph.get_tensor_by_name('recall_c:0')
			self.f1_c = graph.get_tensor_by_name('f1_c:0')
			self.predictions = graph.get_tensor_by_name('predictions:0')
			self.reset_vars_initializer = graph.get_operation_by_name('reset_vars_initializer')
			self.loss_all = graph.get_tensor_by_name('loss_all:0')
			self.fa_init_state = graph.get_tensor_by_name('fa_init_state:0')
			self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
			self.learning_rate = graph.get_tensor_by_name('learning_rate:0')
			self.predictions_normalized = graph.get_tensor_by_name('predictions_normalized:0')
			self.train_op = graph.get_operation_by_name('stack_bidirectional_rnn/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/Adam')
			#self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_all)
			self.graph = graph
			#print(self.sess.run(tf.all_variables()))
			#self.sess.run(tf.all_variables())
			#print(self.sess.run(self.cntr))
			#print(self.sess.run('Vy1:0'))
			#sys.exit()
		else:
			self.graph = tf.Graph()
			with self.graph.as_default():
				self.counter = tf.Variable(0, name='counter', trainable=False)
				self.defineGraph()
				self.saver = tf.train.Saver(save_relative_paths=True, max_to_keep=1)
				self.sess = tf.Session()
				init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
				self.sess.run(init_op)

	""" Function to calculate the metrics for each iteration """
	def defineMeasures(self, y, predictions_softmax):
		""" Code to access Precision, Recall, F1 Score"""
		'''
		argmax_prediction = tf.argmax(self.predictions, 1)
		argmax_y = tf.argmax(y, 1)
		TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
		TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
		FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
		FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
		precision = TP/(TP + FP + 0.000001)
		recall = TP/(TP + FN + 0.000001)
		'''
		predictions_argmax 	= 	tf.argmax(predictions_softmax, axis=1)
		y_argmax 			= 	tf.argmax(y, axis=1)

		recall 		= 	tf.metrics.recall(y_argmax, predictions_argmax, name="reset")[1]
		precision 	= 	tf.metrics.precision(y_argmax, predictions_argmax, name="reset")[1]

		f1 			= 	2 * precision * recall / (precision + recall + self.MINORFLOAT)
		recall_c 	= 	[0.0] * int(self.n_y)
		precision_c = 	[0.0] * int(self.n_y)
		f1_c 		= 	[0.0] * int(self.n_y)
		y_cast 		= 	tf.cast(y, tf.int64)

		for i in range(self.n_y):
			precision_c[i] 	=  	tf.metrics.precision(labels=tf.equal(y_argmax, i), 
														predictions=tf.equal(predictions_argmax, i), 
														name="reset")[1]
			recall_c[i] 	= 	tf.metrics.recall(labels=tf.equal(y_argmax, i), 
														predictions=tf.equal(predictions_argmax, i), 
														name="reset")[1]
			f1_c[i] 		= 	2 * precision_c[i] * recall_c[i] / (precision_c[i] + recall_c[i] + self.MINORFLOAT)

			tf.summary.scalar("precisions_" + str(i), precision_c[i])
			tf.summary.scalar("recalls_" + str(i), recall_c[i])
			tf.summary.scalar("f1_" + str(i), f1_c[i])


		self.precision 	= tf.identity(precision, name='precision')
		self.recall 	= tf.identity(recall, name='recall')
		self.f1 		= tf.identity(f1, name='f1')

		self.precision_c= tf.identity(precision_c, name='precision_c')
		self.recall_c 	= tf.identity(recall_c, name='recall_c')
		self.f1_c 		= tf.identity(f1_c, name='f1_c')

		tf.summary.scalar('F1', f1)
		tf.summary.scalar('Recall', recall)
		tf.summary.scalar('Precision', precision)

		self.predictions_argmax = predictions_argmax
		self.y_argmax = y_argmax

		reset_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="reset")
		self.reset_vars_initializer = tf.variables_initializer(var_list=reset_vars, name='reset_vars_initializer')


	def defineGraph(self):
		xav_init = tf.contrib.layers.xavier_initializer			
		self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
		self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

		n_x, n_a, n_y, num_layers, num_memoryunit = self.n_x, self.n_a, self.n_y, self.num_layers, self.num_memoryunit
		
		self.x = tf.placeholder(shape=[None, None, n_x], dtype=tf.float32, name='x') # Shape is m, Tx, n_x
		self.y = tf.placeholder(shape=[None, None], dtype=tf.int32, name='y')  # Shape is m X Ty
		
		''' Since we need to loop over timestamp instead of samples, we need to transpose our matrix such that time stamp comes first 
		For example if shape of x is (2, 5, 200)  # Where we have 2 samples each having 5 words and w2v dimentions is 200
		Now since we want to iterate over timestamp of every words simultaneously we need something like (5, 2, 200)
		So 1st iteration will be of (2, 200) then next (2, 200) and so on
		
		Similary we will also make modification for y. Now current y state is (m, Tx) howewer we want it to in follwing (-1 shape). 
		Now remember above, for each timestamp we will have predictions for all the samples. Hence we should modify our matrix like.
		where S is sample and W is word.

		[[S1W1, S1W2, S1W3, S1W4, S1W5]
		 [S2W1, S2W2, S2W3, S2W4, S2W5]]   =>   [S1W1, S2W1, S1W2, S2W2, S1W3, S2W3, S1W4, S2W4, S1W5, S2W5]
		'''
		x = tf.transpose(self.x, [1, 0, 2])
		y = tf.transpose(self.y)
		y = tf.reshape(y, [-1])

		y_onehot = tf.one_hot(y, n_y)

		''' Use below if want to have dropout on encodings, then disable DropoutWrapper'''
		#x =  tf.nn.dropout(x, self.keep_prob, noise_shape=[tf.shape(x)[0], tf.shape(x)[1], 1])

		# Forward direction cell
		lstm_fw_cells = []
		for _ in range(num_layers):
			lstm_fw_cells_m = []
			for _ in range(num_memoryunit):
				lstm_fw_cell_m = tf.contrib.rnn.BasicLSTMCell(n_a, state_is_tuple=True, forget_bias=1.0)
				lstm_fw_cells_m.append(lstm_fw_cell_m)
			lstm_fw_cells_m = tf.nn.rnn_cell.MultiRNNCell(lstm_fw_cells_m, state_is_tuple=True)
			lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cells_m, input_keep_prob=self.keep_prob)
			lstm_fw_cells.append(lstm_fw_cell)

		# Backward direction cell
		lstm_bw_cells = []
		for _ in range(num_layers):
			lstm_bw_cells_m = []
			for _ in range(num_memoryunit):
				lstm_bw_cell_m = tf.contrib.rnn.BasicLSTMCell(n_a, state_is_tuple=True, forget_bias=1.0)
				lstm_bw_cells_m.append(lstm_bw_cell_m)
			lstm_bw_cells_m = tf.nn.rnn_cell.MultiRNNCell(lstm_bw_cells_m, state_is_tuple=True)
			lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cells_m, input_keep_prob=self.keep_prob)
			lstm_bw_cells.append(lstm_bw_cell)

		""" This is stacking different layers """
		outputs, _, _= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells, x, dtype=tf.float32)


		by1 = tf.Variable(tf.random_uniform([1, n_a * 2], minval=0, maxval=1.0), tf.float32, name='by1')
		Vy1 = tf.get_variable(name='Vy1', shape=[n_a * 2, n_a * 2], dtype=tf.float32,  initializer=xav_init())
		by2 = tf.Variable(tf.random_uniform([1, n_y], minval=0, maxval=1.0), tf.float32, name='by2')
		Vy2 = tf.get_variable(name='Vy2', shape=[n_a * 2, n_y], dtype=tf.float32,  initializer=xav_init())

		self.fa_init_state = tf.placeholder(shape=[None, n_y], dtype=tf.float32, name='fa_init_state') # 2 x m x n_a

		def computerFinalActivations(prev, st):
			#===================================================================
			# st, ct = es[0], es[1]
			# st = tf.unstack(st)
			# st = tf.concat(st, axis=1)
			#===================================================================
			yt1 = tf.matmul(st, Vy1) + by1
			yt1 = tf.nn.relu(yt1)

			yt2 = tf.matmul(st, Vy2) + by2
			return yt2

			#yt = tf.matmul(st, Vy) + by
			#return yt

		final_activations = tf.scan(computerFinalActivations,
			outputs,
			initializer=self.fa_init_state
		)
		#final_activations = tf.transpose(final_activations, [1, 0, 2])  # ??? 
		final_activations =  tf.nn.dropout(final_activations, self.keep_prob)
		### Whats the shape here ??
		logits = tf.reshape(final_activations, (-1, n_y))
		predictions_softmax = tf.nn.softmax(logits)

		# We have weighted the logits only for training not for prediction
		self.defineMeasures(y_onehot, predictions_softmax)

		#=======================================================================
		# ''' Class imbalancing '''
		# self.pos_class_weight = tf.placeholder(dtype=tf.float32)
		# losses = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=self.pos_class_weight)
		# ''' Class imbalancing end '''
		#=======================================================================

		loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
		loss_all = loss_all + 0.5 * (1 - self.f1)
		loss = tf.reduce_mean(loss_all)
		tf.summary.scalar('loss_modified', loss)
		tf.summary.scalar('loss', tf.reduce_mean(loss_all))

		'''
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss_all, tvars), 10)
		train_op = optimizer.apply_gradients(zip(grads, tvars))
		self.train_op = tf.identity(train_op, name='train_op')
		'''
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_all)
		
		'''
		gradients, variables = zip(*optimizer.compute_gradients(losses))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		self.train_op = optimizer.apply_gradients(zip(gradients, variables))
		'''


		''' We need to recovert or transpose our output predictions, such that 
		[S1W1, S2W1, S1W2, S2W2, S1W3, S2W3, S1W4, S2W4, S1W5, S2W5] =>	[[S1W1, S1W2, S1W3, S1W4, S1W5]
																		[S2W1, S2W2, S2W3, S2W4, S2W5]]
		'''

		predictions_normalized = tf.reshape(predictions_softmax, shape=[tf.shape(self.x)[1], tf.shape(self.x)[0], -1])
		predictions_normalized = tf.transpose(predictions_normalized, [1,0,2])

		predictions = tf.reshape(self.predictions_argmax, shape=[tf.shape(self.x)[1], tf.shape(self.x)[0]])
		predictions = tf.transpose(predictions, [1,0])

		summary = tf.summary.merge_all()

		self.final_activations = final_activations
		self.predictions_normalized = tf.identity(predictions_normalized, name='predictions_normalized')
		self.predictions = tf.identity(predictions, name='predictions')
		self.loss_all = tf.identity(loss_all, name='loss_all')
		self.global_step = tf.assign_add(self.counter, 1, name='global_step')
		self.summary = tf.identity(summary, name='summary')