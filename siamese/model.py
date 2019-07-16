import os
import sys
import tensorflow as tf
    
"""
Define the model function which include 2CNN Model + Features N/w feeded into a fully DNN model
"""

class Siamese():
    def __init__(self, height, width, modelfile, margin=1.0, beta=0.01):

        self.modelfile = modelfile
        self.modeldir = os.path.dirname(self.modelfile)
        self.writer_dir_train = os.path.join(os.path.dirname(modelfile), "train")
        self.writer_dir_eval = os.path.join(os.path.dirname(modelfile), "eval")
        self.margin = margin
        self.model_graph = tf.Graph()
        self.beta = beta
        with self.model_graph.as_default():
            self.cntr = tf.Variable(0, name='counter', trainable=False)
            self.defineGraph(height, width)
 
            self.saver = tf.train.Saver()
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(init_op)

            if os.path.exists(os.path.join(self.modeldir, "checkpoint")):
                #print("CHEKPOINT", self.modeldir,  tf.train.latest_checkpoint(self.modeldir))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.modeldir))

    def cnn_model(self, img, mode, reuse=False):
        #input = tf.cast(img, tf.float16)
        input_ = tf.cast(img, tf.float32)
        print(input_.shape)
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input_, 32, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(net.shape)
        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(net.shape)

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(net.shape)

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 512, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(net.shape)

        with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [3, 3], activation_fn=None, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            print(net.shape)

        net = tf.contrib.layers.flatten(net)        
        print(net.shape)
        #dense = tf.layers.dense(inputs=net, units=2000, activation=None)
        #sys.exit(0)
        return net    
    
    def distance(self, encode1, encode2):
        distance = tf.reduce_sum(tf.math.square(encode1 - encode2), 1, keepdims=True)
        return distance
   
    def dnn_model(self, encoding1, encoding2, mode, reuse=False):
        #input_ = tf.concat([encoding1, encoding2], axis=1)
        input_ = tf.math.square(encoding1 - encoding2)
        input_ = tf.cast(input_, tf.float32)

        with tf.variable_scope("dnn1") as scope:
            dense = tf.layers.dense(inputs=input_, units=2000, activation=tf.nn.relu, reuse=reuse)
            dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope("dnn2") as scope:
            dense = tf.layers.dense(inputs=dense, units=2000, activation=tf.nn.relu, reuse=reuse)
            dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope("dnn3") as scope:
            dense = tf.layers.dense(inputs=dense, units=2000, activation=tf.nn.relu, reuse=reuse)
            dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

        logits = tf.layers.dense(inputs=dropout, units=2, reuse=reuse)
        return logits
 
    def defineGraph(self, height, width):
        with tf.name_scope("model"):

            # Placeholder Variables
            self.A1 = tf.placeholder(tf.float32, [None, height, width, 3], name='A1')
            self.P1 = tf.placeholder(tf.float32, [None, height, width, 3], name='P1')
            self.N1 = tf.placeholder(tf.float32, [None, height, width, 3], name='N1')
            self.learningRate = tf.placeholder(dtype=tf.float32)
            self.mode = tf.placeholder(tf.bool, shape=())

            self.labels_0 = tf.placeholder(tf.float32, [None], name='labels_0')
            self.labels_1 = tf.placeholder(tf.float32, [None], name='labels_1')

            A1, P1, N1, learningRate, mode = self.A1, self.P1, self.N1, self.learningRate, self.mode 
 
            A1_encode = self.cnn_model(A1, mode)
            P1_encode = self.cnn_model(P1, mode, reuse=True)
            N1_encode = self.cnn_model(N1, mode, reuse=True)
    
            '''Getting stats b/w images siamese n/w'''
            S_Distance = self.distance(A1_encode, P1_encode)
            D_Distance = self.distance(A1_encode, N1_encode)

            S_Distance_Mean = tf.reduce_mean(S_Distance)
            D_Distance_Mean = tf.reduce_mean(D_Distance)
            Difference = D_Distance_Mean - S_Distance_Mean

            tf.summary.scalar('Similar_loss', S_Distance_Mean)
            tf.summary.scalar('Diff_loss', D_Distance_Mean)
            tf.summary.scalar('D-S_loss', Difference)

            '''Computing loss using DNN'''
            logits_1 = self.dnn_model(A1_encode, P1_encode, mode, reuse=False)
            logits_0 = self.dnn_model(A1_encode, N1_encode, mode, reuse=True)

            onehot_labels_0 = tf.one_hot(indices=tf.cast(self.labels_0, tf.int32), depth=2)
            loss_all_0 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_0, logits=logits_0, reduction=tf.losses.Reduction.NONE)
            onehot_labels_1 = tf.one_hot(indices=tf.cast(self.labels_1, tf.int32), depth=2)
            loss_all_1 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_1, logits=logits_1, reduction=tf.losses.Reduction.NONE)

            #loss_all = tf.stack([loss_all_0, loss_all_1], axis=0)
            loss_all = tf.concat([loss_all_0, loss_all_1], axis=0)

            loss = tf.reduce_mean(loss_all)

            '''
            similarity = S_Distance
            dissimilarity = tf.maximum((self.margin - D_Distance), 0)

            loss_all = dissimilarity + similarity
            loss = tf.reduce_mean(loss_all)
            '''

            optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
            #tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
            #train_op = optimizer.apply_gradients(zip(grads, tvars))
            l2_loss = tf.losses.get_regularization_loss()
            train_op = optimizer.minimize(loss_all + self.beta * l2_loss)
            #train_op = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss)

            self.globalStep = tf.assign_add(self.cntr, 1, name='increment')
    
            tf.summary.scalar('loss', loss)
            #tf.summary.scalar('global_step', self.globalStep)
            #tf.summary.scalar('step', step)

            self.train_op = train_op
            self.loss = loss 
            self.loss_all = loss_all
            self.l2_loss = l2_loss
            self.logits_0 = tf.nn.softmax(logits_0)
            self.logits_1 = tf.nn.softmax(logits_1)
            self.S_Distance = S_Distance
            self.D_Distance = D_Distance
            self.S_Distance_Mean = S_Distance_Mean 
            self.D_Distance_Mean = D_Distance_Mean 
            self.A1_encode = A1_encode
            self.P1_encode = P1_encode
            self.N1_encode = N1_encode
            self.summary = tf.summary.merge_all()
