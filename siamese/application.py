import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import model  # Contain the model file or CNN model arch used
import argparse
from sklearn import metrics
import helper

np.set_printoptions(suppress=True)
tf.logging.set_verbosity(tf.logging.ERROR)

def parse_function_image(image):
    return tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(image), channels=3), tf.float32)

def parse_function(row):
    imgfiles = tf.string_split([row], delimiter=',', skip_empty=True).values
    A_img1 = imgfiles[0]
    P_img1 = imgfiles[1]
    N_img1 = imgfiles[2]

    A_img1= parse_function_image(A_img1)
    P_img1= parse_function_image(P_img1)
    N_img1= parse_function_image(N_img1)
    #A_img1 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(A_img1), channels=3), tf.float32)
    #P_img1 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(P_img1), channels=3), tf.float32)
    #N_img1 = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(N_img1), channels=3), tf.float32)

    #(2732, 876, 3)
    #A_img1 = tf.image.resize_images(A_img1, [3305, 1169])
    #P_img1 = tf.image.resize_images(P_img1, [3305, 1169])
    #N_img1 = tf.image.resize_images(N_img1, [3305, 1169])

    return A_img1, P_img1, N_img1

def execute_model(cls, mode, args):
    A1 = args['A1']
    P1 = args['P1']
    N1 = args['N1']
    labels_0 = [0] * A1.shape[0]
    labels_1 = [1] * A1.shape[0]

    if mode == "TRAIN":
        learningRate = args['learningRate']
        writer = args['writer_train']

        '''Run Training step'''
        output = cls.sess.run([cls.globalStep, cls.summary, cls.train_op, cls.loss_all, cls.S_Distance_Mean, cls.D_Distance_Mean, cls.l2_loss], 
		                      feed_dict={cls.A1: A1, cls.P1: P1, cls.N1: N1,
                                      cls.labels_0: labels_0, cls.labels_1: labels_1, 
				      cls.mode:"TRAIN", cls.learningRate: learningRate})

        globalStep, summary, loss = output[0], output[1], output[3]
        print("Training Siamese:", "\tGStep:", globalStep, "\tDistance Means(Loss, SLoss, DLoss, L2Loss):", np.mean(output[3]), output[4:])
        writer.add_summary(summary, globalStep)
        #writer.flush()
        return globalStep, loss

    elif mode == "EVAL":
        '''Run Eval step'''
        output = cls.sess.run([cls.summary, cls.loss_all, cls.S_Distance_Mean, cls.D_Distance_Mean, cls.logits_0, cls.logits_1], 
		                      feed_dict={cls.A1: A1, cls.P1: P1, cls.N1: N1,
                                      cls.labels_0: labels_0, cls.labels_1: labels_1, 
				      cls.mode:"EVAL"})
        if 'writer_eval' in args.keys():
            summary = output[0]
            globalStep = args['globalStep']
            writer = args['writer_eval']
            writer.add_summary(summary, globalStep)
        return output


def execute_model_predict(cls, args):
    A1 = args['A1']
    A2 = args['A2']
    logits = cls.sess.run([cls.logits_0], feed_dict={cls.A1: A1, cls.P1: A2, cls.N1: A2, cls.mode:"PREDICT"})
    return logits


def getDatasetIterator(datafile, batchsize=4):

    dataset = tf.data.TextLineDataset(datafile)
    dataset = dataset.shuffle(buffer_size=3)
    dataset = dataset.map(parse_function, num_parallel_calls=4)

    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    init_op = iterator.initializer
    return init_op, next_element


def training(datafile, modelfile, batchsize=4):

    args = {}
    args['learningRate'] = 0.0001
    #args['learningRate'] = 0.01

    try:
        LOSS = 100000000000
        trainfile, evalfile = datafile.split(",")
        init_op_train, next_element_train = getDatasetIterator(trainfile, batchsize)
        init_op_eval, next_element_eval = getDatasetIterator(evalfile, batchsize)

        sess = tf.Session()
        sess.run(init_op_train) 

        height, width = (2732, 876)

        cls = model.Siamese(height, width, modelfile, margin=1.0)

        writer_train = tf.summary.FileWriter(cls.writer_dir_train, cls.sess.graph)
        writer_eval = tf.summary.FileWriter(cls.writer_dir_eval, cls.sess.graph)

        totalEpoch = 3
        for epoch in range(totalEpoch):    
            print("Initializing the dataset iterator")
            sess.run(init_op_train)
            while True: 
                try:
                    batch_train = sess.run(next_element_train)
                except tf.errors.OutOfRangeError:
                    break

                args['A1'] = batch_train[0]
                args['P1'] = batch_train[1]
                args['N1'] = batch_train[2]
                args['writer_train'] = writer_train
                args['writer_eval'] = writer_eval

                with cls.sess.as_default():
                    globalStep, loss_ = execute_model(cls, mode="TRAIN", args=args)

                    if globalStep % 50 == 0:
                        print("saving model", cls.modelfile)
                        cls.saver.save(cls.sess, cls.modelfile)

                    if globalStep % 200 == 0:
                        print("mode => EVAL")
                        args['globalStep'] = globalStep
                        sess.run(init_op_eval)
                        loss = []
                        num_iter = 0
                        while True:
                            try:
                                batch_eval = sess.run(next_element_eval)
                            except tf.errors.OutOfRangeError:
                                break
                            args['A1'] = batch_eval[0]
                            args['P1'] = batch_eval[1]
                            args['N1'] = batch_eval[2]
                            output = execute_model(cls, mode="EVAL", args=args) 
                            loss.extend(output[1])
                        loss = np.mean(loss)
                        if LOSS > loss: 
                            loss = round(loss, 2)
                            LOSS = loss
                            dir_ = "%s_step=%s_loss=%s" % (cls.modeldir, str(globalStep), loss)
                            cls.saver.save(cls.sess, os.path.join(dir_, "model.ckpt"))
                            #cmd = "cp -Rv %s %s_step[%s]_loss[%s]" % (modelfolder, modelfolder, str(globalStep), loss)
                            #os.system(cmd)

        writer_train.close()
        writer_eval.close()
    except Exception as err:
        print("APPLICATION", err)

def evaluation(datafile, modelfile, batchsize=4):
    try:
        init_op, next_element = getDatasetIterator(datafile, batchsize)
        sess = tf.Session()
        sess.run(init_op) 
        height, width = (2732, 876)

        cls = model.Siamese(height, width, modelfile, margin=1.0)
        args = {}
        loss = []
        cntr = 0
        labels = []
        preds = []
        while True: 
            try:
                batch = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break
            cntr += 1
            if cntr % 50 == 0:
                print("cntr =>", cntr, "Loss:", np.mean(loss), "F1:", metrics.f1_score(labels, preds), 
                      "P:", metrics.precision_score(labels, preds), "R:", metrics.recall_score(labels, preds))
            args['A1'] = batch[0]
            args['P1'] = batch[1]
            args['N1'] = batch[2]
            output = execute_model(cls, mode="EVAL", args=args)
            logits_0, logits_1 = output[4:6]
            pred_0 = np.argmax(logits_0, axis=1)
            pred_1 = np.argmax(logits_1, axis=1)
            labels.extend([0] * batch[0].shape[0])
            labels.extend([1] * batch[0].shape[0])
            preds.extend(pred_0)
            preds.extend(pred_1)
            loss.extend(output[1]) 
        print("EVAL Loss:", np.mean(loss), "F1:", metrics.f1_score(labels, preds), "P:", metrics.precision_score(labels, preds), "R:", metrics.recall_score(labels, preds))
    except Exception as err:
        print("APPLICATION", err)


def _predict(sess, cls, predfiles):
    os.system("rm -rf tmp")
    predfiles_ = predfiles.split(",")
    predfiles = predfiles_[:]
    for cntr, predfile in enumerate(predfiles):
        if not os.path.exists(predfile):
            raise(Exception(predfile + " not found"))
        if predfile.lower().endswith("pdf"):
            predfiles[cntr] = helper.convert_tojpg(predfile)

    img = tf.placeholder(tf.string)
    image = parse_function_image(img)

    A1 = sess.run(image, feed_dict={img: predfiles[0]})
    A2 = sess.run(image, feed_dict={img: predfiles[1]})
    args = {'A1': np.array([A1]), 'A2': np.array([A2])}
    logits = execute_model_predict(cls, args)
    logits = logits[0]
    predictions = np.argmax(logits, axis=1)
    confidence = np.max(logits, axis=1) # Different axis from when used with tensor
    #print("predictions:\t", predictions)
    #:wprint("confidence:\t", confidence)
    return {'img1': predfiles_[0], 'img2': predfiles_[1], 'prediction': predictions[0], 'confidence': confidence[0]}

def predict(datafile, predfiles, modelfile, outputfile, batchsize=4):
    try:
        sess = tf.Session()
        height, width = (2732, 876)
        cls = model.Siamese(height, width, modelfile, margin=1.0)

        output = []
        if predfiles:
           r = _predict(sess, cls, predfiles)
           output.append(r)
        else:
           lines = open(datafile).readlines()
           for line in lines:
               line = line.strip()
               r = _predict(sess, cls, line)
               output.append(r)
        df = pd.DataFrame(output)
        print(df)
        if outputfile:
            df.to_csv(outputfile, index=False)
        sess.close()
    except Exception as err:
        print("APPLICATION", err)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Siamese')
        parser.add_argument('--mode', help='TRAIN|EVAL|PREDICT',
                        required=True)
        parser.add_argument('--datafile', help='filepath', required=False)
        parser.add_argument('--modeldir', help='filepath', required=False)
        parser.add_argument('--predfiles', help='file1.pdf,file2.pdf', required=False)
        parser.add_argument('--outputfile', help='file1.pdf', required=False)

        args = parser.parse_args()
        datafile = args.datafile
        mode = args.mode
        predfiles = args.predfiles
        outputfile = args.outputfile

        if not mode == "PREDICT" and not datafile:
            raise(Exception("--datafile Argument missing"))


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
            if not predfiles and not outputfile:
                raise(Exception("--outputfile Argument missing"))
            predict(datafile, predfiles, modelfile, outputfile, batchsize=4)
    except Exception as err:
        print(err)
