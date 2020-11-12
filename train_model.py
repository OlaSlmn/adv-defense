import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime
import os
# from skimage.measure import compare_ssim as ssim
from random import choice as rchoice

import csv
from numpy import array
import copy
import math
from random import seed
from random import random
from random import gauss
from PIL import Image
from pandas import read_csv
import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler
# from tensorflow.contrib.layers.python.layers import batch_norm
import model
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

img_size_original = 16

# Parameters

input_dim = 2
n_l1 = 1000
n_l2 = 1000
beta1 = 0.9
batch_size = 1
n_epochs = 100
learning_rate = 0.001
nb_classes= 2
results_path = '/home/olasalm/PycharmProjects/adv_defense/'




def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The model network
def model(x, reuse=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Model'):
        e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'm_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'm_dense_2'))
        output = dense(e_dense_2, n_l2, nb_classes, 'm_latent_variable')
        return output


def form_results(session_type):
	"""
	Forms folders for each run to store the tensorboard files, saved models and the log files.
	:return: three string pointing to tensorboard, saved models and log paths respectively.
	"""
	folder_name = "/adv_test/"
	tensorboard_path = results_path + folder_name + '/' + session_type + '/Tensorboard'
	saved_model_path = results_path + folder_name + '/' + session_type + '/Saved_models/'
	log_path = results_path + folder_name + '/' + session_type + '/log'

	if not os.path.exists(results_path + folder_name + '/' + session_type):
		if not os.path.exists(results_path + folder_name):
			os.mkdir(results_path + folder_name)

		os.mkdir(results_path + folder_name + '/' + session_type)
		os.mkdir(tensorboard_path)
		os.mkdir(saved_model_path)
		os.mkdir(log_path)

	return results_path + folder_name, tensorboard_path, saved_model_path, log_path



def train(X, Y, folder_name, train_model=True):
	"""
	Used to train the classifier by passing in the necessary inputs.
	:param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
	:return: does not return anything
	"""
	tf.reset_default_graph()
	x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input')
	y_target = tf.placeholder(dtype=tf.float32, shape=[None, nb_classes], name='Target')


	with tf.variable_scope(tf.get_variable_scope()):
		y_predict = model(x_input)



	# Classifier Loss
	classifier_loss= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=y_predict))

	all_variables = tf.trainable_variables()
	m_var = [var for var in all_variables if 'm_' in var.name]


	# Optimizer
	classifier_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(classifier_loss, var_list=m_var)


	init = tf.global_variables_initializer()


	# Tensorboard visualization
	tf.summary.scalar(name='Classifier Loss', tensor=classifier_loss)
	summary_op = tf.summary.merge_all()

	# Saving the model
	saver = tf.train.Saver()
	step = 0

	with tf.Session() as sess:
		if train_model:
			# global_step = tf.Variable(0, trainable=False)
			# learn_rate_init = 0.0003
			# new_learning_rate = tf.train.exponential_decay(learn_rate_init, global_step=global_step,
			#                                                decay_steps=10000,
			#                                                decay_rate=0.98)
			root_path, tensorboard_path, saved_model_path, log_path = form_results(
				'training_' + folder_name + '_nn')
			sess.run(init)
			writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

			for c in range(n_epochs):

				n_batches = int(len(X) / batch_size)
				for b in range(n_batches):
					results = []
					batch_x = X[(b * batch_size):((b + 1) * batch_size), :]
					batch_y = Y[(b * batch_size):((b + 1) * batch_size)]

					print("------------------Epoch {}/{}------------------".format(c, n_epochs))


					sess.run(classifier_optimizer, feed_dict={x_input: batch_x, y_target: batch_y})
					# sess.run(generator_optimizer, feed_dict={x_input_corrupted: batch_x_corrupted, x_target: batch_x})
					# if b % 50 == 0:
					c_loss, summary = sess.run(
						[classifier_loss, summary_op],
						feed_dict={x_input: batch_x, y_target: batch_y})

					writer.add_summary(summary, global_step=step)
					print("Epoch: {}, iteration: {}".format(c, b))
					print("Classifier Loss: {}".format(c_loss))

					with open(log_path + '/log.txt', 'a') as log:
						log.write("Epoch: {}, iteration: {}\n".format(c, b))
						log.write("Classifier Loss: {}".format(c_loss))

					step += 1
					results.append(c)
					results.append(b)
					results.append(c_loss)

					#
					# with open(log_path + '/results_training.csv', 'a') as csv_file_testing:
					# 	wr = csv.writer(csv_file_testing, delimiter=',')
					# 	wr.writerow(results)

				saver.save(sess, save_path=saved_model_path, global_step=step)
		else:

			root_path, tensorboard_path, saved_model_path, log_path, noised_images_path, denoised_images_path, real_images_path, classification_path = form_results(
				'testing_' + folder_name + '_nn')
			# print(root_path + 'training_' + folder_name + '_' + folder_name_1 + '_nn/Saved_models/')

			saver.restore(sess, save_path=tf.train.latest_checkpoint(
				root_path + '/training_' + folder_name + '_nn/Saved_models/'))
			# saver.restore(sess, save_path=tf.train.latest_checkpoint(
			#     root_path + '/training_all_nn/Saved_models/'))

			# print (X_test_corrupted.shape)
			writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)


			for i in range(len(X)):
				results = []

				batch_x = X[i, :]
				batch_y = Y[i]

				c_loss, summary = sess.run(
					[classifier_loss, summary_op],
					feed_dict={x_input: batch_x, y_target: batch_y})


				writer.add_summary(summary, global_step=step)


				print("Iteration: {}".format(i))
				print("Classifier Loss: {}".format(c_loss))

				with open(log_path + '/log.txt', 'a') as log:
					log.write("Iteration: {}\n".format(i))
					log.write("Classifier Loss: {}\n".format(c_loss))

				step += 1

				# results.append(ae_loss)
				# results.append(dc_loss)
				# results.append(Y[i])
				# results.append(round(tf.nn.sigmoid(D_discriminator_out).eval()[0][0]))
				# results.append(round(tf.nn.sigmoid(G_discriminator_out).eval()[0][0]))
				# results.append((np.square(X_corrupted[i, :] - X_in[i, :])).mean(axis=0))
				#
				# with open(log_path + '/results_testing.csv', 'a') as csv_file_testing:
				# 	wr = csv.writer(csv_file_testing, delimiter=',')
				# 	wr.writerow(results)
				#
				# with open(real_images_path + '/real_data.csv', 'a') as csv_file_testing:
				# 	row = X_in[i, :].tolist()
				# 	row.append(Y[i])
				#
				# 	wr = csv.writer(csv_file_testing, delimiter=',')
				# 	wr.writerow(row)


	return



if __name__ == '__main__':
	X, Y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_informative=2, n_classes=2,
							   n_clusters_per_class=1, random_state=1, class_sep=2)
	scalar = MinMaxScaler()
	scalar.fit(X)
	X = scalar.transform(X).reshape(len(X), 2)
	Y = np_utils.to_categorical(Y, 2)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.25)
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
	test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
	train(X,Y,"sim_results",True)






