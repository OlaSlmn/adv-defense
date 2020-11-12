"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from model import Model
from pgd_attack import LinfPGDAttack
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
# tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']



# Setting up the data and the model
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

X, Y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_informative=2, n_classes=2,
                           n_clusters_per_class=1, random_state=1, class_sep=2)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.show()

scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X).reshape(len(X), 2, 1)
Y = np_utils.to_categorical(Y, 2)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.25)
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

global_step = tf.compat.v1.train.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
# tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0


  # Main training loop

  # for ii in range(max_num_training_steps):
  #   x_batch, y_batch = train_dataset.next_batch(batch_size)
  n_batches = int(len(X) / batch_size)
  for b in range(n_batches):

    x_batch = X[(b * batch_size):((b + 1) * batch_size), :]
    y_batch = Y[(b * batch_size):((b + 1) * batch_size)]

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    # adv_dict = {model.x_input: x_batch_adv,
    #             model.y_input: y_batch}

    # Output to stdout
    if b % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      # adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      # print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      # summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start