"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 2])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.input_dim = 2
    self.n_l1 = 100
    self.n_l2 = 100
    self.n_classes = 2


    # self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])


    # first dense layer
    W_dense1 = self._weight_variable([self.input_dim, self.n_l1])
    b_dense1 = self._bias_variable([self.n_l1])

    h_dense1 = tf.nn.relu(self._dense(self.x_input, W_dense1, b_dense1))


    # second dense layer
    W_dense2 = self._weight_variable([self.n_l1,self.n_l2])
    b_dense2 = self._bias_variable([self.n_l2])

    h_dense2 = tf.nn.relu(self._dense(h_dense1, W_dense2, b_dense2))


    # output layer
    W_fc = self._weight_variable([self.n_l2,self.n_classes])
    b_fc = self._bias_variable([self.n_classes])

    self.pre_softmax = self._dense(h_dense2, W_fc, b_fc)

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _dense(x, W, b):
      return tf.add(tf.matmul(x, W, name='matmul'),b)

