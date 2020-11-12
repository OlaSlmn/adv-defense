import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from cleverhans.future.tf2.attacks import projected_gradient_descent, fast_gradient_method
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# plt.figure()
#
#
# # plt.subplot()
#
#
#
# #generate data
# plt.title("Two informative features, one cluster per class", fontsize='small')
# X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
# plt.show()

FLAGS = flags.FLAGS


class Net(Model):
  def __init__(self):
    super(Net, self).__init__()
    # self.conv1 = Conv2D(64, 8, strides=(2, 2), activation='relu', padding='same')
    # self.conv2 = Conv2D(128, 6, strides=(2, 2), activation='relu', padding='valid')
    # self.conv3 = Conv2D(128, 5, strides=(1, 1), activation='relu', padding='valid')
    # self.dropout = Dropout(0.25)
    # self.flatten = Flatten()
    # self.dense1 = Dense(128, activation='relu')
    # self.dense2 = Dense(10)
    self.dense1 = Dense(2, activation='relu')
    self.dense2 = Dense(4, activation='relu')
    # self.dense3 = Dense(100, activation='relu')
    self.dense3 = Dense(2)

  def call(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.dropout(x)
    # x = self.flatten(x)
    # x = self.dense1(x)
    # return self.dense2(x)

    x = self.dense1(x)
    x = self.dense2(x)
    # x = self.dense3(x)
    return self.dense3(x)


# def ld_mnist():
#   """Load training and test data."""
#
#   def convert_types(image, label):
#     image = tf.cast(image, tf.float32)
#     image /= 255
#     return image, label
#
#   dataset, info = tfds.load('mnist',
#                             data_dir='gs://tfds-data/datasets',
#                             with_info=True,
#                             as_supervised=True)
#   mnist_train, mnist_test = dataset['train'], dataset['test']
#   mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
#   mnist_test = mnist_test.map(convert_types).batch(128)
#   return EasyDict(train=mnist_train, test=mnist_test)


def main(_):
  # Load training and test data
  # data = ld_mnist()
  X, Y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_informative=2, n_classes=2, n_clusters_per_class=1,random_state=1, class_sep=2)

  plt.figure()
  plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
  plt.show()

  scalar = MinMaxScaler()
  scalar.fit(X)
  X = scalar.transform(X).reshape(len(X),2,1)
  Y = np_utils.to_categorical(Y, 2)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y,stratify=Y,test_size=0.25)
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

  model = Net()
  loss_object = tf.losses.BinaryCrossentropy(from_logits=True)
  loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.optimizers.Adam(learning_rate=0.03)

  # Metrics to track the different accuracies.
  train_loss = tf.metrics.Mean(name='train_loss')
  test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
  test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
  test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()
  # test_acc_clean = tf.metrics.BinaryAccuracy()
  # test_acc_fgsm = tf.metrics.BinaryAccuracy()
  # test_acc_pgd = tf.metrics.BinaryAccuracy()

  @tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      predictions = model(x)
      loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

  # Train model with adversarial training
  for epoch in range(FLAGS.nb_epochs):
    # keras like display of progress
    progress_bar_train = tf.keras.utils.Progbar(10000)
    for (x, y) in train_dataset:

      if FLAGS.adv_train:
        # Replace clean example with adversarial example for adversarial training
        x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
      train_step(x, y)
      progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])

  # Evaluate on clean and adversarial data
  progress_bar_test = tf.keras.utils.Progbar(10000)
  for x, y in test_dataset:
    y_pred = model(x)
    test_acc_clean(y, y_pred)

    x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
    y_pred_fgm = model(x_fgm)
    test_acc_fgsm(y, y_pred_fgm)

    x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
    y_pred_pgd = model(x_pgd)
    test_acc_pgd(y, y_pred_pgd)

    progress_bar_test.add(x.shape[0])

  print('test acc on clean examples (%): {:.3f}'.format(test_acc_clean.result() * 100))
  print('test acc on FGM adversarial examples (%): {:.3f}'.format(test_acc_fgsm.result() * 100))
  print('test acc on PGD adversarial examples (%): {:.3f}'.format(test_acc_pgd.result() * 100))


if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', 50, 'Number of epochs.')
  flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')
  flags.DEFINE_bool('adv_train', False, 'Use adversarial training (on PGD adversarial examples).')
  app.run(main)