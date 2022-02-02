import data_preparation as dp
import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(7, (3, 3), padding='same', input_shape=(100, 100, 3)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.LeakyReLU())
  model.add(tf.keras.layers.Dense(50, activation='relu'))
  model.add(tf.keras.layers.Dense(1))
  return model

def main():
  to_load = input('Create Abnormality GAN: Input 0\nCreate Condition GAN: Input 1\n')

  x_train = None
  x_test = None
  y_train = None
  y_test = None
  if to_load == '0':
    labels = dp.get_labels(paths.ABNORMALITY_LABELS)
    x_train, y_train = dp.get_dataset(paths.ABNORMALITY_TRAIN, labels)
    x_test, y_test = dp.get_dataset(paths.ABNORMALITY_TEST, labels)
  elif to_load == '1':
    labels = dp.get_labels(paths.CONDITION_LABELS)
    x_train, y_train = dp.get_dataset(paths.CONDITION_TRAIN, labels)
    x_test, y_test = dp.get_dataset(paths.CONDITION_TEST, labels)
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a GAN on')

  x_train = (x_train - 127.5) / 127.5
  x_test = (x_test - 127.5) / 127.5

  BUFFER_SIZE = x_train.shape[0]
  BATCH_SIZE = 30
  x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  x_test = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)

if __name__ == '__main__':
  main()