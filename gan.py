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

def get_discriminator_loss(real_predictions, fake_predictions):
  real_predictions = tf.sigmoid(real_predictions)
  fake_predictions = tf.sigmoid(fake_predictions)
  real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions)
  fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
  return real_loss + fake_loss

def main():
  to_load = input('Create Abnormality GAN: Input 0\nCreate Condition GAN: Input 1\n')

  x = None
  y = None
  if to_load == '0':
    labels = dp.get_labels(paths.ABNORMALITY_LABELS)
    x, y = dp.get_dataset(paths.ABNORMALITY_DATASET, labels)
  elif to_load == '1':
    labels = dp.get_labels(paths.CONDITION_LABELS)
    x, y = dp.get_dataset(paths.CONDITION_DATASET, labels)
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a GAN on')

  x = (x - 127.5) / 127.5

  BUFFER_SIZE = x.shape[0]
  BATCH_SIZE = 30
  x = tf.data.Dataset.from_tensor_slices(x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  discriminator = make_discriminator_model()
  discriminator_optimizer = tf.optimizers.Adam(1e-3)

if __name__ == '__main__':
  main()