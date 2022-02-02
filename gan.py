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

def make_generator_model():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(25*25*256, use_bias=False, input_shape=(100,)))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Reshapae((25*25*256)))
  model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same'))
  return model

def get_generator_loss(fake_predictions):
  fake_predictions = tf.sigmoid(fake_predictions)
  fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions), fake_predictions)
  return fake_loss

def train_step(images, generator, generator_optimizer, discriminator, discriminator_optimizer):
  fake_image_noise = np.random.randn(BATCH_SIZE, 100, type='float32')
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(fake_image_noise)
    real_output = discriminator(images)
    fake_output = discriminator(generated_images)

    gen_loss = get_generator_loss(fake_output)
    disc_loss = get_discriminator_loss(real_output, fake_output)    

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    print("generator loss:", np.mean(gen_loss))
    print("discriminator loss: ", np.mean(disc_loss))

def train(dataset, epochs, generator, generator_optimizer, discriminator, discriminator_optimizer):
  for _ in range(epochs):
    for images in dataset:
      images = tf.cast(images, tf.dtypes.float32)
      train_step(images, generator, generator_optimizer, discriminator, discriminator_optimizer)

def main():
  to_load = input('Create Abnormality GAN: Input 0\nCreate Condition GAN: Input 1\n')

  x = None
  y = None
  if to_load == '0':
    labels = dp.get_labels(paths.ABNORMALITY_LABELS)
    x = dp.get_dataset(paths.ABNORMALITY_DATASET, labels)[0]
  elif to_load == '1':
    labels = dp.get_labels(paths.CONDITION_LABELS)
    x = dp.get_dataset(paths.CONDITION_DATASET, labels)[0]
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a GAN on')

  x = (x - 127.5) / 127.5

  BUFFER_SIZE = x.shape[0]
  BATCH_SIZE = 30
  x = tf.data.Dataset.from_tensor_slices(x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  discriminator = make_discriminator_model()
  discriminator_optimizer = tf.optimizers.Adam(1e-3)

  generator = make_generator_model()
  generator_optimizer = tf.optimizers.Adam(1e-4)

  train(x, 2, generator, generator_optimizer, discriminator, discriminator_optimizer)

if __name__ == '__main__':
  main()