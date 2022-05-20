import data_preparation as dp
import paths
import keras
import numpy as np
import tensorflow as tf
from PIL import Image

def main():
  labels = dp.get_labels(paths.CONDITION_LABELS)

  label = input('Choose a label from the following:\n' + '\n'.join([key for key in labels if type(key) == type('')]) + '\n')
  if label not in labels:
    raise ValueError('Input a valid label')

  model = keras.models.load_model(paths.GANS + label + '_generator')
    
  num_to_generate = int(input('Number of images to generate: '))
  random_noises = tf.random.normal((num_to_generate, 100))
  predictions = model.predict(random_noises)
  for i in range(num_to_generate):
    img = Image.fromarray((np.reshape(predictions[i], (dp.HEIGHT, dp.WIDTH, 3)) * 127.5 + 127.5).astype(np.uint8))
    img = img.convert('RGB')
    if label in ['healthy', 'hemorrhoids', 'polyps']:
      img.save(paths.ABNORMALITY_DATASET + label + '/generated/img' + str(i) + '.jpg')
    img.save(paths.CONDITION_DATASET + label + '/generated/img' + str(i) + '.jpg')

if __name__ == '__main__':
  main()