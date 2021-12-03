from keras.models import load_model
import paths
import data_preparation as dp
import numpy as np

def max_prob(probs):
  max_prob = 0
  max_index = -1
  for i in range(len(probs)):
    if probs[i] > max_prob:
      max_prob = probs[i]
      max_index = i
  return max_index

def main():
  to_load = input('Evaluate Abnormality Model: Input 0\nEvaluate Condition Model: Input 1\n')
  model_name = input('Model Name: ')
  image_filepath = input('Image Filepath: ')

  img_data = np.array([dp.load_image(image_filepath)])
  expected_output = image_filepath.split('/')[3]
  model = None
  if to_load == '0':
    model = load_model(paths.ABNORMALITY_MODELS + model_name)
    labels = dp.get_labels(paths.ABNORMALITY_LABELS)
  elif to_load == '1':
    model = load_model(paths.CONDITION_MODELS + model_name)
    labels = dp.get_labels(paths.CONDITION_LABELS)
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a model on')

  probs = model.predict(img_data, labels[expected_output])[0]
  img_label = max_prob(probs)
  print('Prediction: ' + labels[img_label])
  print('Expected Output: ' + expected_output)
  print('Correct!' if img_label == labels[expected_output] else 'Incorrect!')


if __name__ == '__main__':
  main()