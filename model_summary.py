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

def create_model_summary(model, x, y, labels):
  num_labels = len(labels) // 2
  classes = [[0 for j in range(num_labels)] for i in range(num_labels)]
  probs = model.predict(x)
  for i in range(len(probs)):
    max_index = max_prob(probs[i])
    classes[y[i]][max_index] += 1
  return classes

def main():
  to_load = input('Evaluate Abnormality Model: Input 0\nEvaluate Condition Model: Input 1\n')
  model_name = input('Model Name: ')

  labels = None
  model = None
  x_train = None
  y_train = None
  x_test = None
  y_test = None
  if to_load == '0':
    model = load_model(paths.ABNORMALITY_MODELS + model_name)
    labels = dp.get_labels(paths.ABNORMALITY_LABELS)
    x_train, y_train = dp.get_dataset(paths.ABNORMALITY_TRAIN, labels)
    x_test, y_test = dp.get_dataset(paths.ABNORMALITY_TEST, labels)
  elif to_load == '1':
    model = load_model(paths.CONDITION_MODELS + model_name)
    labels = dp.get_labels(paths.CONDITION_LABELS)
    x_train, y_train = dp.get_dataset(paths.CONDITION_TRAIN, labels)
    x_test, y_test = dp.get_dataset(paths.CONDITION_TEST, labels)

  train_classes = create_model_summary(model, x_train, y_train, labels)
  test_classes = create_model_summary(model, x_test, y_test, labels)

  f_train = open(paths.SUMMARIES + ('abnormality' if to_load == '0' else 'condition') + '_training.txt', 'w')
  f_train.write('Training Data\n')
  for i in range(len(train_classes)):
    f_train.write(str(labels[i]) + '\n')
    for j in range(len(train_classes[i])):
      f_train.write('\t' + labels[j] + ': ' + str(train_classes[i][j]) + '\n')
  f_train.close()

  f_test = open(paths.SUMMARIES + ('abnormality' if to_load == '0' else 'condition') + '_testing.txt', 'w')
  f_test.write('Testing Data\n')
  for i in range(len(test_classes)):
    f_test.write(str(labels[i]) + '\n')
    for j in range(len(test_classes[i])):
      f_test.write('\t' + labels[j] + ': ' + str(test_classes[i][j]) + '\n')
  f_test.close()

if __name__ == "__main__":
  main()