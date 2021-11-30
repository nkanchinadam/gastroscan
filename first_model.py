from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import os
from PIL import Image
import paths

HEIGHT = 100
WIDTH = 100

def get_labels(filepath):
  f = open(filepath, 'r')
  labels = {}
  count = 0
  for line in f.readlines():
    line = line.replace('\n', '')
    labels[line] = count
    labels[count] = line
    count += 1
  return labels
  
def resized_image(filepath):
  img = Image.open(filepath)
  imrgb = img.convert('RGB')
  img_width, img_height = imrgb.size
  data = imrgb.getdata()  
  resized = [[[k for k in data[i * img_width + j]] for j in range(img_width)] for i in range(img_height)]

  num_remove_from_height = img_height - HEIGHT
  resized = resized[(num_remove_from_height // 2) + (num_remove_from_height % 2) : img_height - (num_remove_from_height // 2)]

  num_remove_from_width = img_width - WIDTH
  for i in range(len(resized)):
    resized[i] = resized[i][(num_remove_from_width // 2) + (num_remove_from_width % 2) : img_width - (num_remove_from_width // 2)]
  return resized

def get_data(dir):
  data = []
  for entry in os.scandir(dir):
    if entry.is_file():
      data.append(resized_image(dir + entry.name))
    else:
      data += get_data(dir + entry.name + '/')
  return data

def get_dataset(dir, labels):
  x = []
  y = []
  count = 0
  for entry in os.scandir(dir):
    count += 1
  for entry in os.scandir(dir):
    label_data = get_data(dir + entry.name + '/')
    index = labels[entry.name]
    x += label_data
    for i in range(len(label_data)):
      y.append(index)
  return np.array(x), np.array(y)

def main():
  to_create = input('Create and Train Abnormality Model: Input 0\nCreate and Train Condition Model: Input 1\n')
  model_name = input('Model Name: ')

  labels = None
  x_train = None
  y_train = None
  x_test = None
  y_test = None
  if to_create == '0':
    labels = get_labels(paths.ABNORMALITY_LABELS)
    x_train, y_train = get_dataset(paths.ABNORMALITY_TRAIN, labels)
    x_test, y_test = get_dataset(paths.ABNORMALITY_TEST, labels)
  elif to_create == '1':
    labels = get_labels(paths.CONDITION_LABELS)
    x_train, y_train = get_dataset(paths.CONDITION_TRAIN, labels)
    x_test, y_test = get_dataset(paths.CONDITION_TEST, labels)
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a model on')

  model = Sequential()
  model.add(Flatten(input_shape=(HEIGHT, WIDTH, 3)))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(20, activation='relu'))
  model.add(Dense(len(labels) // 2, activation='sigmoid'))

  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
  model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
  model.save('./models/' + ('abnormalities' if to_create == '0' else 'conditions') + '/' + model_name)

if __name__ == '__main__':
  main()