import numpy as np
import os
from PIL import Image

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

def get_data(dir):
  data = []
  for entry in os.scandir(dir):
    if entry.is_file():
      resized_data = Image.open(dir + entry.name).resize((WIDTH, HEIGHT)).getdata()
      data.append([[[k for k in resized_data[i * WIDTH + j]] for j in range(WIDTH)] for i in range(HEIGHT)])
      print(dir + entry.name)
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