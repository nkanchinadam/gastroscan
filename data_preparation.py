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