import numpy as np
import os
from PIL import Image

HEIGHT = 100
WIDTH = 100
IDEAL_WINDOW_SIZE = (5, 5)
RESIZE_METHOD = 'resize' # 'crop' or 'resize'

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
  
def crop_image(data):
  img_height = len(data)
  img_width = len(data[0])

  num_remove_from_height = img_height - HEIGHT
  data = data[(num_remove_from_height // 2) + (num_remove_from_height % 2) : img_height - (num_remove_from_height // 2)]

  num_remove_from_width = img_width - WIDTH
  for i in range(len(data)):
    data[i] = data[i][(num_remove_from_width // 2) + (num_remove_from_width % 2) : img_width - (num_remove_from_width // 2)]

def average_color(data, top_left_r, top_left_c, height, width):
  r = 0
  g = 0
  b = 0
  for i in range(top_left_r, top_left_r + height):
    for j in range(top_left_c, top_left_c + width):
      r += data[i][j][0]
      g += data[i][j][1]
      b += data[i][j][2]
  return [r // (height * width), g // (height * width), b // (height * width)]

def pool_image(data, window_height, window_width):
  pooled_data = []
  for i in range(len(data) - (window_height - 1)):
    pooled_data.append([])
    for j in range(len(data[0]) - (window_width - 1)):
      pooled_data[i].append(average_color(data, i, j, window_height, window_width))
  return pooled_data

def resize_image(data):
  while len(data) > HEIGHT or len(data[0]) > WIDTH:
    window_height = 0 if len(data) == HEIGHT else IDEAL_WINDOW_SIZE[0] if (len(data) - HEIGHT) % (IDEAL_WINDOW_SIZE[0] - 1) == 0 else (len(data) - HEIGHT) % (IDEAL_WINDOW_SIZE[0] - 1) + 1
    window_width = 0 if len(data[0]) == WIDTH else IDEAL_WINDOW_SIZE[1] if (len(data[0]) - WIDTH) % (IDEAL_WINDOW_SIZE[1] - 1) == 0 else (len(data[0]) - WIDTH) % (IDEAL_WINDOW_SIZE[1] - 1) + 1
    data = pool_image(data, window_height, window_width)
  return data

def load_image(filepath):
  img = Image.open(filepath)
  imrgb = img.convert('RGB')
  img_width, img_height = imrgb.size
  data = imrgb.getdata()  
  resized = [[[k for k in data[i * img_width + j]] for j in range(img_width)] for i in range(img_height)]
  if RESIZE_METHOD == 'crop':
    crop_image(resized)
  elif RESIZE_METHOD == 'resize':
    resized = resize_image(resized)
  else:
    raise ValueError('Invalid resize method')
  return resized

def get_data(dir):
  data = []
  for entry in os.scandir(dir):
    if entry.is_file():
      data.append(load_image(dir + entry.name))
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