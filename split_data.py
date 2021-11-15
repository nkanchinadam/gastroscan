import os, random, shutil

HYPERKVASIR = './datasets/hyper-kvasir/'

ABNORMALITY_DATASET = './datasets/abnormality_dataset/'
ABNORMALITY_TRAIN = './datasets/abnormality_train/'
ABNORMALITY_TEST = './datasets/abnormality_test/'

CONDITION_DATASET = './datasets/condition_dataset/'
CONDITION_TRAIN = './datasets/condition_train/'
CONDITION_TEST = './datasets/condition_test/'

def split_folder(image_folder):
  images = []
  for image in os.scandir(image_folder):
    images.append(image)
  random.shuffle(images)
  return images[:int(0.7 * len(images))], images[int(0.7 * len(images)):]

def transfer_images(source_folder, target_folder, to_copy):
  for f in to_copy:
    shutil.copyfile(source_folder + f.name, target_folder + f.name)
    os.remove(source_folder + f.name)

def recur(source_dataset, train_dataset, test_dataset, filepath):
  source_dir = source_dataset + filepath
  for entry in os.scandir(source_dir):
    if entry.is_file():
      train_images, test_images = split_folder(source_dir)
      transfer_images(train_dataset + filepath, test_dataset + filepath, test_dataset)
      break
    else:
      recur(source_dataset, train_dataset, test_dataset, filepath + entry.name + '/')

def delete_files(dir):
  for entry in os.scandir(dir):
    if entry.is_file():
      os.remove(dir + entry.name)
    else:
      delete_files(dir + entry.name + '/')

def split_all(source_dataset, train_dataset, test_dataset):
  os.remove(train_dataset)
  os.remove(test_dataset)
  shutil.copytree(source_dataset, train_dataset)
  shutil.copytree(source_dataset, test_dataset)
  delete_files(test_dataset)
  recur(source_dataset, train_dataset, test_dataset, '')

def main():
  to_split = input('Split Abnormality Dataset: Input 0\nsplit Condition Dataset: Input 1\n')

  if to_split == '0':
    split_all(ABNORMALITY_DATASET, ABNORMALITY_TRAIN, ABNORMALITY_TEST)
  elif to_split == '1':
    split_all(CONDITION_DATASET, CONDITION_TRAIN, CONDITION_TEST)
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to split')

if __name__ == "__main__":
  main()