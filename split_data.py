import os, random, shutil

HYPERKVASIR = './datasets/hyper-kvasir/'

ABNORMALITY_DATASET = './datasets/abnormality_dataset/'
ABNORMALITY_TRAIN = './datasets/abnormality_train/'
ABNORMALITY_TEST = './datasets/abnormality_test/'

CONDITION_DATASET = './datasets/condition_dataset/'
CONDITION_TRAIN = './datasets/condition_train/'
CONDITION_TEST = './datasets/condition_test/'

def split(image_folder):
  images = []
  for image in os.scandir(image_folder):
    images.append(image)
  random.shuffle(images)
  return images[:int(0.7 * len(images))], images[int(0.7 * len(images)):]

def copy_images(source_folder, target_folder, toCopy):
  for f in toCopy:
    print(f.name)
    shutil.copy(source_folder + f.name, target_folder + f.name)

def recur(source_dataset, train_dataset, test_dataset, filepath):
  source_dir = source_dataset + filepath
  for entry in os.scandir(source_dir):
    if entry.is_file():
      train_images, test_images = split(source_dir)
      copy_images(source_dir, train_dataset + filepath, train_images)
      copy_images(source_dir, test_dataset + filepath, test_images)
      break
    else:
      recur(source_dataset, train_dataset, test_dataset, filepath + entry.name + '/')

def main():
  to_split = input('Split Abnormality Dataset: Input 0\nsplit Condition Dataset: Input 1\n')

  if to_split == '0':
    recur(ABNORMALITY_DATASET, ABNORMALITY_TRAIN, ABNORMALITY_TEST, '')
  elif to_split == '1':
    recur(CONDITION_DATASET, CONDITION_TRAIN, CONDITION_TEST, '')
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to split')

if __name__ == "__main__":
  main()