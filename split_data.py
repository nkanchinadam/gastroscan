import os, random, shutil

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

def recur(source_dir, train_dir, test_dir):
  for entry in os.scandir(source_dir):
    if entry.is_file():
      test_images = split_folder(source_dir)[1]
      transfer_images(train_dir, test_dir, test_images)
      break
    else:
      recur(source_dir + entry.name + '/', train_dir + entry.name + '/', test_dir + entry.name + '/')

def delete_files(dir):
  for entry in os.scandir(dir):
    if entry.is_file():
      os.remove(dir + entry.name)
    else:
      delete_files(dir + entry.name + '/')

def split_all(source_dataset, train_dataset, test_dataset):
  shutil.rmtree(train_dataset)
  shutil.rmtree(test_dataset)
  shutil.copytree(source_dataset, train_dataset)
  shutil.copytree(source_dataset, test_dataset)
  delete_files(test_dataset)
  recur(source_dataset, train_dataset, test_dataset)

def main():
  to_split = input('Split Abnormality Dataset: Input 0\nsplit Condition Dataset: Input 1\n')

  if to_split == '0':
    split_all(os.environ['ABNORMALITY_DATASET'], os.environ['ABNORMALITY_TRAIN'], os.environ['ABNORMALITY_TEST'])
  elif to_split == '1':
    split_all(os.environ['CONDITION_DATASET'], os.environ['CONDITION_TRAIN'], os.environ['CONDITION_TEST'])
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to split')

if __name__ == "__main__":
  main()