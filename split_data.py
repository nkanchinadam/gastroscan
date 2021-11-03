import os, random

ABNORMALITY_DATASET = './AbnormalityDataset'
CONDITION_DATASET = './ConditionDataset'

def split(image_folder):
  images = []
  for image in image_folder:
    images.append(image)
  random.shuffle(images)
  return images[:int(0.7 * len(images))], images[int(0.7 * len(images)):]

def main():
  file_num = input('Split Abnormality Dataset: Input 0\nsplit Condition Dataset: Input 1\n')

  filepaths = []
  if file_num == '0':
    filepaths = open(ABNORMALITY_DATASET, 'r').readlines()
  elif file_num == '1':
    filepaths = open(CONDITION_DATASET, 'r').readlines()
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to split')




if __name__ == "__main__":
  main()