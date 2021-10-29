ABNORMALITY_FILEPATHS = './filepaths/abnormality_filepaths.py'
CONDITION_FILEPATHS = './filepaths/condition_filepaths.py'

def main():
  file_num = input('Split Abnormality Dataset: Input 0\nsplit Condition Dataset: Input 1\n')

  filepaths = []
  if file_num == '0':
    filepaths = open(ABNORMALITY_FILEPATHS, 'r').readlines()
  else if file_num == '1':
    filepaths = open(CONDITION_FILEPATHS, 'r').readlines()
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to split')

  for filepath in filepaths:
    


if __name__ == "__main__":
  main()