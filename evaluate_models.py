from keras.models import load_model
import paths
import data_preparation as dp

def main():
  to_load = input('Evaluate Abnormality Model: Input 0\nEvaluate Condition Model: Input 1\n')
  model_name = input('Model Name: ')

  model = None
  x_train = None
  x_test = None
  y_train = None
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
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a model on')

  train_scores = model.evaluate(x_train, y_train)
  test_scores = model.evaluate(x_test, y_test)

  print('Accuray Percentages for ' + ('Abnormality' if to_load == '0' else 'Condition') + ' Model "' + model_name + '"')
  print('Training Data ' + model.metrics_names[1] + ': ' + str(train_scores[1] * 100) + '%')
  print('Testing Data ' + model.metrics_names[1] + ': ' + str(test_scores[1] * 100) + '%')

if __name__ == '__main__':
  main()