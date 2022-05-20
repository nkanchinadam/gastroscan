from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import paths
import data_preparation as dp

def main():
  to_create = input('Create and Train Abnormality Model: Input 0\nCreate and Train Condition Model: Input 1\n')
  model_name = input('Model Name: ')

  labels = None
  x_train = None
  y_train = None
  x_test = None
  y_test = None
  if to_create == '0':
    labels = dp.get_labels(paths.ABNORMALITY_LABELS)
    x_train, y_train = dp.get_dataset(paths.ABNORMALITY_TRAIN, labels)
    x_test, y_test = dp.get_dataset(paths.ABNORMALITY_TEST, labels)
  elif to_create == '1':
    labels = dp.get_labels(paths.CONDITION_LABELS)
    x_train, y_train = dp.get_dataset(paths.CONDITION_TRAIN, labels)
    x_test, y_test = dp.get_dataset(paths.CONDITION_TEST, labels)
  else:
    raise ValueError('Input either a 0 or 1 to indicate which dataset to create and train a model on')

  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(dp.HEIGHT, dp.WIDTH, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dense(20, activation='relu'))
  model.add(Dense(len(labels) // 2, activation='sigmoid'))

  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
  model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
  model.save('./models/' + ('abnormalities' if to_create == '0' else 'conditions') + '/' + model_name)

if __name__ == '__main__':
  main()