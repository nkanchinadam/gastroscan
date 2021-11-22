from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np

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

def main():
  model = Sequential()
  model.add(Flatten(input_shape=(352, 332, 3)))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(20, activation='relu'))
  model.add(Dense(5, activation='simgoid')) #replace with actual number of conditions

  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
  #model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))   -- replace x_train, y_train, x_test, and y_test with actual dataset objects once they are created
  model.save('./model')

  #scores = model.evaluate(x_test, y_test)
  #print(model.metrics_names[1], scores[1] * 100)

if __name__ == '__main__':
  main()