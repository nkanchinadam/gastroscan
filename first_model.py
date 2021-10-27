from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np

def main():
  model = Sequential()
  model.add(Flatten(input_shape=(352, 332, 3)))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(20, activation='relu'))
  model.add(Dense(5, activation='')) #replace with actual number of conditions

  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
  model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
  model.save('./model')

  scores = model.evaluate(x_test, y_test)
  print(model.metrics_names[1], scores[1] * 100)

if __name__ == '__main__':
  main()