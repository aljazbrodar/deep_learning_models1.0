import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import SGD
from matplotlib import pyplot as plt

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()


X_train = X_train.reshape(60000,784).astype('float32')
X_valid = X_valid.reshape(10000,784).astype('float32')

X_train/255
X_valid/255

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='SGD', loss='categorical_crossentropy')

model.fit(X_train, y_train, batch_size = 128, epochs = 10,verbose = 1, validation_data=(X_valid, y_valid))

model.summary


