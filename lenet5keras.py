import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten


(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_valid = X_valid.reshape(10000, 28, 28, 1).astype('float32')

n = 10

y_train = keras.utils.to_categorical(y_train, n)
y_valid = keras.utils.to_categorical(y_valid, n)

model = Sequential()

#first convolutional layer, learns simple features
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28,1)))

#second conv layer, with polling and dropout
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

#dense hidden layer, with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#output layer
model.add(Dense(n, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid))