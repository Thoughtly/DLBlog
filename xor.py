import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from util.callback import WeightLogger

# We are going to do a simple binary XOR, so we need two inputs and a single output
X = np.zeros((4, 2))
y = np.zeros(4)

# Build the truth table for XOR
X[0] = [0, 0]
y[0] = 0.0
X[1] = [0, 1]
y[1] = 1.0
X[2] = [1, 0]
y[2] = 1.0
X[3] = [1, 1]
y[3] = 0.0

model = Sequential()

# XOR cannot be done with a single layer. This model has a first layer with two inputs of dimension 2
model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))

# The second layer is a single binary output
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

# Gradient descent. We don't use the defaults because for a simple model like this the Keras defaults don't work
sgd = SGD(lr=0.1, decay=1e-6, momentum=.99)
 
model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")

# Train the model and log weights
model.fit(X, y, nb_epoch=1000, batch_size=4, show_accuracy=True, verbose=1, callbacks=[WeightLogger()])

# print the truth table output for our original X
print model.predict(X)