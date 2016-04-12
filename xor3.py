import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD, Adam
from util.callback import WeightLogger

# We are going to output the XOR of 3 values. In this case we are calling XOR true if there is an odd number of values
# set and false for any other combination of values.
X = np.zeros((8, 3))
y = np.zeros(8)

# Train with the following truth table
X[0] = [0, 0, 0]
y[0] = 0.0
X[1] = [0, 0, 1]
y[1] = 1.0
X[2] = [0, 1, 0]
y[2] = 1.0
X[3] = [0, 1, 1]
y[3] = 0.0

X[4] = [1, 0, 0]
y[4] = 1.0
X[5] = [1, 0, 1]
y[5] = 0.0
X[6] = [1, 1, 0]
y[6] = 0.0
X[7] = [1, 1, 1]
y[7] = 1.0

model = Sequential()
# XOR cannot be done with a single layer. This model has a first layer with three inputs of dimension 2
model.add(Dense(4, input_dim=3))
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