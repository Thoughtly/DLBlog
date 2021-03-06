import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD, Adam
from util.callback import WeightLogger

# We are going to do a simple binary OR, so we need two inputs and a single output
X = np.zeros((4, 2))
y = np.zeros(4)

# Build the truth table for OR
X[0] = [0, 0]
y[0] = 0.0
X[1] = [0, 1]
y[1] = 1.0
X[2] = [1, 0]
y[2] = 1.0
X[3] = [1, 1]
y[3] = 1.0

model = Sequential()

# We make a single layer with one output. It takes in a 2D input.
model.add(Dense(1, input_dim=2))

# Hard sigmoid is not necessary, but it usually results in firm 1/0 output so convenient for binary
model.add(Activation('hard_sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=.99, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd)

# Train the model and log weights
model.fit(X, y, nb_epoch=100, batch_size=4, show_accuracy=True, verbose=1, callbacks=[WeightLogger()])

# print the truth table output for our original X
print model.predict(X)