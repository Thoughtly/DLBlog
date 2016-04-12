import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD, Adam
from util.callback import WeightLogger

# We are going to train a model to perform either an and/or/xor based upon the first
# value in a 3D input. Value 0 identifies the operation. Values 1/2 are the inputs. Value
# 3 is the output.
X = np.zeros((12, 3))
y = np.zeros(12)

# Set up the three truth tables

# And
X[0] = [0, 0, 0]
y[0] = 0.0
X[1] = [0, 0, 1]
y[1] = 0.0
X[2] = [0, 1, 0]
y[2] = 0.0
X[3] = [0, 1, 1]
y[3] = 1.0

# Or
X[4] = [1, 0, 0]
y[4] = 0.0
X[5] = [1, 0, 1]
y[5] = 1.0
X[6] = [1, 1, 0]
y[6] = 1.0
X[7] = [1, 1, 1]
y[7] = 1.0

# Xor
X[8] = [2, 0, 0]
y[8] = 0.0
X[9] = [2, 0, 1]
y[9] = 1.0
X[10] = [2, 1, 0]
y[10] = 1.0
X[11] = [2, 1, 1]
y[11] = 0.0

model = Sequential()

# This model requires two layers. The first has 3D input and 4 neurons.
model.add(Dense(4, input_dim=3, init='zero'))
model.add(Activation('sigmoid'))

# The second and output layer has a single output neuron for the binary classification
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

# Gradient descent. We don't use the defaults because for a simple model like this the Keras defaults don't work
sgd = SGD(lr=0.1, decay=1e-6, momentum=.9)

model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")

# Train the model and log weights
model.fit(X, y, nb_epoch=1000, batch_size=4, show_accuracy=True, verbose=1, callbacks=[WeightLogger()])

# print the truth table output for our original X
print model.predict(X)
