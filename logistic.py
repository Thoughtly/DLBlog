import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import random
from util.callback import WeightLogger
import util.plot

# Generate a bunch of 2D point to classification pairs. We are doing a binary classification
# so one of the classes gets marked with a 1, the other a 0. Points are generated to be around
# the point 1,3 or 1,1 and put into class 1 or 0 respectively.
def generate_data(data_size):

    # Using 2D points
    X = np.zeros((data_size, 2))

    # Mapped to a single binary output
    y = np.zeros((data_size, 1))

    # We need to make sure the output data is in a random order so the train/test split is representative of all data
    indexes = range(len(y))
    random.shuffle(indexes)

    for ordered_index in xrange(0, data_size):
        # Grab a randomly ordered index
        index = indexes[ordered_index]

        # Use 1,3 for half of the points
        if random.random() < .5:
            x1 = np.random.normal(1, .25)
            x2 = np.random.normal(3, .25)
            y_val = 1
        # and 1,1 for the other hald
        else:
            x1 = np.random.normal(1, .25)
            x2 = np.random.normal(1, .25)
            y_val = 0

        X[index] = [x1, x2]
        y[index] = y_val

    # Set aside 20% of the samples to use for testing
    train_size = int(data_size*.8)

#    util.plot.scatter(X)
    # return training and test data in 2 different buckets
    return (X[0:train_size], y[0:train_size]), (X[train_size:], y[train_size:])


def main():

    data_size = 200

    (X_train, y_train), (X_test, y_test) = generate_data(data_size)

    model = Sequential()

    # We have a single neural layer with 1 neuron. It takes in 2D samples.
    layers = [Dense(1, input_shape=(2,)), Activation('sigmoid')]

    for layer in layers:
        model.add(layer)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=.9)

    # We have two classes we are trying to put the data into, so binary cross entropy is a fair choice.
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    try:
        # Go about actually training the model. We are going to run for a fixed max number of epochs (training steps)
        # unless we see that the validation data has converged on a minimum.
        model.fit(X_train, y_train, nb_epoch=500, batch_size=100, show_accuracy=True, verbose=1, validation_split=0.2, callbacks=[WeightLogger()])
    except KeyboardInterrupt:
        pass

    # Try the model out on the test data
    loss = model.evaluate(X_test, y_test, batch_size=100, verbose=1)

    num_classification_errors = 0.0
    y_predictions = model.predict(X_test)
    for i, x in enumerate(X_test):
        y = y_test[i]
        y_model = round(y_predictions[i])

        # Simply counting classification errors
        num_classification_errors += 1 if y_model != y else 0
        print x, y, y_model

    print "accuracy: " + str((len(X_test) - num_classification_errors) / len(X_test))
    print "loss: " + str(loss)

main()