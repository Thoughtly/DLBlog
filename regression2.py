import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import random
from util.callback import WeightLogger
from keras.callbacks import EarlyStopping
import util.plot

# Generate a single y value given an x, a and b
def generate_y(x, a, b):
    return a*x + b
    
# Generate a bunch of xy pairs within the domain [min_x, max_x). These
# points will all fall along two lines
def generate_data(num_samples, min_x, max_x):
 
    # We are just making 2 single lines with 1 X value per sample
    X = np.zeros((num_samples, 1))

    # Two y values for each X, one for each line
    y = np.zeros((len(X), 2))

    for i in xrange(0, len(X)):
        # pick a random x value
        x = min_x + (max_x - min_x) * random.random()

        # generate each of the 2 y values for the given x
        y_val1 = generate_y(x, .5, 5)
        y_val2 = generate_y(x, 2, 7)

        # the one x value is then mapped to the two output values
        X[i][0] = x
        y[i] = [y_val1, y_val2]


    # Set aside 20% of the samples to use for testing
    train_size = int(num_samples*.8)
    
    # return training and test data in 2 different buckets
    return (X[0:train_size], y[0:train_size]), (X[train_size:], y[train_size:])

def main():

    # Generate our data, splitting it into two sets - training and test data. We'll use the training data
    # to train the model and then test it with the holdout test data
    (X_train, y_train), (X_test, y_test) = generate_data(2000, 1.0, 10.0)

    # Build our model
    model = Sequential()

    # The model consists of a single neuron, keras calls them dense layers. The activation function we
    # are using is linear (which is the default). The single layer has 2 neurons in it and an input
    # dimension of 1. The input_dim corresponds to the fact that we are passing in a single x per sample
    # and the 2 neurons are one for each of the y outputs.
    layers = [Dense(2, input_dim=1), Activation('linear')]

    # Add the layers to the model
    for layer in layers:
        model.add(layer)

    # Configure an optimizer used to minimize the loss function
    sgd = SGD(lr=0.1, decay=.01)

    # Compile our model
    model.compile(loss='mean_absolute_error', optimizer=sgd)

    try:
        # Go about actually training the model. We are going to run for a fixed max number of epochs (training steps)
        # unless we see that the validation data has converged on a minimum.
        model.fit(X_train, y_train, nb_epoch=1000, batch_size=100, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=25, mode='min'), WeightLogger()])
    except KeyboardInterrupt:
        pass

    # Now see how the trained model does against the holdout test data
    loss = model.evaluate(X_test, y_test, batch_size=100, verbose=1)
    print "Loss: " + str(loss)

    # As a quick test we also calculate the mean absolute error manually. Because the output is two
    # dimensional we actually are accumulating 2 errors, one for y1 and one for y2.
    sum_absolute_error = 0
    y_predictions = model.predict(X_test)
    for i, x in enumerate(X_test):
        y = y_test[i]
        y_model = y_predictions[i]
        sum_absolute_error += abs(y - y_model)

    # The error across all samples is the average of the y1 and y2 error
    sum_absolute_error /= len(X_test)
    print "Calculated loss: " + str((sum_absolute_error[0] + sum_absolute_error[1]) / 2)
        
if __name__ == '__main__':
    main()
