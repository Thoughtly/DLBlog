import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD, Adam, Adamax
import random
from util.callback import WeightLogger
from keras.callbacks import EarlyStopping
import util.plot

# Generate a single y value given an x
def generate_y(x):
    return .5*x + 5.0

# Generate a bunch of xy pairs within the domain [min_x, max_x). Values
# are then shifted by a random amount along a vector perpendicular
# to the generated line.
def generate_data(num_samples, min_x, max_x, standard_deviation=0):

    # We need a vector from the x = 0 to whatever point on the line
    # we generate. We need the y intercept for that.
    y_intercept = generate_y(0)

    # We are just making a single line with 1 X value per sample
    X = np.zeros((num_samples, 1))

    # One y value for each X
    y = np.zeros(len(X))

    for i in xrange(0, len(X)):

        # We need to randomize the output x/y pair order. If not, when
        # we split into data and test sets we'll actually be grabbing
        # only test data for large x values. So generate a random X
        # instead of generating them in order
        x = min_x + (max_x - min_x) * random.random()

        # calculate the corresponding y value for the given x
        y_val = generate_y(x)

        # generate a vector from the x/y point back to x=0, y_intercept
        vector_to_y_intercept = [x, y_val - y_intercept]

        # generate a random length vector perpendicular to vector_to_y_intercept
        r = perpendicular_vector(vector_to_y_intercept, np.random.normal(0, standard_deviation) if standard_deviation > 0 else 0)

        # the new point is the original generated point plus the perpendicular offset
        point = r + [x, y_val]

        # save the point in teh generated data
        X[i][0] = point[0]
        y[i] = point[1]

    # Set aside 20% of the samples to use for testing
    train_size = int(num_samples*.8)
    
    # return training and test data in 2 different buckets
    return (X[0:train_size], y[0:train_size]), (X[train_size:], y[train_size:])

def perpendicular_vector(v, l):

    # nothing is perpendicular to a vector with no length
    if v[0] == 0 and v[1] == 0:
        return v

    # generate a perpendicular vector
    p = np.array([-v[1], v[0]])

    # get the scale for a vector of length l
    norm = np.linalg.norm(p)
    scale = l / norm    

    # scale the perpendicular vector
    p[0] *= float(scale)
    p[1] *= float(scale)
    
    return p
    

def main():

    # Generate our data, splitting it into two sets - training and test data. We'll use the training data
    # to train the model and then test it with the holdout test data
    (X_train, y_train), (X_test, y_test) = generate_data(2000, 1.0, 10.0, 0.2)

    # Uncomment to plot the output of data generation
    util.plot.scatter(X_train, y_train)

    # Build our model
    model = Sequential()

    # The model consists of a single neuron, Keras calls them dense layers. The activation function we
    # are using is linear (which is the default)
    layers = [Dense(1, input_dim=1), Activation('linear')]

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

    # As a quick test we also calculate the mean absolute error manually.
    sum_absolute_error = 0
    y_predictions = model.predict(X_test)
    for i, x in enumerate(X_test):
        y = y_test[i]
        y_model = y_predictions[i]
        sum_absolute_error += abs(y - y_model)


    print "Calculated loss: " + str(sum_absolute_error / len(X_test))

if __name__ == '__main__':
    main()
