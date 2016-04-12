import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, Adam, Adamax
import random
import math
import matplotlib.pyplot as plot
import itertools
from util.callback import WeightLogger
from keras.callbacks import EarlyStopping, TensorBoard
import util.plot
from random import shuffle

def vector_on_circle(r, r_size, offset, off_x, off_y, gap):

    theta = random.random()*2*math.pi
    l = r * r_size
    if l <= 0:
        l -= gap
    else:
        l += gap

    l += offset
    # to generate a circle x1^2 + x2^2 must equal r
    x1 = math.cos(theta)
    x2 = math.sin(theta)

    mag = math.sqrt(x1*x1 + x2*x2)

    x1 *= l / mag
    x2 *= l / mag

    return (x1 + off_x), (x2 + off_y)

def generate_data(data_size, num_classes, num_circles):

    X = np.zeros((data_size, 2))
    y = np.zeros((data_size, num_circles*num_classes))

    plot_x = []
    for i in xrange(0, num_classes):
        plot_x.append([[], []]*num_circles)
    print plot_x

    r_size = 10
    print "generating data " + str(len(X))

    max_value = None
    offset_x = 2*num_classes*r_size
    offset_y = 2*num_classes*r_size
    gap = r_size

    ordered_index = 0
    indexes = range(len(y))
    random.shuffle(indexes)
    for circle in xrange(0, num_circles):

        for _ in xrange(0, data_size/num_circles):

            index = indexes[ordered_index]
            r = random.randint(0, num_classes - 1)

            x1,x2 = vector_on_circle(r + 1, r_size, np.random.normal(0, .125*r_size), offset_x, offset_y, gap)
            max_value = max(max_value, x1, x2)

            X[index][0] = x1
            X[index][1] = x2
            y[index][r + circle*num_classes] = 1

            plot_x[r][0].append(x1)
            plot_x[r][1].append(x2)

            ordered_index += 1

        offset_x += 1.5*num_classes*(r_size + 1)
        offset_y += 1.5*num_classes*(r_size + 1)

    for c in plot_x:
        c[0] = np.array(c[0]) / max_value
        c[1] = np.array(c[1]) / max_value

    for x in X:
        x[0] /= max_value
        x[1] /= max_value

    colors = itertools.cycle(["red", "blue", "green", "black", "yellow", "grey"])
    plot.figure(figsize=(1040/80, 800/80))
    for i in xrange(0, num_classes):
        plot.scatter(plot_x[i][0], plot_x[i][1], color=next(colors))

#    plot.show()

    train_size = int(data_size*.8)

    return (X[0:train_size], y[0:train_size]), (X[train_size:], y[train_size:])


def main():
    
    num_classes = 5
    num_circles = 5
    data_size = 25000

    (X_train, y_train), (X_test, y_test) = generate_data(data_size, num_classes, num_circles)

    print X_train
    print y_train
    print X_test
    print y_test

    print "configuring model"
    model = Sequential()

    layers = [Dense(int(num_circles*num_classes), input_shape=(2,)), Activation('sigmoid'),
              Dense(num_circles*num_classes), Activation('softmax')]
              
    for layer in layers:
        model.add(layer)

    print "compiling model"

    adamax = Adamax(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='binary_crossentropy', optimizer=adamax)

    print "training model"
    
    try:
        model.fit(X_train, y_train, nb_epoch=500, batch_size=100, show_accuracy=True, verbose=1, validation_split=0.2, callbacks=[])
    except KeyboardInterrupt:
        pass

    loss = model.evaluate(X_test, y_test, batch_size=100, verbose=1)

    num_errors = 0.0
    error_distance = 0.0
    y_predictions = model.predict(X_test)
    for i, x in enumerate(X_test):
        y = np.argmax(y_test[i])
        y_model = np.argmax(y_predictions[i])
        if y_model != y:
            num_errors += 1
            error_distance += abs(y_model - y)
        print x, y, y_model

    print "accuracy: " + str((len(X_test) - num_errors) / len(X_test))
    print "average error distance: " + str(error_distance / num_errors)
    print "loss: " + str(loss)

main()