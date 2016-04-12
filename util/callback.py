from keras.callbacks import Callback
import numpy as np

# Extends the Callback class so that we can get updates as the model is being trained.
class WeightLogger(Callback):

    # Every epoch we will print out all the layer weights
    def on_epoch_end(self, epoch, logs={}):
        self.print_weights(self.model.layers)

    # Print each layer's input weights. This code is a bit awkward because by default keras returns a layer's
    # output weights. The bulk of the effort here is to track and flip those.
    def print_weights(self, layers):

        # Print layer by layer
        for layer_index, layer in enumerate(layers):

            # Get the forward weights for this layer. These are weights from layer n to layer n+1
            neuron_forward_weights = layer.get_weights()

            # Each neuron is an array of FORWARD weights. We want input weights, so have to look at the previous layer
            input_weights = self.get_input_weights(neuron_forward_weights)

            # Print if we have any weights (activation layers for example do not have weights)
            if input_weights is not None and len(input_weights) > 0:

                print "Layer " + str(layer_index)

                for neuron_index, neuron_weights in enumerate(input_weights):
                    print "\tNeuron " + str(neuron_index) + ") " + ", ".join( ["w[" + str(weight_index) +"]=" + str(weight) for weight_index, weight in enumerate(neuron_weights[0:-1])]) + ", w[b]=" + str(neuron_weights[-1])

    # Flip the output weights to input weights. Just a matter of reorganizing the weights so they are grouped
    # based on the neuron they feed into rather than the neuron they come out of.
    def get_input_weights(self, neuron_forward_weights):

        weights = None

        # The weights are broken into 2 chunks, first the inputs then the bias. Each is handled seperately below.
        if len(neuron_forward_weights) == 2:

            # Split into forward weights and bias weights
            forward_weights, bias_weights = [w for w in neuron_forward_weights]

            # If there are any forward weights we can map them to input weights and return them
            if len(forward_weights) > 0:

                num_input_weights = len(forward_weights)

                # every neuron should have the same number of weights, so just look at the first one as a representative
                # example.
                num_output_neurons = len(forward_weights[0])

                # allocate a bunch of empty arrays, one for each output neuron. Each output neuron will then
                # have a bunch (num_input_weights) of input weights
                weights = [[] for _ in xrange(num_output_neurons)]

                # Go through all of the n+1th layer neurons and collect the weights that point at each
                for output_neuron_index in xrange(0, num_output_neurons):

                    if len(weights[output_neuron_index]) == 0:
                        weights[output_neuron_index] = np.zeros(num_input_weights + 1)

                    for input_weight_index in xrange(0, num_input_weights):
                        weights[output_neuron_index][input_weight_index] = forward_weights[input_weight_index][output_neuron_index]

            # collect all the bias weights
            for bias_index, bias in enumerate(bias_weights):
                weights[bias_index][-1] = bias

        return weights