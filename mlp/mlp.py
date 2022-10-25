import numpy as np
from mlp.neuron import Neuron
from utils.datareader import denormalize

class MLP:
    """Multilayer Perceptron
    """
    def __init__(self, layers_and_nodes: list):
        """MLP Initialize

        Args:
            layers_and_nodes (list): list of layers and nodes
        """
        self.layers = {}
        self.layers_and_nodes = layers_and_nodes
        self.sse = []
        self.linear_weights = []
        for i in range(len(layers_and_nodes)):
            t_layers = []
            for j in range(layers_and_nodes[i]):
                if i == 0:
                    t_layers.append(Neuron(0))
                else:
                    t_layers.append(Neuron(layers_and_nodes[i-1]))
            if i == 0:
                self.layers['INPUT_LAYER'] = np.array(t_layers)
            elif i == len(layers_and_nodes) - 1:
                self.layers['OUTPUT_LAYER'] = np.array(t_layers)
            else:
                key = 'HIDDEN_LAYER' + str(i)
                self.layers[key] = np.array(t_layers)
    
    def forward_pass(self, input_data: list):
        """forward pass

        Args:
            input_data (list): list of input data
        """
        if len(input_data) != len(self.layers['INPUT_LAYER']):
            print('input data not match input neurons')
            return
        else:
            prev_layer_output = []
            for i in range(len(input_data)):
                self.layers['INPUT_LAYER'][i].set_input(input_data[i])
                prev_layer_output.append(input_data[i])
            for layer, neurons in self.layers.items():
                if 'HIDDEN_LAYER' in layer or 'OUTPUT_LAYER' in layer:
                    o_from_hidden = []
                    for neuron in neurons:
                        output = neuron.calc_output(prev_layer_output)
                        o_from_hidden.append(output)
                    prev_layer_output = o_from_hidden

    def calc_error(self, desired_output: list):
        """calculate error between predicted and desired

        Args:
            desired_output (list): desired output

        Returns:
            float: error
        """
        error = 0
        for i in range(len(self.layers['OUTPUT_LAYER'])):
            error += np.abs((self.layers['OUTPUT_LAYER'][i].get_output() - desired_output[i]))
        return error, desired_output
    
    def run(self, dataset):
        """forward pass and calculate error

        Args:
            dataset (list): dataset

        Returns:
            float: mean absolute error after feed dataset of this model
        """
        se = 0.0
        for train_data in dataset:
            self.forward_pass(train_data['INPUT'])
            error, desired_output = self.calc_error(train_data['OUTPUT'])
            se += error
        mae = se / len(dataset)
        return mae
    
    def run_show(self, dataset, min_o, max_o):
        """forward pass and calculate error with show result as confusion matrix

        Args:
            dataset (list): dataset

        Returns:
            float: mean absolute error after feed dataset of this model
        """
        se = 0.0
        for train_data in dataset:
            self.forward_pass(train_data['INPUT'])
            error,desired_output = self.calc_error(train_data['OUTPUT'])
            se =+ error
            predicted = self.layers['OUTPUT_LAYER'][0].get_output()
            desired = desired_output[0]
            # denorm
            # predicted = denormalize(predicted, min_o, max_o)
            # desired = denormalize(desired, min_o, max_o)
            print('Predicted:', predicted, 'Desired:', desired)
        mae = se / len(dataset)
        return mae
    
    def get_weights(self):
        """return weights of all neurons

        Returns:
            list: weights of all neurons
        """
        weights = []
        for layer, neurons in self.layers.items():
            if 'HIDDEN_LAYER' in layer or 'OUTPUT_LAYER' in layer:
                for neuron in neurons:
                    weights.append(neuron.get_weights())
        linear_weights = []
        for c_node in weights:
            for c in c_node:
                linear_weights.append(c)
        self.linear_weights = linear_weights
        return linear_weights
    
    def set_new_weights(self, weights: list):
        """set new weights of all neurons

        Args:
            weights (list): 
        """
        linear_weights = weights
        for layer, neurons in self.layers.items():
            if 'HIDDEN_LAYER' in layer or 'OUTPUT_LAYER' in layer:
                for neuron in neurons:
                    new_weights = []
                    for l in range(neuron.prev_layer_neurons):
                        new_weights.append(linear_weights.pop(0))
                    neuron.set_weights(np.array(new_weights))
        self.linear_weights = weights