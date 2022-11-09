import numpy as np
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class VEC_MLP:
    
    def __init__(self, layers_and_nodes: list):
        """initialize multilayer perceptron

        Args:
            layers_and_nodes (list): layers and neurons in that layer
        """
        self.layers = [None] * (len(layers_and_nodes) - 1)
        self.bias = 1.0
        weights_size = 0
        for i in range(1,len(layers_and_nodes)):
            weights_size += (layers_and_nodes[i-1] * layers_and_nodes[i])
        all_weights = np.random.uniform(-1000,1000,weights_size)
        w_idx = 0
        for i in range(1, len(layers_and_nodes)):
            layer = []
            bias = []
            nodes = [[]] * layers_and_nodes[i-1]
            for j in range(layers_and_nodes[i-1] * layers_and_nodes[i]):
                idx = j
                if j >= layers_and_nodes[i]:
                    idx = j % layers_and_nodes[i]
                if (j % layers_and_nodes[i-1]) == 0:
                    bias.append(self.bias)
                    nodes[idx].append(0)
            layer.append(bias)
            for n in nodes:
                t_n = []
                for k in range(len(n)):
                    t_n.append(all_weights[w_idx])
                    w_idx += 1
                layer.append(t_n)
            self.layers[i-1] = np.array(layer)
    
    def forward_pass(self, inputs):
        """forward pass

        Args:
            inputs : dataset input

        Returns:
            output: output getting after forward pass
        """
        outputs = inputs
        for layer in self.layers:
            inputs = np.hstack([np.ones(shape=(outputs.shape[0],1)), outputs])
            outputs = sigmoid(np.matmul(inputs, layer))
        return outputs
    
    def run(self,input_datas, desired_datas):
        """forward pass and calc error

        Args:
            input_datas: input data for forward pass
            desired_datas: desired output for calc error

        Returns:
            error: mae error
        """
        self.output = self.forward_pass(input_datas)
        mae = self.calc_mae(self.output, desired_datas)
        return mae[0]
    
    def get_weights(self):
        """get weights in mlp

        Returns:
            weights: weights
        """
        return self.layers
    
    def set_weights(self, new_weights):
        """set new weight

        Args:
            new_weights: new weights
        """
        self.layers = new_weights
    
    def calc_mae(self, predicted, desired):
        """get predict and desired for calc mae

        Args:
            predicted: predicted output
            desired: desired output

        Returns:
            error: mae
        """
        return sum(abs(desired - predicted)) / len(desired)