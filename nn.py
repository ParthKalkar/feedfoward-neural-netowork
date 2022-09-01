import math
import random
from matrix import Matrix
import numpy as np

class NeuralNetwork: 
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = np.array(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.array(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        self.bias_h.randomize()
        self.bias_o.randomize()


        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    

    def randomize(self): 
        for i in range(self.rows):
            for j in range(self.columns):
                self.data[i, j] = math.floor(random.randint(-1,1))
                
    def feedfoward(self, input): 
        hidden = np.mul(self.weights_ih, input)
        hidden.sum(self.bias_h)

        hidden.sigmoid()

        output = np.mul(self.weights_ho, hidden)
        output.sum(self.bias_o)
        output.sigmoid()


        print("Output", output)



        
        