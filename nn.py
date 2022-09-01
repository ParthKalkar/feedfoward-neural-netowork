from matrix import Matrix

class NeuralNetwork: 
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_h = Matrix(self.output_nodes, 1)


    def feedfoward(self, input): 
        hidden = Matrix.multiply(self.weights_ih, input)
        hidden.add(self.bias_h)
        
        