from nn import NeuralNetwork

def setup(): 
    nn = NeuralNetwork(2,2,1)

    input = [1,0]

    output = nn.feedfoward(input)

    print(output)
