import math
import numpy as np
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)


scores = np.array([ -0.54761371,  17.04850603,   4.86054302])
print(sigmoid_v(scores))