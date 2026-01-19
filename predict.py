import numpy as np
import pickle
import sys

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, x):
        self.x = x
        self.out = x @ self.weights + self.biases


class ReLU:
    def forward(self, x):
        self.x = x
        self.out = np.maximum(0, x)

with open('trained_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

l1 = Dense(3, 16)
a1 = ReLU()
l2 = Dense(16, 16)
a2 = ReLU()
l3 = Dense(16, 8)
a3 = ReLU()
l4 = Dense(8, 1)

l1.weights = model_data['l1_weights']
l1.biases = model_data['l1_biases']
l2.weights = model_data['l2_weights']
l2.biases = model_data['l2_biases']
l3.weights = model_data['l3_weights']
l3.biases = model_data['l3_biases']
l4.weights = model_data['l4_weights']
l4.biases = model_data['l4_biases']

def predict(a, b, c):
    test = np.array([[float(a), float(b), float(c)]], dtype=float)
    
    l1.forward(test)
    a1.forward(l1.out)
    l2.forward(a1.out)
    a2.forward(l2.out)
    l3.forward(a2.out)
    a3.forward(l3.out)
    l4.forward(a3.out)
    
    result = l4.out[0][0]
    return result

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <int1> <int2> <int3>")
        print("Example: python predict.py 2 1 7")
        sys.exit(1)
    
    try:
        a, b, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        
        prediction = predict(a, b, c)
        true_sum = a + b + c
        
        print(f"Input: [{a}, {b}, {c}]")
        print(f"True sum: {true_sum}")
        print(f"Model prediction: {prediction:.4f}")
        print(f"Rounded prediction: {round(prediction)}")
        
    except ValueError:
        print("Error: Please provide 3 valid integers")
        sys.exit(1)
