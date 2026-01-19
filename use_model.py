import numpy as np
import pickle

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, x):
        self.x = x
        self.out = x @ self.weights + self.biases

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        self.dx = dout @ self.weights.T


class ReLU:
    def forward(self, x):
        self.x = x
        self.out = np.maximum(0, x)

    def backward(self, dout):
        self.dx = dout * (self.x > 0)

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

print("[*] Model loaded successfully!")
print("\n" + "="*50)
print("Cross-Sum Neural Network Predictor")
print("="*50)

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

test_cases = [
    (2, 1, 7),
    (5, 3, 2),
    (9, 9, 9),
    (0, 0, 0),
    (4, 5, 6),
]

print("\nTesting the model:")
print("-" * 50)
for a, b, c in test_cases:
    true_sum = a + b + c
    predicted = predict(a, b, c)
    error = abs(true_sum - predicted)
    print(f"Input: [{a}, {b}, {c}] | True: {true_sum:2d} | Predicted: {predicted:.4f} | Error: {error:.4f}")

print("\n" + "="*50)
print("Interactive Mode")
print("="*50)
print("Enter 3 integers separated by spaces (or 'quit' to exit):")

while True:
    user_input = input("\nEnter 3 integers: ").strip()
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    try:
        numbers = list(map(int, user_input.split()))
        if len(numbers) != 3:
            print("Error: Please enter exactly 3 integers")
            continue
        
        a, b, c = numbers
        true_sum = a + b + c
        predicted = predict(a, b, c)
        
        print(f"\nInput: [{a}, {b}, {c}]")
        print(f"True sum: {true_sum}")
        print(f"Model prediction: {predicted:.4f}")
        print(f"Rounded prediction: {round(predicted)}")
        print(f"Error: {abs(true_sum - predicted):.4f}")
        
    except ValueError:
        print("Error: Please enter valid integers")
