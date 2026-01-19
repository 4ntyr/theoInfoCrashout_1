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


class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]


X_list = []
Y_list = []

for a in range(10):
    for b in range(10):
        for c in range(10):
            X_list.append([a, b, c])
            Y_list.append([a + b + c])

X = np.array(X_list, dtype=float)
Y = np.array(Y_list, dtype=float)

l1 = Dense(3, 16)
a1 = ReLU()
l2 = Dense(16, 16)
a2 = ReLU()
l3 = Dense(16, 8)
a3 = ReLU()
l4 = Dense(8, 1)

loss_fn = MSE()
lr = 0.001

for epoch in range(100000):

    l1.forward(X)
    a1.forward(l1.out)
    l2.forward(a1.out)
    a2.forward(l2.out)
    l3.forward(a2.out)
    a3.forward(l3.out)
    l4.forward(a3.out)

    loss = loss_fn.forward(l4.out, Y)

    dloss = loss_fn.backward()
    l4.backward(dloss)
    a3.backward(l4.dx)
    l3.backward(a3.dx)
    a2.backward(l3.dx)
    l2.backward(a2.dx)
    a1.backward(l2.dx)
    l1.backward(a1.dx)

    l1.weights -= lr * l1.dW
    l1.biases  -= lr * l1.db
    l2.weights -= lr * l2.dW
    l2.biases  -= lr * l2.db
    l3.weights -= lr * l3.dW
    l3.biases  -= lr * l3.db
    l4.weights -= lr * l4.dW
    l4.biases  -= lr * l4.db

    if epoch % 5000 == 0:
        predictions = l4.out
        errors = np.abs(predictions - Y)
        accuracy = np.mean(errors < 0.5) * 100
        print(f"Epoch {epoch:6d} | Loss: {loss:.6f} | Accuracy: {accuracy:.2f}%")

test = np.array([[2, 1, 7]], dtype=float)

l1.forward(test)
a1.forward(l1.out)
l2.forward(a1.out)
a2.forward(l2.out)
l3.forward(a2.out)
a3.forward(l3.out)
l4.forward(a3.out)

print("\nInput:", test)
print("True sum:", test.sum())
print("Model output:", l4.out[0][0])

model_data = {
    'l1_weights': l1.weights,
    'l1_biases': l1.biases,
    'l2_weights': l2.weights,
    'l2_biases': l2.biases,
    'l3_weights': l3.weights,
    'l3_biases': l3.biases,
    'l4_weights': l4.weights,
    'l4_biases': l4.biases,
}

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n[*] Model saved to 'trained_model.pkl'")
