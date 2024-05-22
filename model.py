import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# based off https://www.youtube.com/watch?v=w8yWXqWQYmU&t=563s&pp=ygUWbmV1cmFsIGVudG93a3Igc2NyYXRjaA%3D%3D
''' I made a feedforward neural network from scrath in Python to learn how it works.'''


learning_rate = 0.01

W0 = np.random.randn(784, 10) * np.sqrt(1.0 / 784)
W1 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)

b0 = np.random.randn(1, 10)
b1 = np.random.randn(1, 10)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def relu(x):
    return np.maximum(0, x)

def cross_entropy_loss(y, probs):
    log_probs = -np.log(probs)
    return sum(log_probs * y) 

def d_relu(x):
    return np.where(x > 0, 1, 0)

def train(xs, ys):
    l = 0
    for col in range(len(ys)):
        z1, A1, z2 = forward(xs[:,col].reshape(784,1).transpose())
        logits = z2.transpose()
        probs = softmax(logits)
        y_one_hot = np.eye(10)[ys[col].astype(int)]
        loss = cross_entropy_loss(y_one_hot.reshape(10,1), probs.reshape(10,1))
        l += loss

        dW0, db0, dW1, db1 = backward(xs[:,col].reshape(784,1).transpose(), z1, A1, probs.reshape(10,1) - y_one_hot.reshape(10,1))
        update_params(dW0, db0, dW1, db1)

    return l / len(ys)

def forward(A0)->np.ndarray:
    z1 = A0.dot(W0) + b0
    A1 = relu(z1)
    z2 = A1.dot(W1) + b1
    return z1, A1,z2

def backward(A0, z1, A1, diff):
    dW1 = A1.dot(diff)
    db1 = np.sum(diff, axis=0, keepdims=True)

    dA1 = diff.transpose().dot(W1.transpose())
    dZ1 = dA1 * d_relu(z1)

    dW0 = A0.transpose().dot(dZ1)
    db0 = np.sum(dZ1, axis=0, keepdims=True)
    return dW0, db0, dW1, db1

def update_params(dW0, db0, dW1, db1):
    global W0, b0, W1, b1
    W0 -= learning_rate * dW0
    b0 -= learning_rate * db0
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

def read_training_data(training_data)->np.ndarray:
    xs = np.zeros((784, training_data.shape[0]), dtype=np.float64)
    ys = np.zeros(training_data.shape[0], dtype=np.float64)
    for i, row in enumerate(training_data):
        ys[i] = row[0]
        xs[:,i] = np.array(list(row[1:])).transpose()
        i+=1

    return xs, ys

def read_test_data(test_data)->np.ndarray:
    xs = np.zeros((784, test_data.shape[0]), dtype=np.float64)
    for i, row in enumerate(test_data):
        xs[:,i] = np.array(list(row)).transpose()
        i+=1

    return xs

def predict(x):
    _, _, logits= forward(x)
    probs = softmax(logits)
    return probs

def main():
    training_data = pd.read_csv("data/train.csv").values
    test_data = pd.read_csv("data/test.csv").values
    x_test = read_test_data(test_data)
    xs, ys = read_training_data(training_data)
    xs = xs / 255.0
    x_test = x_test / 255.0

    for x in range(100):
        loss = train(xs, ys)
        if (x % 1 == 0):
            print(f"Loss: {loss}")

    for i in range(0,10):
        current_image = x_test[:,i].reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

        x = x_test[:,i].reshape(784,1).transpose()
        probs = predict(x)
        print(f"Prediction: {np.argmax(probs)}")

if __name__ == "__main__":
    main()

