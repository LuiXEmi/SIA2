import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Definir la clase de la red neuronal
class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        # Inicializar la arquitectura de la red
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # Inicializar pesos y sesgos de las capas ocultas y de salida
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        for i in range(1, len(layer_sizes)):
            w = np.random.rand(layer_sizes[i - 1], layer_sizes[i])
            b = np.zeros((1, layer_sizes[i]))
            self.weights.append(w)
            self.biases.append(b)

    # Propagación hacia adelante
    def feedforward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            activations.append(a)
        return activations

    # Retropropagación y ajuste de pesos
    def backpropagation(self, x, y, learning_rate):
        activations = self.feedforward(x)
        deltas = [activations[-1] - y]
    
        for i in range(len(activations) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * sigmoid_derivative(activations[i])
            deltas.insert(0, delta)
    
        for i in range(len(self.weights)):
            # Cambia la forma de activations[i] para que sea 2D (1, 2)
            activations_i_2d = activations[i].reshape(1, -1)
            self.weights[i] -= learning_rate * np.dot(activations_i_2d.T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)


    # Entrenamiento de la red
    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            for i in range(len(X)):
                self.backpropagation(X[i], y[i], learning_rate)

    # Predicción
    def predict(self, x):
        return self.feedforward(x)[-1]

# Cargar el conjunto de datos
data = pd.read_csv('concentlite.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Definir la arquitectura de la red
input_size = X.shape[1]
hidden_layer_sizes = [8, 8]  # Puedes ajustar la cantidad y tamaño de las capas ocultas
output_size = 1

# Inicializar y entrenar la red
nn = NeuralNetwork(input_size, hidden_layer_sizes, output_size)
nn.train(X, y, epochs=1000, learning_rate=0.01)

# Hacer predicciones
predictions = [nn.predict(x) for x in X]

# Dibujar el gráfico de dispersión de las predicciones
plt.scatter(X[:, 0], X[:, 1], c=np.round(predictions), cmap='coolwarm')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Clasificación con Perceptrón Multicapa')
plt.show()
