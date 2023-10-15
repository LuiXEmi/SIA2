import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, LeavePOut

# Cargar el archivo CSV
data = pd.read_csv('irisbin.csv')

# Extraer características (columnas 0 a 3) y etiquetas (columna 4)
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador de perceptrón multicapa
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print("Precisión en el conjunto de prueba:", accuracy)

# Validación leave-one-out
loo = LeaveOneOut()
accuracies_loo = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_loo.append(accuracy)

average_accuracy_loo = np.mean(accuracies_loo)
std_accuracy_loo = np.std(accuracies_loo)
print("Precisión promedio (leave-one-out):", average_accuracy_loo)
print("Desviación estándar (leave-one-out):", std_accuracy_loo)

# Validación leave-p-out (en este caso, p=5)
lpout = LeavePOut(p=5)
accuracies_lpout = []

for train_index, test_index in lpout.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_lpout.append(accuracy)

average_accuracy_lpout = np.mean(accuracies_lpout)
std_accuracy_lpout = np.std(accuracies_lpout)
print("Precisión promedio (leave-p-out):", average_accuracy_lpout)
print("Desviación estándar (leave-p-out):", std_accuracy_lpout)
