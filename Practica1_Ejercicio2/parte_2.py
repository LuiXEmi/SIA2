import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Lista de archivos de datos
data_files = ["spheres2d10.csv", "spheres2d50.csv", "spheres2d70.csv"]

# Número de particiones
num_partitions = 10

# Porcentaje de datos para entrenamiento y prueba
train_percentage = 0.8
test_percentage = 0.2

for data_file in data_files:
    # Cargar el conjunto de datos desde el archivo CSV
    data = pd.read_csv(data_file)

    for partition in range(num_partitions):
        # Dividir el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=test_percentage, random_state=partition)

        # Crear un modelo de perceptrón simple
        perceptron = Perceptron()

        # Entrenar el modelo en los datos de entrenamiento
        perceptron.fit(X_train, y_train)

        # Realizar predicciones en los datos de prueba
        y_pred = perceptron.predict(X_test)

        # Calcular la precisión del modelo en esta partición
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f'Archivo: {data_file} - Partición {partition + 1} - Precisión: {accuracy}')
