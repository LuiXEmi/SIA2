import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cargar los datos desde un archivo CSV
file_path = "zoo.csv"  # Reemplaza esto con la ruta correcta
df = pd.read_csv(file_path)

# Dividir los datos en características (X) y etiquetas (y)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Cambiado a weighted
recall = recall_score(y_test, y_pred, average='weighted')  # Cambiado a weighted
f1 = f1_score(y_test, y_pred, average='weighted')  # Cambiado a weighted

# Imprimir las métricas
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall (Sensitivity): {recall}")
print(f"F1 Score: {f1}")

