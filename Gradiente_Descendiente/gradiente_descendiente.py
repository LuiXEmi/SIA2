import numpy as np
import matplotlib.pyplot as plt

# Función de costo
def funcion_costo(X1, X2):
    return 10 - np.exp(-(X1**2 + 3*X2**2))

# Gradiente de la función de costo
def gradiente(X1, X2):
    df_dx1 = 2*X1 * np.exp(-(X1**2 + 3*X2**2))
    df_dx2 = 6*X2 * np.exp(-(X1**2 + 3*X2**2))
    return df_dx1, df_dx2

# Descenso de gradiente
def descenso_gradiente(funcion_costo, gradiente, tasa_aprendizaje, iteraciones, x1_inicial, x2_inicial):
    x1_actual = x1_inicial
    x2_actual = x2_inicial
    historial_x1 = []
    historial_x2 = []
    historial_costo = []

    for i in range(iteraciones):
        df_dx1, df_dx2 = gradiente(x1_actual, x2_actual)
        x1_actual = x1_actual - tasa_aprendizaje * df_dx1
        x2_actual = x2_actual - tasa_aprendizaje * df_dx2
        historial_x1.append(x1_actual)
        historial_x2.append(x2_actual)
        historial_costo.append(funcion_costo(x1_actual, x2_actual))

    return x1_actual, x2_actual, historial_x1, historial_x2, historial_costo

# Parámetros
tasa_aprendizaje = 0.1
iteraciones = 1000
x1_inicial = 1
x2_inicial = 1

resultado_x1, resultado_x2, historial_x1, historial_x2, historial_costo = descenso_gradiente(funcion_costo, gradiente, tasa_aprendizaje, iteraciones, x1_inicial, x2_inicial)

# Crear una malla para la superficie
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = funcion_costo(X1, X2)

# Graficar la función de costo
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Costo')
ax.set_title('Función de Costo')

# Graficar el progreso del descenso de gradiente
ax.plot(historial_x1, historial_x2, historial_costo, 'ro-', label='Descenso de Gradiente')
ax.legend()

plt.show()

print(f"El valor mínimo de la función se encuentra en X1 = {resultado_x1}, X2 = {resultado_x2}")
