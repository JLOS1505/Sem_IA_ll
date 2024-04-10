import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, capas, beta=0.9):
        # Inicializa la red neuronal con el número de neuronas en cada capa
        self.capas = capas
        # Inicializa los pesos con valores aleatorios y los sesgos a cero
        self.pesos = [np.random.randn(capas[i], capas[i + 1]) for i in range(len(capas) - 1)]
        self.sesgo = [np.zeros((1, capas[i + 1])) for i in range(len(capas) - 1)]
        # Inicializa los parámetros de momento
        self.v_pesos = [np.zeros((capas[i], capas[i + 1])) for i in range(len(capas) - 1)]
        self.v_sesgo = [np.zeros((1, capas[i + 1])) for i in range(len(capas) - 1)]
        self.beta = beta

    def sigmoid(self, x):
        # Define la función de activación sigmoide
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        # Calcula la derivada de la función sigmoide
        return x * (1 - x)

    def forward(self, X):
        # Realiza la propagación hacia adelante
        self.activaciones = [X]
        self.valores_z = []

        for i in range(len(self.capas) - 1):
            # Calcula los valores ponderados y las activaciones para cada capa
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.sesgo[i]
            a = self.sigmoid(z)
            # Almacena los valores para su uso posterior en la retropropagación
            self.valores_z.append(z)
            self.activaciones.append(a)

    def backward(self, X, y, tasa_aprendizaje):
        # Realiza la retropropagación para ajustar los pesos y los sesgos
        errores = [y - self.activaciones[-1]]
        deltas = [errores[-1] * self.derivada_sigmoid(self.activaciones[-1])]

        for i in range(len(self.capas) - 2, 0, -1):
            # Calcula el error y el delta para cada capa oculta
            error = deltas[-1].dot(self.pesos[i].T)
            delta = error * self.derivada_sigmoid(self.activaciones[i])
            errores.append(error)
            deltas.append(delta)

        for i in range(len(self.capas) - 2, -1, -1):
            # Calcula el gradiente de los pesos y los sesgos
            gradiente_pesos = self.activaciones[i].T.dot(deltas[len(self.capas) - 2 - i])
            gradiente_sesgo = np.sum(deltas[len(self.capas) - 2 - i], axis=0, keepdims=True)
            # Actualiza el momento
            self.v_pesos[i] = self.beta * self.v_pesos[i] + (1 - self.beta) * gradiente_pesos
            self.v_sesgo[i] = self.beta * self.v_sesgo[i] + (1 - self.beta) * gradiente_sesgo
            # Actualiza los pesos y los sesgos utilizando el momento
            self.pesos[i] += self.v_pesos[i] * tasa_aprendizaje
            self.sesgo[i] += self.v_sesgo[i] * tasa_aprendizaje

    def train(self, X, y, epocas, tasa_aprendizaje):
        # Entrena la red durante un número específico de épocas
        for epoca in range(epocas):
            # Realiza la propagación hacia adelante y la retropropagación en cada época
            self.forward(X)
            self.backward(X, y, tasa_aprendizaje)

    def predecir(self, X):
        # Realiza una predicción utilizando la red neuronal entrenada
        self.forward(X)
        return np.round(self.activaciones[-1])

# Carga el conjunto de datos
datos = pd.read_csv('concentlite.csv')

# Dividir en características (X) y etiquetas (y)
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values

# Generar 1000 puntos de datos aleatorios para la prueba
np.random.seed(42)  # Fijar la semilla para reproducibilidad
X_prueba_extra = np.random.rand(1000, 2)  # Generar 1000 puntos aleatorios en 2 dimensiones
y_prueba_extra = np.random.randint(0, 2, size=(1000,))  # Generar etiquetas aleatorias para los 1000 puntos

# Visualizar los datos de prueba
plt.scatter(X_prueba_extra[y_prueba_extra==0, 0], X_prueba_extra[y_prueba_extra==0, 1], color='blue', label='Datos de Prueba Extra Clase 0', alpha=0.7)
plt.scatter(X_prueba_extra[y_prueba_extra==1, 0], X_prueba_extra[y_prueba_extra==1, 1], color='red', label='Datos de Prueba Extra Clase 1', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Datos de Prueba Extra')
plt.legend()
plt.show()

# Divide en conjunto de entrenamiento y prueba (20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Define la arquitectura de la red, por ejemplo, [tamaño_entrada, tamaño_oculto, tamaño_salida]
capas = [X.shape[1], 8, 1]

# Inicializa la red
mlp = MLP(capas)

# Entrena la red con más épocas y una tasa de aprendizaje más alta
mlp.train(X_entrenamiento, y_entrenamiento.reshape(-1, 1), epocas=5000, tasa_aprendizaje=0.2)

# Realiza predicciones en el conjunto de prueba y redondea a 0 o 1
predicciones = np.round(mlp.predecir(X_prueba))

# Visualiza el resultado
plt.scatter(X_prueba[y_prueba==1, 0], X_prueba[y_prueba==1, 1], color='red', label='Clase 1 (Real)', alpha=0.7)
plt.scatter(X_prueba[:, 0], X_prueba[:, 1], c=predicciones.flatten(), cmap='viridis', marker='x', label='Clase 2', linewidth=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificación del Perceptrón Multicapa con Momentum')
plt.legend()
plt.show()
