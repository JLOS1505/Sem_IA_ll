import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class MLP:
    def inicio(self, capas):
        #Se inicializa la Red neuronal con el número de neuronas en cada capa
        self.capas = capas
        #Pesos con valores aleatorios y sesgos a cero
        self.pesos = [np.random.randn(capas[i], capas[i + 1]) for i in range(len(capas) - 1)]
        self.sesgo = [np.zeros((1, capas[i + 1])) for i in range(len(capas) - 1)]

    def sigmoidal(self, x):
        #Función sigmoide
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def derivada_sigmoidal(self, x):
        #Derivada de la función sigmoide
        return x * (1 - x)

    def hacia_delante(self, X):
        #Propagación hacia adelante
        self.activaciones = [X]
        self.valores_z = []

        for i in range(len(self.capas) - 1):
            #Valor ponderado y activación de capa
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.sesgo[i]
            #Selección de la función de activación adecuada
            if i == len(self.capas) - 2:
                #Si es la última capa, aplicar función softma
                a = self.softmax(z)
            else:
                #Para capas anteriores, aplicar función sigmoidal
                a = self.sigmoidal(z)
            #Almacenar los valores para usarlos en la retropropagación
            self.valores_z.append(z)
            self.activaciones.append(a)

    def softmax(self, x):
        #Mejora la estabilidad numérica
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def retroceder(self, X, y, tasa_aprendizaje):
        #Ajusta pesos y sesgos
        errores = [y - self.activaciones[-1]]
        deltas = [errores[-1]]

        for i in range(len(self.capas) - 2, 0, -1):
            #Calcular el error y la delta para cada capa
            error = deltas[-1].dot(self.pesos[i].T)
            delta = error * self.derivada_sigmoidal(self.activaciones[i])
            errores.append(error)
            deltas.append(delta)

        for i in range(len(self.capas) - 2, -1, -1):
            #Actualiza pesos y sesgos utilizando errores y deltas
            self.pesos[i] += self.activaciones[i].T.dot(deltas[len(self.capas) - 2 - i]) * tasa_aprendizaje
            self.sesgo[i] += np.sum(deltas[len(self.capas) - 2 - i], axis=0, keepdims=True) * tasa_aprendizaje

    def entrenar(self, X, y, épocas, tasa_aprendizaje):
        #Entrenamiento de la red neuronal
        for época in range(épocas):
            #Propagación hacia adelante y retropropagación en cada época
            self.hacia_delante(X)
            self.retroceder(X, y, tasa_aprendizaje)

    def predecir(self, X):
        #Predicción utilizando la red neuronal entrenada
        self.hacia_delante(X)
        return self.activaciones[-1]

    def evaluar_loo(self, X, y):
        #Evaluación utilizando Leave-One-Out
        loo = LeaveOneOut()
        precisiones = []

        #Iteración sobre los conjuntos y pruebas generadas por Leave-One-Out
        for índice_entrenamiento, índice_prueba in loo.split(X):
            X_entrenamiento, X_prueba = X[índice_entrenamiento], X[índice_prueba]
            y_entrenamiento, y_prueba = y[índice_entrenamiento], y[índice_prueba]
            self.entrenar(X_entrenamiento, y_entrenamiento, épocas=1000, tasa_aprendizaje=0.2)
            #Realiza predicciones en el conjunto de prueba
            y_predicción_onehot = self.predecir(X_prueba)
            #Convertir de one-hot a clase única
            y_predicción = np.argmax(y_predicción_onehot, axis=1) 
            y_real = np.argmax(y_prueba, axis=1)
            #Calcular precisión
            precisión = accuracy_score(y_real, y_predicción)
            precisiones.append(precisión)
        #Calcula la media y la desviación estándar
        return np.mean(precisiones), np.std(precisiones)

    def evaluar_lko(self, X, y, k):
        #Evaluación utilizando Leave-k-Out
        lko = KFold(n_splits=5)
        precisiones = []
        # Iteración sobre los conjuntos y pruebas generadas por Leave-k-Out
        for índice_entrenamiento, índice_prueba in lko.split(X):
            X_entrenamiento, X_prueba = X[índice_entrenamiento], X[índice_prueba]
            y_entrenamiento, y_prueba = y[índice_entrenamiento], y[índice_prueba]
            self.entrenar(X_entrenamiento, y_entrenamiento, épocas=1000, tasa_aprendizaje=0.2)
            y_predicción_onehot = self.predecir(X_prueba)
            y_predicción = np.argmax(y_predicción_onehot, axis=1)
             #Convertir de one-hot a clase única
            y_real = np.argmax(y_prueba, axis=1)
            precisión = accuracy_score(y_real, y_predicción)
            precisiones.append(precisión)
        #Calcular la media y la desviación estándar
        return np.mean(precisiones), np.std(precisiones)

# Cargar el conjunto de datos
datos = pd.read_csv('irisbin.csv', header=None)
#Dividir en características (X) y etiquetas (y) 4 entradas, 3 salidas
X = datos.iloc[:, :-3].values
y = datos.iloc[:, -3:].values
#Objeto StandardScaler
escalador = StandardScaler()
X = escalador.fit_transform(X)
#Se define la arquitectura de la red
capas = [X.shape[1], 8, 3]

#Inicializa la red
mlp = MLP(capas)
#Dividir en conjunto de entrenamiento y prueba, 80 y 20
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
mlp.entrenar(X_entrenamiento, y_entrenamiento, épocas=1000, tasa_aprendizaje=0.2)
#Realizar predicciones en el conjunto de prueba
predicciones = mlp.predecir(X_prueba)

# Calcula el error esperado, promedio y desviación estándar
lko_promedio_precisión, lko_desviación_estándar = mlp.evaluar_lko(X, y, k=5)
loo_promedio_precisión, loo_desviación_estándar = mlp.evaluar_loo(X, y)
lko_error = 1 - lko_promedio_precisión
loo_error = 1 - loo_promedio_precisión

#Resultados
print("leave-k-out")
print("Error Esperado:", lko_error)
print("Promedio:", lko_promedio_precisión)
print("Desviación Estándar:", lko_desviación_estándar)
print("leave-one-out")
print("Error Esperado:", loo_error)
print("Promedio:", loo_promedio_precisión)
print("Desviación Estándar:", loo_desviación_estándar)

#Datos de prueba
print("Predicciones y Especies Reales:")
for i in range(len(predicciones)):
    especie_real = None
    #Determinar la especie real basada en la codificación one-hot
    if y_prueba[i][2] == 1: #[-1, -1, 1]
        especie_real = 'Setosa'
    elif y_prueba[i][1] == 1: #[-1, 1, -1]
        especie_real = 'Versicolor'
    elif y_prueba[i][0] == 1: #[1, -1, -1]
        especie_real = 'Virginica'
    
    especie_predicha = None
    #Determinar la especie predicha basada en la clase con mayor probabilidad
    if np.argmax(predicciones[i]) == 2:
        especie_predicha = 'Setosa'
    elif np.argmax(predicciones[i]) == 1:
        especie_predicha = 'Versicolor'
    elif np.argmax(predicciones[i]) == 0:
        especie_predicha = 'Virginica'

    print(f"{i+1}: Predicción = {especie_predicha}, Especie real = {especie_real}")

#Puntos para Setosa
plt.scatter(X_prueba[y_prueba[:, 0] == 1, 0], X_prueba[y_prueba[:, 0] == 1, 1], color='blue', label='Setosa', alpha=0.7)
#Puntos para Versicolor
plt.scatter(X_prueba[y_prueba[:, 1] == 1, 0], X_prueba[y_prueba[:, 1] == 1, 1], color='green', label='Versicolor', alpha=0.7)
#Puntos para Virginica
plt.scatter(X_prueba[y_prueba[:, 2] == 1, 0], X_prueba[y_prueba[:, 2] == 1, 1], color='purple', label='Virginica', alpha=0.7)

plt.xlabel('Pétalo')
plt.ylabel('Sépalo')
plt.title('Especies de Iris - Clasificación MLP')
plt.legend(loc='lower right', bbox_transform=plt.gcf().transFigure)
plt.show()