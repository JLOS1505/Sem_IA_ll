import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def regresion_logistica(X_train, X_test, y_train, y_test):
    modelo = LogisticRegression(max_iter=10000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (Regresión Logística):", mse)

def vecinos_mas_cercanos(X_train, X_test, y_train, y_test, n_neighbors=3):
    modelo = KNeighborsRegressor(n_neighbors=n_neighbors)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (Vecinos Más Cercanos):", mse)

def maquina_de_vectores_de_soporte(X_train, X_test, y_train, y_test):
    modelo = SVR()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (Máquina de Vectores de Soporte):", mse)

def naive_Bayes(X_train, X_test, y_train, y_test):
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (Naive Bayes):", mse)
    
def AutoInsurSweden():
    dataset = pd.read_csv('AutoInsurSweden.csv')
    #Número de reclamos
    X = dataset['X']
    #Pago total por los reclamos
    y = dataset['Y']
    #Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def pima_indians_diabetes():
    dataset = pd.read_csv('pima-indians-diabetes.csv', sep=",")
    X = dataset.drop("Class variable (0 or 1)", axis=1)
    y = dataset["Class variable (0 or 1)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def wine_Quality():
    dataset = pd.read_csv('wine-Quality.csv', sep=",")
    X = dataset.drop("quality", axis=1)
    y = dataset["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    
#Red Neuronal Implementada
def MLP(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,50), max_iter=500):
    modelo = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (MLP):", mse)

nombres_archivos = ['AutoInsurSweden.csv','wine-Quality.csv', 'pima-indians-diabetes.csv']

while True:
    print("\nDataSets:\n")
    for i, nombre_archivo in enumerate(nombres_archivos, start=1):
        print(f"{i}. {nombre_archivo}")

    opcion = input("Seleccione el dataSet: ('s' para terminar): ")

    if opcion.lower() == 's':
        break

    try:
        opcion = int(opcion)
        if opcion == 1:
            print("\nAutoInsurSweden.csv")
            X_train, X_test, y_train, y_test = AutoInsurSweden()
            regresion_logistica(X_train, X_test, y_train, y_test)
            vecinos_mas_cercanos(X_train, X_test, y_train, y_test)
            maquina_de_vectores_de_soporte(X_train, X_test, y_train, y_test)
            naive_Bayes(X_train, X_test, y_train, y_test)
            MLP(X_train, X_test, y_train, y_test)
        elif opcion == 2:
            print("\nwine-Quality.csv")
            X_train, X_test, y_train, y_test = wine_Quality()
            regresion_logistica(X_train, X_test, y_train, y_test)
            vecinos_mas_cercanos(X_train, X_test, y_train, y_test)
            maquina_de_vectores_de_soporte(X_train, X_test, y_train, y_test)
            naive_Bayes(X_train, X_test, y_train, y_test)
            MLP(X_train, X_test, y_train, y_test)
        elif opcion == 3:
            print("\npima-indians-diabetes.csv")
            X_train, X_test, y_train, y_test = pima_indians_diabetes()
            regresion_logistica(X_train, X_test, y_train, y_test)
            vecinos_mas_cercanos(X_train, X_test, y_train, y_test)
            maquina_de_vectores_de_soporte(X_train, X_test, y_train, y_test)
            naive_Bayes(X_train, X_test, y_train, y_test)
            MLP(X_train, X_test, y_train, y_test)
        else:
            print("Opción inválida. Inténtelo de nuevo.")
    except ValueError:
        print("Ingrese una opción válida ('s' para salir)")

