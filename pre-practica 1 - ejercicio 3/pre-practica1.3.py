import numpy as np
import matplotlib.pyplot as plt

def f(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

def grad_f(x1, x2):
    return np.array([-2*x1*np.exp(-(x1**2 + 3*x2**2)),
                     -6*x2*np.exp(-(x1**2 + 3*x2**2))])

def update_params(x, lr, grad):
    return x - lr * grad

def optimize_function(lr, epochs):
    x0 = np.random.uniform(-1, 1, 2)

    error_history = []

    best_solution = {'x1': None, 'x2': None, 'error': float('inf')}

    for epoch in range(epochs):
        grad = grad_f(x0[0], x0[1])
        x0 = update_params(x0, lr, grad)
        error = f(x0[0], x0[1])
        error_history.append(error)
        if error < best_solution['error']:
            best_solution['x1'] = x0[0]
            best_solution['x2'] = x0[1]
            best_solution['error'] = error

    print("Mejor solución encontrada:")
    print(f"Valor de X1: {best_solution['x1']}")
    print(f"Valor de X2: {best_solution['x2']}")
    print(f"Error mínimo encontrado: {best_solution['error']}")

    # Graficar la convergencia del error
    plt.plot(range(epochs), error_history)
    plt.title('Convergencia del Error')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

# Interfaz de usuario
lr = float(input("Ingrese el learning rate: "))
epochs = int(input("Ingrese la cantidad de épocas: "))

# Validación de límites
if not (0 < lr < 1):
    print("El learning rate debe estar en el rango (0, 1)")
elif epochs <= 0:
    print("La cantidad de épocas debe ser un número positivo")
else:
    optimize_function(lr, epochs)
