import numpy as np
import math

def f(u):
    return (2 / (1 + math.exp(-u))) - 1

def find_output(data, w):
    u = np.dot(w, data)
    lamb = 0.1
    return f(lamb * u)

p = np.array([[1, 1, -1],
              [1, -1, -1],
              [-1, 1, -1],
              [-1, -1, -1]])

d = np.array([1, 1, 1, -1])

w = np.random.rand(p.shape[1])

c = 0.5 
d_error= 0.01
max_iter = 1000

for iteration in range(max_iter):
    error = 0  # Erro acumulado

    for i in range(len(p)):
        o = find_output(p[i], w)
        error += 0.5 * (d[i] - o) ** 2

        delta = (d[i] - o) * (1 - o * o) / 2  
        
        w += c * delta * p[i]

    print(f"Iteração {iteration + 1} | Erro: {error:.4f} | Pesos: {w}")

    if error < d_error:
        print(f"Convergência atingida após {iteration + 1} iterações.")
        break
else:
    print("Número máximo de iterações atingido sem convergência.")

for entrada in p:
    print(f"Entrada: {entrada} | Saída prevista: {find_output(entrada, w)}")
