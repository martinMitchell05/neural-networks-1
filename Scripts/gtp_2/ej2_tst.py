import numpy as np
import pandas as pd
import random as random
import copy
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

class Capa:
    w : np.ndarray
    y: np.ndarray
    delta: np.ndarray

    def __init__(self, w_i, y_i, delta_i):
        self.w = w_i
        self.y = y_i
        self.delta = delta_i

def sigm(x):
    return (2/(1+np.exp(-x))) - 1


entrada_usuario = [8,1]

tabla = pd.read_csv('../../Data/gtp_2/concent_tst.csv', header=None).to_numpy()
x0 = np.ones(len(tabla))*-1
entradas = np.c_[x0,tabla[:,:-1]]
yd =  tabla[:,-1]

w = np.array([
    [ 9.17843741e+00,  1.70257334e+01,  7.04301088e+00],
    [-5.28604358e+00, -1.63616359e+01,  1.27049684e+01],
    [ 7.29808883e-03,  1.11958187e-03,  3.03807301e-02],
    [-1.32459661e+01, -2.62116245e+00, -1.84157371e+01],
    [ 4.34348725e-01,  6.14238966e+00, -9.21291030e+00],
    [ 1.74994520e-01,  1.73813322e-02,  6.91986958e-01],
    [-3.38872981e-02, -5.34263433e-03, -1.40077442e-01],
    [-2.48423465e-01, -1.25331004e-02, -9.43228304e-01]
])
y_init = np.zeros(entrada_usuario[0])
delta = np.zeros(entrada_usuario[0])
cap = Capa(w,y_init,delta)
vect_capas = [copy.deepcopy(cap)]


w = np.array([
    [-2.32166434e+01, -1.99992968e+01, -1.66814159e+01,  1.26737754e-02,
     -2.02304486e+01,  3.95984354e+00,  2.85919951e-01, -5.83926882e-02,
     -3.86115096e-01]
])
y_init = np.zeros(entrada_usuario[1])
delta = np.zeros(entrada_usuario[1])
cap = Capa(w,y_init,delta)
vect_capas.append(copy.deepcopy(cap))

plt.ion()
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Clasificación: Aciertos y Errores")
ax.set_xlabel("x1")
ax.set_ylabel("x2")

aciertos = 0
for n in range(len(entradas)):
    for i in range(len(vect_capas)):
        for j in range(len(vect_capas[i].y)):
            if i == 0:
                z = np.dot(entradas[n, :], vect_capas[i].w[j, :])
            else: 
                ent = np.r_[-1, vect_capas[i-1].y]
                z = np.dot(ent, vect_capas[i].w[j, :])
            vect_capas[i].y[j] = sigm(z)
    prediccion = vect_capas[-1].y[-1]
    
    x1_val = entradas[n, 1]
    x2_val = entradas[n, 2]

    if prediccion > 0 and yd[n] > 0:
        ax.scatter(x1_val, x2_val, c='k', marker='x', s=40)
        aciertos += 1 
    elif prediccion < 0 and yd[n] < 0:
        ax.scatter(x1_val, x2_val, edgecolors='r', facecolors='None', marker='s', s=40)
        aciertos += 1  
    elif prediccion > 0 and yd[n] < 0:
        ax.scatter(x1_val, x2_val, c='k', marker='x', s=40)
    elif prediccion < 0 and yd[n] > 0:
        ax.scatter(x1_val, x2_val, edgecolors='r', facecolors='None', marker='s', s=40)

plt.ioff()
plt.show()

tasa_aciertos = aciertos / len(entradas)
print(f"Tasa de aciertos: {tasa_aciertos * 100:.2f}%")
  
    

                        
