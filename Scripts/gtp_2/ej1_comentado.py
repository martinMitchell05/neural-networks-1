import numpy as np
import pandas as pd
import random as random
import copy


class Capa:
    w : np.ndarray
    y: np.ndarray
    delta: np.ndarray

    def __init__(self, w_i, y_i, delta_i):
        self.w = w_i
        self.y = y_i
        self.delta = delta_i

def sigm(x):
    # SOLUCIÓN: Solo -x en el exponente
    return (2/(1+np.exp(-x))) - 1


entrada_usuario = [2,1]

tabla = pd.read_csv('XOR_trn.csv', header=None).to_numpy()
x0 = np.ones(len(tabla))*-1
entradas = np.c_[x0,tabla[:,:-1]]
yd =  tabla[:,-1]

# SOLUCIÓN: Dimensiones invertidas y uso de np.random.rand para no repetir el mismo peso
w = np.random.rand(entrada_usuario[0], len(entradas[0])) - 0.5
# SOLUCIÓN: Cambiamos el nombre a y_init para no borrar tu variable yd (las respuestas)
y_init = np.zeros(entrada_usuario[0])
delta = np.zeros(entrada_usuario[0])
cap = Capa(w,y_init,delta)
vect_capas = [copy.deepcopy(cap)]

for i in range(1,len(entrada_usuario)):
    # SOLUCIÓN: Inicialización aleatoria y dimensiones correctas
    w = np.random.rand(entrada_usuario[i], entrada_usuario[i-1]+1) - 0.5
    y_init = np.zeros(entrada_usuario[i])
    delta = np.zeros(entrada_usuario[i])
    cap = Capa(w,y_init,delta)
    vect_capas.append(copy.deepcopy(cap))


epoca = 1
epocas_max = 5000 # SOLUCIÓN: Aumentado (10 épocas no alcanzan para el XOR)
tasa = 0.1 # SOLUCIÓN: Aumenté un poco la tasa para que aprenda más rápido

# n -> ejemplo actual
# i -> la capa
# j -> la neurona
while epoca < epocas_max: 
    # SOLUCIÓN: Iterar sobre len(entradas) (las filas), no len(entradas[0]) (las columnas)
    for n in range(len(entradas)):

        #paso hacia adelante
        for i in range(len(vect_capas)):
            for j in range(len(vect_capas[i].y)):
                if i==0:
                    z = np.dot(entradas[n,:],vect_capas[i].w[j,:])
                else: 
                    ent = np.r_[-1,vect_capas[i-1].y]
                    z = np.dot(ent,vect_capas[i].w[j,:])
                vect_capas[i].y[j] = sigm(z)

        #propagacion hacia atras
        # SOLUCIÓN: Empezar en len(vect_capas)-1 para no tener un IndexError
        for i in range(len(vect_capas)-1,-1,-1):
            for j in range(len(vect_capas[i].y)):
                # SOLUCIÓN: Restar 1 al if porque los índices empiezan en 0
                if i==len(vect_capas)-1:
                    # SOLUCIÓN: Usar yd[n] (el deseado del ejemplo actual) en vez de yd[j]
                    vect_capas[i].delta[j] = (1/2) * (yd[n] - vect_capas[i].y[j]) * (1 + vect_capas[i].y[j]) * (1 - vect_capas[i].y[j])
                else:
                    # SOLUCIÓN: np.dot contra el array de deltas completo de la capa siguiente (quitamos el [j])
                    vect_capas[i].delta[j] = (1/2) * np.dot(vect_capas[i+1].delta, vect_capas[i+1].w[:,j+1])  * (1 + vect_capas[i].y[j]) * (1 - vect_capas[i].y[j])

        #actualizar los pesos
        for i in range(len(vect_capas)):
            for j in range(len(vect_capas[i].y)):
                # SOLUCIÓN: Usar len de los pesos de la neurona actual
                for m in range(len(vect_capas[i].w[j])):
                    if i==0:
                        # SOLUCIÓN: Usar += en lugar de -=
                        vect_capas[i].w[j,m] += tasa*vect_capas[i].delta[j]*entradas[n,m]
                    else:
                        # SOLUCIÓN: Hay que reconstruir la entrada con el -1 para poder usar el índice 'm' sin salir de rango, y usar +=
                        ent = np.r_[-1, vect_capas[i-1].y]
                        vect_capas[i].w[j,m] += tasa*vect_capas[i].delta[j]*ent[m]
                        
    # SOLUCIÓN: Sumar a la época o el while será infinito
    epoca += 1
                        
print(vect_capas[1].y[0])