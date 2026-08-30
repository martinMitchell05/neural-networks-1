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
    return (2/(1+np.exp(-x))) - 1


entrada_usuario = [8,1]

tabla = pd.read_csv('concent_trn.csv', header=None).to_numpy()
x0 = np.ones(len(tabla))*-1
entradas = np.c_[x0,tabla[:,:-1]]
yd =  tabla[:,-1]

w = np.random.rand(entrada_usuario[0], len(entradas[0])) - 0.5
y_init = np.zeros(entrada_usuario[0])
delta = np.zeros(entrada_usuario[0])
cap = Capa(w,y_init,delta)
vect_capas = [copy.deepcopy(cap)]

for i in range(1,len(entrada_usuario)):
    w = np.random.rand(entrada_usuario[i], entrada_usuario[i-1]+1) - 0.5
    y_init = np.zeros(entrada_usuario[i])
    delta = np.zeros(entrada_usuario[i])
    cap = Capa(w,y_init,delta)
    vect_capas.append(copy.deepcopy(cap))

for i in range(len(vect_capas)):
    print(vect_capas[i].w)

epoca = 1
epocas_max = 5000
tasa = 0.8
tasa_aciertos = 0
# n -> ejemplo actual
# i -> la capa
# j -> la neurona
while epoca < epocas_max and tasa_aciertos<0.99: 
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
        for i in range(len(vect_capas)-1,-1,-1):
            for j in range(len(vect_capas[i].y)):
                if i==len(vect_capas)-1:
                    vect_capas[i].delta[j] = (1/2) * (yd[n] - vect_capas[i].y[j]) * (1 + vect_capas[i].y[j]) * (1 - vect_capas[i].y[j])
                else:
                    vect_capas[i].delta[j] = (1/2) * np.dot(vect_capas[i+1].delta, vect_capas[i+1].w[:,j+1])  * (1 + vect_capas[i].y[j]) * (1 - vect_capas[i].y[j])

        #actualizar los pesos
        for i in range(len(vect_capas)):
            for j in range(len(vect_capas[i].y)):
                for m in range(len(vect_capas[i].w[j])):
                    if i==0:
                        vect_capas[i].w[j,m] += tasa*vect_capas[i].delta[j]*entradas[n,m]
                    else:
                        ent = np.r_[-1, vect_capas[i-1].y]
                        vect_capas[i].w[j,m] += tasa*vect_capas[i].delta[j]*ent[m]

    #validar
    aciertos = 0
    for n in range(len(entradas)):
        for i in range(len(vect_capas)):
            for j in range(len(vect_capas[i].y)):
                if i==0:
                    z = np.dot(entradas[n,:],vect_capas[i].w[j,:])
                else: 
                    ent = np.r_[-1,vect_capas[i-1].y]
                    z = np.dot(ent,vect_capas[i].w[j,:])
                vect_capas[i].y[j] = sigm(z)
        if (vect_capas[-1].y[-1] > 0 and yd[n] > 0) or (vect_capas[-1].y[-1] < 0 and yd[n] < 0) :
            aciertos +=1
    tasa_aciertos = aciertos / len(entradas)
    print(f"{epoca}- {tasa_aciertos}") 

    if (tasa_aciertos>=0.8 and tasa==0.8):
        tasa = 0.5
    if (tasa_aciertos>=0.92 and tasa==0.5):
        tasa = 0.3
    if (tasa_aciertos>=0.96 and tasa==0.3):
        tasa = 0.1
    if (tasa_aciertos>=0.98 and tasa==0.1):
        tasa = 0.05
    
    epoca += 1
                        
for i in range(len(vect_capas)):
    print(vect_capas[i].w)