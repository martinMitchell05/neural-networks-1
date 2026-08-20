import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import random


def graficar(x1, x2_d, w):
    plt.ion()

    fig, ax = plt.subplots(figsize=(8,5))

    for i in range(len(w)):
        w0 = w[i][2]
        w1 = w[i][0]
        w2 = w[i][1]
        
        x2_linea = w0/w2 - (w1/w2) * x1

        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=1) 
        ax.axvline(0, color='black', linewidth=1) 
        ax.set_title(i)

        ax.scatter(x1, x2_d, color='blue', s=0.5) 
        ax.plot(x1, x2_linea, color='red') 
        
        plt.pause(0.2)  
    plt.ioff()
    plt.show()


cabeceras = ['x1', 'x2', 'd']

df_trn = pd.read_csv('../../Data/gtp-1/OR_trn.csv', header=None, names=cabeceras)

x1_trn = df_trn['x1'].to_numpy()
x2_trn = df_trn['x2'].to_numpy()
d_trn = df_trn['d'].to_numpy()

epoc_max = 500
epoc_actual = 1
mu = 0.01

x = np.array([x1_trn, x2_trn, -np.ones(d_trn.size)])

w = np.array([random.random()-0.5, random.random()-0.5, random.random()-0.5])
w_historial = []

while epoc_actual < epoc_max:

    acierto = 0

    # Entrenar:
    for i in range(0, x1_trn.size):   
        w_historial.append(w.copy())

        y = np.sign(np.dot(x[:, i], w[:]))

        error = mu/2 * (d_trn[i] - y)

        w = w + error * x[:, i]



    # Verificar:
    for i in range(0,d_trn.size):

        y_actualizada = np.sign(np.dot(x[:,i], w[:]))
        if (y_actualizada == d_trn[i]):
            acierto += 1


    tasa_aciertos = acierto/d_trn.size
    print(f"Tasa aciertos = {tasa_aciertos}, Errores = {d_trn.size-acierto}")

    if (tasa_aciertos > 0.99):
        print(f"\nTasa de acierto = {tasa_aciertos}, Epoca: {epoc_actual}")
        print(f"Pesos finales = {w}")
        graficar(x[0,:], x[1,:], w_historial)
        break
    else:
        epoc_actual += 1


print(f"Epoca final: {epoc_actual}")