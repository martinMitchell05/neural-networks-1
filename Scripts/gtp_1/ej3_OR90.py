import numpy as np
import pandas as pd
import random
from ej2_graficar import graficar 

cabeceras = ['x1','x2','r']
tabla = pd.read_csv('OR_90_trn.csv', header=None, names=cabeceras)
x1 = tabla['x1'].to_numpy()
x2 = tabla['x2'].to_numpy()
r = tabla['r'].to_numpy()
x0 = np.ones(x1.size)*-1

epocas_max = 10
epocas = 1
tasa = 0.01

x = np.array([x1, x2, x0])
#w = np.array([random.random() - 0.5,  random.random() - 0.5, random.random() - 0.5]) 
#print(w)
w = np.array([-0.0503728,   0.14896308,  0.23925176]) 

tasa_aciertos = 0
w_historial = [w.copy()]

while epocas < epocas_max and tasa_aciertos<0.95:
    
    for i in range(x1.size):
        z = np.dot(x[:,i],w)
        y = np.sign(z)
        w[:] = w[:] + (tasa/2)*(r[i]-y)*x[:,i]
        
        if (r[i]!=y):
            w_historial.append(w.copy())

    aciertos = 0
    for i in range(x1.size):
        z = np.dot(x[:,i],w)
        y = np.sign(z)
        if (r[i]==y):
            aciertos = aciertos + 1

    tasa_aciertos = aciertos / x1.size
    print(tasa_aciertos)
    epocas = epocas + 1

if tasa_aciertos>=0.95:
    print(w)

#[ 0.0702273   0.05729098 -0.07074824]
graficar(x1,x2,w_historial)