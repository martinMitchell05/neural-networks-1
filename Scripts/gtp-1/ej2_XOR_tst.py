import numpy as np
import pandas as pd

cabeceras = ['x1','x2','r']
tabla = pd.read_csv('XOR_tst.csv', header=None, names=cabeceras)
x1 = tabla['x1'].to_numpy()
x2 = tabla['x2'].to_numpy()
r = tabla['r'].to_numpy()
bias = np.ones(x1.size)*-1


x = np.array([x1, x2, bias])
w = np.array([-0.00134322, -0.00017824, -0.00123526]) 

aciertos = 0
for i in range(x1.size):
    z = np.dot(x[:,i],w)
    y = np.sign(z)
    if (r[i]==y):
        aciertos = aciertos + 1

tasa_aciertos = aciertos / x1.size
print(tasa_aciertos)
    