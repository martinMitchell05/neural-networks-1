import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Neurona:
  def __init__(self, w_i):
    self.w = w_i  # Vector de pesos (2,)


def obtener_frontera_2d(idx_ganadora, filas, columnas, radio):
    fila_ganadora = idx_ganadora // columnas
    col_ganadora = idx_ganadora % columnas
    vecinos = [] 
    for f in range(filas):
        for c in range(columnas):
            pasos_en_filas = abs(f - fila_ganadora)
            pasos_en_cols = abs(c - col_ganadora)
            # Vale las diagonales
            distancia = max(pasos_en_filas, pasos_en_cols)
            if distancia <= radio:    
                idx_actual = f * columnas + c             
                vecinos.append(idx_actual)
                
    return vecinos


tabla = pd.read_csv('circulo.csv', header=None).to_numpy()
#tabla = pd.read_csv('te.csv', header=None).to_numpy()
entradas = tabla[:, :]  # Forma (750, 2)

filas, columnas = 4, 4
cant_neuronas = filas * columnas
epocas = 1000

# Parámetros iniciales
tasa_inicial = 0.5
radio_inicial = 2

# 1 Inicialización
vect_Neuronas = []
for i in range(cant_neuronas):
  w = np.random.rand(2) - 0.5
  vect_Neuronas.append(Neurona(w))

# Configuración del gráfico 
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))


for epoca in range(epocas):


  tasa = tasa_inicial * (1 - epoca / epocas)
  radio_vecindad = int(np.round(radio_inicial * (1 - epoca / epocas)))

  # 2. Cada patrón x
  for x in entradas:
    # Obtener los pesos actualizados de todas las neuronas -> Forma (16, 2)
    pesos_neuronas = np.array([neu.w for neu in vect_Neuronas])

    # Distancia euclidea del patrón x a cada  neurona
    distancias = np.linalg.norm(pesos_neuronas - x, axis=1)

    
    neurona_ganadora_idx = np.argmin(distancias)

    
    indices_entorno = obtener_frontera_2d(neurona_ganadora_idx,filas=filas,columnas=columnas,radio=radio_vecindad)

    # 3. Adaptación de los pesos
    for idx in indices_entorno:
      w_actual = vect_Neuronas[idx].w
      vect_Neuronas[idx].w = w_actual + tasa * (x - w_actual)

  # --- GRAFICO CADA 10 EPOCAS ---
  if epoca % 10 == 0 or epoca == epocas - 1:
    ax.clear()
    pesos_totales = np.array([neu.w for neu in vect_Neuronas])

    
    dists_puntos = np.linalg.norm(
        entradas[:, np.newaxis, :] - pesos_totales[np.newaxis, :, :], axis=2
    )
    bmu_por_punto = np.argmin(dists_puntos, axis=1)

    # A) Patrones de entrada coloreados según su neurona ganadora
    ax.scatter(entradas[:, 0], entradas[:, 1],c=bmu_por_punto,cmap='tab20', s=12, alpha=0.5,)

    # B) Líneas de unión entre neuronas vecinas (Malla 2D)
    # ax.plot(pesos_totales[:, 0], pesos_totales[:, 1], 'k-', lw=1.5) <-- COMENTADO
    for r in range(filas):
      for c in range(columnas):
        idx = r * columnas + c
        if c + 1 < columnas:  # Unión horizontal
          idx_d = r * columnas + (c + 1)
          ax.plot([pesos_totales[idx, 0], pesos_totales[idx_d, 0]],[pesos_totales[idx, 1], pesos_totales[idx_d, 1]],'k-',lw=1)
        if r + 1 < filas:  # Unión vertical
          idx_a = (r + 1) * columnas + c
          ax.plot([pesos_totales[idx, 0], pesos_totales[idx_a, 0]],[pesos_totales[idx, 1], pesos_totales[idx_a, 1]],'k-',lw=1) 

    # C) Centroides (Pesos de cada neurona) con marcador 'X'
    ax.scatter(pesos_totales[:, 0],pesos_totales[:, 1],c='red',marker='X',s=70,edgecolors='black',)

    ax.set_title(f'SOM 2D - Época {epoca}/{epocas}')
    plt.pause(0.01)

plt.ioff()
plt.show()