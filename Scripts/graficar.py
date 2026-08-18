import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

def graficar_estatico(x1, x2, w, d):

    plt.figure("Resultado final", figsize=(8,5))
    
    w0 = w[2]
    w1 = w[0]
    w2 = w[1]
        
    x2_linea = w0/w2 - (w1/w2) * x1  
    plt.scatter(x1, x2, c=d, cmap='bwr', s=50, edgecolors='black', zorder=3)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.plot(x1, x2_linea, color='g', linewidth=1.5, zorder=2)

    plt.show()


def graficar(x1, x2_d, w, d):
    plt.ion()

    fig, ax = plt.subplots(figsize=(8,5))

    for i in range(len(w)):
        w0 = w[i][2]
        w1 = w[i][0]
        w2 = w[i][1]
        
        x2_linea = w0/w2 - (w1/w2) * x1

        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-2,2)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1) 
        ax.axvline(0, color='black', linewidth=1) 
        ax.set_title(i)

        ax.scatter(x1, x2_d, c=d, cmap='bwr', s=50, edgecolors='black', zorder=3) 
        ax.plot(x1, x2_linea, color='g', linewidth=1.5, zorder=2) 
        
        plt.pause(0.2)  
        
    plt.ioff()
    plt.show()