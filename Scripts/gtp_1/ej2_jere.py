import matplotlib.pyplot as plt

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
        
        plt.pause(0.01)  
    plt.ioff()
    plt.show()

        