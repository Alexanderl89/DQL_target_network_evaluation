import matplotlib.pyplot as plt
import numpy as np


def comp_plot_graphs(y_1,y_2,x_label,y_label,y_1_label=None,y_2_label=None,start=0,legend_loc='upper left',colors=["red","blue"],type=plt.plot,path_name="uknown"):
    
    y_1 = y_1[start:]
    y_2 = y_2[start:]

    x = np.arange(start,start+len(y_1),1)
    
    plt.figure(figsize=(10,5))

    if type == plt.bar:
        type(x, y_1,color=colors[0], label=y_1_label,width=1.0)
        type(x, y_2,color=colors[1], label=y_2_label,width=1.0)
    else:
        type(x, y_1,color=colors[0], label=y_1_label)
        type(x, y_2,color=colors[1], label=y_2_label)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize=12, loc=legend_loc)
    plt.savefig(f"{path_name}.png",format="png")
    plt.close()


def plot_graph(y,x_label,y_label,start=0,color="green",ylim=None,type=plt.plot,path_name="uknown"):
    y_1 = y[start:]
    x = np.arange(start,start+len(y_1),1)
    plt.figure(figsize=(10,5))

    if type == plt.bar:
        type(x, y,color=color,width=1.0)
    else:
        type(x, y,color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if ylim is not None and len(ylim) > 1:
        plt.ylim(ylim[0],ylim[1])
    
    plt.savefig(f"{path_name}.png",format="png")
    plt.close()