import numpy as np

from typing import Sequence, List
from typing import NewType
from matplotlib import pyplot as plt 

axes = NewType("axes", plt.axes)

def loss_curve(losses: Sequence[float], ax: axes = None) -> None:
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set(title="The loss vesus iteration",
    xlabel="iter",
    ylabel="loss")
    plt.show()
    
if __name__ == "__main__":
    loss_curve(np.linspace(0,1,100))