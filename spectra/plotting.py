import numpy as np
import matplotlib.pyplot as plt

def banded(x, upper, lower, ax, **kwargs):
    """Plots upper & lower error bands on an existing axis.

    Args:
        x: 1D array
            Abscissa values for each each value in y ordinates.
        upper: 1D array
            Upper error bounds to plot (e.g. STD, SEM, or CI bound)
        lower: 1D array
            Lower error bounds to plot (e.g. STD, SEM, or CI bound)
        ax: axis instance
            A Matplotlib figure axis instance to plot x and y to. If None
            (Default) creates a new axis instance.
        **kwargs: dict
            Any valid kwarg for matplotlib's plot and fill_between funcs.

    Returns: A matplotlib axis instance.
    """

    x = np.arange(len(arr)) if x is None else x

    color = kwargs.pop('color', 'k')
    facecolor = kwargs.pop('facecolor', 'tab:gray')
    alpha = kwargs.pop('alpha', 0.4)

    ax.fill_between(x, lower, upper,color='k', facecolor=facecolor,
            alpha=alpha, **kwargs)
    return ax    
