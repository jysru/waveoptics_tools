import numpy as np
import matplotlib.pyplot as plt
from .utils import complex_to_hsv


def complex_colormap(points: int = 100, hue_start: int = 0):
    mod = np.arange(0, points) / points
    phi = np.arange(0, points) / points * 2 * np.pi
    X, Y = np.meshgrid(mod , phi)
    bar = np.abs(X) * np.exp(1j * Y)
    return complex_to_hsv(bar, hue_start=hue_start)


def complex_imshow(complex_array: np.ndarray,
                   colorbar: bool = True,
                   figsize: tuple[int,int] = (15,5),
                   ):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.imshow(complex_to_hsv(complex_array))

    if colorbar:
        cbar_img = complex_colormap()
        cbar_ax = fig.add_axes([ax.get_position().x1+0.01,
                                ax.get_position().y0,
                                0.03,
                                ax.get_position().height])
        cbar_ax.imshow(cbar_img, aspect=15)
        cbar_ax.set_xticks([0, 99], labels=['0', '1'], fontsize=10)
        cbar_ax.set_yticks([0, 50, 99], labels=[r'$+\pi$', '0', r'$-\pi$'], fontsize=10)
        cbar_ax.xaxis.set_ticks_position('bottom')
        cbar_ax.yaxis.set_ticks_position('right')

    return fig, ax
