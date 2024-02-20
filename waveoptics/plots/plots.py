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
                   rmin: float = None,
                   rmax: float = None,
                   hue_start: int = 0,
                   colorbar: bool = True,
                   figsize: tuple[int,int] = (15,5),
                   ):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.imshow(complex_to_hsv(complex_array, rmin, rmax, hue_start))

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


def complex_imshow_real_imag(field: np.ndarray, figsize: tuple[int, int] = (15,5), cmap: str = 'bwr', return_handles: bool = False):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    pl0 = axs[0].imshow(np.real(field), cmap=cmap, vmin=-np.max(np.abs(np.real(field))), vmax=+np.max(np.abs(np.real(field))))
    pl1 = axs[1].imshow(np.imag(field), cmap=cmap, vmin=-np.max(np.abs(np.imag(field))), vmax=+np.max(np.abs(np.imag(field))))
    plt.colorbar(pl0, ax=axs[0])
    plt.colorbar(pl1, ax=axs[1])
    if return_handles:
        return fig, axs


def compare_complex_imshow_real_imag(fields_list: list[np.ndarray], figsize: tuple[int, int] = (15,5), cmap: str = 'bwr', return_handles: bool = False):
    fig, axs = plt.subplots(2, len(fields_list), figsize=figsize)
    
    for i in range(len(fields_list)):
        field = fields_list[i]
        re = axs[0, i].imshow(np.real(field), cmap=cmap,
                               vmin=-np.max(np.abs(np.real(field))),
                               vmax=+np.max(np.abs(np.real(field))),
                               )
        im = axs[1, i].imshow(np.imag(field), cmap=cmap,
                               vmin=-np.max(np.abs(np.imag(field))),
                               vmax=+np.max(np.abs(np.imag(field))),
                               )
        plt.colorbar(re, ax=axs[0, i])
        plt.colorbar(im, ax=axs[1, i])
        axs[0, i].set_title(f"Real field {i}")
        axs[0, i].set_title(f"Imag field {i}")

    if return_handles:
        return fig, axs
