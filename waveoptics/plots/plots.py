import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def animate_images(speckles: np.ndarray, savepath: str, figsize: tuple[float, float]) -> None:
    frames = speckles.shape[0]
    size = speckles.shape[1:]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot([0, 1], [0, 1], label='Full Figure Axes', ls='none')
    ax.axis('off')

    # fig.set_facecolor('black')
    img = plt.imshow(np.zeros(shape=size), cmap='gray')
    img.set_clim(vmin=0, vmax=1)
    # plt.axis('off')

    def animate(frame):
        speckle = speckles[frame, ...]
        speckle /= np.max(np.abs(speckle))
        img.set_data(speckle)

    anim_created = animation.FuncAnimation(fig, animate, frames=frames, interval=500/frames, repeat=True)
    anim_created.save(savepath,)
    plt.close()


def animate_fields(speckles: np.ndarray, savepath: str, figsize: tuple[float, float]) -> None:
    frames = speckles.shape[0]
    size = speckles.shape[1:]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot([0, 1], [0, 1], label='Full Figure Axes', ls='none')
    ax.axis('off')

    fig.set_facecolor('black')
    img = plt.imshow(np.zeros(shape=size), cmap='gray')
    img.set_clim(vmin=0, vmax=1)
    # plt.axis('off')

    def animate(frame):
        speckle = speckles[frame, ...]
        speckle /= np.max(np.abs(speckle))
        speckle_img = complex_to_hsv(speckle)
        img.set_data(speckle_img)

    anim_created = animation.FuncAnimation(fig, animate, frames=frames, interval=500/frames, repeat=True)
    anim_created.save(savepath,)
    plt.close()

