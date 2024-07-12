import numpy as np
import tensorflow as tf
import warnings

def make_bellshaped_plane_2d(domain, width=4.5e-3, centers=(5e-3,5e-3)):
    """ Computes bell-shaped 2d plane for a given domain. """

    shape = domain.shape
    xx = np.linspace(0, 1e-2, shape[0])
    yy = np.linspace(0, 1e-2, shape[1])
    xx, yy = np.meshgrid(xx, yy)

    plane = np.exp(-((xx - centers[0])**2 + (yy - centers[1])**2) / width**2)
    plane[np.isnan(domain)] = np.nan

    return plane.astype('float32')


def make_reduce_grid_2d(domain):
    h, w = np.shape(domain)

    # get ids of pixels on half height, width
    half_h, half_w = h // 2, w // 2

    # compute diffs on each middle plane
    half_h_diffs = np.diff(domain[half_h, :])
    half_w_diffs = np.diff(domain[:, half_w])

    # set NaNs to zero
    half_h_diffs[np.isnan(half_h_diffs)] = 0.
    half_w_diffs[np.isnan(half_w_diffs)] = 0.

    # get indices where values changes (diff is nonzero)
    h_grid = np.nonzero(half_h_diffs)[0] + 1
    w_grid = np.nonzero(half_w_diffs)[0] + 1

    h_grid = np.concatenate([[0], h_grid, [h]])
    w_grid = np.concatenate([[0], w_grid, [w]])

    return h_grid, w_grid


def reduce_2d(domains, reduce_grid=None, agg_func=None):
    domain = domains[0]
    if reduce_grid is None:
        reduce_grid = make_reduce_grid_2d(domain)
    if agg_func is None:
        agg_func = np.nanmean

    h_grid, w_grid = reduce_grid

    reduced_h, reduced_w = len(h_grid) - 1, len(w_grid) - 1
    reduced_domains = np.zeros(
        (len(domains), reduced_h, reduced_w),
        dtype=domain.dtype,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for hi in range(reduced_h):
            for wi in range(reduced_w):
                reduce_areas = domains[:, h_grid[hi]:h_grid[hi + 1], w_grid[wi]:w_grid[wi + 1]]
                reduced_domains[:, hi, wi] = agg_func(reduce_areas, axis=(-2, -1))
        return reduced_domains
    

def tf_quantize(x, n_bits: int, dyn_range: tuple[float] = None, levels: np.array = None):
    if levels is None:
        dyn_range = tf.convert_to_tensor([tf.reduce_min(x), tf.reduce_max(x)]) if dyn_range is None else tf.convert_to_tensor(dyn_range)
        levels = tf.linspace(dyn_range[0], dyn_range[1], tf.math.pow(2, n_bits))
    
    # Compute the absolute difference between each element of x and each quantization level
    diff = tf.abs(tf.expand_dims(x, axis=-1) - levels)
    # Find the index of the minimum difference
    indices = tf.argmin(diff, axis=-1)
    # Gather the quantization levels based on the indices
    quantized_x = tf.gather(levels, indices)
    return quantized_x
    