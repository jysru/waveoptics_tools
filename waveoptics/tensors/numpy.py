import warnings
import numpy as np
from scipy.interpolate import interp2d

warnings.filterwarnings("ignore", category=DeprecationWarning)


def crop_2d(feature_map: np.ndarray, new_shape: tuple[int]) -> np.ndarray:
    feature_shape = feature_map.shape
    img_shape = (feature_shape[-2], feature_shape[-1])
    lines_idx, lines_ceil = int(np.floor((img_shape[0] - new_shape[0]) / 2)), int(np.ceil((img_shape[0] - new_shape[0]) / 2))
    cols_idx, cols_ceil = int(np.floor((img_shape[1] - new_shape[1]) / 2)), int(np.ceil((img_shape[1] - new_shape[1]) / 2))
    add_line = 1 if lines_ceil != lines_idx else 0
    add_col = 1 if cols_ceil != cols_idx else 0
    feature_map = feature_map[..., lines_idx:(-lines_idx -add_line), cols_idx:(-cols_idx -add_col)]
    return feature_map


def extend_2d(feature_map: np.ndarray, new_shape: tuple[int]) -> np.ndarray:
    feature_shape = feature_map.shape
    img_shape = (feature_shape[-2], feature_shape[-1])
    lines_idx, lines_ceil = int(np.floor((new_shape[0] - img_shape[0]) / 2)), int(np.ceil((new_shape[0] - img_shape[0]) / 2))
    cols_idx, cols_ceil = int(np.floor((new_shape[1] - img_shape[1]) / 2)), int(np.ceil((new_shape[1] - img_shape[1]) / 2))
    add_line = 1 if lines_ceil != lines_idx else 0
    add_col = 1 if cols_ceil != cols_idx else 0
    if feature_map.ndim == 3:
        feature_map = np.pad(feature_map, ((0, 0), (lines_idx, lines_idx + add_line), (cols_idx, cols_idx + add_col)))
    elif feature_map.ndim == 2:
        feature_map = np.pad(feature_map, ((lines_idx, lines_idx + add_line), (cols_idx, cols_idx + add_col)))
    else:
        raise ValueError("Invalid feature map dimensionality")
    return feature_map


def resize_2d(feature_map: np.ndarray, new_shape: tuple[int]) -> np.ndarray:
    img_shape = np.array([feature_map.shape[-2], feature_map.shape[-1]])
    if np.all(new_shape > img_shape):
        return extend_2d(feature_map, new_shape)
    elif np.all(new_shape < img_shape):
        return crop_2d(feature_map, new_shape)
    elif np.all(img_shape == new_shape):
        return feature_map
    else:
        raise NotImplementedError('Invalid new_shape')



def pooling_2d(feature_map: np.ndarray, kernel: tuple[int] = (2, 2), func: callable = np.max) -> np.ndarray:
    """
    Applies 2D pooling to a feature map.

    Parameters
    ----------
    feature_map : np.ndarray
        A 2D or 3D feature map to apply max pooling to. If the feature map is 3D, the channels should be the first dimension.
    kernel: tuple
        The size of the kernel to use for max pooling.
    func:
        Numpy reduction method to apply: default = numpy.max

    Returns
    -------
    np.ndarray
        The feature map after pooling was applied.
    """

    dim_add = 1 if feature_map.ndim > 2 else 0

    # Check if it fits without padding the feature map
    if feature_map.shape[0 + dim_add] % kernel[0] != 0:
        # Add padding to the feature map
        feature_map = np.pad(feature_map, ((0, kernel[0] - feature_map.shape[0 + dim_add] % kernel[0]), (0,0), (0,0)), 'constant')
    
    if feature_map.shape[1 + dim_add] % kernel[1] != 0:
        feature_map = np.pad(feature_map, ((0, 0), (0, kernel[1] - feature_map.shape[1 + dim_add] % kernel[1]), (0,0)), 'constant')

    if dim_add:
        newshape = (-1, feature_map.shape[1] // kernel[0], kernel[0], feature_map.shape[2] // kernel[1], kernel[1])
    else:
        newshape = (feature_map.shape[0] // kernel[0], kernel[0], feature_map.shape[1] // kernel[1], kernel[1])

    pooled = feature_map.reshape(newshape)
    pooled = func(pooled, axis=(1 + dim_add, 3 + dim_add))
    return pooled


def max_pooling_2d(feature_map: np.ndarray, kernel: tuple[int] = (2, 2)) -> np.ndarray:
    return pooling_2d(feature_map, kernel, func=np.max)


def avg_pooling_2d(feature_map: np.ndarray, kernel: tuple[int] = (2, 2)) -> np.ndarray:
    return pooling_2d(feature_map, kernel, func=np.mean)


def subsample_complex(feature_map: np.ndarray, kernel: tuple[int] = (2, 2)) -> np.ndarray:
    return (
        avg_pooling_2d(np.real(feature_map), kernel=kernel)
        + 1j * avg_pooling_2d(np.imag(feature_map), kernel=kernel)
    )


def interpolate(image: np.ndarray, scale: float, kind = 'linear') -> np.ndarray:
    height, width = image.shape

    # Create a grid of coordinates for the original image
    x = np.arange(0, width)
    y = np.arange(0, height)

    # Create interpolation functions for each channel
    interpolators = interp2d(x, y, image, kind=kind, bounds_error=False, fill_value=0)

    # Calculate the new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Generate a new grid of coordinates for the interpolated image
    new_x = np.linspace(0, width - 1, new_width)
    new_y = np.linspace(0, height - 1, new_height)

    return interpolators(new_x, new_y)


def interpolate_complex(field: np.ndarray, scale: float, kind: str = 'linear') -> np.ndarray:
    return (
        interpolate(np.real(field), scale=scale, kind=kind)
        + 1j * interpolate(np.imag(field), scale=scale, kind=kind)
    )
    

# def autobin_2d_to(feature_map: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
#     pooling_kernel = (
#         int(feature_map.shape[-2] // new_shape[0]),
#         int(feature_map.shape[-1] // new_shape[1]),
#     )
#     crop_amounts = (
#         int((feature_map.shape[-2] - (pooling_kernel[0] * new_shape[0])) // 2),
#         int((feature_map.shape[-1] - (pooling_kernel[1] * new_shape[1])) // 2),
#     )
#     cropped_shape = (
#         feature_map.shape[-2] - 2 * crop_amounts[0],
#         feature_map.shape[-1] - 2 * crop_amounts[1],
#     )

#     print(f"crops={crop_amounts}")
#     print(f"pool={pooling_kernel}")
#     print(f"cropped shape={cropped_shape}")

#     print(feature_map.shape)
#     feature_map = crop_2d(feature_map, cropped_shape)
#     print(feature_map.shape)
#     feature_map = avg_pooling_2d(feature_map, kernel=pooling_kernel)
#     print(feature_map.shape)
#     return feature_map