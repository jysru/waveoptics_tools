import tensorflow as tf


def crop_2d(feature_map: tf.Tensor, new_shape: tuple[int]) -> tf.Tensor:
    feature_shape = feature_map.shape
    img_shape = (feature_shape[-2], feature_shape[-1])
    lines_idx, lines_ceil = int(tf.math.floor((img_shape[0] - new_shape[0]) / 2)), int(tf.math.ceil((img_shape[0] - new_shape[0]) / 2))
    cols_idx, cols_ceil = int(tf.math.floor((img_shape[1] - new_shape[1]) / 2)), int(tf.math.ceil((img_shape[1] - new_shape[1]) / 2))
    add_line = 1 if lines_ceil != lines_idx else 0
    add_col = 1 if cols_ceil != cols_idx else 0
    feature_map = feature_map[..., lines_idx:(-lines_idx -add_line), cols_idx:(-cols_idx -add_col)]
    return feature_map


def extend_2d(feature_map: tf.Tensor, new_shape: tuple[int]) -> tf.Tensor:
    feature_shape = feature_map.shape
    img_shape = (feature_shape[-2], feature_shape[-1])
    lines_idx, lines_ceil = int(tf.math.floor((new_shape[0] - img_shape[0]) / 2)), int(tf.math.ceil((new_shape[0] - img_shape[0]) / 2))
    cols_idx, cols_ceil = int(tf.math.floor((new_shape[1] - img_shape[1]) / 2)), int(tf.math.ceil((new_shape[1] - img_shape[1]) / 2))
    add_line = 1 if lines_ceil != lines_idx else 0
    add_col = 1 if cols_ceil != cols_idx else 0
    if feature_map.ndim == 3:
        feature_map = tf.pad(feature_map, ((0, 0), (lines_idx, lines_idx + add_line), (cols_idx, cols_idx + add_col)))
    elif feature_map.ndim == 2:
        feature_map = tf.pad(feature_map, ((lines_idx, lines_idx + add_line), (cols_idx, cols_idx + add_col)))
    else:
        raise ValueError("Invalid feature map dimensionality")
    return feature_map


def resize_2d(feature_map: tf.Tensor, new_shape: tuple[int]) -> tf.Tensor:
    img_shape = np.array([feature_map.shape[-2], feature_map.shape[-1]])
    if tf.reduce_all(new_shape > img_shape):
        return extend_2d(feature_map, new_shape)
    elif tf.reduce_all(new_shape < img_shape):
        return crop_2d(feature_map, new_shape)
    elif tf.reduce_all(img_shape == new_shape):
        return feature_map
    else:
        raise NotImplementedError('Invalid new_shape')


def pooling_2d(feature_map: tf.Tensor, kernel: tuple[int] = (2, 2), type: str = 'max') -> tf.Tensor:
    type = type.lower()
    _allowed_types = ['max', 'avg']

    if type.lower() not in _allowed_types:
        raise ValueError('Invalid type')
    else:
        if type == 'max':
            return max_pooling_2d(feature_map, kernel)
        elif type == 'avg':
            return avg_pooling_2d(feature_map, kernel)


def max_pooling_2d(feature_map: tf.Tensor, kernel: tuple[int] = (2, 2)) -> tf.Tensor:
    layer = tf.keras.layers.MaxPooling2D(pool_size=kernel, padding='valid')
    return layer(feature_map)


def avg_pooling_2d(feature_map: tf.Tensor, kernel: tuple[int] = (2, 2)) -> tf.Tensor:
    layer = tf.keras.layers.AveragePooling2D(pool_size=kernel, padding='valid')
    return layer(feature_map)