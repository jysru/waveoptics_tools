import tensorflow as tf


def pearson(y_true: tf.Tensor, y_pred: tf.Tensor, squared: bool = False, inversed: bool = False) -> tf.Tensor:
    y_true, y_pred = tf.math.abs(y_true), tf.math.abs(y_pred)
    if squared:
        y_true, y_pred = tf.math.square(y_true), tf.math.square(y_pred)
    s = tf.math.reduce_sum((y_true - tf.math.reduce_mean(y_true)) * (y_pred - tf.math.reduce_mean(y_pred)) / tf.cast(tf.size(y_true), y_true.dtype))
    p = s / (tf.math.reduce_std(y_true) * tf.math.reduce_std(y_pred))
    return 1 - p if inversed else p


def mae(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Return the mean absolute error between the two arrays."""
    x, y = tf.math.abs(x), tf.math.abs(y)
    return tf.math.reduce_mean(tf.math.abs(x - y))


def mse(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Return the mean square error between the two arrays."""
    x, y = tf.math.abs(x), tf.math.abs(y)
    return tf.math.reduce_mean(tf.math.square(tf.math.abs(x - y)))


def dot_product(x: tf.Tensor, y: tf.Tensor, normalized: bool = True) -> tf.Tensor:
    """Return the scalar product between the two complex arrays."""
    prod = tf.math.reduce_sum(x * tf.math.conj(y))
    norm = tf.cast(tf.math.reduce_sum(tf.math.abs(x) * tf.math.abs(y)), prod.dtype)
    return prod / norm if normalized else prod


def quality(x: tf.Tensor, y: tf.Tensor, squared: bool = False, inversed: bool = False) -> tf.Tensor:
    """Return the magnitude of the normalized dot product between the two complex arrays."""
    q = tf.math.abs(dot_product(x, y, normalized=True))
    if squared:
        q = tf.math.square(q)
    return 1 - q if inversed else q


