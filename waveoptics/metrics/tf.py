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


def energy_in_target(y, target, inversed: bool = False):
    p = tf.reduce_sum(tf.square(tf.abs(y * target))) /tf.reduce_sum(tf.square(tf.abs(y)))
    return 1 - p if inversed else p


def power_overlap_integral(y, target, inversed: bool = False):
    numer = tf.square(tf.abs(tf.reduce_sum(y * tf.math.conj(target))))
    denom = tf.reduce_sum(tf.square(tf.abs(y))) * tf.reduce_sum(tf.square(tf.abs(target)))
    over = numer / denom
    return 1 - over if inversed else over


def power_overlap_integral_moduli(y, target, inversed: bool = False):
    numer = tf.square(tf.abs(tf.reduce_sum(tf.abs(y) * tf.abs(target))))
    denom = tf.reduce_sum(tf.square(tf.abs(y))) * tf.reduce_sum(tf.square(tf.abs(target)))
    over = numer / denom
    return 1 - over if inversed else over


def energy_ratio_in_target_trsh(y, target, trsh: float = 0.1, inversed: bool = False):
    norm_target = tf.abs(target) / tf.reduce_max(tf.abs(target))
    norm_field = tf.abs(y) / tf.reduce_max(tf.abs(y))
    total_energy = tf.reduce_sum(tf.square(tf.abs(norm_field)))
    above_trsh = tf.math.greater_equal(tf.square(tf.abs(norm_target)), trsh)
    energy_in_trsh = tf.reduce_sum(tf.square(tf.abs(tf.where(above_trsh, norm_field, 0))))
    ratio = energy_in_trsh / total_energy
    return 1 - ratio if inversed else ratio