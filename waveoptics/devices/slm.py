import os
import numpy as np
import tensorflow as tf
from ._utils import reduce_2d, make_bellshaped_plane_2d


def tf_partition_to_nxn(partition: tf.Tensor, desired_field_size: int = 48) -> np.ndarray:
    partition = tf.squeeze(partition)
        
    if tf.rank(partition) > 2:
        raise IndexError(f"Partition dimension > 2!")
    elif tf.rank(partition) < 2:
        raise IndexError(f"Partition dimension < 2!")
    else:
        n = tf.sqrt(tf.cast(tf.size(partition), tf.float32))
        if tf.math.mod(n, 1) != 0:
            raise NotImplementedError(f"Partition is not square !")
        
    # if n==8 or n==16:
    #     repeat_amount = tf.cast(tf.math.floor(desired_field_size / n), tf.int32)
    # else:
    repeat_amount = tf.cast(tf.math.ceil(desired_field_size / n), tf.int32)

    partition = tf.repeat(tf.repeat(partition, repeat_amount, axis=0), repeat_amount, axis=1)
    crop_amount = int(((repeat_amount * tf.cast(n, tf.int32)) - desired_field_size) // 2)
    if crop_amount != 0:
        if crop_amount > 0:
            partition = partition[crop_amount:-crop_amount, crop_amount:-crop_amount]
        elif crop_amount < 0:
            partition = tf.concat([tf.repeat(tf.reshape(partition[:, 0], (-1, 1)), tf.math.abs(crop_amount), axis=1), partition], axis=1)
            partition = tf.concat([partition, tf.repeat(tf.reshape(partition[:, -1], (-1, 1)), tf.math.abs(crop_amount), axis=1)], axis=1)
            partition = tf.concat([tf.repeat(tf.reshape(partition[0, :], (1, -1)), tf.math.abs(crop_amount), axis=0), partition], axis=0)
            partition = tf.concat([partition, tf.repeat(tf.reshape(partition[-1, :], (1, -1)), tf.math.abs(crop_amount), axis=0)], axis=0)
    return partition