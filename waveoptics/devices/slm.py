import os
import numpy as np
import tensorflow as tf
from waveoptics.tensors.numpy import crop_2d
from ._utils import reduce_2d, make_bellshaped_plane_2d


_DEFAULT_SCREEN_WIDTH = 1680
_DEFAULT_SCREEN_HEIGHT = 1050
_DEFAULT_SCREEN_WIDTH = 1920
_DEFAULT_SCREEN_HEIGHT = 1080



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




class SLMPistonSquare:
    _screen_width: int = _DEFAULT_SCREEN_WIDTH
    _screen_height: int = _DEFAULT_SCREEN_HEIGHT

    def __init__(self,
                 n_act_1d: int = 8,
                 width: int = None,
                 height: int = None,
                 roi_size: int = 192,
                 roi_shifts: tuple[int] = (541, 624),
                 ) -> None:
        self.n_act_1d = n_act_1d
        self.n_act_2d = self.n_act_1d ** 2
        self.size_act = None
        self.roi_size = roi_size
        self.roi_centers_xy = roi_shifts
        self._screen_width = width if width is not None else SLMPistonSquare._screen_width
        self._screen_height = height if height is not None else SLMPistonSquare._screen_height
        self.actuator_phases = None
        self.phase_matrix = None

    def generate_phases(self, n: int) -> np.ndarray:
        phi = 2 * np.pi * np.random.rand(n, n)
        self.n_act_1d = n
        self.n_act_2d = n ** 2
        self.image_from_phases(phi)

    def image_from_phases(self, phases_array):
        self.actuator_phases = phases_array
        n = np.round(np.sqrt(phases_array.size)).astype(np.int32)
        
        ideal_actu_size = int(np.ceil(self.roi_size / n))
        self.size_act = ideal_actu_size

        phase_map = np.repeat(phases_array, repeats=ideal_actu_size, axis=0)
        phase_map = np.repeat(phase_map, repeats=ideal_actu_size, axis=1)

        if phase_map.shape[0] != self.roi_size: 
            phase_map = crop_2d(phase_map, new_shape=(self.roi_size, self.roi_size))

        map = np.zeros((self._screen_height, self._screen_width), dtype=np.complex64)
        # map[1::2, ::2] = 0
        # map[::2, 1::2] = 0

        map[0:self.roi_size, 0:self.roi_size] = np.exp(1j * phase_map)
        map = np.roll(map, shift=self.roi_centers_xy[0] - self.roi_size // 2, axis=0)
        map = np.roll(map, shift=self.roi_centers_xy[1] - self.roi_size // 2, axis=1)

        self.phase_matrix = map

    def generate_field(self, n: int) -> np.ndarray:
        phi = 2 * np.pi * np.random.rand(n, n)
        self.n_act_1d = n
        self.n_act_2d = n ** 2
        self.field_from_phases(phi)

    def field_from_phases(self, phases_array):
        self.actuator_phases = phases_array
        n = np.round(np.sqrt(phases_array.size)).astype(np.int32)
        
        ideal_actu_size = int(np.ceil(self.roi_size / n))
        self.size_act = ideal_actu_size

        phase_map = np.repeat(phases_array, repeats=ideal_actu_size, axis=0)
        phase_map = np.repeat(phase_map, repeats=ideal_actu_size, axis=1)

        if phase_map.shape[0] != self.roi_size: 
            phase_map = crop_2d(phase_map, new_shape=(self.roi_size, self.roi_size))

        map = np.zeros((self._screen_height, self._screen_width), dtype=np.complex64)

        map[0:self.roi_size, 0:self.roi_size] = np.exp(1j * phase_map)
        map = np.roll(map, shift=self.roi_centers_xy[0] - self.roi_size // 2, axis=0)
        map = np.roll(map, shift=self.roi_centers_xy[1] - self.roi_size // 2, axis=1)

        self.field_matrix = map
