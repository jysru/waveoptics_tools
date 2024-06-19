import os
import numpy as np
import tensorflow as tf
from ._utils import reduce_2d, make_bellshaped_plane_2d

_nan_file: str = 'dm_nan_mask_34x34.npy'


def load_nan_mask() -> np.ndarray:
    basepath = os.path.dirname(os.path.abspath(__file__))
    return np.load(os.path.join(basepath, _nan_file))

_nan_mask = load_nan_mask()


def compute_weights_from_energies(energies: np.ndarray) -> dict:
    weights = {}
    weights['energies'] = energies
    weights['all_idx'] = np.arange(0, weights['energies'].size)
    weights['nz_idx'] = np.flatnonzero(weights['energies'])
    weights['dead_idx'] = np.array(list(set(weights['all_idx']) - set(weights['nz_idx'])))
    weights['live_idx'] = np.array(list(set(weights['all_idx']) - set(weights['dead_idx'])))
    weights['amplitudes'] = np.delete(np.sqrt(weights['energies']).flatten(), weights['dead_idx'])
    return weights


def compute_weights_from_phase_maps(phases_maps: np.ndarray, beam_width: float = 4.425e-3) -> dict:
    phases_maps = np.reshape(phases_maps, (-1, *phases_maps.shape[-2:]))
    phases_reduced = reduce_2d(phases_maps)
    domain = np.squeeze(phases_reduced[0, :, :])
    amps = make_bellshaped_plane_2d(domain, width=beam_width)
    mask = np.isnan(domain)
    phases_reduced[:, mask] = 0.
    amps[mask] = 0.

    weights = {}
    weights['energies'] = np.square(amps)
    weights['all_idx'] = np.arange(0, domain.size)
    weights['nz_idx'] = np.flatnonzero(amps)
    weights['dead_idx'] = np.array(list(set(weights['all_idx']) - set(weights['nz_idx'])))
    weights['live_idx'] = np.array(list(set(weights['all_idx']) - set(weights['dead_idx'])))
    weights['amplitudes'] = np.delete(np.sqrt(weights['energies']).flatten(), weights['dead_idx'])
    return weights


def generate_weights(map_34x34: np.array, width: float = 4.425e-3) -> tuple[dict, np.ndarray]:
    weights={}
    if map_34x34.ndim < 3:
        map_34x34 = np.expand_dims(map_34x34, axis=0)
    domain = np.squeeze(reduce_2d(map_34x34))
    amps = make_bellshaped_plane_2d(domain, width=width)
    mask = np.isnan(domain)
    amps[mask] = 0.

    weights['energies'] = np.square(amps)
    weights['shape'] = domain.shape
    weights['all_idx'] = np.arange(0, domain.size)
    weights['nz_idx'] = np.flatnonzero(amps)
    weights['dead_idx'] = np.array(list(set(weights['all_idx']) - set(weights['nz_idx'])))
    weights['live_idx'] = np.array(list(set(weights['all_idx']) - set(weights['dead_idx'])))
    weights['amplitudes'] = np.delete(np.sqrt(weights['energies']).flatten(), weights['dead_idx'])
    return weights, domain


def vec_to_partition(vec, weights):
    if np.iscomplexobj(vec):
        vec = np.angle(vec)
    partition = np.zeros(shape=np.prod(weights['shape']), dtype=np.float64)
    partition[weights['live_idx']] = vec
    return partition.reshape(weights['shape'])


def mat34x34_to_vec(domain: np.ndarray, dead_idx: list[int]) -> np.ndarray:
    if domain.ndim < 3:
        domain = np.expand_dims(domain, axis=0)
    vec = reduce_2d(domain)
    vec = np.delete(vec, dead_idx)
    return vec


def reconstruct_34x34(phases: np.ndarray, weights: dict) -> np.ndarray:
    phi = np.zeros(np.prod(weights['shape']), dtype=np.float64)
    phi_live = np.delete(phases.flatten(), weights['dead_idx'])
    phi[weights['live_idx']] = phi_live
    phi = np.reshape(phi, weights['shape'])

    phi = np.repeat(phi, np.ceil(34 / weights['shape'][0]), axis=0)
    phi = np.repeat(phi, np.ceil(34 / weights['shape'][0]), axis=1)
    crop = (phi.shape[0] - 34)//2

    if crop > 0:
        phi = phi[crop:-crop, crop:-crop]
    phi[_nan_mask] = np.nan

    return phi



def partition_to_34x34(partition: np.ndarray, field_size: int = 34) -> np.ndarray:
    desired_field_size = field_size
    partition = np.squeeze(partition)
        
    if partition.ndim > 2:
        raise IndexError(f"Partition dimension > 2!")
    elif partition.ndim < 2:
        raise IndexError(f"Partition dimension < 2!")
    else:
        n = np.sqrt(partition.size)
        if np.mod(n, 1) != 0:
            raise NotImplementedError(f"Partition is not square !")
        
    if n==8 or n==16:
        repeat_amount = np.floor(desired_field_size / n)
    else:
        repeat_amount = np.ceil(desired_field_size / n)

    partition = np.repeat(np.repeat(partition, repeat_amount, axis=0), repeat_amount, axis=1)
    crop_amount = int(((repeat_amount * n) - desired_field_size) // 2)
    if crop_amount != 0:
        if crop_amount > 0:
            partition = partition[crop_amount:-crop_amount, crop_amount:-crop_amount]
        elif crop_amount < 0:
            partition = np.hstack([np.repeat(partition[:, 0].reshape(-1, 1), np.abs(crop_amount), axis=1), partition])
            partition = np.hstack([partition, np.repeat(partition[:, -1].reshape(-1, 1), np.abs(crop_amount), axis=1)])
            partition = np.vstack([np.repeat(partition[0, :].reshape(1, -1), np.abs(crop_amount), axis=0), partition])
            partition = np.vstack([partition, np.repeat(partition[-1, :].reshape(1, -1), np.abs(crop_amount), axis=0)])

    partition[_nan_mask] = np.nan
    return partition


def batch_partition_to_34x34(partition: np.ndarray, field_size: int = 34) -> np.ndarray:
    """Assumes partition dimensions are the last two in the case of batched data, thus, batch dimension is the first one"""
    desired_field_size = field_size
    partition = np.squeeze(partition)
    batch_size = partition.shape[0]
    partitionned = np.nan * np.zeros(shape=(batch_size, field_size, field_size))

    for i in range(batch_size):
        partitionned[i, ...] = partition_to_34x34(partition[i, ...], field_size=desired_field_size)

    return partitionned


def tf_partition_to_34x34(partition: tf.Tensor) -> np.ndarray:
    nan_mask = tf.convert_to_tensor(_nan_mask)
    desired_field_size = 34
    partition = tf.squeeze(partition)
        
    if tf.rank(partition) > 2:
        raise IndexError(f"Partition dimension > 2!")
    elif tf.rank(partition) < 2:
        raise IndexError(f"Partition dimension < 2!")
    else:
        n = tf.sqrt(tf.cast(tf.size(partition), tf.float32))
        if tf.math.mod(n, 1) != 0:
            raise NotImplementedError(f"Partition is not square !")
        
    if n==8 or n==16:
        repeat_amount = tf.cast(tf.math.floor(desired_field_size / n), tf.int32)
    else:
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

    partition = tf.where(nan_mask, 0, partition)
    return partition