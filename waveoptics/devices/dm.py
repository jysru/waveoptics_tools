import os
import numpy as np
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