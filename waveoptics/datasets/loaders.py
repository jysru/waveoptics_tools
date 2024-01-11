import os
import numpy as np
import h5py
import scipy.io
import mat73


def _keep_keys(dictionary: dict, keys_to_keep: list[str]) -> dict:
    if keys_to_keep is None:
        return dictionary
    else:
        return {key: value for key, value in dictionary.items() if key in keys_to_keep}
    

def load_matfile(filepath: str, keys: list[str] = None)-> dict:
    try:
        loaded_data = scipy.io.loadmat(filepath)
    except NotImplementedError:
        loaded_data = mat73.loadmat(filepath)
    loaded_data = _keep_keys(loaded_data.copy, keys)
    return loaded_data


def load_h5file(filepath: str, keys: list[str] = None) -> dict:
    with h5py.File(filepath, 'r') as hf:
        loaded_data = {}
        for key in hf.keys():
            loaded_data[key] = np.array(hf[key])
    return _keep_keys(loaded_data, keys)
