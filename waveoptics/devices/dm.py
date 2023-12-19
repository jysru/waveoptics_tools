import os
import numpy as np

_nan_file: str = 'dm_nan_mask_34x34.npy'

def load_nan_mask() -> np.ndarray:
    basepath = os.path.dirname(os.path.abspath(__file__))
    return np.load(os.path.join(basepath, _nan_file))
