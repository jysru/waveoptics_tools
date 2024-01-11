import os
import numpy as np
import h5py
import scipy.io as sio
import mat73

from . import loaders


def matfile_to_hdf5(
        filepath: str,
        avoid_keys: list[str] = [],
        verbose: bool = True,
        return_path: bool = True,
        ) -> str:
    
    basename, _ = os.path.splitext(filepath)
    savename = f"{basename}.hdf5"
    dset = loaders.load_matfile(filepath)

    if verbose:
        print(f"Converting matfile {filepath}...")
    
    # Iterate through keys, save as h5py datasets
    with h5py.File(savename, 'w') as hf:
        for key_name in dset:
            print(key_name)
            if key_name not in avoid_keys:
                hf.create_dataset(name=key_name, data=dset[key_name])
                if verbose:
                    print(f"\tSaved {key_name}.")
            else:
                if verbose:
                    print(f"\tSkipped {key_name}.")

    if verbose:
        print(f"Conversion is complete.")
        print(f"Saved HDF5 file: {savename}")

    if return_path:
        return savename
