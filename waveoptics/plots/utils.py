import numpy as np
from matplotlib.colors import hsv_to_rgb


def complex_to_hsv(z: np.ndarray, rmin: float = None, rmax: float = None, hue_start: int = 0):
    # Check if arguments are not None, else assign default values
    rmin = rmin if rmin is not None else 0
    rmax = rmax if rmax is not None else np.max(np.abs(z))

    # Get amplitude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)

    # Convert phase to degrees and offset to hue_start
    ph = np.angle(z, deg=True) + hue_start

    # Build HSV arrays (HSV are values in range [0,1])
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp - rmin) / (rmax - rmin)

    return hsv_to_rgb(np.dstack((h, s, v)))


def stitch_tm(tm: np.ndarray, weights, nans: bool = False, crop_outputs: bool = False, cropped_length: int = -1) -> np.ndarray:
    all_chans = np.size(weights['energies'])
    input_shape = weights['energies'].shape
    output_shape = int(np.sqrt(tm.shape[0]))
    output_shape = (output_shape, output_shape)

    if crop_outputs:
        crop_idx = (output_shape[0] - cropped_length) // 2 if cropped_length > 0 else 0
        cropped_shape = (cropped_length, cropped_length)
        final_shape = (input_shape[0] * cropped_shape[0], input_shape[1] * cropped_shape[1])
        stitched_image = np.zeros(shape=(*input_shape, *cropped_shape), dtype=tm.dtype)
    else:
        final_shape = (input_shape[0] * output_shape[0], input_shape[1] * output_shape[1])
        stitched_image = np.zeros(shape=(*input_shape, *output_shape), dtype=tm.dtype)
    
    if nans:
        stitched_image *= np.nan
    k = 0
    for chan in range(all_chans):
        idxs = np.unravel_index(chan, shape=input_shape)
        if chan in weights['live_idx']:
            tm_column = tm[..., k].reshape(output_shape)
            if crop_outputs:
                tm_column = tm_column[crop_idx:-crop_idx, crop_idx:-crop_idx]
            stitched_image[idxs[0], idxs[1], ...] = tm_column
            k += 1

    stitched_image = np.moveaxis(stitched_image, 2, 1)
    stitched_image = np.reshape(stitched_image, final_shape)
    return stitched_image


def stitch_array(array: np.ndarray, nans: bool = False, crop_outputs: bool = False, cropped_length: int = -1) -> np.ndarray:
    chans = array.shape[1]
    divs = squarest_divisors(chans)
    output_shape = int(np.sqrt(array.shape[0]))
    output_shape = (output_shape, output_shape)

    if crop_outputs:
        crop_idx = (output_shape[0] - cropped_length) // 2 if cropped_length > 0 else 0
        cropped_shape = (cropped_length, cropped_length)
        final_shape = (divs[0] * cropped_shape[0], divs[1] * cropped_shape[1])
        stitched_image = np.zeros(shape=(*divs, *cropped_shape), dtype=array.dtype)
    else:
        final_shape = (divs[0] * output_shape[0], divs[1] * output_shape[1])
        stitched_image = np.zeros(shape=(*divs, *output_shape), dtype=array.dtype)
    
    if nans:
        stitched_image *= np.nan
    k = 0
    for chan in range(chans):
        idxs = np.unravel_index(chan, shape=divs)
        array_column = array[..., k].reshape(output_shape)
        if crop_outputs:
            array_column = array_column[crop_idx:-crop_idx, crop_idx:-crop_idx]
        stitched_image[idxs[0], idxs[1], ...] = array_column
        k += 1

    stitched_image = np.moveaxis(stitched_image, 2, 1)
    stitched_image = np.reshape(stitched_image, final_shape)
    return stitched_image


def squarest_divisors(n):
    divs = np.array(list(get_divisors(n)))
    best_div = np.max(divs[divs <= np.sqrt(n)])
    other_div = n // best_div
    return best_div, other_div


def get_divisors(n) -> list[int]:
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            yield i
    yield n
