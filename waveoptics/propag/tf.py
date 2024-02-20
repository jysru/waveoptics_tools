import numpy as np
import tensorflow as tf
import waveoptics.propag.numpy


def fft_1d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ft = tf.signal.fftshift(tf.signal.fft(tf.signal.fftshift(field)))
    if normalize:
        field_shape = field.shape
        numel_image = tf.math.reduce_prod(field_shape[-2:])
        return ft / tf.cast(tf.math.sqrt(tf.cast(numel_image, tf.float64)), ft.dtype)
    else:
        return ft

def ifft_1d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ft = tf.signal.ifftshift(tf.signal.ifft(tf.signal.ifftshift(field)))
    if normalize:
        field_shape = field.shape
        numel_image = tf.math.reduce_prod(field_shape[-2:])
        return ft * tf.cast(tf.math.sqrt(tf.cast(numel_image, tf.float64)), ft.dtype)
    else:
        return ft

def fft_2d(field: tf.Tensor, normalize: bool = True) -> tf.Tensor:
    ft = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(field)))
    if normalize:
        field_shape = field.shape
        numel_image = tf.math.reduce_prod(field_shape[-2:])
        return ft / tf.cast(tf.math.sqrt(tf.cast(numel_image, tf.float64)), ft.dtype)
    else:
        return ft

def ifft_2d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ft = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(field)))
    if normalize:
        field_shape = field.shape
        numel_image = tf.math.reduce_prod(field_shape[-2:])
        return ft * tf.cast(tf.math.sqrt(tf.cast(numel_image, tf.float64)), ft.dtype)
    else:
        return ft

def frt_1d(field: np.ndarray,
           propagator: np.ndarray = None,
           dz: float = 0.0,
           wavelength: float = 1064e-9,
           pixel_size: float = 5.04e-6,
           ) -> np.ndarray:
    if propagator is None:
        propagator = frt_1d_propagator(field, dz, wavelength, pixel_size)
    return ifft_1d(fft_1d(field) * tf.cast(tf.complex(tf.cos(propagator), tf.sin(propagator)), tf.complex64))

def frt_2d(field: np.ndarray,
           propagator: np.ndarray = None,
           dz: float = 0.0,
           wavelength: float = 1064e-9,
           pixel_size: float = 5.04e-6,
           ) -> np.ndarray:
    if propagator is None:
        propagator = frt_2d_propagator(field, dz, wavelength, pixel_size)
    return ifft_2d(fft_2d(field) * tf.cast(tf.complex(tf.cos(propagator), tf.sin(propagator)), tf.complex64))

def frt_2d_fourier_mask(field: np.ndarray,
           propagator: np.ndarray = None,
           dz: float = 0.0,
           wavelength: float = 1064e-9,
           pixel_size: float = 5.04e-6,
           mask: np.ndarray = None,
           ) -> np.ndarray:
    if propagator is None:
        propagator = frt_2d_propagator(field, dz, wavelength, pixel_size)
    if mask is None:
        return ifft_2d(fft_2d(field) * tf.cast(tf.complex(tf.cos(propagator), tf.sin(propagator)), tf.complex64))
    else:
        return ifft_2d(fft_2d(field) * tf.cast(tf.complex(tf.cos(propagator), tf.sin(propagator)), tf.complex64), * tf.cast(mask, tf.complex64))

def frt_1d_propagator(field: np.ndarray,
                      dz: float = 0.0,
                      wavelength: float = 1064e-9,
                      pixel_size: float = 5.04e-6,
                      )  -> np.ndarray:
    return tf.convert_to_tensor(
        waveoptics.propag.numpy.frt_1d_propagator(field, dz, wavelength, pixel_size)
        )
    
def frt_2d_propagator(field: np.ndarray,
                      dz: float = 0.0,
                      wavelength: float = 1064e-9,
                      pixel_size: float = 5.04e-6,
                      )  -> np.ndarray:
    return tf.convert_to_tensor(
        waveoptics.propag.numpy.frt_2d_propagator(field, dz, wavelength, pixel_size)
        )



