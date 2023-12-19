import numpy as np

def fft_1d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(field)))
    return ft / np.sqrt(ft.size) if normalize else ft

def ifft_1d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ift = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(field)))
    return ift * np.sqrt(ift.size) if normalize else ift

def fft_2d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    return ft / np.sqrt(ft.size) if normalize else ft

def ifft_2d(field: np.ndarray, normalize: bool = True) -> np.ndarray:
    ift = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field)))
    return ift * np.sqrt(ift.size) if normalize else ift

def frt_1d(field: np.ndarray,
           propagator: np.ndarray = None,
           dz: float = 0.0,
           wavelength: float = 1064e-9,
           pixel_size: float = 5.04e-6,
           ) -> np.ndarray:
    if propagator is None:
        propagator = frt_1d_propagator(field, dz, wavelength, pixel_size)
    return ifft_1d(fft_1d(field) * np.exp(1j * propagator))

def frt_2d(field: np.ndarray,
           propagator: np.ndarray = None,
           dz: float = 0.0,
           wavelength: float = 1064e-9,
           pixel_size: float = 5.04e-6,
           ) -> np.ndarray:
    if propagator is None:
        propagator = frt_2d_propagator(field, dz, wavelength, pixel_size)
    return ifft_2d(fft_2d(field) * np.exp(1j * propagator))

def frt_1d_propagator(field: np.ndarray,
                      dz: float = 0.0,
                      wavelength: float = 1064e-9,
                      pixel_size: float = 5.04e-6,
                      )  -> np.ndarray:
    _, kx = fft_1d_grids(field, pixel_size)
    return dz * np.sqrt(np.abs(4 * np.square(np.pi/wavelength) - np.square(kx)))
    
def frt_2d_propagator(field: np.ndarray,
                      dz: float = 0.0,
                      wavelength: float = 1064e-9,
                      pixel_size: float = 5.04e-6,
                      )  -> np.ndarray:
    _, _, kx, ky = fft_2d_grids(field, pixel_size)
    return dz * np.sqrt(np.abs(4 * np.square(np.pi/wavelength) - np.square(kx) - np.square(ky)))

def fft_1d_grids(field: np.ndarray, pixel_size: float) -> tuple[np.ndarray]:
    # Spatial plane
    dx = pixel_size
    n_pts = field.shape[0]
    grid_size = dx * (n_pts - 1)
    lim_x = n_pts / 2 * dx
    x = np.arange(-lim_x, lim_x, dx)

    # Conjugate plane
    dnx = 1 / grid_size
    lim_nx = (n_pts / 2) * dnx
    kx = 2 * np.pi * np.arange(-lim_nx, lim_nx, dnx)
    return (x, kx)

def fft_2d_grids(field: np.ndarray, pixel_size: float) -> tuple[np.ndarray]:
    # Spatial plane
    dx = pixel_size
    n_pts = field.shape[0]
    grid_size = dx * (n_pts - 1)
    lim_x = n_pts / 2 * dx
    x = np.arange(-lim_x, lim_x, dx)
    x, y = np.meshgrid(x, x)

    # Conjugate plane
    dnx = 1 / grid_size
    lim_nx = (n_pts / 2) * dnx
    kx = 2 * np.pi * np.arange(-lim_nx, lim_nx, dnx)
    kx, ky = np.meshgrid(kx, kx)
    return (x, y, kx, ky)
