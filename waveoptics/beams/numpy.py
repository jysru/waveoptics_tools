import numpy as np

def gaussian_1d(grid: np.ndarray,
                width: float = 10e-6,
                amplitude: float = 1,
                center: float = 0,
                ) -> np.ndarray:
    return amplitude * np.exp(- np.square((grid - center) / width))

def gaussian_2d(grids: tuple[np.ndarray],
                widths: tuple[float] = (10e-6, 10e-6),
                amplitude: float = 1,
                centers: tuple[float] = (0, 0),
                ) -> np.ndarray:
    return amplitude * np.exp(
         - np.square((grids[0] - centers[0]) / widths[0])
         - np.square((grids[1] - centers[1]) / widths[1])
         )
