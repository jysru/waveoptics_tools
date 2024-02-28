import numpy as np

def svd_filter_tm(tm: np.ndarray, alpha: int = -1) -> np.ndarray:
    U, S, Vh = np.linalg.svd(tm, full_matrices=False)
    alpha = S.size if (alpha == -1 or alpha > S.size) else alpha
    S[alpha:] = 0
    return (U @ np.diag(S) @ Vh)

