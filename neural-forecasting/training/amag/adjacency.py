import numpy as np


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute channel-wise correlation matrix from training data.

    Per the paper: flatten features across time, compute Pearson correlation.
    Corr(u,v) = X_u · X_v^T / (||X_u|| · ||X_v||)

    Args:
        data: (N, T, C, F) normalized training data
    Returns:
        corr_matrix: (C, C) correlation matrix
    """
    n, t, c, f = data.shape
    # Flatten time and features per channel: (N, C, T*F)
    flat = data.reshape(n, t, c * f).transpose(0, 2, 1)  # wrong
    # Actually: (N, T, C, F) -> per sample, per channel: (T*F,) vector
    # Reshape to (N, C, T*F)
    flat = data.transpose(0, 2, 1, 3).reshape(n, c, t * f)  # (N, C, T*F)

    # Average correlation across samples
    corr_sum = np.zeros((c, c), dtype=np.float64)
    for i in range(n):
        x = flat[i]  # (C, T*F)
        # Normalize each row
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        x_normed = x / norms
        corr = x_normed @ x_normed.T  # (C, C)
        corr_sum += corr

    corr_matrix = corr_sum / n
    return corr_matrix.astype(np.float32)
