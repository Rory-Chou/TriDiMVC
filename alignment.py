"""
Alignment utilities for partially misaligned multi-view data.
Reference: C-OT project.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def compute_similarity_matrix(data_view0: np.ndarray,
                              data_view1: np.ndarray,
                              metric: str = 'cosine',
                              normalize: bool = False) -> np.ndarray:
    """
    Compute the similarity matrix between two views.

    Args:
        data_view0: View-0 data, shape (n_view0, feature_dim0)
        data_view1: View-1 data, shape (n_view1, feature_dim1)
        metric: Similarity metric ('euclidean', 'cosine', 'dot')
        normalize: Whether to L2-normalize inputs

    Returns:
        similarity_matrix: Shape (n_view0, n_view1)
    """
    if normalize:
        # L2 normalization
        data_view0 = data_view0 / (np.linalg.norm(data_view0, axis=1, keepdims=True) + 1e-8)
        data_view1 = data_view1 / (np.linalg.norm(data_view1, axis=1, keepdims=True) + 1e-8)

    if metric == 'euclidean':
        # Euclidean distance converted to similarity: 1 / (1 + distance)
        distance_matrix = cdist(data_view0, data_view1, metric='euclidean')
        similarity_matrix = 1.0 / (1.0 + distance_matrix)

    elif metric == 'cosine':
        # Cosine similarity
        similarity_matrix = np.dot(data_view0, data_view1.T)
        if not normalize:
            # Divide by norms if inputs were not normalized
            norm0 = np.linalg.norm(data_view0, axis=1, keepdims=True)
            norm1 = np.linalg.norm(data_view1, axis=1, keepdims=True)
            similarity_matrix = similarity_matrix / (norm0 * norm1.T + 1e-8)

    elif metric == 'dot':
        # Dot-product similarity
        similarity_matrix = np.dot(data_view0, data_view1.T)

    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

    return similarity_matrix


def align_data_max_similarity(data_view0: np.ndarray,
                              data_view1: np.ndarray,
                              metric: str = 'cosine',
                              normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Max-similarity matching: for each sample in view 0, pick the most similar sample in view 1.
    Complexity O(N^2), much faster than Hungarian O(N^3).
    Suitable for large-scale data but not guaranteed globally optimal.

    One-to-many matching is allowed: when n_view1 < n_view0, one view-1 sample may match
    multiple view-0 samples so every view-0 sample has a match and |z_sem| = |z_shared| = n_view0.

    Args:
        data_view0: View-0 data, shape (n_view0, feature_dim0)
        data_view1: View-1 data, shape (n_view1, feature_dim1)
        metric: Similarity metric
        normalize: Whether to normalize data

    Returns:
        alignment_indices: Pairs of indices, shape (n_view0, 2); col 0 = view-0 idx, col 1 = view-1 idx
        aligned_view0: Aligned view-0 data, shape (n_view0, feature_dim0)
        aligned_view1: Aligned view-1 data, shape (n_view0, feature_dim1); duplicates allowed
    """
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(data_view0, data_view1, metric=metric, normalize=normalize)

    n_view0, n_view1 = similarity_matrix.shape

    # For each view-0 row, take argmax over view-1 columns (one-to-many allowed)
    row_ind = np.arange(n_view0)
    col_ind = np.argmax(similarity_matrix, axis=1)

    # When n_view1 < n_view0, one view-1 sample may match multiple view-0 samples

    # Build alignment index pairs
    alignment_indices = np.column_stack([row_ind, col_ind])

    # Build aligned data
    aligned_view0 = data_view0[row_ind]
    aligned_view1 = data_view1[col_ind]

    return alignment_indices, aligned_view0, aligned_view1


def align_data_auto(data_view0: np.ndarray,
                    data_view1: np.ndarray,
                    metric: str = 'cosine',
                    normalize: bool = True,
                    maximize: bool = True,
                    use_hungarian: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Automatically align two views (reference: C-OT project).

    Args:
        data_view0: View-0 data, shape (n_view0, feature_dim0)
        data_view1: View-1 data, shape (n_view1, feature_dim1)
        metric: Similarity metric
        normalize: Whether to normalize data
        maximize: Whether to maximize similarity
        use_hungarian: Hungarian (True) vs max-similarity matching (False)
                      - Hungarian: O(N^3), globally optimal but slower
                      - Max similarity: O(N^2), fast but not globally optimal

    Returns:
        alignment_indices: Alignment pairs, shape (n_aligned, 2)
        aligned_view0: Aligned view-0 data
        aligned_view1: Aligned view-1 data
        cost_matrix: Cost matrix (only when Hungarian is used)
    """
    if use_hungarian:
        # Hungarian algorithm (global optimum, slower)
        similarity_matrix = compute_similarity_matrix(data_view0, data_view1, metric=metric, normalize=normalize)

        # Build cost matrix
        if maximize:
            cost_matrix = -similarity_matrix
        else:
            cost_matrix = similarity_matrix.copy()

        # Optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build alignment index pairs
        alignment_indices = np.column_stack([row_ind, col_ind])

        # Build aligned data
        aligned_view0 = data_view0[row_ind]
        aligned_view1 = data_view1[col_ind]

        return alignment_indices, aligned_view0, aligned_view1, cost_matrix
    else:
        # Max-similarity matching (fast, not globally optimal)
        alignment_indices, aligned_view0, aligned_view1 = align_data_max_similarity(
            data_view0, data_view1, metric=metric, normalize=normalize
        )
        # Empty cost_matrix for API consistency
        cost_matrix = None
        return alignment_indices, aligned_view0, aligned_view1, cost_matrix
