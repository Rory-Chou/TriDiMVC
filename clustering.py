"""
Clustering evaluation: K-means and metrics (ACC, NMI, ARI, F1).
Optional GPU acceleration via cuML or FAISS.
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment

# Optional GPU backends
try:
    import cuml
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


def run_kmeans(embeddings, n_clusters, n_init=20, random_state=None, use_gpu=True):
    """
    Run K-means clustering (optional GPU).

    Args:
        embeddings: Embeddings (numpy array or torch tensor)
        n_clusters: Number of clusters
        n_init: Number of initializations
        random_state: Random seed
        use_gpu: Try GPU if available

    Returns:
        preds: Predicted cluster labels
        kmeans: sklearn KMeans on CPU; None on GPU backends
    """
    # Convert torch tensors to numpy
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()
    elif hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Count unique points to avoid degenerate K-means
    unique_embeddings = np.unique(embeddings, axis=0)
    n_unique = len(unique_embeddings)

    # Cap cluster count by number of unique samples
    actual_n_clusters = min(n_clusters, n_unique)

    if actual_n_clusters < n_clusters:
        logger.warning(f"Only {n_unique} unique samples found, but n_clusters={n_clusters}. "
                      f"Adjusting n_clusters to {actual_n_clusters} to avoid convergence warning.")

    # Try GPU
    if use_gpu:
        # Prefer cuML
        if CUML_AVAILABLE:
            try:
                import cupy as cp
                # Move data to GPU
                gpu_embeddings = cp.asarray(embeddings)
                # cuML KMeans
                kmeans = cuml.KMeans(n_clusters=actual_n_clusters, n_init=n_init, random_state=random_state)
                preds = kmeans.fit_predict(gpu_embeddings)
                # Back to CPU numpy
                if hasattr(preds, 'get'):
                    preds = preds.get()
                else:
                    preds = cp.asnumpy(preds)
                return preds, None
            except Exception as e:
                logger.debug(f"cuML GPU KMeans failed: {e}, falling back to CPU")

        # FAISS GPU
        if FAISS_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    # FAISS GPU KMeans
                    d = embeddings.shape[1]
                    # FAISS KMeans object
                    kmeans = faiss.Kmeans(d, actual_n_clusters, niter=300, nredo=n_init,
                                         gpu=True, seed=random_state if random_state is not None else 1234,
                                         verbose=False)
                    # Train
                    kmeans.train(embeddings.astype(np.float32))
                    # Predict: nearest centroid via flat index
                    # Build flat index and add centroids
                    index = faiss.IndexFlatL2(d)
                    if torch.cuda.is_available():
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    index.add(kmeans.centroids.astype(np.float32))
                    # Nearest centroid search
                    _, preds = index.search(embeddings.astype(np.float32), 1)
                    preds = preds.flatten().astype(np.int64)
                    return preds, None
            except Exception as e:
                logger.debug(f"FAISS GPU KMeans failed: {e}, falling back to CPU")

    # CPU sklearn fallback
    kmeans = KMeans(n_clusters=actual_n_clusters, n_init=n_init, random_state=random_state)
    preds = kmeans.fit_predict(embeddings)
    return preds, kmeans


def evaluate_clustering(y_true, y_pred):
    """
    Evaluate clustering: ACC (Hungarian), NMI, ARI, F1.

    Args:
        y_true: Ground-truth labels
        y_pred: Predicted cluster labels

    Returns:
        acc, nmi, ari, f1
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    # ACC and F1 with Hungarian label matching
    # Confusion matrix: rows = predicted, cols = true
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Hungarian assignment (maximize matches)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = w[row_ind, col_ind].sum() * 1.0 / y_pred.size

    # Label map (same as used for ACC)
    label_map = {}
    for pred_idx, true_idx in zip(row_ind, col_ind):
        label_map[pred_idx] = true_idx

    # Map unmapped predicted labels to most frequent true label
    actual_pred_labels = np.unique(y_pred)
    unmapped_preds = set(actual_pred_labels) - set(label_map.keys())
    if len(unmapped_preds) > 0:
        if len(y_true) > 0:
            most_common_true = np.bincount(y_true).argmax()
        else:
            most_common_true = 0
        for pred_idx in unmapped_preds:
            label_map[pred_idx] = most_common_true

    # Apply mapping
    y_pred_mapped = np.array([label_map[pred] for pred in y_pred])

    # Macro F1 on mapped labels
    f1 = f1_score(y_true, y_pred_mapped, average='macro', zero_division=0)

    return acc, nmi, ari, f1


def _get_y_pred_mapped(y_true, y_pred):
    """
    Map predicted labels to ground-truth space using the same Hungarian mapping as ACC.
    Returns y_pred_mapped for per-class accuracy.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    label_map = {}
    for pred_idx, true_idx in zip(row_ind, col_ind):
        label_map[pred_idx] = true_idx
    actual_pred_labels = np.unique(y_pred)
    unmapped_preds = set(actual_pred_labels) - set(label_map.keys())
    if len(unmapped_preds) > 0 and len(y_true) > 0:
        most_common_true = np.bincount(y_true).argmax()
        for pred_idx in unmapped_preds:
            label_map[pred_idx] = most_common_true
    return np.array([label_map[pred] for pred in y_pred])


def compute_class_ranking_by_clustering(embeddings, y_true, n_clusters, random_state=42, use_gpu=True):
    """
    Run K-means once and rank classes by per-class clustering accuracy (best first).
    Useful for visualizing the top-k best-clustered classes.

    Args:
        embeddings: Representations for clustering (N, d)
        y_true: Ground-truth labels (N,); non-contiguous IDs supported
        n_clusters: Number of clusters
        random_state: K-means random seed
        use_gpu: Use GPU if available

    Returns:
        class_ranking: 1D array; class_ranking[0] is the best-clustered class label, etc.
    """
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()
    elif hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    elif hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    y_true = np.asarray(y_true).ravel()

    preds, _ = run_kmeans(embeddings, n_clusters, random_state=random_state, use_gpu=use_gpu)
    y_pred_mapped = _get_y_pred_mapped(y_true, preds)

    unique_labels = np.unique(y_true)
    per_class_acc = np.zeros(len(unique_labels))
    for i, c in enumerate(unique_labels):
        mask = (y_true == c)
        n_c = mask.sum()
        if n_c > 0:
            correct = ((y_pred_mapped == c) & mask).sum()
            per_class_acc[i] = correct / n_c
        else:
            per_class_acc[i] = 0.0
    # Descending order by per-class accuracy
    order = np.argsort(-per_class_acc)
    class_ranking = unique_labels[order].astype(np.float64)
    return class_ranking


def evaluate_kmeans(sem_all, y_all, n_clusters, base_seed=None, n_runs=5, use_gpu=True):
    """
    K-means evaluation averaged over multiple runs (optional GPU).

    Args:
        sem_all: Semantic representations (N, d); torch or numpy
        y_all: Ground-truth labels (N,); torch or numpy
        n_clusters: Number of clusters
        base_seed: Base seed; if None, each run is random
        n_runs: Number of runs (default 5)
        use_gpu: Try GPU if available

    Returns:
        results: {'ACC', 'NMI', 'ARI', 'F1'} means
    """
    # Convert to numpy
    if hasattr(sem_all, 'cpu'):
        sem_all = sem_all.cpu().numpy()
    elif hasattr(sem_all, 'numpy'):
        sem_all = sem_all.numpy()
    if hasattr(y_all, 'cpu'):
        y_all = y_all.cpu().numpy()
    elif hasattr(y_all, 'numpy'):
        y_all = y_all.numpy()

    all_acc = []
    all_nmi = []
    all_ari = []
    all_f1 = []

    # K-means on semantic representations
    for i in range(n_runs):
        if base_seed is not None:
            random_state = base_seed + i
        else:
            random_state = None

        kmeans_preds, _ = run_kmeans(sem_all, n_clusters, random_state=random_state, use_gpu=use_gpu)
        kmeans_acc, kmeans_nmi, kmeans_ari, kmeans_f1 = evaluate_clustering(y_all, kmeans_preds)

        all_acc.append(kmeans_acc)
        all_nmi.append(kmeans_nmi)
        all_ari.append(kmeans_ari)
        all_f1.append(kmeans_f1)

    # Average metrics
    results = {
        'ACC': np.mean(all_acc),
        'NMI': np.mean(all_nmi),
        'ARI': np.mean(all_ari),
        'F1': np.mean(all_f1)
    }

    return results
