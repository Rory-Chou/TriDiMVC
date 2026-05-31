import math
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances between rows of x and y.
    x: [n_x, d]
    y: [n_y, d]
    return: [n_x, n_y]
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [n_x, 1]
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T  # [1, n_y]
    dist = x_norm + y_norm - 2.0 * x @ y.T
    return torch.clamp(dist, min=0.0)


def uot_sinkhorn(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.1,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Entropic unbalanced optimal transport using Sinkhorn iterations.

    Solve:
        min_T <T, C> + eps * KL(T || a ⊗ b)
                    + tau_a * KL(T 1 || a)
                    + tau_b * KL(T^T 1 || b)

    Args:
        C: cost matrix [n_a, n_b]
        a: source weights [n_a]
        b: target weights [n_b]
        eps: entropy regularization
        tau_a, tau_b: marginal relaxation strengths
        max_iters: max Sinkhorn iterations
        tol: stopping criterion on scaling updates

    Returns:
        T: optimal transport plan [n_a, n_b]
    """
    device = C.device
    n_a, n_b = C.shape
    a = a.to(device)
    b = b.to(device)

    K = torch.exp(-C / eps)  # [n_a, n_b]
    u = torch.ones(n_a, device=device)
    v = torch.ones(n_b, device=device)

    # Unbalanced exponents; as tau -> +inf, rho -> 1 recovers balanced OT
    rho_a = tau_a / (tau_a + eps)
    rho_b = tau_b / (tau_b + eps)

    K_plus = K + 1e-16

    for _ in range(max_iters):
        u_prev = u.clone()
        v_prev = v.clone()

        Kv = K_plus @ v  # [n_a]
        u = (a / Kv).pow(rho_a)

        KTu = K_plus.T @ u  # [n_b]
        v = (b / KTu).pow(rho_b)

        if torch.max(torch.abs(u - u_prev)).item() < tol and torch.max(
            torch.abs(v - v_prev)
        ).item() < tol:
            break

    T = torch.diag(u) @ K @ torch.diag(v)
    return T


def row_max_and_entropy(T: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute row-wise max probability and entropy from a transport matrix.

    Args:
        T: [n_a, n_b], non-negative.
    Returns:
        row_max: [n_a] max normalized probability per row
        row_entropy: [n_a] entropy of normalized row distribution
    """
    row_mass = T.sum(dim=1, keepdim=True)  # [n_a, 1]
    row_mass_safe = torch.clamp(row_mass, min=eps)
    P = T / row_mass_safe

    row_max = P.max(dim=1).values  # [n_a]
    P_safe = torch.clamp(P, min=eps)
    row_entropy = -(P_safe * P_safe.log()).sum(dim=1)  # [n_a]
    return row_max, row_entropy


def topk_impute_private(
    T: torch.Tensor,
    z_p_src: torch.Tensor,
    z_p_tgt: torch.Tensor,
    k: int = 5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    For each source sample i, top-k weighted average on target using T[i,:]
    to impute target-view private semantics.

    Args:
        T: [n_src, n_tgt] transport matrix (non-negative)
        z_p_src: [n_src, d_p] (unused; kept for symmetric API)
        z_p_tgt: [n_tgt, d_p]
        k: top-k neighbors (capped at n_tgt)
    Returns:
        z_p_tgt_imputed: [n_src, d_p]
    """
    n_src, n_tgt = T.shape
    k = min(k, n_tgt)

    row_mass = T.sum(dim=1, keepdim=True)  # [n_src, 1]
    row_mass_safe = torch.clamp(row_mass, min=eps)
    P = T / row_mass_safe  # Row-normalize

    topk_vals, topk_idx = torch.topk(P, k=k, dim=1)  # [n_src, k]

    weight_sum = torch.clamp(topk_vals.sum(dim=1, keepdim=True), min=eps)
    weights = topk_vals / weight_sum  # [n_src, k]

    z_neighbors = z_p_tgt[topk_idx]  # [n_src, k, d_p]
    z_p_tgt_imputed = (weights.unsqueeze(-1) * z_neighbors).sum(dim=1)  # [n_src, d_p]
    return z_p_tgt_imputed


def prototype_impute_private(
    z_s_src: torch.Tensor,
    z_s_tgt: torch.Tensor,
    z_p_tgt: torch.Tensor,
    n_clusters: int,
    unmatched_src_indices: List[int],
) -> torch.Tensor:
    """
    Impute private semantics for unmatched samples using in-view cluster prototypes.

    For unmatched source i, find the cluster center in view B most similar to z_s_src[i]
    and use that cluster's private semantic prototype as the imputation.

    Clustering uses concatenated z_s_B and z_p_B to capture full semantic structure in view B.

    Args:
        z_s_src: [n_src, d_s] shared semantics in view A (for nearest cluster)
        z_s_tgt: [n_tgt, d_s] shared semantics in view B (for clustering)
        z_p_tgt: [n_tgt, d_p] private semantics in view B (clustering and prototypes)
        n_clusters: Number of clusters
        unmatched_src_indices: Indices of unmatched source samples

    Returns:
        z_p_tgt_imputed: [n_src, d_p] imputed view-B private semantics for each view-A sample
    """
    device = z_s_src.device
    n_src = z_s_src.shape[0]
    d_p = z_p_tgt.shape[1]

    # Initialize output
    z_p_tgt_imputed = torch.zeros(n_src, d_p, device=device, dtype=z_p_tgt.dtype)

    if len(unmatched_src_indices) == 0:
        return z_p_tgt_imputed

    # 1. Cluster concatenated shared + private semantics in view B
    z_sem_tgt = torch.cat([z_s_tgt, z_p_tgt], dim=-1)  # [n_tgt, d_s + d_p]
    z_sem_tgt_np = z_sem_tgt.cpu().numpy()
    kmeans_sem = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans_sem.fit_predict(z_sem_tgt_np)  # [n_tgt]
    cluster_centers_sem = torch.tensor(kmeans_sem.cluster_centers_, device=device, dtype=z_sem_tgt.dtype)  # [n_clusters, d_s + d_p]

    # 2. Private semantic prototypes per cluster in view B
    cluster_centers_p = torch.zeros(n_clusters, d_p, device=device, dtype=z_p_tgt.dtype)
    for c in range(n_clusters):
        mask = torch.tensor(cluster_labels == c, device=device)
        if mask.sum() > 0:
            cluster_centers_p[c] = z_p_tgt[mask].mean(dim=0)
        else:
            # Empty cluster: use global mean
            cluster_centers_p[c] = z_p_tgt.mean(dim=0)

    # 3. Unmatched sources: nearest cluster by shared semantics, use private prototype
    unmatched_src_tensor = torch.tensor(unmatched_src_indices, dtype=torch.long, device=device)
    z_s_unmatched = z_s_src[unmatched_src_tensor]  # [n_unmatched, d_s]

    d_s = z_s_tgt.shape[1]
    cluster_centers_s = cluster_centers_sem[:, :d_s]  # [n_clusters, d_s]

    distances = pairwise_squared_distances(z_s_unmatched, cluster_centers_s)  # [n_unmatched, n_clusters]

    closest_cluster_idx = distances.argmin(dim=1)  # [n_unmatched]

    z_p_prototypes = cluster_centers_p[closest_cluster_idx]  # [n_unmatched, d_p]

    for idx, i in enumerate(unmatched_src_indices):
        z_p_tgt_imputed[i] = z_p_prototypes[idx]

    return z_p_tgt_imputed


def build_z_sem1(
    z_s_A: torch.Tensor,
    z_p_A: torch.Tensor,
    z_p_B: torch.Tensor,
    z_p_B_imputed_for_A: torch.Tensor,
    matches: List[Tuple[int, int]],
    unmatched_A: List[int],
) -> torch.Tensor:
    """
    Build semantic fusion for view 1 (assumes n_view1 >= n_view2).

    z_sem1:
      - Matched: [z_s^A, z_p^A, z_p^B] (true matched z_p_B)
      - Unmatched: [z_s^A, z_p^A, hat z_p^B] (imputed z_p_B)

    Args:
        z_s_A: [n_A, d_s] shared semantics view 1
        z_p_A: [n_A, d_p] private semantics view 1
        z_p_B: [n_B, d_p] private semantics view 2
        z_p_B_imputed_for_A: [n_A, d_p] imputed view-2 private for each view-1 sample
        matches: One-to-one matches [(i, j), ...]
        unmatched_A: Unmatched indices in view 1

    Returns:
        z_sem1: [n_A, d_s + 2*d_p] view-1 fusion (original view-1 order)
    """
    device = z_s_A.device
    n_A = z_s_A.shape[0]
    d_s = z_s_A.shape[1]
    d_p = z_p_A.shape[1]
    d_sem = d_s + 2 * d_p

    z_sem1 = torch.zeros(n_A, d_sem, device=device, dtype=z_s_A.dtype)

    matched_A_to_B = {i: j for i, j in matches}

    for i in range(n_A):
        if i in matched_A_to_B:
            j = matched_A_to_B[i]
            z_p_B_real = z_p_B[j]
            z_sem1[i] = torch.cat([z_s_A[i], z_p_A[i], z_p_B_real], dim=-1)
        else:
            z_sem1[i] = torch.cat([z_s_A[i], z_p_A[i], z_p_B_imputed_for_A[i]], dim=-1)

    return z_sem1


def build_one_to_one_matches(
    T: torch.Tensor,
    row_max: torch.Tensor,
    col_max: torch.Tensor,
    row_entropy: torch.Tensor,
    col_entropy: torch.Tensor,
    max_threshold: float = None,
    entropy_threshold: float = None,
    percentile_threshold: float = 1.0,  # Percentile scale (default 1.0 = original logic)
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Build matches (i, j) from T (one-to-many allowed):
    - Use row max m_i and row entropy H_i only
    - High m_i & low H_i -> sharp row -> create match
    - Low m_i & high H_i -> diffuse row -> no match
    - If matched, pair with argmax_j T_ij

    Adaptive thresholds:
    - From view-1 / view-2 sample ratio via percentiles
    - e.g. 100 vs 90 samples -> ratio=0.9
    - Row max threshold: ((1 - ratio) * percentile_threshold * 100) percentile
    - Row entropy threshold: (ratio * percentile_threshold * 100) percentile (lower is better)
    - percentile_threshold:
      * = 1.0: original logic (default)
      * < 1.0: looser matching (more samples pass)
      * > 1.0: stricter matching (fewer samples pass)

    Returns:
      matches: [(i,j)] (one-to-many allowed)
      unmatched_A: view-A indices with no match
      unmatched_B: view-B indices not in any match
    """
    n_A, n_B = T.shape

    row_argmax = T.argmax(dim=1)  # [n_A] best j per row

    if max_threshold is None or entropy_threshold is None:
        ratio = n_B / n_A

        row_max_np = row_max.cpu().numpy()
        row_entropy_np = row_entropy.cpu().numpy()

        if max_threshold is None:
            percentile_max = (1 - ratio) * percentile_threshold * 100
            percentile_max = max(0.0, min(100.0, percentile_max))
            max_threshold = float(np.percentile(row_max_np, percentile_max))

        if entropy_threshold is None:
            percentile_entropy = ratio * percentile_threshold * 100
            percentile_entropy = max(0.0, min(100.0, percentile_entropy))
            entropy_threshold = float(np.percentile(row_entropy_np, percentile_entropy))

    entropy_threshold_row = entropy_threshold

    matches: List[Tuple[int, int]] = []
    matched_A = set()
    matched_B = set()

    for i in range(n_A):
        m_i = row_max[i].item()
        H_i = row_entropy[i].item()

        if m_i >= max_threshold and H_i <= entropy_threshold_row:
            j = row_argmax[i].item()
            matches.append((i, j))
            matched_A.add(i)
            matched_B.add(j)

    unmatched_A = [i for i in range(n_A) if i not in matched_A]
    unmatched_B = [j for j in range(n_B) if j not in matched_B]

    return matches, unmatched_A, unmatched_B


def build_z_sem2(
    z_s_A: torch.Tensor,
    z_p_A: torch.Tensor,
    z_s_B: torch.Tensor,
    z_p_B: torch.Tensor,
    z_p_B_imputed_for_A: torch.Tensor,
    matches: List[Tuple[int, int]],
    unmatched_A: List[int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build semantic fusion for view 1 (n_view1 >= n_view2; returns n_A samples).

    z_sem2:
      - Matched (i,j):
          z_sem_pair = [ mean(z_s^A[i], z_s^B[j]), z_p^A[i], z_p^B[j] ]
      - Unmatched view-1 i:
          [z_s^A[i], z_p^A[i], hat z_p^B[i]]

    Args:
        z_s_A: [n_A, d_s] shared semantics view 1
        z_p_A: [n_A, d_p] private semantics view 1
        z_s_B: [n_B, d_s] shared semantics view 2
        z_p_B: [n_B, d_p] private semantics view 2
        z_p_B_imputed_for_A: [n_A, d_p] imputed view-2 private per view-1 sample
        matches: One-to-one matches [(i, j), ...]
        unmatched_A: Unmatched indices in view 1

    Returns:
        z_sem2: [n_A, d_s + 2*d_p] view-1 fusion (original order)
        meta: Dict with match metadata
    """
    device = z_s_A.device
    n_A = z_s_A.shape[0]
    d_s = z_s_A.shape[1]
    d_p = z_p_A.shape[1]
    d_sem = d_s + 2 * d_p

    z_sem2 = torch.zeros(n_A, d_sem, device=device, dtype=z_s_A.dtype)

    matched_A_set = set()
    pair_A_idx_list = []
    pair_B_idx_list = []

    if len(matches) > 0:
        for i, j in matches:
            pair_A_idx_list.append(i)
            pair_B_idx_list.append(j)
            matched_A_set.add(i)

            z_s_pair = 0.5 * (z_s_A[i] + z_s_B[j])
            z_sem_pair = torch.cat([z_s_pair, z_p_A[i], z_p_B[j]], dim=-1)
            z_sem2[i] = z_sem_pair

    if len(unmatched_A) > 0:
        unmatched_A_idx_tensor = torch.tensor(unmatched_A, dtype=torch.long, device=device)
        z_sem_A_unmatched = torch.cat(
            [
                z_s_A[unmatched_A_idx_tensor],
                z_p_A[unmatched_A_idx_tensor],
                z_p_B_imputed_for_A[unmatched_A_idx_tensor],
            ],
            dim=-1,
        )
        for idx, i in enumerate(unmatched_A):
            z_sem2[i] = z_sem_A_unmatched[idx]
    else:
        unmatched_A_idx_tensor = torch.empty(0, dtype=torch.long, device=device)

    pair_A_idx = torch.tensor(pair_A_idx_list, dtype=torch.long, device=device) if pair_A_idx_list else torch.empty(0, dtype=torch.long, device=device)
    pair_B_idx = torch.tensor(pair_B_idx_list, dtype=torch.long, device=device) if pair_B_idx_list else torch.empty(0, dtype=torch.long, device=device)

    meta: Dict[str, torch.Tensor] = {
        "pairs_A_idx": pair_A_idx,
        "pairs_B_idx": pair_B_idx,
        "unmatched_A_idx": unmatched_A_idx_tensor,
    }

    return z_sem2, meta


def run_uot_semantic_fusion(
    z_s_A: torch.Tensor,
    z_p_A: torch.Tensor,
    z_s_B: torch.Tensor,
    z_p_B: torch.Tensor,
    eps: float = 0.1,
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    sinkhorn_iters: int = 500,
    sinkhorn_tol: float = 1e-6,
    topk: int = 5,
    match_max_threshold: Optional[float] = None,  # None = adaptive threshold
    percentile_threshold: float = 1.0,  # Percentile scale (default 1.0)
    n_clusters: Optional[int] = None,  # Clusters for prototype imputation
    labels_A: Optional[torch.Tensor] = None,
    labels_B: Optional[torch.Tensor] = None,
) -> Dict[str, Union[torch.Tensor, Dict[str, Union[int, float]]]]:
    """
    High-level pipeline (assumes n_view1 >= n_view2):

    1. Cost matrix C from shared z_s.
    2. Unbalanced Sinkhorn -> transport matrix T.
    3. Row/column max and entropy; mutual argmax + sharpness for one-to-one matches.
    4. Prototype imputation for unmatched samples (v7).
    5. z_sem1: view-1 [z_s, z_p, hat z_p from view 2].
    6. z_sem2: matched pairs + unmatched fusion (n_A samples).

    Args:
        z_s_A, z_p_A: [n_A, d_s], [n_A, d_p] view-1 features (n_A >= n_B)
        z_s_B, z_p_B: [n_B, d_s], [n_B, d_p] view-2 features
        percentile_threshold: Percentile scale (default 1.0)
            - = 1.0: original logic
            - < 1.0: looser matching
            - > 1.0: stricter matching
        n_clusters: Clusters for prototype imputation; if None, use top-k imputation
        labels_A: [n_A] view-1 labels (optional, for stats)
        labels_B: [n_B] view-2 labels (optional, for stats)

    Returns:
        dict:
            "T": [n_A, n_B] transport matrix
            "row_max_A": [n_A] row max probability view 1
            "row_entropy_A": [n_A] row entropy view 1
            "row_max_B": [n_B] column max probability view 2
            "row_entropy_B": [n_B] column entropy view 2
            "z_p_B_imputed_for_A": [n_A, d_p] imputed view-2 private per view-1 sample
            "z_sem1": [n_A, d_s + 2*d_p] view-1 semantic fusion
            "z_sem2": [n_A, d_s + 2*d_p] view-1 fusion (matched + imputed)
            "pairs_A_idx": LongTensor[num_pairs] matched view-1 indices
            "pairs_B_idx": LongTensor[num_pairs] matched view-2 indices
            "unmatched_A_idx": LongTensor[num_unmatched_A] unmatched view-1 indices
            "stats": dict (if labels provided):
                "num_matched_pairs": int number of one-to-one matches
                "num_imputed_samples": int imputed count (unmatched in view A)
                "match_accuracy": float fraction of matches with same class label
    """
    device = z_s_A.device
    n_A, d_s = z_s_A.shape
    n_B = z_s_B.shape[0]
    assert n_A >= n_B, "View 1 sample count must be >= view 2 sample count"
    assert z_s_B.shape[1] == d_s, "z_s_A and z_s_B must have same shared dim"
    assert z_p_A.shape[0] == n_A and z_p_B.shape[0] == n_B, "z_p shapes mismatch"

    # 1. Cost matrix in shared space
    C = pairwise_squared_distances(z_s_A, z_s_B)  # [n_A, n_B]

    # 2. Uniform source/target weights
    a = torch.full((n_A,), 1.0 / n_A, device=device)
    b = torch.full((n_B,), 1.0 / n_B, device=device)

    # 3. Unbalanced OT
    T = uot_sinkhorn(
        C,
        a,
        b,
        eps=eps,
        tau_a=tau_a,
        tau_b=tau_b,
        max_iters=sinkhorn_iters,
        tol=sinkhorn_tol,
    )

    # 4. Row/column statistics
    row_max_A, row_entropy_A = row_max_and_entropy(T)
    col_max_B, col_entropy_B = row_max_and_entropy(T.T)

    # 5. One-to-one matching (adaptive threshold if match_max_threshold is None)
    matches, unmatched_A, unmatched_B = build_one_to_one_matches(
        T,
        row_max=row_max_A,
        col_max=col_max_B,
        row_entropy=row_entropy_A,
        col_entropy=col_entropy_B,
        max_threshold=match_max_threshold if match_max_threshold is not None else None,
        entropy_threshold=None,
        percentile_threshold=percentile_threshold,
    )

    # 6. Impute view-2 private for view-1 (only view-1 gets imputed z_p from view 2)
    # v7: cluster prototypes instead of k-NN
    if n_clusters is not None and len(unmatched_A) > 0:
        z_p_B_imputed_for_A = prototype_impute_private(
            z_s_src=z_s_A,
            z_s_tgt=z_s_B,
            z_p_tgt=z_p_B,
            n_clusters=n_clusters,
            unmatched_src_indices=unmatched_A,
        )
        # Matched samples use true z_p_B in build_z_sem1/build_z_sem2; imputed values only for unmatched
    else:
        z_p_B_imputed_for_A = topk_impute_private(T, z_p_src=z_p_A, z_p_tgt=z_p_B, k=topk)

    # 7. z_sem1 (view-1 semantic fusion)
    z_sem1 = build_z_sem1(
        z_s_A=z_s_A,
        z_p_A=z_p_A,
        z_p_B=z_p_B,
        z_p_B_imputed_for_A=z_p_B_imputed_for_A,
        matches=matches,
        unmatched_A=unmatched_A,
    )

    # 8. z_sem2 (view-1 fusion: matched fusion + unmatched imputation)
    z_sem2, meta = build_z_sem2(
        z_s_A=z_s_A,
        z_p_A=z_p_A,
        z_s_B=z_s_B,
        z_p_B=z_p_B,
        z_p_B_imputed_for_A=z_p_B_imputed_for_A,
        matches=matches,
        unmatched_A=unmatched_A,
    )

    # 9. Statistics (if labels provided)
    stats = {}
    if labels_A is not None:
        labels_A = labels_A.to(device)

        num_matched_pairs = len(matches)
        num_imputed_samples = len(unmatched_A)

        if labels_B is not None:
            labels_B = labels_B.to(device)
            if num_matched_pairs > 0:
                correct_matches = 0
                for i, j in matches:
                    if labels_A[i].item() == labels_B[j].item():
                        correct_matches += 1
                match_accuracy = correct_matches / num_matched_pairs
            else:
                match_accuracy = 0.0
        else:
            # No labels_B (misaligned mode): cannot compute match accuracy
            match_accuracy = -1.0

        stats = {
            "num_matched_pairs": num_matched_pairs,
            "num_imputed_samples": num_imputed_samples,
            "match_accuracy": match_accuracy,
        }

    out: Dict[str, torch.Tensor] = {
        "T": T,
        "row_max_A": row_max_A,
        "row_entropy_A": row_entropy_A,
        "row_max_B": col_max_B,
        "row_entropy_B": col_entropy_B,
        "z_p_B_imputed_for_A": z_p_B_imputed_for_A,
        "z_sem1": z_sem1,
        "z_sem2": z_sem2,
        "pairs_A_idx": meta["pairs_A_idx"],
        "pairs_B_idx": meta["pairs_B_idx"],
        "unmatched_A_idx": meta["unmatched_A_idx"],
        "stats": stats,
    }
    return out
