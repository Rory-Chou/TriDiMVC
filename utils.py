"""
Utility module.
Contains logging setup, training utilities, and related helpers.
"""

import os
import logging
import numpy as np
import torch
from datetime import datetime
from typing import List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint
from alignment import align_data_auto

logger = logging.getLogger(__name__)


def setup_logging_to_file(
    dataset_name: str,
    start_time: str,
    best_accuracy: float = 0.0,
    logs_root: Optional[str] = None,
) -> str:
    """
    Configure file-based logging (following the C-OT project logging pattern).

    Args:
        dataset_name: Dataset name.
        start_time: Start time string (format: YYYYMMDD_HHMMSS).
        best_accuracy: Best accuracy (used in the log filename).
        logs_root: Log root directory; defaults to ``logs`` under the current directory when None.

    Returns:
        log_file_path: Path to the log file.
    """
    # Create logs root directory (default under TriD-MVC; external path also supported)
    if logs_root is None:
        logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    logs_root = os.path.abspath(logs_root)
    os.makedirs(logs_root, exist_ok=True)

    # Create dataset subdirectory
    dataset_dir = os.path.join(logs_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Log filename: timestamp_bestACC.txt
    time_str = start_time
    acc_str = f"{best_accuracy:.4f}".replace('.', '_')
    log_filename = f"{time_str}_ACC{acc_str}.txt"

    log_file_path = os.path.join(dataset_dir, log_filename)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')  # Omit timestamp prefix
    file_handler.setFormatter(file_formatter)

    # Attach handler to logger
    logger.addHandler(file_handler)

    return log_file_path


def rename_log_file(old_path: str, new_best_accuracy: float) -> str:
    """
    Rename the log file according to the new best accuracy.

    On Windows, file handlers must be closed before renaming the file.

    Args:
        old_path: Previous log file path.
        new_best_accuracy: New best accuracy.

    Returns:
        new_path: New log file path.
    """
    if not os.path.exists(old_path):
        return old_path

    # Extract directory and old filename
    directory = os.path.dirname(old_path)
    old_filename = os.path.basename(old_path)

    # Extract timestamp from old filename
    # Format: timestamp_ACCvalue.txt
    parts = old_filename.rsplit('_ACC', 1)
    if len(parts) == 2:
        time_part = parts[0]
        # Build new filename
        acc_str = f"{new_best_accuracy:.4f}".replace('.', '_')
        new_filename = f"{time_part}_ACC{acc_str}.txt"
    else:
        # Fall back to current time if format is unexpected
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        acc_str = f"{new_best_accuracy:.4f}".replace('.', '_')
        new_filename = f"{time_str}_ACC{acc_str}.txt"

    new_path = os.path.join(directory, new_filename)

    # Rename file
    if old_path != new_path:
        file_handler_to_remove = None
        try:
            # On Windows, close the file handler before renaming
            # Find and close the FileHandler tied to the old file
            for handler in logger.handlers[:]:  # Slice copy to avoid mutation during iteration
                if isinstance(handler, logging.FileHandler):
                    handler_path = os.path.abspath(handler.baseFilename)
                    old_path_abs = os.path.abspath(old_path)
                    if handler_path == old_path_abs:
                        file_handler_to_remove = handler
                        break

            # Flush and close the handler if found
            if file_handler_to_remove is not None:
                # Flush buffer so all log lines are written
                file_handler_to_remove.flush()
                file_handler_to_remove.close()
                logger.removeHandler(file_handler_to_remove)

            # Flush all handlers before renaming
            for handler in logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()

            # Rename file
            os.rename(old_path, new_path)

            # Recreate file handler for the new path
            if file_handler_to_remove is not None:
                new_file_handler = logging.FileHandler(new_path, mode='a', encoding='utf-8')
                new_file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter('%(message)s')  # Omit timestamp prefix
                new_file_handler.setFormatter(file_formatter)
                logger.addHandler(new_file_handler)

            logger.info(f"Log file renamed: {os.path.basename(old_path)} -> {os.path.basename(new_filename)}")
        except Exception as e:
            logger.warning(f"Failed to rename log file: {e}, continuing with original file")
            # If rename fails, try to restore the file handler (if it was closed)
            if file_handler_to_remove is not None:
                try:
                    old_file_handler = logging.FileHandler(old_path, mode='a', encoding='utf-8')
                    old_file_handler.setLevel(logging.INFO)
                    file_formatter = logging.Formatter('%(message)s')  # Omit timestamp prefix
                    old_file_handler.setFormatter(file_formatter)
                    logger.addHandler(old_file_handler)
                except Exception as restore_error:
                    logger.error(f"Failed to restore file handler: {restore_error}")
            return old_path

    return new_path


def get_all_semantic_embeddings(model, dataloader, device):
    """
    Iterate over the full dataset and collect semantic representations, labels, and indices,
    then reorder by the original sample order.

    Returns:
        all_sem: Semantic representations (shared + private).
        all_y: Ground-truth labels.
    """
    model.eval()
    all_sem = []
    all_y = []
    all_indices = []
    with torch.no_grad():
        for views, labels, indices in dataloader:
            views = [v.to(device) for v in views]
            out = model(views)
            sem = out["sem"]
            all_sem.append(sem.cpu())
            all_y.append(labels)
            all_indices.append(indices)
    all_sem = torch.cat(all_sem, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    # reorder to dataset order
    order = all_indices.argsort()
    all_sem = all_sem[order]
    all_y = all_y[order]
    return all_sem, all_y


def get_all_shared_embeddings(model, dataloader, device):
    """
    Iterate over the full dataset and collect shared representations (for standalone clustering evaluation).

    Returns:
        all_zs: Shared representations (cross-view averaged shared features).
        all_y: Ground-truth labels.
    """
    model.eval()
    all_zs = []
    all_y = []
    all_indices = []
    with torch.no_grad():
        for views, labels, indices in dataloader:
            views = [v.to(device) for v in views]
            out = model(views)
            zs_list = out["zs_list"]
            # Average shared features across views
            zs_stack = torch.stack(zs_list, dim=0)  # [V, B, D_s]
            zs_mean = torch.mean(zs_stack, dim=0)   # [B, D_s]
            all_zs.append(zs_mean.cpu())
            all_y.append(labels)
            all_indices.append(indices)
    all_zs = torch.cat(all_zs, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    # reorder to dataset order
    order = all_indices.argsort()
    all_zs = all_zs[order]
    all_y = all_y[order]
    return all_zs, all_y


def get_per_view_embeddings(model, dataloader, device):
    """
    Iterate over the full dataset and collect per-view [z_s, z_p] features (for single-view clustering evaluation).

    Returns:
        per_view_embeddings: List of per-view [z_s, z_p] features; each element has shape (N, D_s + D_p).
        all_y: Ground-truth labels.
    """
    model.eval()
    per_view_embeddings = [[] for _ in range(model.n_views)]
    all_y = []
    all_indices = []

    with torch.no_grad():
        for views, labels, indices in dataloader:
            views = [v.to(device) for v in views]
            out = model(views)
            zs_list = out["zs_list"]
            zp_list = out["zp_list"]

            # Concatenate z_s and z_p for each view
            for v_idx in range(model.n_views):
                sem_view = torch.cat([zs_list[v_idx], zp_list[v_idx]], dim=-1)  # [B, D_s + D_p]
                per_view_embeddings[v_idx].append(sem_view.cpu())

            all_y.append(labels)
            all_indices.append(indices)

    # Concatenate all batches
    for v_idx in range(model.n_views):
        per_view_embeddings[v_idx] = torch.cat(per_view_embeddings[v_idx], dim=0)

    all_y = torch.cat(all_y, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    # Reorder to dataset order
    order = all_indices.argsort()
    for v_idx in range(model.n_views):
        per_view_embeddings[v_idx] = per_view_embeddings[v_idx][order]
    all_y = all_y[order]

    return per_view_embeddings, all_y


def get_aligned_embeddings_for_misaligned_data(model, dataloader, device, use_hungarian=False, dataset=None):
    """
    For misaligned data, extract features and align them with an alignment algorithm (following the C-OT project).
    Note: alignment matching always uses z_shared as guidance, because z_p is private and may differ across views.

    Args:
        model: Trained model.
        dataloader: Data loader (contains misaligned data).
        device: Device.
        use_hungarian: Use Hungarian algorithm (True) or maximum-similarity matching (False).
        dataset: Dataset object (optional, for mask information).

    Returns:
        alignment_indices: Alignment index pairs, shape (n_aligned, 2).
        all_zs_view0: Shared features for view 0 (N, D_s).
        all_zs_view1: Shared features for view 1 (N, D_s).
        all_zp_view0: Private features for view 0 (N, D_p).
        all_zp_view1: Private features for view 1 (N, D_p).
        all_y: Labels (N,).
        all_indices: Original indices (N,).
        mask_view0: Mask for view 0 (N,) if the dataset provides masks.
        mask_view1: Mask for view 1 (N,) if the dataset provides masks.
    """
    model.eval()
    all_zs_view0 = []
    all_zs_view1 = []
    all_zp_view0 = []
    all_zp_view1 = []
    all_y = []
    all_indices = []

    with torch.no_grad():
        for views, labels, indices in dataloader:
            views = [v.to(device) for v in views]
            out = model(views)

            zs_list = out["zs_list"]
            zp_list = out["zp_list"]

            all_zs_view0.append(zs_list[0].cpu().numpy())
            all_zs_view1.append(zs_list[1].cpu().numpy())
            all_zp_view0.append(zp_list[0].cpu().numpy())
            all_zp_view1.append(zp_list[1].cpu().numpy())
            all_y.append(labels.numpy())
            all_indices.append(indices.numpy())

    # Concatenate all samples
    all_zs_view0 = np.concatenate(all_zs_view0, axis=0)  # (N, D_s)
    all_zs_view1 = np.concatenate(all_zs_view1, axis=0)  # (N, D_s)
    all_zp_view0 = np.concatenate(all_zp_view0, axis=0)  # (N, D_p)
    all_zp_view1 = np.concatenate(all_zp_view1, axis=0)  # (N, D_p)
    all_y = np.concatenate(all_y, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    # Load mask information if available
    mask_view0 = None
    mask_view1 = None
    if dataset is not None and hasattr(dataset, 'test_X1_mask') and hasattr(dataset, 'test_X2_mask'):
        # Keep test-set masks only (training set is complete)
        if dataset.mode == 'test':
            mask_view0 = dataset.test_X1_mask
            mask_view1 = dataset.test_X2_mask
        elif dataset.mode == 'all':
            # In 'all' mode, training set is complete; test set has masks
            n_train = len(dataset.train_X1)
            n_test = len(dataset.test_X1)
            # Apply masks only to the test portion
            train_mask_view0 = np.ones(n_train, dtype=bool)
            train_mask_view1 = np.ones(n_train, dtype=bool)
            test_mask_view0 = dataset.test_X1_mask
            test_mask_view1 = dataset.test_X2_mask
            mask_view0 = np.concatenate([train_mask_view0, test_mask_view0])
            mask_view1 = np.concatenate([train_mask_view1, test_mask_view1])

    # Align using z_shared (z_p is private and may not be similar across views)
    alignment_indices, aligned_zs_view0, aligned_zs_view1, _ = align_data_auto(
        all_zs_view0, all_zs_view1,
        metric='cosine',
        normalize=True,
        maximize=True,
        use_hungarian=use_hungarian
    )

    return alignment_indices, all_zs_view0, all_zs_view1, all_zp_view0, all_zp_view1, all_y, all_indices, mask_view0, mask_view1


def get_sn(view_num, alldata_len, missing_rate):
    """
    Randomly generate incomplete-view metadata by simulating partial views from complete data.
    Reference implementation: C-OT/data/dataset.py.

    Args:
        view_num: Number of views.
        alldata_len: Number of samples.
        missing_rate: Missing rate (as defined in paper Section 4.3).

    Returns:
        Sn: Missingness mask matrix, shape (alldata_len, view_num); 1 = present, 0 = missing.
    """
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate

    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve

    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix

    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

    return matrix
