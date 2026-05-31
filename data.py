"""
Dataset module.
Multi-view dataset loading and preprocessing.
"""

import os
import logging
import numpy as np
import scipy.io as sio
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_l2(data):
    """L2 normalization."""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return data / norms


def load_multi_view_data(
    dataset: str,
    dataset_path: Optional[str] = None,
    shuffle_views: bool = True,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load a multi-view dataset from the datasets/ directory.
    Implementation follows Topo/main.py.

    Args:
        dataset: Dataset name.
        dataset_path: Path to the .mat file; if None, search automatically.
        shuffle_views: Whether to shuffle the two views to simulate misalignment.
        seed: Random seed.

    Returns:
        X1, X2: Two view matrices (N, d1), (N, d2).
        label: Labels (N,).
        n_clusters: Number of classes.
    """
    # Resolve dataset file path
    if dataset_path is None:
        # Try several candidate paths
        possible_paths = [
            os.path.join('datasets', dataset + '.mat'),
            os.path.join('../..', 'datasets', dataset + '.mat'),
            os.path.join('.', 'datasets', dataset + '.mat'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', dataset + '.mat'),
        ]

        mat_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mat_path = path
                break

        if mat_path is None:
            raise FileNotFoundError(
                f"Could not find {dataset}.mat\n"
                f"Tried paths:\n" + "\n".join([f"  - {p}" for p in possible_paths])
            )
    else:
        mat_path = dataset_path

    # Load .mat file
    logger.info(f"Loading {dataset} from {mat_path}...")
    mat = sio.loadmat(mat_path)

    # Load views and labels per dataset
    data = []
    label = None

    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['gt'])

    elif dataset == 'Caltech101-20':
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['gt'])

    elif dataset == 'Orl_mtv':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][0].T)
        data.append(mat['X'][0][1].T)

    elif dataset == 'HandWritten':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])

    elif dataset == 'ALOI':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])

    elif dataset == 'Reuters_dim10':
        data.append(normalize_l2(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize_l2(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif dataset == 'NoisyMNIST-30000':
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

    elif dataset == '2view-caltech101-8677sample':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][0].T)
        data.append(mat['X'][0][1].T)

    elif dataset == 'MNIST-USPS':
        data.append(mat['X1'])
        data.append(normalize_l2(mat['X2']))
        label = np.squeeze(mat['Y'])

    elif dataset == 'AWA-7view-10158sample':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][5].T)
        data.append(mat['X'][0][6].T)

    elif dataset == 'caltech7':
        label = np.squeeze(mat['Y'])
        data.append(mat['X1'])
        data.append(mat['X2'])

    elif dataset == 'BDGP':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][0].T)
        data.append(mat['X'][0][1].T)

    elif dataset == 'BBCsports':
        label = np.squeeze(mat['Y'])
        data.append(mat['X1'])
        data.append(mat['X2'])

    elif dataset == '3Sources':
        label = np.squeeze(mat['Y'])
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])

    elif dataset == 'YouTube_X':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])

    elif dataset == 'HandWritten_X':
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])

    elif dataset == '100leaves':
        mat['data'][0][0], mat['data'][0][1] = mat['data'][0][0].T, mat['data'][0][1].T
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])

    elif dataset == 'Prokaryotic':
        value1 = mat['X'][0][0]
        value2 = mat['X'][2][0]
        data = [value1, value2]
        label = np.squeeze(mat['y'])

    elif dataset == 'yale_mtv':
        mat['X'][0][0], mat['X'][0][1] = mat['X'][0][0].T, mat['X'][0][1].T
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])

    elif dataset == 'feature_matrix':
        features = mat['feature']
        feature_data = features[:, :6144]  # 302 x 6144
        label = features[:, 6144].astype(np.int64)  # 302
        view0 = feature_data[:, :2048]    # 302 x 2048
        view1 = feature_data[:, 2048:4096]  # 302 x 2048
        data = [view0, view1]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}\n"
                        f"Supported datasets: Scene15, Caltech101-20, Orl_mtv, HandWritten, ALOI, "
                        f"Reuters_dim10, NoisyMNIST-30000, 2view-caltech101-8677sample, "
                        f"MNIST-USPS, AWA-7view-10158sample, caltech7, BDGP, BBCsports, "
                        f"3Sources, YouTube_X, HandWritten_X, 100leaves, Prokaryotic, "
                        f"yale_mtv, feature_matrix")

    if label is None:
        raise ValueError(f"Failed to load labels for dataset {dataset}")

    if len(data) < 2:
        raise ValueError(f"Dataset {dataset} requires at least 2 views, but only {len(data)} were loaded")

    # Remap labels to start at 0
    min_label = np.min(label)
    if min_label == 1:
        logger.info("Labels start at 1; subtracting 1 so they start at 0")
        label = label - 1

    # Extract two views
    X1 = data[0]  # first view
    X2 = data[1]  # second view

    # Ensure 2D arrays
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    logger.info(f"{dataset} loaded:")
    logger.info(f"  samples: {X1.shape[0]}")
    logger.info(f"  view 1 dim: {X1.shape[1]}")
    logger.info(f"  view 2 dim: {X2.shape[1]}")
    logger.info(f"  num classes: {len(np.unique(label))}")

    # L2 normalization for numerical stability
    # Note: Reuters_dim10 and MNIST-USPS are normalized during loading
    if dataset not in ['Reuters_dim10', 'MNIST-USPS']:
        X1 = normalize_l2(X1)
        X2 = normalize_l2(X2)

    # Optional view shuffle to simulate misalignment
    # Note: TriD-MVC needs aligned sample pairs for contrastive learning;
    # shuffle_views=True turns positive pairs into negatives, so we disable it
    perm = None
    if shuffle_views:
        logger.warning("shuffle_views=True breaks TriD-MVC contrastive learning (aligned pairs required); disabled automatically")
        shuffle_views = False
        perm = None

    n_clusters = len(np.unique(label))

    return X1.astype(np.float32), X2.astype(np.float32), label.astype(np.int64), n_clusters


class MultiViewDigits(Dataset):
    """
    Simple multi-view dataset:
    - sklearn digits (1797 x 64)
    - Split features into n_views
    """
    def __init__(self, n_views=2):
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target.astype(np.int64)

        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        self.X = X
        self.y = y
        self.n_views = n_views

        # Simple feature split into views
        n_features = X.shape[1]
        assert n_views <= 3, "This simple splitter only supports up to 3 views."
        splits = np.array_split(np.arange(n_features), n_views)
        self.view_indices = splits

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        views = [x[idxs] for idxs in self.view_indices]
        y = self.y[idx]
        # Returns: list of view features, label (eval only), sample index
        return [torch.from_numpy(v) for v in views], torch.tensor(y, dtype=torch.long), idx


class MultiViewRealDataset(Dataset):
    """
    Real multi-view dataset:
    - Load from .mat files
    - Same return format as MultiViewDigits
    """
    def __init__(self, dataset: str, dataset_path: Optional[str] = None,
                 shuffle_views: bool = True, seed: int = 0):
        X1, X2, label, n_clusters = load_multi_view_data(
            dataset=dataset,
            dataset_path=dataset_path,
            shuffle_views=shuffle_views,
            seed=seed
        )

        self.X1 = X1
        self.X2 = X2
        self.y = label
        self.n_clusters = n_clusters
        self.n_views = 2

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        view1 = torch.from_numpy(self.X1[idx])
        view2 = torch.from_numpy(self.X2[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        # Returns: list of view features, label (eval only), sample index
        return [view1, view2], y, idx


class PartiallyMisalignedDataset(Dataset):
    """
    Partially misaligned multi-view dataset (C-OT style):
    - Train: aligned and complete
    - Test: shuffle view 2 for misalignment; optional incomplete data
    - mode='all': full data (train aligned, test misaligned)
    """
    def __init__(self, dataset: str, dataset_path: Optional[str] = None,
                 misalign_ratio: float = 0.3, missing_rate: float = 0.0, seed: int = 0,
                 mode: str = 'train'):
        """
        Args:
            dataset: Dataset name.
            dataset_path: Path to the .mat file.
            misalign_ratio: Fraction of test samples that are misaligned.
            missing_rate: Fraction of incomplete test data (0.0 = complete).
            seed: Random seed.
            mode: 'train', 'test', or 'all'
                - 'train': aligned training set only
                - 'test': misaligned test set (optionally incomplete)
                - 'all': train (aligned) + test (misaligned)
        """
        from utils import get_sn

        # Load original aligned data
        X1, X2, label, n_clusters = load_multi_view_data(
            dataset=dataset,
            dataset_path=dataset_path,
            shuffle_views=False,  # load aligned first
            seed=seed
        )

        n_samples = len(X1)
        rng = np.random.RandomState(seed)

        # Train / test split
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_train = int((1 - misalign_ratio) * n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # Train: aligned and complete
        self.train_X1 = X1[train_indices]
        self.train_X2 = X2[train_indices]
        self.train_y = label[train_indices]

        # Test: shuffle view 2
        test_X1 = X1[test_indices]
        test_X2 = X2[test_indices]
        test_y = label[test_indices]

        # Permute view 2 in the test set
        perm = rng.permutation(len(test_X2))
        test_X2_misaligned = test_X2[perm]

        # Optional incomplete data (missing_rate > 0)
        if missing_rate > 0:
            n_test = len(test_X1)
            # Missing mask via get_sn (2 = two views)
            test_mask = get_sn(2, n_test, missing_rate)
            X1_mask = test_mask[:, 0].astype(np.bool_)
            X2_mask = test_mask[:, 1].astype(np.bool_)

            # Zero vectors for missing samples
            zero_X1 = np.zeros_like(test_X1[0])
            zero_X2 = np.zeros_like(test_X2_misaligned[0])

            # Apply mask: missing entries replaced with zeros
            test_X1_processed = test_X1.copy()
            test_X2_processed = test_X2_misaligned.copy()
            test_X1_processed[~X1_mask] = zero_X1
            test_X2_processed[~X2_mask] = zero_X2

            # Store masks for downstream use
            self.test_X1_mask = X1_mask
            self.test_X2_mask = X2_mask

            logger.info("Test incomplete data generated:")
            logger.info(f"  test X1 kept: {np.sum(X1_mask)}/{n_test}")
            logger.info(f"  test X2 kept: {np.sum(X2_mask)}/{n_test}")
            logger.info(f"  missing rate: {missing_rate:.2%}")
        else:
            # Complete data: all masks True
            self.test_X1_mask = np.ones(len(test_X1), dtype=bool)
            self.test_X2_mask = np.ones(len(test_X2_misaligned), dtype=bool)
            test_X1_processed = test_X1
            test_X2_processed = test_X2_misaligned

        self.test_X1 = test_X1_processed
        self.test_X2 = test_X2_processed
        self.test_y = test_y  # original labels for evaluation
        self.test_y_X1 = test_y  # view 1 labels
        self.test_y_X2 = test_y[perm]  # view 2 labels (misaligned)

        self.n_clusters = n_clusters
        self.n_views = 2
        self.mode = mode
        self.missing_rate = missing_rate

        logger.info("Partially misaligned dataset ready:")
        logger.info(f"  train (aligned, complete): {len(self.train_X1)} samples")
        logger.info(f"  test (misaligned): {len(self.test_X1)} samples")
        logger.info(f"  misalign ratio: {misalign_ratio:.2%}")

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_X1)
        elif self.mode == 'test':
            return len(self.test_X1)
        else:  # 'all'
            return len(self.train_X1) + len(self.test_X1)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # Train: aligned set only
            view1 = torch.from_numpy(self.train_X1[idx])
            view2 = torch.from_numpy(self.train_X2[idx])
            y = torch.tensor(self.train_y[idx], dtype=torch.long)
            return [view1, view2], y, idx
        elif self.mode == 'test':
            # Test: misaligned set only
            view1 = torch.from_numpy(self.test_X1[idx])
            view2 = torch.from_numpy(self.test_X2[idx])
            y = torch.tensor(self.test_y[idx], dtype=torch.long)
            return [view1, view2], y, idx
        else:  # 'all'
            # All: train (aligned) + test (misaligned)
            n_train = len(self.train_X1)
            if idx < n_train:
                view1 = torch.from_numpy(self.train_X1[idx])
                view2 = torch.from_numpy(self.train_X2[idx])
                y = torch.tensor(self.train_y[idx], dtype=torch.long)
            else:
                test_idx = idx - n_train
                view1 = torch.from_numpy(self.test_X1[test_idx])
                view2 = torch.from_numpy(self.test_X2[test_idx])
                y = torch.tensor(self.test_y[test_idx], dtype=torch.long)
            return [view1, view2], y, idx
