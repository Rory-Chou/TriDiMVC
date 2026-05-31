"""
Main training loop and entry point v5
Three-path latent multi-view MLP + clustering head
Losses: reconstruction + clustering + decorrelation + style adversarial + contrastive learning (no VAE, KL removed)
v4 improvement: style adversarial loss predicts view labels, forcing z_style to learn view-specific style information
v5 improvement: integrates UOT semantic fusion for inference on misaligned data
"""

import os
import sys
import argparse
import logging
import random
from datetime import datetime
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from scipy.io import savemat

# Import custom modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import MultiViewDigits, MultiViewRealDataset, PartiallyMisalignedDataset
from models_v5 import TriDMVC_MLP
from losses_v5 import compute_total_loss, NoiseRobustLoss
from clustering import evaluate_kmeans, compute_class_ranking_by_clustering
from utils import (
    setup_logging_to_file,
    rename_log_file,
    get_all_semantic_embeddings,
    get_all_shared_embeddings,
    get_per_view_embeddings,
    get_aligned_embeddings_for_misaligned_data
)
from uot_semantic_fusion import run_uot_semantic_fusion
import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Use the logger from utils module to ensure file output works correctly
logger = utils.logger


def generate_pseudo_labels(model, dataloader, device, n_clusters):
    """
    Generate pseudo labels using K-means

    Args:
        model: Model instance
        dataloader: Data loader
        device: Device
        n_clusters: Number of clusters

    Returns:
        pseudo_labels: Pseudo label array
    """
    # Ensure CUDA context is initialized and synchronized
    if device.startswith("cuda"):
        try:
            # Initialize CUDA context
            _ = torch.zeros(1).to(device)
            # Synchronize to ensure initialization completes
            torch.cuda.synchronize(device)
        except Exception as e:
            logger.warning(f"CUDA initialization failed in generate_pseudo_labels: {e}")
            device = "cpu"
            model.to(device)

    # Ensure model is on the correct device
    model.to(device)
    model.eval()

    all_sem_embeddings = []
    first_batch_processed = False

    with torch.no_grad():
        for views, labels, indices in dataloader:
            try:
                views = [v.to(device) for v in views]

                # For the first batch on CUDA, synchronize first to ensure context is initialized
                if not first_batch_processed and device.startswith("cuda"):
                    try:
                        torch.cuda.synchronize(device)
                    except:
                        pass
                    first_batch_processed = True

                out = model(views)
                sem = out['sem']  # [B, D]
                all_sem_embeddings.append(sem.cpu().numpy())
            except RuntimeError as e:
                if "CUBLAS" in str(e) or "CUDA" in str(e):
                    logger.error(f"CUDA error in generate_pseudo_labels: {e}")
                    logger.error("Trying to recover by clearing CUDA cache and reinitializing...")
                    torch.cuda.empty_cache()
                    # Reinitialize CUDA context
                    try:
                        _ = torch.zeros(1).to(device)
                        torch.cuda.synchronize(device)
                    except:
                        pass
                    # Retry once
                    views = [v.to(device) for v in views]
                    out = model(views)
                    sem = out['sem']
                    all_sem_embeddings.append(sem.cpu().numpy())
                else:
                    raise

    # Merge all embeddings
    all_sem_embeddings = np.vstack(all_sem_embeddings)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    pseudo_labels = kmeans.fit_predict(all_sem_embeddings)

    return pseudo_labels


def train_tridmvc_mlp(
    dataset_name: str = "digits",
    dataset_path: Optional[str] = None,
    n_views=2,
    n_clusters=None,
    batch_size=256,
    epochs=100,
    lr=1e-3,
    shuffle_views: bool = True,
    device=None,
    # Loss weight parameters
    weight_recon: float = 1.0,
    weight_cluster: float = 1.0,
    weight_decorr: float = 0.1,
    weight_style_adv: float = 0.1,
    # Note: contrastive loss weight should be large enough to optimize effectively
    weight_contrastive: float = 1.0,  # Shared contrastive loss weight
    # Clustering parameters
    cluster_update_interval: int = 10,  # Update pseudo labels/cluster centers every N epochs
    use_dec_loss: bool = True,  # Whether to use DEC-style clustering loss
    dec_alpha: float = 1.0,  # Student t-distribution degrees of freedom for DEC clustering loss
    # Style adversarial parameters
    lambda_adv: float = 1.0,  # Adversarial loss weight
    # Shared contrastive learning parameters
    contrastive_margin: float = 1.5,  # Contrastive margin (for L2-normalized features)
    use_infonce: bool = True,  # Use InfoNCE loss (otherwise noise-robust loss)
    contrastive_temperature: float = 0.5,  # InfoNCE temperature
    num_neg_samples: Optional[int] = None,  # Negative samples per sample (None = all negative pairs)
    # Clustering evaluation parameters
    use_gpu_clustering: bool = False,  # Use GPU-accelerated k-means clustering
    n_runs: int = 5,  # Number of runs per clustering evaluation (average over runs)
    # Partial misalignment parameters
    misalign_ratio: float = 0.0,
    use_partial_misalignment: bool = True,
    missing_rate: float = 0.0,  # Missing rate in test set (0.0 = complete data)
    # Alignment algorithm parameters
    use_hungarian_alignment: bool = False,
    # UOT semantic fusion parameters
    topk: int = 5,  # K for Top-K nearest-neighbor imputation
    # percentile_threshold is auto-computed from misalign_ratio and missing_rate: 1 - (misalign_ratio * missing_rate) / 2
    # Random seed parameters
    seed: Optional[int] = None,
    # Feature saving parameters (for t-SNE visualization, etc.)
    save_best_features: bool = False,  # Save latent features at best ACC (raw two-view features, z_UOT1, z_shared)
    logs_root: Optional[str] = None,
):
    """
    Train TriD-MVC MLP model (no VAE, KL removed)

    Args:
        dataset_name: Dataset name
        dataset_path: Dataset file path (optional)
        n_views: Number of views (digits dataset only)
        n_clusters: Number of clusters; inferred from dataset if None
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        shuffle_views: Whether to shuffle view order (real datasets only)
        device: Device (cuda/cpu)
        weight_recon: Reconstruction loss weight
        weight_cluster: Clustering loss weight
        weight_decorr: Decorrelation loss weight
        weight_style_adv: Style adversarial loss weight
        weight_contrastive: Contrastive loss weight
        cluster_update_interval: Pseudo label update interval
        lambda_adv: Adversarial loss weight
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize CUDA context (if using GPU)
    if device.startswith("cuda"):
        try:
            # Create a dummy tensor to initialize CUDA context
            _ = torch.zeros(1).to(device)
            # Synchronize to ensure initialization completes
            torch.cuda.synchronize(device)
            # Run another simple op to stabilize context
            _ = torch.ones(1).to(device)
            torch.cuda.synchronize(device)
            logger.info(f"CUDA context initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA context: {e}, falling back to CPU")
            device = "cpu"

    logger.info(f"Using device: {device}")

    if logs_root is None:
        logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    logs_root = os.path.abspath(logs_root)

    # Initialize logging (before setting seed so seed info is recorded)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = setup_logging_to_file(
        dataset_name,
        start_time,
        best_accuracy=0.0,
        logs_root=logs_root,
    )
    logger.info(f"Missing rate={missing_rate:.1f}")
    logger.info(f"Misalignment ratio={misalign_ratio:.1f}")
    logger.info("=" * 60)
    logger.info("TriD-MVC MLP Multi-view Clustering Training (v5: UOT Semantic Fusion)")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Logs root: {logs_root}")
    logger.info("=" * 60)

    # Set random seeds
    if seed is not None:
        actual_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed (fixed): {actual_seed}")
    else:
        # If no seed specified, generate one and record it
        actual_seed = random.randint(0, 2**31 - 1)
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        torch.manual_seed(actual_seed)
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed (generated): {actual_seed}")

    # Load dataset
    if dataset_name == "digits":
        if use_partial_misalignment:
            logger.warning("Partial misalignment mode not supported for digits dataset")
            use_partial_misalignment = False
        dataset = MultiViewDigits(n_views=n_views)
        view_dims = [len(idx) for idx in dataset.view_indices]
        if n_clusters is None:
            n_clusters = 10
        train_dataset = dataset
        test_dataset = dataset
    else:
        if use_partial_misalignment:
            train_dataset = PartiallyMisalignedDataset(
                dataset=dataset_name,
                dataset_path=dataset_path,
                misalign_ratio=misalign_ratio,
                missing_rate=missing_rate,
                seed=seed if seed is not None else 0,
                mode='train'
            )
            test_dataset = PartiallyMisalignedDataset(
                dataset=dataset_name,
                dataset_path=dataset_path,
                misalign_ratio=misalign_ratio,
                missing_rate=missing_rate,
                seed=seed if seed is not None else 0,
                mode='all'
            )
            view_dims = [train_dataset.train_X1.shape[1], train_dataset.train_X2.shape[1]]
            if n_clusters is None:
                n_clusters = train_dataset.n_clusters
        else:
            dataset = MultiViewRealDataset(
                dataset=dataset_name,
                dataset_path=dataset_path,
                shuffle_views=shuffle_views,
                seed=seed if seed is not None else 0
            )
            view_dims = [dataset.X1.shape[1], dataset.X2.shape[1]]
            if n_clusters is None:
                n_clusters = dataset.n_clusters
            train_dataset = dataset
            test_dataset = dataset

    logger.info(f"Dataset: {dataset_name} | Views: {view_dims} | Clusters: {n_clusters} | Samples: {len(train_dataset)}")
    logger.info(f"Loss weights: Recon={weight_recon:.2f}, Cluster={weight_cluster:.2f}, Decor={weight_decorr:.2f}, StyleAdv={weight_style_adv:.2f}, Contrastive={weight_contrastive:.2f}")
    logger.info(f"Lambda_adv={lambda_adv:.2f}, Cluster update interval={cluster_update_interval}, Use DEC={use_dec_loss}, DEC_alpha={dec_alpha:.2f}")
    logger.info(f"Contrastive: margin={contrastive_margin:.2f}, use_infonce={use_infonce}, temp={contrastive_temperature:.2f}, num_neg={num_neg_samples}")
    if use_partial_misalignment:
        # Auto-compute percentile_threshold: 1 - (misalign_ratio * missing_rate) / 2
        percentile_threshold = 1.0 - (misalign_ratio * missing_rate) / 2.0
        logger.info(f"UOT Semantic Fusion: topk={topk}, percentile_threshold={percentile_threshold:.2f} (auto-calculated from misalign_ratio={misalign_ratio:.2f}, missing_rate={missing_rate:.2f})")

    # Check GPU clustering library availability
    try:
        from clustering import CUML_AVAILABLE, FAISS_AVAILABLE
    except ImportError:
        CUML_AVAILABLE = False
        FAISS_AVAILABLE = False

    if use_gpu_clustering:
        if CUML_AVAILABLE:
            logger.info("GPU clustering: cuML available (will use GPU acceleration)")
        elif FAISS_AVAILABLE:
            logger.info("GPU clustering: FAISS available (will use GPU acceleration)")
        else:
            logger.warning("GPU clustering requested but no GPU library available (cuML/FAISS), falling back to CPU")
    else:
        logger.info("GPU clustering: Disabled (using CPU sklearn)")

    # Create data loaders
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=512,
                            shuffle=False, drop_last=False)

    # Create model
    model = TriDMVC_MLP(
        view_dims=view_dims,
        z_shared_dim=20,
        z_priv_dim=20,
        z_style_dim=20,
        n_clusters=n_clusters,
        hidden_dims=[1024, 1024, 1024],
        dropout=0.2,
        use_batch_norm=True,
        activation='relu'
    )

    # Ensure CUDA context is initialized before moving model
    if device.startswith("cuda"):
        try:
            # Re-ensure CUDA context is initialized and synchronized
            _ = torch.zeros(1).to(device)
            torch.cuda.synchronize(device)
        except Exception as e:
            logger.warning(f"CUDA initialization failed before moving model: {e}, falling back to CPU")
            device = "cpu"
            model.to(device)
        else:
            model.to(device)
            # Synchronize again after moving model
            try:
                torch.cuda.synchronize(device)
            except:
                pass
    else:
        model.to(device)

    # Initialize noise-robust contrastive loss
    noise_robust_loss_fn = NoiseRobustLoss(
        margin=contrastive_margin,
        use_robust_loss=True,
        start_fine=False,
        reduction='mean'
    ).to(device)

    # Separate encoder and style classifier parameters
    encoder_params = []
    style_classifier_params = []

    for name, param in model.named_parameters():
        if 'style_classifier' in name:
            style_classifier_params.append(param)
        else:
            encoder_params.append(param)

    # Create two optimizers
    optimizer = torch.optim.Adam(encoder_params, lr=lr)
    style_optimizer = torch.optim.Adam(style_classifier_params, lr=lr)

    # Track best results
    best_acc_sem = 0.0
    best_epoch_sem = 0
    best_results_sem = None
    best_acc_shared_at_best_sem = 0.0  # z_shared ACC when z_sem is best

    best_acc_shared = 0.0
    best_epoch_shared = 0
    best_results_shared = None
    best_acc_sem_at_best_shared = 0.0  # z_sem ACC when z_shared is best

    # Partial misalignment mode: track best results for each z_sem group
    best_acc_maxsim = 0.0
    best_epoch_maxsim = 0
    best_results_maxsim = None

    best_acc_uot1 = 0.0
    best_epoch_uot1 = 0
    best_results_uot1 = None

    best_acc_uot2 = 0.0
    best_epoch_uot2 = 0
    best_results_uot2 = None

    # Feature save path (when save_best_features is enabled)
    best_features_path = None
    if save_best_features:
        logs_dir = os.path.join(logs_root, dataset_name)
        os.makedirs(logs_dir, exist_ok=True)
        best_features_path = os.path.join(logs_dir, 'best_acc_features.npz')

    # Initialize pseudo labels and cluster centers
    pseudo_labels = None
    cluster_centers_list = None

    for epoch in range(epochs):
        model.train()

        # Update pseudo labels / cluster centers
        if epoch % cluster_update_interval == 0 or pseudo_labels is None:
            logger.info(f"Epoch {epoch}: Updating clustering assignments...")
            if use_dec_loss:
                # DEC loss: update cluster centers
                # First collect all semantic representations
                model.eval()
                all_sem_embeddings = []
                with torch.no_grad():
                    for views, labels, indices in train_loader:
                        views = [v.to(device) for v in views]
                        out = model(views)
                        sem_per_view = out['sem_per_view']
                        all_sem_embeddings.append([sem.cpu().numpy() for sem in sem_per_view])

                # Merge all batches
                n_views = len(all_sem_embeddings[0])
                sem_merged = []
                for v_idx in range(n_views):
                    sem_v = np.vstack([batch[v_idx] for batch in all_sem_embeddings])
                    sem_merged.append(torch.tensor(sem_v, device=device))

                # Initialize cluster centers with K-means
                cluster_centers_list = []
                for sem_v in sem_merged:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                    kmeans.fit(sem_v.cpu().numpy())
                    centers = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)
                    cluster_centers_list.append(centers)

                logger.info(f"Cluster centers updated: {n_clusters} clusters per view")
            else:
                # Pseudo label loss
                pseudo_labels = generate_pseudo_labels(model, train_loader, device, n_clusters)
                logger.info(f"Pseudo labels generated: {len(np.unique(pseudo_labels))} clusters")

        # Initialize loss accumulators (KL removed)
        total_loss = 0.0
        total_recon_loss = 0.0
        total_cluster_loss = 0.0
        total_decorr_loss = 0.0
        total_style_adv_loss = 0.0
        total_contrastive_loss = 0.0
        # Contrastive learning statistics
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        total_pos_count = 0
        total_neg_count = 0
        num_batches = 0

        # Prepare pseudo labels (if using pseudo label loss)
        if not use_dec_loss and pseudo_labels is not None:
            pseudo_labels_dict = {i: pseudo_labels[i] for i in range(len(pseudo_labels))}

        for batch_idx, (views, labels, indices) in enumerate(train_loader):
            views = [v.to(device) for v in views]

            # Validate alignment: both views must have the same batch size
            if len(views) == 2:
                if views[0].size(0) != views[1].size(0):
                    raise ValueError(f"Batch {batch_idx}: batch size mismatch between two views! view1: {views[0].size(0)}, view2: {views[1].size(0)}")

            # Forward pass
            out = model(views)

            # Validate model output: both views in zs_list must have the same batch size
            if len(out['zs_list']) == 2:
                zs1, zs2 = out['zs_list']
                if zs1.size(0) != zs2.size(0):
                    raise ValueError(f"Batch {batch_idx}: batch size mismatch in zs_list! zs1: {zs1.size(0)}, zs2: {zs2.size(0)}")

                # In first epoch's first few batches, retain z_s gradients for diagnostics
                if epoch == 0 and batch_idx < 3:
                    zs1.retain_grad()
                    zs2.retain_grad()

                # Diagnostic output on first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    logger.info(f"Data alignment check: batch_size={zs1.size(0)}, zs1.shape={zs1.shape}, zs2.shape={zs2.shape}")
                    logger.info(f"Raw input check: views[0].shape={views[0].shape}, views[1].shape={views[1].shape}")
                    logger.info(f"Sample indices (first 5): {indices[:min(5, len(indices))].cpu().numpy()}")

                    # Compute shared feature distance for first few samples (should be small when aligned)
                    with torch.no_grad():
                        zs1_norm = F.normalize(zs1[:min(5, zs1.size(0))], dim=1)
                        zs2_norm = F.normalize(zs2[:min(5, zs2.size(0))], dim=1)
                        pos_dists_sample = torch.norm(zs1_norm - zs2_norm, dim=1)
                        logger.info(f"Positive pair distances for first 5 samples (should be small when aligned): {pos_dists_sample.cpu().numpy()}")

            # Prepare clustering-related parameters
            batch_pseudo_labels = None
            batch_cluster_centers = None
            if use_dec_loss and cluster_centers_list is not None:
                batch_cluster_centers = cluster_centers_list
            elif not use_dec_loss and pseudo_labels is not None:
                batch_pseudo_labels = torch.tensor(
                    [pseudo_labels_dict[idx.item()] for idx in indices],
                    dtype=torch.long,
                    device=device
                )

            # Style adversarial training: optimize style classifier first (no gradient reversal)
            # Key fix: all views' z_u use the same shared classifier with true view labels
            # so the classifier learns real style features instead of memorizing input source
            batch_size = views[0].size(0)
            n_views = len(views)

            style_optimizer.zero_grad()
            style_classifier_loss = 0.0

            # Collect z_u and view labels from all views
            all_zu = []
            all_view_labels = []

            for view_idx, zu in enumerate(out['zu_list']):
                # View labels for all samples in current view (all view_idx)
                view_labels_for_view = torch.full((batch_size,), view_idx, dtype=torch.long, device=device)
                all_zu.append(zu.detach())  # Detach gradients; train classifier only
                all_view_labels.append(view_labels_for_view)

            # Concatenate z_u and labels from all views
            all_zu_concat = torch.cat(all_zu, dim=0)  # [B*n_views, D_u]
            all_view_labels_concat = torch.cat(all_view_labels, dim=0)  # [B*n_views]

            # Predict all views' z_u with shared classifier
            logits = model.style_classifier(all_zu_concat)  # [B*n_views, n_views]
            style_classifier_loss = F.cross_entropy(logits, all_view_labels_concat)

            style_classifier_loss.backward()
            style_optimizer.step()

            # Note: style classifier is updated; total loss still uses gradient reversal
            loss_dict = compute_total_loss(
                model=model,
                model_output=out,
                views=views,
                cluster_labels=None,
                pseudo_labels=batch_pseudo_labels,
                weight_recon=weight_recon,
                weight_cluster=weight_cluster,
                weight_decorr=weight_decorr,
                weight_style_adv=weight_style_adv,
                weight_contrastive=weight_contrastive,
                lambda_adv=lambda_adv,
                dec_alpha=dec_alpha,
                cluster_centers_list=batch_cluster_centers,
                contrastive_margin=contrastive_margin,
                use_infonce=use_infonce,
                contrastive_temperature=contrastive_temperature,
                num_neg_samples=num_neg_samples,
                noise_robust_loss_fn=noise_robust_loss_fn
            )

            loss = loss_dict['total_loss']

            # Backward pass (encoder update, includes gradient-reversed adversarial loss)
            optimizer.zero_grad()
            loss.backward()

            # Diagnostic: check contrastive loss gradients (first few batches of first epoch only)
            if epoch == 0 and batch_idx < 3:
                contrastive_loss_val = loss_dict['contrastive_loss']
                weighted_contrastive = weight_contrastive * contrastive_loss_val

                logger.info(f"Batch {batch_idx} loss breakdown:")
                logger.info(f"  Contrastive loss (raw)={contrastive_loss_val.item():.6f}")
                logger.info(f"  Contrastive loss (weighted)={weighted_contrastive.item():.6f}")
                logger.info(f"  Total loss={loss.item():.6f}")
                logger.info(f"  Contrastive contribution={weighted_contrastive.item() / (loss.item() + 1e-8) * 100:.2f}%")

                # Check z_s gradients (after backward)
                zs1, zs2 = out['zs_list']
                if zs1.grad is not None and zs2.grad is not None:
                    zs1_grad_norm = zs1.grad.norm().item()
                    zs2_grad_norm = zs2.grad.norm().item()
                    logger.info(f"  zs1.grad.norm()={zs1_grad_norm:.6f}, zs2.grad.norm()={zs2_grad_norm:.6f}")

                    # Check encoder parameter gradients
                    encoder_params_check = list(model.encoders[0].parameters())
                    if len(encoder_params_check) > 0:
                        encoder_grads = [p.grad for p in encoder_params_check if p.grad is not None]
                        if len(encoder_grads) > 0:
                            encoder_grad_norm = torch.norm(torch.cat([g.flatten() for g in encoder_grads])).item()
                            logger.info(f"  Encoder grad norm={encoder_grad_norm:.6f}")
                else:
                    logger.warning(f"  zs1.grad or zs2.grad is None!")

            # Gradient clipping (prevent explosion)
            torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=10.0)

            optimizer.step()

            # Accumulate losses (KL removed)
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_cluster_loss += loss_dict['cluster_loss'].item()
            total_decorr_loss += loss_dict['decorr_loss'].item()
            total_style_adv_loss += loss_dict['style_adv_loss'].item()
            total_contrastive_loss += loss_dict['contrastive_loss'].item()

            # Accumulate contrastive learning statistics
            contrastive_stats = loss_dict.get('contrastive_stats', {})
            if contrastive_stats:
                pos_avg_dist = contrastive_stats.get('pos_avg_dist', torch.tensor(0.0))
                neg_avg_dist = contrastive_stats.get('neg_avg_dist', torch.tensor(0.0))
                pos_count = contrastive_stats.get('pos_count', 0)
                neg_count = contrastive_stats.get('neg_count', 0)

                if isinstance(pos_avg_dist, torch.Tensor):
                    total_pos_dist += pos_avg_dist.item() * pos_count
                else:
                    total_pos_dist += pos_avg_dist * pos_count

                if isinstance(neg_avg_dist, torch.Tensor):
                    total_neg_dist += neg_avg_dist.item() * neg_count
                else:
                    total_neg_dist += neg_avg_dist * neg_count

                total_pos_count += pos_count
                total_neg_count += neg_count

            num_batches += 1

        # Compute average losses (KL removed)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_cluster_loss = total_cluster_loss / num_batches
        avg_decorr_loss = total_decorr_loss / num_batches
        avg_style_adv_loss = total_style_adv_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches

        # Compute average positive/negative pair distances
        avg_pos_dist = total_pos_dist / total_pos_count if total_pos_count > 0 else 0.0
        avg_neg_dist = total_neg_dist / total_neg_count if total_neg_count > 0 else 0.0

        # Compute relative contribution of contrastive loss
        contrastive_contrib = (weight_contrastive * avg_contrastive_loss) / (avg_loss + 1e-8) * 100

        # Line 1: loss info
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss={avg_loss:.4f} | "
            f"Recon={avg_recon_loss:.4f} | "
            f"Cluster={avg_cluster_loss:.4f} | "
            f"Decor={avg_decorr_loss:.4f} | "
            f"StyleAdv={avg_style_adv_loss:.4f} | "
            f"Contrastive={avg_contrastive_loss:.4f}({contrastive_contrib:.1f}%)"
        )

        # Line 2: sample pair statistics
        logger.info(
            f"=====> Sample Pairs | "
            f"PosDist={avg_pos_dist:.4f} | "
            f"NegDist={avg_neg_dist:.4f} | "
            f"PosPairs={total_pos_count} | "
            f"NegPairs={total_neg_count}"
        )

        # Diagnostic: warn if positive distance exceeds negative (console only, not logged)
        if avg_pos_dist > avg_neg_dist and avg_pos_dist > 0 and avg_neg_dist > 0:
            print(f"  Warning: PosDist ({avg_pos_dist:.4f}) > NegDist ({avg_neg_dist:.4f}), contrastive loss may not be working correctly!")

        # Evaluation (every epoch)

        if use_partial_misalignment:
            # Partial misalignment mode: alignment required
            alignment_indices, all_zs_view0, all_zs_view1, all_zp_view0, all_zp_view1, all_y, all_indices, mask_view0, mask_view1 = get_aligned_embeddings_for_misaligned_data(
                model, test_loader, device, use_hungarian=use_hungarian_alignment, dataset=test_dataset
            )

            # In misaligned mode, get correct labels from dataset
            # all_y is view0 labels (original order), but view1 sample order is shuffled
            # so we need view1's correct labels from the dataset
            if test_dataset is not None and hasattr(test_dataset, 'test_y_X1') and hasattr(test_dataset, 'test_y_X2'):
                if test_dataset.mode == 'test':
                    # Test mode: test set labels only
                    all_y_view0_correct = test_dataset.test_y_X1  # view0 labels
                    all_y_view1_correct = test_dataset.test_y_X2  # view1 labels (shuffled order)
                elif test_dataset.mode == 'all':
                    # All mode: aligned train set, misaligned test set
                    n_train = len(test_dataset.train_X1)
                    n_test = len(test_dataset.test_X1)
                    # Train set labels (aligned)
                    train_y = test_dataset.train_y
                    # Test set labels (misaligned)
                    test_y_view0 = test_dataset.test_y_X1
                    test_y_view1 = test_dataset.test_y_X2
                    # Concatenate
                    all_y_view0_correct = np.concatenate([train_y, test_y_view0])
                    all_y_view1_correct = np.concatenate([train_y, test_y_view1])
                else:
                    # Train mode: use original labels
                    all_y_view0_correct = all_y
                    all_y_view1_correct = all_y
            else:
                # No dataset info: fall back to original labels (may be inaccurate)
                all_y_view0_correct = all_y
                all_y_view1_correct = all_y
                logger.warning("Cannot get correct labels from dataset; using original labels (may be inaccurate)")

            # Sample counts before filtering
            n_view0_before = len(all_zs_view0)
            n_view1_before = len(all_zs_view1)
            logger.info(f"Sample count before filtering: view0={n_view0_before}, view1={n_view1_before}")

            # Filter incomplete samples per view (if masks exist)
            if mask_view0 is not None and mask_view1 is not None:
                # Mask statistics
                n_view0_complete = np.sum(mask_view0)
                n_view1_complete = np.sum(mask_view1)
                logger.info(f"Mask stats: view0 complete={n_view0_complete}/{n_view0_before}, view1 complete={n_view1_complete}/{n_view1_before}")

                # Filter incomplete samples per view (views may have different counts)
                # View0: keep only view0-complete samples
                view0_mask = mask_view0
                all_zs_view0 = all_zs_view0[view0_mask]
                all_zp_view0 = all_zp_view0[view0_mask]
                # Use correct labels
                all_y_view0 = all_y_view0_correct[view0_mask]
                all_indices_view0 = all_indices[view0_mask]

                # View1: keep only view1-complete samples
                view1_mask = mask_view1
                all_zs_view1 = all_zs_view1[view1_mask]
                all_zp_view1 = all_zp_view1[view1_mask]
                # Use correct labels
                all_y_view1 = all_y_view1_correct[view1_mask]
                all_indices_view1 = all_indices[view1_mask]

                # Sample counts after filtering
                n_view0_after = len(all_zs_view0)
                n_view1_after = len(all_zs_view1)
                logger.info(f"Sample count after filtering: view0={n_view0_after} (filtered {n_view0_before - n_view0_after}), view1={n_view1_after} (filtered {n_view1_before - n_view1_after})")

                # Recompute alignment indices (sample counts may differ after filtering)
                from utils import align_data_auto
                alignment_indices, _, _, _ = align_data_auto(
                    all_zs_view0, all_zs_view1,
                    metric='cosine',
                    normalize=True,
                    maximize=True,
                    use_hungarian=use_hungarian_alignment
                )

                # For alignment indices, use view0 labels (indices are based on view0)
                all_y = all_y_view0
            else:
                # No mask: log current sample counts
                logger.info(f"No mask info; using all samples: view0={n_view0_before}, view1={n_view1_before}")

            # Choose anchor view (view A) as the one with more samples
            n_view0 = len(all_zs_view0)
            n_view1 = len(all_zs_view1)

            if n_view0 >= n_view1:
                # view0 count >= view1: view0 is anchor (view A)
                z_s_anchor = all_zs_view0  # anchor shared semantics
                z_p_anchor = all_zp_view0  # anchor private semantics
                z_s_supplement = all_zs_view1  # supplement shared semantics
                z_p_supplement = all_zp_view1  # supplement private semantics
                # Labels: strictly use anchor view labels
                if mask_view0 is not None and mask_view1 is not None:
                    labels_anchor = all_y_view0  # anchor (view0) labels
                    labels_supplement = all_y_view1  # supplement labels
                else:
                    labels_anchor = all_y  # anchor labels
                    labels_supplement = all_y  # same as anchor when no mask
                anchor_name = "view0"
                supplement_name = "view1"
                logger.info(f"Anchor view: {anchor_name}={n_view0} samples, supplement view: {supplement_name}={n_view1} samples, ratio={n_view1/n_view0:.4f}")
            else:
                # view1 count > view0: view1 is anchor (view A)
                z_s_anchor = all_zs_view1  # anchor shared semantics
                z_p_anchor = all_zp_view1  # anchor private semantics
                z_s_supplement = all_zs_view0  # supplement shared semantics
                z_p_supplement = all_zp_view0  # supplement private semantics
                # Labels: strictly use anchor view labels
                if mask_view0 is not None and mask_view1 is not None:
                    labels_anchor = all_y_view1  # anchor (view1) labels
                    labels_supplement = all_y_view0  # supplement labels
                else:
                    labels_anchor = all_y  # anchor labels
                    labels_supplement = all_y  # same as anchor when no mask
                anchor_name = "view1"
                supplement_name = "view0"
                logger.info(f"Anchor view: {anchor_name}={n_view1} samples, supplement view: {supplement_name}={n_view0} samples, ratio={n_view0/n_view1:.4f}")

            # Convert to torch tensors for UOT
            z_s_A = torch.tensor(z_s_anchor, device=device, dtype=torch.float32)
            z_p_A = torch.tensor(z_p_anchor, device=device, dtype=torch.float32)
            z_s_B = torch.tensor(z_s_supplement, device=device, dtype=torch.float32)
            z_p_B = torch.tensor(z_p_supplement, device=device, dtype=torch.float32)
            labels_A = torch.tensor(labels_anchor, device=device, dtype=torch.long)  # anchor labels
            labels_B = None  # supplement labels not needed (stats may be inaccurate)

            # 1. z_sem from max-similarity matching (anchor-based)
            # Match each anchor sample to supplement samples
            from utils import align_data_auto
            alignment_indices_anchor_to_supplement, aligned_zs_anchor_auto, aligned_zs_supplement_auto, _ = align_data_auto(
                z_s_anchor, z_s_supplement,
                metric='cosine',
                normalize=True,
                maximize=True,
                use_hungarian=use_hungarian_alignment
            )
            # alignment_indices_anchor_to_supplement: [n_anchor, 2], col0=anchor index, col1=supplement index
            # Note: one-to-many allowed — one supplement sample may match multiple anchor samples
            # Ensures every anchor sample has a match; final z_sem/z_share count equals anchor count
            n_aligned = len(alignment_indices_anchor_to_supplement)
            n_anchor = len(z_s_anchor)
            n_supplement = len(z_s_supplement)
            if n_aligned == n_anchor:
                logger.info(f"Alignment matching: all {n_aligned} anchor samples matched to supplement (one-to-many allowed)")
            else:
                logger.warning(f"Alignment matching: {n_aligned}/{n_anchor} anchor samples matched to supplement (incomplete)")

            # Use aligned data (align_data_auto returns aligned arrays)
            aligned_zs_anchor = aligned_zs_anchor_auto
            aligned_zs_supplement = aligned_zs_supplement_auto
            aligned_zs = (aligned_zs_anchor + aligned_zs_supplement) / 2.0  # average fusion

            # Aligned private semantics (via alignment indices)
            aligned_zp_anchor = z_p_anchor[alignment_indices_anchor_to_supplement[:, 0]]
            aligned_zp_supplement = z_p_supplement[alignment_indices_anchor_to_supplement[:, 1]]
            aligned_zp_cat = np.concatenate([aligned_zp_anchor, aligned_zp_supplement], axis=1)
            aligned_sem_maxsim = np.concatenate([aligned_zs, aligned_zp_cat], axis=1)

            # Labels: strictly anchor labels in alignment index order
            # Note: col0 of alignment indices is anchor original index
            aligned_labels_maxsim = labels_anchor[alignment_indices_anchor_to_supplement[:, 0]]

            # Verify aligned data lengths
            assert len(aligned_zs) == n_aligned, f"Aligned z_s length mismatch: {len(aligned_zs)} != {n_aligned}"
            assert len(aligned_labels_maxsim) == n_aligned, f"Aligned label length mismatch: {len(aligned_labels_maxsim)} != {n_aligned}"
            assert len(aligned_sem_maxsim) == n_aligned, f"Aligned z_sem length mismatch: {len(aligned_sem_maxsim)} != {n_aligned}"

            # Debug: check if alignment index col0 is sequential
            anchor_indices = alignment_indices_anchor_to_supplement[:, 0]
            is_sequential = np.array_equal(anchor_indices, np.arange(n_aligned))
            if is_sequential:
                logger.debug(f"Alignment index col0 is sequential [0, 1, ..., {n_aligned-1}]")
            else:
                logger.warning(f"Alignment index col0 is not sequential; first 10 indices: {anchor_indices[:10]}")
                # If not sequential, features and labels still use the same indices so order is consistent

            # 2. UOT semantic fusion for z_sem1 and z_sem2 (adaptive threshold)
            # Note: strictly anchor (view A) to supplement (view B)
            # Final z_sem1 and z_sem2 both have n_A samples (anchor count)
            # Auto-compute percentile_threshold: 1 - (misalign_ratio * missing_rate) / 2
            percentile_threshold = 1.0 - (misalign_ratio * missing_rate) / 2.0
            uot_result = run_uot_semantic_fusion(
                z_s_A=z_s_A,
                z_p_A=z_p_A,
                z_s_B=z_s_B,
                z_p_B=z_p_B,
                eps=0.1,
                tau_a=1.0,
                tau_b=1.0,
                sinkhorn_iters=500,
                sinkhorn_tol=1e-6,
                topk=topk,  # configurable Top-K (when n_clusters is None)
                match_max_threshold=None,  # None = adaptive threshold
                percentile_threshold=percentile_threshold,  # auto from misalign/missing rates
                n_clusters=n_clusters,  # v7: cluster center (prototype) imputation
                labels_A=labels_A,
                labels_B=labels_B,  # may be None; stats may be inaccurate
            )

            z_sem1_uot = uot_result['z_sem1'].cpu().numpy()  # [n_A, d_s + 2*d_p] anchor sample count
            z_sem2_uot = uot_result['z_sem2'].cpu().numpy()  # [n_A, d_s + 2*d_p] anchor sample count

            # Labels: strictly anchor labels (n_A samples)
            labels_uot = labels_A.cpu().numpy()

            # UOT statistics
            if uot_result.get('stats'):
                stats = uot_result['stats']
                if stats['match_accuracy'] >= 0:
                    logger.info(
                        f"=====> UOT Statistics | "
                        f"MatchedPairs={stats['num_matched_pairs']} | "
                        f"ImputedSamples={stats['num_imputed_samples']} | "
                        f"MatchAccuracy={stats['match_accuracy']:.4f}"
                    )
                else:
                    logger.info(
                        f"=====> UOT Statistics | "
                        f"MatchedPairs={stats['num_matched_pairs']} | "
                        f"ImputedSamples={stats['num_imputed_samples']} | "
                        f"MatchAccuracy=N/A (supplement view labels unknown)"
                    )

            # Evaluate clustering for three z_sem variants
            # 1. Max-similarity z_sem
            results_sem_maxsim = evaluate_kmeans(
                aligned_sem_maxsim, aligned_labels_maxsim, n_clusters,
                n_runs=n_runs, use_gpu=use_gpu_clustering
            )

            # 2. UOT z_sem1
            results_sem_uot1 = evaluate_kmeans(
                z_sem1_uot, labels_uot, n_clusters,
                n_runs=n_runs, use_gpu=use_gpu_clustering
            )

            # 3. UOT z_sem2
            results_sem_uot2 = evaluate_kmeans(
                z_sem2_uot, labels_uot, n_clusters,
                n_runs=n_runs, use_gpu=use_gpu_clustering
            )

            # Evaluate z_shared (max-similarity alignment)
            results_shared = evaluate_kmeans(
                aligned_zs, aligned_labels_maxsim, n_clusters,
                n_runs=n_runs, use_gpu=use_gpu_clustering
            )

            results = {
                'z_sem_maxsim': results_sem_maxsim,  # max-similarity matching
                'z_sem_uot1': results_sem_uot1,      # UOT z_sem1
                'z_sem_uot2': results_sem_uot2,      # UOT z_sem2
                'z_shared': results_shared,
                'per_view': []  # no per-view eval in misaligned mode
            }
        else:
            # Standard mode
            z_sem_all, labels_all = get_all_semantic_embeddings(
                model, test_loader, device
            )
            z_shared_all, _ = get_all_shared_embeddings(
                model, test_loader, device
            )

            # Evaluate z_sem clustering
            results_sem = evaluate_kmeans(z_sem_all, labels_all, n_clusters, n_runs=n_runs, use_gpu=use_gpu_clustering)

            # Evaluate z_shared clustering
            results_shared = evaluate_kmeans(z_shared_all, labels_all, n_clusters, n_runs=n_runs, use_gpu=use_gpu_clustering)

            results = {
                'z_sem': results_sem,
                'z_shared': results_shared,
            }

        # Record results
        if use_partial_misalignment:
            # Partial misalignment: three z_sem result groups
            acc_maxsim = results['z_sem_maxsim']['ACC']
            acc_uot1 = results['z_sem_uot1']['ACC']
            acc_uot2 = results['z_sem_uot2']['ACC']
            acc_shared = results['z_shared']['ACC']

            # Line 3: max-similarity Z_sem
            logger.info(
                f"=====> Z_sem (MaxSim) | "
                f"ACC={acc_maxsim:.4f} | "
                f"NMI={results['z_sem_maxsim']['NMI']:.4f} | "
                f"ARI={results['z_sem_maxsim']['ARI']:.4f} | "
                f"F1={results['z_sem_maxsim']['F1']:.4f}"
            )

            # Line 4: UOT z_sem1
            logger.info(
                f"=====> Z_sem (UOT1) | "
                f"ACC={acc_uot1:.4f} | "
                f"NMI={results['z_sem_uot1']['NMI']:.4f} | "
                f"ARI={results['z_sem_uot1']['ARI']:.4f} | "
                f"F1={results['z_sem_uot1']['F1']:.4f}"
            )

            # Line 5: UOT z_sem2
            logger.info(
                f"=====> Z_sem (UOT2) | "
                f"ACC={acc_uot2:.4f} | "
                f"NMI={results['z_sem_uot2']['NMI']:.4f} | "
                f"ARI={results['z_sem_uot2']['ARI']:.4f} | "
                f"F1={results['z_sem_uot2']['F1']:.4f}"
            )

            # Line 6: z_shared
            logger.info(
                f"=====> Z_shared | "
                f"ACC={acc_shared:.4f} | "
                f"NMI={results['z_shared']['NMI']:.4f} | "
                f"ARI={results['z_shared']['ARI']:.4f} | "
                f"F1={results['z_shared']['F1']:.4f}"
            )

            # Update best results per z_sem group
            if acc_maxsim > best_acc_maxsim:
                best_acc_maxsim = acc_maxsim
                best_epoch_maxsim = epoch + 1
                best_results_maxsim = results['z_sem_maxsim']

            if acc_uot1 > best_acc_uot1:
                best_acc_uot1 = acc_uot1
                best_epoch_uot1 = epoch + 1
                best_results_uot1 = results['z_sem_uot1']
                # Optional: save features at best ACC (for t-SNE visualization)
                if save_best_features and best_features_path:
                    view1_feat = np.concatenate([z_s_anchor, z_p_anchor], axis=1)   # anchor [n_anchor, D_s+D_p]
                    view2_feat = np.concatenate([z_s_supplement, z_p_supplement], axis=1)  # supplement [n_supplement, D_s+D_p]
                    # Raw fusion: concatenate view features on feature dim (only when sample counts match)
                    raw_fusion = np.concatenate([view1_feat, view2_feat], axis=1).astype(np.float64) if view1_feat.shape[0] == view2_feat.shape[0] else None
                    # Rank classes by clustering quality for visualization (top-k best classes)
                    class_ranking = compute_class_ranking_by_clustering(
                        z_sem1_uot, labels_uot, n_clusters, random_state=42, use_gpu=use_gpu_clustering
                    )
                    save_dict = {
                        'view1_feat': view1_feat.astype(np.float64),
                        'view2_feat': view2_feat.astype(np.float64),
                        'z_fusion': z_sem1_uot.astype(np.float64),   # Z_sem (UOT1): shared+private fusion
                        'z_shared': aligned_zs.astype(np.float64),   # shared fusion only
                        'labels_view1': labels_anchor,
                        'labels_view2': labels_supplement,
                        'labels_fusion': labels_uot,
                        'class_ranking': class_ranking,
                        'mode': np.array('misaligned', dtype=object),
                        'best_epoch': np.array(epoch + 1),
                        'best_acc': np.array(acc_uot1),
                    }
                    if raw_fusion is not None:
                        save_dict['raw_fusion'] = raw_fusion
                    np.savez(best_features_path, **save_dict)
                    # Also save .mat for MATLAB
                    mat_path = best_features_path.replace('.npz', '.mat')
                    mat_dict = {
                        'view1_feat': view1_feat, 'view2_feat': view2_feat,
                        'z_fusion': z_sem1_uot, 'z_shared': aligned_zs,
                        'labels_view1': labels_anchor, 'labels_view2': labels_supplement,
                        'labels_fusion': labels_uot, 'class_ranking': class_ranking,
                        'mode': 'misaligned',
                        'best_epoch': np.array([[epoch + 1]]), 'best_acc': np.array([[acc_uot1]]),
                    }
                    if raw_fusion is not None:
                        mat_dict['raw_fusion'] = raw_fusion
                    savemat(mat_path, mat_dict)
                    logger.info(f"Saved best-ACC features to {best_features_path} (epoch {epoch + 1}, ACC={acc_uot1:.4f})")

            if acc_uot2 > best_acc_uot2:
                best_acc_uot2 = acc_uot2
                best_epoch_uot2 = epoch + 1
                best_results_uot2 = results['z_sem_uot2']

            # Update best results (UOT2 as primary metric — best fusion)
            if acc_uot2 > best_acc_sem:
                best_acc_sem = acc_uot2
                best_epoch_sem = epoch + 1
                best_results_sem = results['z_sem_uot2']
                best_acc_shared_at_best_sem = acc_shared

            if acc_shared > best_acc_shared:
                best_acc_shared = acc_shared
                best_epoch_shared = epoch + 1
                best_results_shared = results['z_shared']
                best_acc_sem_at_best_shared = acc_uot2

            # Line 7: best results comparison
            logger.info(
                f"=====> Best | "
                f"MaxSim ACC={best_acc_maxsim:.4f} (epoch {best_epoch_maxsim}) | "
                f"UOT1 ACC={best_acc_uot1:.4f} (epoch {best_epoch_uot1}) | "
                f"UOT2 ACC={best_acc_uot2:.4f} (epoch {best_epoch_uot2}) | "
                f"Z_shared ACC={best_acc_shared:.4f} (epoch {best_epoch_shared})"
            )
        elif 'z_sem' in results:
            # Standard mode
            acc_sem = results['z_sem']['ACC']
            acc_shared = results['z_shared']['ACC']

            # Line 3: Z_sem
            logger.info(
                f"=====> Z_sem | "
                f"ACC={acc_sem:.4f} | "
                f"NMI={results['z_sem']['NMI']:.4f} | "
                f"ARI={results['z_sem']['ARI']:.4f} | "
                f"F1={results['z_sem']['F1']:.4f}"
            )

            # Line 4: z_shared
            logger.info(
                f"=====> Z_shared | "
                f"ACC={acc_shared:.4f} | "
                f"NMI={results['z_shared']['NMI']:.4f} | "
                f"ARI={results['z_shared']['ARI']:.4f} | "
                f"F1={results['z_shared']['F1']:.4f}"
            )

            # Update best results
            if acc_sem > best_acc_sem:
                best_acc_sem = acc_sem
                best_epoch_sem = epoch + 1
                best_results_sem = results['z_sem']
                best_acc_shared_at_best_sem = acc_shared  # z_shared ACC when z_sem is best
                # Optional: save features at best ACC (for t-SNE visualization)
                if save_best_features and best_features_path:
                    per_view_embeddings, labels_all_np = get_per_view_embeddings(model, test_loader, device)
                    view1_feat = per_view_embeddings[0].numpy().astype(np.float64)
                    view2_feat = per_view_embeddings[1].numpy().astype(np.float64)
                    # Raw fusion: concatenate view features on feature dim [N, D1+D2]
                    raw_fusion = np.concatenate([view1_feat, view2_feat], axis=1).astype(np.float64)
                    z_sem_np = z_sem_all.cpu().numpy().astype(np.float64)
                    z_shared_np = z_shared_all.cpu().numpy().astype(np.float64)
                    labels_np = labels_all_np.numpy()
                    # Rank classes by clustering quality for visualization (top-k best classes)
                    class_ranking = compute_class_ranking_by_clustering(
                        z_sem_np, labels_np, n_clusters, random_state=42, use_gpu=use_gpu_clustering
                    )
                    np.savez(
                        best_features_path,
                        view1_feat=view1_feat,
                        view2_feat=view2_feat,
                        raw_fusion=raw_fusion,   # raw fusion: direct view1 + view2 concat
                        z_fusion=z_sem_np,
                        z_shared=z_shared_np,
                        labels_view1=labels_np,
                        labels_view2=labels_np,
                        labels_fusion=labels_np,
                        class_ranking=class_ranking,
                        mode=np.array('aligned', dtype=object),
                        best_epoch=np.array(epoch + 1),
                        best_acc=np.array(acc_sem),
                    )
                    # Also save .mat for MATLAB
                    mat_path = best_features_path.replace('.npz', '.mat')
                    savemat(mat_path, {
                        'view1_feat': view1_feat, 'view2_feat': view2_feat,
                        'raw_fusion': raw_fusion,
                        'z_fusion': z_sem_np, 'z_shared': z_shared_np,
                        'labels_view1': labels_np, 'labels_view2': labels_np,
                        'labels_fusion': labels_np, 'class_ranking': class_ranking,
                        'mode': 'aligned',
                        'best_epoch': np.array([[epoch + 1]]), 'best_acc': np.array([[acc_sem]]),
                    })
                    logger.info(f"Saved best-ACC features to {best_features_path} (epoch {epoch + 1}, ACC={acc_sem:.4f})")

            if acc_shared > best_acc_shared:
                best_acc_shared = acc_shared
                best_epoch_shared = epoch + 1
                best_results_shared = results['z_shared']
                best_acc_sem_at_best_shared = acc_sem  # z_sem ACC when z_shared is best

            # Line 5: best results comparison
            logger.info(
                f"=====> Best | "
                f"Z_sem ACC={best_acc_sem:.4f} (Z_shared ACC={best_acc_shared_at_best_sem:.4f} at epoch {best_epoch_sem}) | "
                f"Z_shared ACC={best_acc_shared:.4f} (Z_sem ACC={best_acc_sem_at_best_shared:.4f} at epoch {best_epoch_shared})"
            )

    # Final results
    logger.info("=" * 60)
    logger.info("Training completed!")

    if use_partial_misalignment:
        # Partial misalignment: best results per z_sem group
        logger.info("Best Results for each Z_sem method:")
        logger.info(f"  Z_sem (MaxSim): ACC={best_acc_maxsim:.4f} at epoch {best_epoch_maxsim}")
        if best_results_maxsim:
            logger.info(f"    NMI={best_results_maxsim['NMI']:.4f}, ARI={best_results_maxsim['ARI']:.4f}, F1={best_results_maxsim['F1']:.4f}")

        logger.info(f"  Z_sem (UOT1): ACC={best_acc_uot1:.4f} at epoch {best_epoch_uot1}")
        if best_results_uot1:
            logger.info(f"    NMI={best_results_uot1['NMI']:.4f}, ARI={best_results_uot1['ARI']:.4f}, F1={best_results_uot1['F1']:.4f}")

        logger.info(f"  Z_sem (UOT2): ACC={best_acc_uot2:.4f} at epoch {best_epoch_uot2}")
        if best_results_uot2:
            logger.info(f"    NMI={best_results_uot2['NMI']:.4f}, ARI={best_results_uot2['ARI']:.4f}, F1={best_results_uot2['F1']:.4f}")

        logger.info(f"  Z_shared: ACC={best_acc_shared:.4f} at epoch {best_epoch_shared}")
        if best_results_shared:
            logger.info(f"    NMI={best_results_shared['NMI']:.4f}, ARI={best_results_shared['ARI']:.4f}, F1={best_results_shared['F1']:.4f}")
    else:
        # Standard mode
        logger.info(f"Best Z_sem: ACC={best_acc_sem:.4f} at epoch {best_epoch_sem}")
        if best_results_sem:
            logger.info(f"  NMI={best_results_sem['NMI']:.4f}, ARI={best_results_sem['ARI']:.4f}, F1={best_results_sem['F1']:.4f}")
        logger.info(f"Best Z_shared: ACC={best_acc_shared:.4f} at epoch {best_epoch_shared}")
        if best_results_shared:
            logger.info(f"  NMI={best_results_shared['NMI']:.4f}, ARI={best_results_shared['ARI']:.4f}, F1={best_results_shared['F1']:.4f}")

    logger.info("=" * 60)

    # After training, rename log file based on best z_sem ACC
    if best_acc_sem > 0.0:
        final_log_path = rename_log_file(log_file_path, best_acc_sem)
        logger.info(f"Log file renamed to: {os.path.basename(final_log_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TriD-MVC MLP model (v5: UOT Semantic Fusion)")

    # Dataset parameters
    parser.add_argument("--dataset", '-data', type=str, default="Scene15", help="Dataset name")
    parser.add_argument("--dataset_path", '-data_path', type=str, default=None, help="Dataset file path")
    parser.add_argument("--n_views", '-n_views', type=int, default=2, help="Number of views (for digits dataset)")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters")
    parser.add_argument("--shuffle_views", '-shuffle_views', action="store_true", help="Shuffle view order")

    # Training parameters
    parser.add_argument("--batch_size", '-bs', type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", '-eps', type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", '-lr', type=float, default=1e-3, help="Learning rate")

    # Loss weight parameters
    parser.add_argument("--weight_recon", '-wr', type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--weight_cluster", '-wcl', type=float, default=0.5, help="Clustering loss weight")
    parser.add_argument("--weight_decorr", '-wde', type=float, default=1.0, help="Decorrelation loss weight")
    parser.add_argument("--weight_style_adv", '-wst', type=float, default=1.0, help="Style adversarial loss weight")
    parser.add_argument("--weight_contrastive", '-wco', type=float, default=1.0, help="Shared contrastive loss weight")

    # Clustering parameters
    parser.add_argument("--cluster_update_interval", '-cui', type=int, default=10, help="Pseudo label/cluster center update interval")
    parser.add_argument("--use_dec_loss", action="store_true", help="Use DEC-style clustering loss")
    parser.add_argument("--dec_alpha", '-da', type=float, default=1.0, help="DEC clustering loss Student t distribution alpha parameter")

    # Style adversarial parameters
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Adversarial loss lambda parameter")

    # Shared contrastive learning parameters
    parser.add_argument("--contrastive_margin", type=float, default=1.5, help="Contrastive loss margin (for L2 normalized features)")
    parser.add_argument("--use_infonce", action="store_true", default=True, help="Use InfoNCE loss instead of noise-robust loss (default: True)")
    parser.add_argument("--no_infonce", dest="use_infonce", action="store_false", help="Disable InfoNCE loss, use noise-robust loss instead")
    parser.add_argument("--contrastive_temperature", '-ctemp', type=float, default=0.5, help="InfoNCE temperature parameter")
    parser.add_argument("--num_neg_samples", type=int, default=None, help="Number of negative samples per positive sample (None for all)")

    # Clustering evaluation parameters
    parser.add_argument("--use_gpu_clustering", action="store_true", help="Use GPU acceleration for k-means clustering")
    parser.add_argument("--no_gpu_clustering", dest="use_gpu_clustering", action="store_false", default=False, help="Disable GPU acceleration for k-means clustering (default)")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs for clustering evaluation (average over multiple runs)")

    # Partial misalignment parameters
    parser.add_argument("--misalign_ratio", "-mar", type=float, default=0.0, help="Misalignment ratio")
    parser.add_argument("--use_partial_misalignment", action="store_true", default=True, help="Use partial misalignment mode (default)")
    parser.add_argument("--no_partial_misalignment", dest="use_partial_misalignment", action="store_false", help="Disable partial misalignment mode")
    parser.add_argument("--missing_rate", '-msr', type=float, default=0.0, help="Missing rate for incomplete data in test set (0.0 means complete data)")
    parser.add_argument("--use_hungarian_alignment", action="store_true", help="Use Hungarian algorithm for alignment")

    # UOT semantic fusion parameters
    parser.add_argument("--topk", '-topk', type=int, default=5, help="Top-K nearest neighbors for imputation in UOT semantic fusion (default: 5)")
    # percentile_threshold is auto-computed from misalign_ratio and missing_rate: 1 - (misalign_ratio * missing_rate) / 2

    # Random seed
    parser.add_argument("--seed", '-seed', type=int, default=None, help="Random seed")

    # Feature saving (for t-SNE visualization)
    parser.add_argument("--save_best_features", action="store_true", help="Save latent features at best ACC epoch (view1/view2, z_UOT1, z_shared) to <logs_root>/<dataset>/best_acc_features.npz and .mat")
    parser.add_argument("--logs_root", type=str, default=None, help="Root directory for logs and saved features (default: <script_dir>/logs)")

    args = parser.parse_args()

    train_tridmvc_mlp(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        n_views=args.n_views,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        shuffle_views=args.shuffle_views,
        weight_recon=args.weight_recon,
        weight_cluster=args.weight_cluster,
        weight_decorr=args.weight_decorr,
        weight_style_adv=args.weight_style_adv,
        weight_contrastive=args.weight_contrastive,
        cluster_update_interval=args.cluster_update_interval,
        use_dec_loss=args.use_dec_loss,
        dec_alpha=args.dec_alpha,
        lambda_adv=args.lambda_adv,
        contrastive_margin=args.contrastive_margin,
        use_infonce=args.use_infonce,
        contrastive_temperature=args.contrastive_temperature,
        num_neg_samples=args.num_neg_samples,
        use_gpu_clustering=args.use_gpu_clustering,
        n_runs=args.n_runs,
        misalign_ratio=args.misalign_ratio,
        use_partial_misalignment=args.use_partial_misalignment,
        missing_rate=args.missing_rate,
        use_hungarian_alignment=args.use_hungarian_alignment,
        topk=args.topk,
        # percentile_threshold is auto-computed from misalign_ratio and missing_rate
        seed=args.seed,
        save_best_features=args.save_best_features,
        logs_root=args.logs_root,
    )
