"""
Loss functions module v5.
Reconstruction + clustering + decorrelation + style adversarial + shared contrastive losses.
No VAE; all KL divergence terms removed.
v4 improvement: style adversarial loss predicts view labels to force z_style to learn view-specific style.
v5 improvement: integrates UOT semantic fusion for inference on misaligned data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models_v5 import grad_reverse


# ----------------------------
# Reconstruction loss
# ----------------------------

def reconstruction_loss(recons, views):
    """
    Reconstruction loss (MSE).

    Args:
        recons: List of reconstructed views.
        views: List of original views.

    Returns:
        loss: Reconstruction loss.
    """
    loss = 0.0
    for rec, v in zip(recons, views):
        loss = loss + F.mse_loss(rec, v, reduction='mean')
    return loss / len(recons)


# ----------------------------
# Clustering loss (DEC-style)
# ----------------------------

def dec_clustering_loss(sem_embeddings, cluster_centers, alpha=1.0):
    """
    DEC-style clustering loss: Student t distribution + target distribution sharpening.

    Goal: shape semantic subspace z_sem = [z_sh, z_pr] into a cluster-friendly structure.
    Effect: Student t distribution + target sharpening encourages intra-class compactness
    and inter-class separation.

    Args:
        sem_embeddings: Semantic embeddings [B, D].
        cluster_centers: Cluster centers [n_clusters, D].
        alpha: Student t degrees of freedom (default 1.0).

    Returns:
        q: Soft assignment distribution [B, n_clusters].
        p: Target distribution [B, n_clusters].
        loss: KL(p || q) loss.
    """
    # Distance from samples to cluster centers
    # sem_embeddings: [B, D], cluster_centers: [n_clusters, D]
    distances = torch.cdist(sem_embeddings, cluster_centers, p=2) ** 2  # [B, n_clusters]

    # Student t: q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
    # Normalize to obtain soft assignments
    q = (1 + distances / alpha).pow(-(alpha + 1) / 2)  # [B, n_clusters]
    q = q / (q.sum(dim=1, keepdim=True) + 1e-8)  # Normalize

    # Target sharpening: p_ij = q_ij^2 / sum_j(q_ij^2)
    p = q.pow(2) / (q.pow(2).sum(dim=1, keepdim=True) + 1e-8)  # [B, n_clusters]

    # KL divergence: KL(p || q) = sum_i sum_j p_ij * log(p_ij / q_ij)
    loss = (p * torch.log((p + 1e-8) / (q + 1e-8))).sum(dim=1).mean()

    return q, p, loss


def dec_clustering_loss_multi_view(sem_embeddings_list, cluster_centers_list, alpha=1.0):
    """
    Multi-view DEC clustering loss.

    Args:
        sem_embeddings_list: List of semantic embeddings (one per view), each [B, D].
        cluster_centers_list: List of cluster centers (one per view), each [n_clusters, D].
        alpha: Student t degrees of freedom.

    Returns:
        total_loss: Total clustering loss.
        all_q: List of soft assignment distributions for all views.
        all_p: List of target distributions for all views.
    """
    total_loss = 0.0
    all_q = []
    all_p = []

    for sem, centers in zip(sem_embeddings_list, cluster_centers_list):
        q, p, loss = dec_clustering_loss(sem, centers, alpha)
        total_loss = total_loss + loss
        all_q.append(q)
        all_p.append(p)

    return total_loss / len(sem_embeddings_list), all_q, all_p


def clustering_loss_with_pseudo_labels(sem_embeddings, cluster_heads, pseudo_labels):
    """
    Clustering loss with pseudo labels (when pseudo labels are available).

    Args:
        sem_embeddings: List of semantic embeddings (one per view).
        cluster_heads: List of clustering heads.
        pseudo_labels: Pseudo labels [B].

    Returns:
        loss: Clustering loss.
        predictions: Predicted cluster labels.
    """
    total_loss = 0.0
    all_predictions = []

    for sem, cluster_head in zip(sem_embeddings, cluster_heads):
        logits = cluster_head(sem)  # [B, n_clusters]
        predictions = torch.argmax(logits, dim=1)
        all_predictions.append(predictions)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, pseudo_labels)
        total_loss = total_loss + loss

    return total_loss / len(sem_embeddings), all_predictions


# ----------------------------
# Shared contrastive loss
# ----------------------------

class NoiseRobustLoss(nn.Module):
    """
    Noise-robust contrastive loss.
    Handles noisy positive/negative pairs in contrastive learning.
    """
    def __init__(self, margin=1.5, use_robust_loss=True, start_fine=False, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.use_robust_loss = use_robust_loss
        self.start_fine = start_fine
        self.reduction = reduction

    def forward(self, pair_dist, labels, margin=None, use_robust_loss=None, start_fine=None):
        margin = margin if margin is not None else self.margin
        use_robust_loss = use_robust_loss if use_robust_loss is not None else self.use_robust_loss
        start_fine = start_fine if start_fine is not None else self.start_fine

        labels = labels.to(torch.float32)
        dist_sq = pair_dist * pair_dist
        N = len(labels)

        if use_robust_loss:
            if start_fine:
                pos_loss = labels * dist_sq
                neg_term = torch.pow(pair_dist, 0.5) * (margin - pair_dist)
                neg_term_clamped = torch.clamp(neg_term, min=0.0)
                neg_loss = (1 - labels) * (1.0 / margin) * torch.pow(neg_term_clamped, 2)
                loss = pos_loss + neg_loss
            else:
                pos_loss = labels * dist_sq
                neg_loss = (1 - labels) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
                loss = pos_loss + neg_loss
        else:
            pos_loss = labels * dist_sq
            neg_loss = (1 - labels) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
            loss = pos_loss + neg_loss

        loss_sum = torch.sum(loss)
        if self.reduction == 'mean':
            return loss_sum / N
        else:
            return loss_sum


def shared_contrastive_loss_robust(zs_list, noise_robust_loss_fn, margin=1.5, use_infonce=False,
                                   temperature=0.5, num_neg_samples=None):
    """
    Shared-feature contrastive loss using noise-robust contrastive learning.
    Encodes aligned data and builds positive/negative pairs from cross-view shared latents.
    Supports 2-view setting only.

    Args:
        zs_list: List of shared features [zs1, zs2], each [B, D_s].
        noise_robust_loss_fn: NoiseRobustLoss instance.
        margin: Margin (default 1.5, suitable for L2-normalized features).
        use_infonce: Whether to use InfoNCE loss.
        temperature: InfoNCE temperature.
        num_neg_samples: Number of negative samples per anchor (noise_robust_loss_fn only).
                        If None, use all off-diagonal pairs (B*(B-1)).
                        If int, randomly sample num_neg_samples negatives per anchor.

    Returns:
        loss: Contrastive loss.
        stats: Dict with average distance statistics for positive/negative pairs:
            - 'pos_avg_dist': Average distance of positive pairs.
            - 'neg_avg_dist': Average distance of negative pairs.
            - 'pos_count': Number of positive pairs.
            - 'neg_count': Number of negative pairs.
    """
    assert len(zs_list) == 2, "shared_contrastive_loss_robust only supports 2 views"

    z1, z2 = zs_list
    batch_size = z1.size(0)

    # Normalize shared features
    z1_norm = F.normalize(z1, dim=1)  # [B, D_s]
    z2_norm = F.normalize(z2, dim=1)  # [B, D_s]

    # Build positive and negative pairs
    # Positive pairs: same index across views (same sample, different view)
    pos_dists = torch.norm(z1_norm - z2_norm, dim=1)  # [B] - distance per positive pair

    # Select loss function
    if use_infonce:
        # InfoNCE (similarity-based, better suited for contrastive learning)
        similarity_matrix = torch.matmul(z1_norm, z2_norm.t()) / temperature  # [B, B]
        labels = torch.arange(batch_size, device=z1.device)  # Positive indices: 0, 1, 2, ..., B-1
        loss = F.cross_entropy(similarity_matrix, labels)

        theoretical_lower_bound = torch.log(torch.tensor(float(batch_size), device=z1.device))

        # Cross-view distance matrix
        dist_matrix = torch.cdist(z1_norm.unsqueeze(0), z2_norm.unsqueeze(0), p=2).squeeze(0)  # [B, B]
        # Exclude diagonal (positive pairs)
        mask = ~torch.eye(batch_size, device=z1.device, dtype=torch.bool)
        neg_dists = dist_matrix[mask]  # [B*(B-1)]
        neg_avg_dist = neg_dists.mean() if len(neg_dists) > 0 else torch.tensor(0.0, device=z1.device)
        neg_count = batch_size * (batch_size - 1)  # Each positive has B-1 negatives in InfoNCE
    else:
        # Noise-robust contrastive loss (distance-based)
        if noise_robust_loss_fn is None:
            raise ValueError("noise_robust_loss_fn cannot be None when use_infonce=False")

        # Negative pairs: random samples from the other view per anchor
        if num_neg_samples is None:
            # Use all off-diagonal pairs
            dist_matrix = torch.cdist(z1_norm.unsqueeze(0), z2_norm.unsqueeze(0), p=2).squeeze(0)  # [B, B]
            mask = ~torch.eye(batch_size, device=z1.device, dtype=torch.bool)
            neg_dists = dist_matrix[mask]  # [B*(B-1)]
        else:
            # Randomly sample num_neg_samples negatives per anchor
            num_neg_samples = min(num_neg_samples, batch_size - 1)

            neg_dists_list = []
            for i in range(batch_size):
                candidate_indices = torch.cat([
                    torch.arange(i, device=z1.device),
                    torch.arange(i + 1, batch_size, device=z1.device)
                ])

                if len(candidate_indices) > 0:
                    if len(candidate_indices) >= num_neg_samples:
                        selected_indices = candidate_indices[torch.randperm(len(candidate_indices), device=z1.device)[:num_neg_samples]]
                    else:
                        selected_indices = candidate_indices

                    z1_i = z1_norm[i:i+1]  # [1, D_s]
                    z2_selected = z2_norm[selected_indices]  # [num_selected, D_s]
                    dists = torch.norm(z1_i - z2_selected, dim=1)  # [num_selected]
                    neg_dists_list.append(dists)

            if len(neg_dists_list) > 0:
                neg_dists = torch.cat(neg_dists_list, dim=0)
            else:
                neg_dists = torch.tensor([], device=z1.device)

        # Merge positive and negative pairs
        all_pair_dists = torch.cat([pos_dists, neg_dists], dim=0)
        all_pair_labels = torch.cat([
            torch.ones(batch_size, device=z1.device),
            torch.zeros(len(neg_dists), device=z1.device)
        ], dim=0)
        loss = noise_robust_loss_fn(all_pair_dists, all_pair_labels, margin=margin)

        neg_avg_dist = neg_dists.mean() if len(neg_dists) > 0 else torch.tensor(0.0, device=z1.device)
        neg_count = len(neg_dists)

    # Average distances for positive and negative pairs
    pos_avg_dist = pos_dists.mean() if len(pos_dists) > 0 else torch.tensor(0.0, device=z1.device)

    stats = {
        'pos_avg_dist': pos_avg_dist,
        'neg_avg_dist': neg_avg_dist,
        'pos_count': len(pos_dists),
        'neg_count': neg_count
    }

    # For InfoNCE, add theoretical lower bound info
    if use_infonce:
        stats['theoretical_lower_bound'] = theoretical_lower_bound
        stats['loss_above_lower_bound'] = loss - theoretical_lower_bound
        stats['loss_ratio'] = loss / (theoretical_lower_bound + 1e-8)

    return loss, stats


# ----------------------------
# Decorrelation loss
# ----------------------------

def decorrelation_loss(zs_list, zp_list, zu_list):
    """
    Decorrelation loss: cross-covariance for statistical independence.

    Goal: separate semantic and style latents so they do not entangle in the same subspace.
    Effect: minimize cross-covariance between semantic and style latents toward independence.

    Args:
        zs_list: List of shared features.
        zp_list: List of private features.
        zu_list: List of style features.

    Returns:
        loss_zs_zp: Decorrelation loss between z_s and z_p (cross-covariance).
        loss_zu_sem: Decorrelation loss between z_u and semantic (cross-covariance).
        total_loss: Total decorrelation loss.
    """
    loss_zs_zp = 0.0
    loss_zu_sem = 0.0

    for zs, zp, zu in zip(zs_list, zp_list, zu_list):
        batch_size = zs.size(0)

        # 1. Decorrelate z_s and z_p: cross-covariance matrix
        zs_centered = zs - zs.mean(dim=0, keepdim=True)  # [B, D_s]
        zp_centered = zp - zp.mean(dim=0, keepdim=True)  # [B, D_p]

        # Cross-covariance: Cov(zs, zp) = (1/B) * zs^T * zp
        cross_cov_zs_zp = torch.matmul(zs_centered.t(), zp_centered) / batch_size  # [D_s, D_p]
        # Minimize squared Frobenius norm of cross-covariance
        loss_zs_zp = loss_zs_zp + cross_cov_zs_zp.pow(2).sum()

        # 2. Decorrelate z_u from [z_s, z_p]
        sem = torch.cat([zs, zp], dim=1)  # [B, D_s + D_p]
        sem_centered = sem - sem.mean(dim=0, keepdim=True)
        zu_centered = zu - zu.mean(dim=0, keepdim=True)

        # Cross-covariance matrix
        cross_cov_zu_sem = torch.matmul(zu_centered.t(), sem_centered) / batch_size  # [D_u, D_sem]
        # Minimize squared cross-covariance
        loss_zu_sem = loss_zu_sem + cross_cov_zu_sem.pow(2).sum()

    num_views = len(zs_list)
    loss_zs_zp = loss_zs_zp / num_views
    loss_zu_sem = loss_zu_sem / num_views
    total_loss = loss_zs_zp + loss_zu_sem

    return loss_zs_zp, loss_zu_sem, total_loss


# ----------------------------
# Style adversarial loss
# ----------------------------

def style_adversarial_loss(zu_list, style_classifier, lambda_adv=1.0):
    """
    Style adversarial loss: encourage z_u to encode view-specific style.

    Goal: z_u should predict view labels (view-specific style information).

    Key fix: all views share one classifier with true view labels as targets,
    so the classifier learns style rather than memorizing input source.

    Note: no gradient reversal here; encoder is encouraged to produce view-predictive z_u.
    Style classifier is trained separately (in train_v5.py) for accurate prediction.
    Encoder is pushed via this loss to encode view style in z_u.

    Args:
        zu_list: List of style features, one z_u per view [B, D_u].
        style_classifier: Shared style classifier (all views).
        lambda_adv: Loss weight (kept for API compatibility; not used for gradient reversal).

    Returns:
        loss: Style adversarial loss (lower is better = z_u predicts view labels well).
    """
    # Collect z_u and view labels from all views
    all_zu = []
    all_view_labels = []

    for view_idx, zu in enumerate(zu_list):
        batch_size = zu.size(0)
        # View label for every sample in this view (all view_idx)
        view_labels = torch.full((batch_size,), view_idx, dtype=torch.long, device=zu.device)
        all_zu.append(zu)
        all_view_labels.append(view_labels)

    # Concatenate z_u and labels across views
    all_zu_concat = torch.cat(all_zu, dim=0)  # [B*n_views, D_u]
    all_view_labels_concat = torch.cat(all_view_labels, dim=0)  # [B*n_views]

    # Shared classifier on all z_u
    logits = style_classifier(all_zu_concat)  # [B*n_views, n_views]

    # Cross-entropy: encourage accurate view prediction from z_u
    loss = F.cross_entropy(logits, all_view_labels_concat)

    return loss


def style_adversarial_loss_with_view_labels(zu_list, style_classifier, lambda_adv=1.0):
    """
    Style adversarial loss with view labels (v4).

    Args:
        zu_list: List of style features.
        style_classifier: Shared style classifier (all views).
        lambda_adv: Adversarial loss weight.

    Returns:
        loss: Style adversarial loss.
    """
    return style_adversarial_loss(zu_list, style_classifier, lambda_adv)


# ----------------------------
# Combined loss
# ----------------------------

# Global counter for diagnostics
_compute_total_loss_call_count = 0

def compute_total_loss(model, model_output, views, cluster_labels=None, pseudo_labels=None,
                       weight_recon=1.0, weight_cluster=1.0,
                       weight_decorr=1.0, weight_style_adv=1.0,
                       weight_contrastive=1.0,  # Shared contrastive loss weight
                       lambda_adv=1.0, dec_alpha=1.0,
                       cluster_centers_list=None,
                       # Contrastive learning parameters
                       contrastive_margin=1.5,
                       use_infonce=False,
                       contrastive_temperature=0.5,
                       num_neg_samples=None,
                       noise_robust_loss_fn=None):
    """
    Total loss (v4: no VAE, no KL; style adversarial uses view labels).

    Args:
        model: Model instance (cluster_heads, style_classifiers).
        model_output: Model output dict.
        views: List of original views.
        cluster_labels: True cluster labels [B] (optional).
        pseudo_labels: Pseudo labels [B] (optional, for clustering/adversarial).
        weight_recon: Reconstruction weight.
        weight_cluster: Clustering weight.
        weight_decorr: Decorrelation weight.
        weight_style_adv: Style adversarial weight.
        weight_contrastive: Shared contrastive weight.
        lambda_adv: Lambda for adversarial loss.
        dec_alpha: Student t dof for DEC clustering.
        cluster_centers_list: Cluster centers for DEC; if None, use pseudo-label loss.
        contrastive_margin: Contrastive margin.
        use_infonce: Use InfoNCE.
        contrastive_temperature: InfoNCE temperature.
        num_neg_samples: Negatives per sample.
        noise_robust_loss_fn: NoiseRobustLoss instance.

    Returns:
        dict: Per-term and total losses.
    """
    # 1. Reconstruction loss
    recon_loss = reconstruction_loss(model_output['recons'], views)

    # 2. Clustering loss (DEC-style or pseudo labels)
    if cluster_centers_list is not None:
        # DEC-style clustering
        cluster_loss, all_q, all_p = dec_clustering_loss_multi_view(
            model_output['sem_per_view'], cluster_centers_list, alpha=dec_alpha
        )
        cluster_predictions = [torch.argmax(q, dim=1) for q in all_q]
    elif pseudo_labels is not None:
        # Pseudo-label clustering
        cluster_loss, cluster_predictions = clustering_loss_with_pseudo_labels(
            model_output['sem_per_view'], model.cluster_heads, pseudo_labels
        )
        all_q, all_p = None, None
    else:
        # Skip clustering if no centers or pseudo labels
        cluster_loss = torch.tensor(0.0, device=views[0].device)
        cluster_predictions = None
        all_q, all_p = None, None

    # 3. Decorrelation loss (cross-covariance)
    decorr_zs_zp, decorr_zu_sem, decorr_loss = decorrelation_loss(
        model_output['zs_list'],
        model_output['zp_list'],
        model_output['zu_list']
    )

    # 4. Style adversarial loss (v4: view labels, not cluster labels)
    # View label per sample: view index for each view's batch
    # Assumes aligned data: same batch_size per view
    batch_size = views[0].size(0)
    n_views = len(views)

    # e.g. 2 views: view 0 -> label 0, view 1 -> label 1
    view_labels_list = []
    for view_idx in range(n_views):
        view_labels = torch.full((batch_size,), view_idx, dtype=torch.long, device=views[0].device)
        view_labels_list.append(view_labels)

    # Use first view's labels (same across views when aligned)
    view_labels = view_labels_list[0]

    # clone() avoids graph conflicts; do not detach() or gradients won't reach encoder
    zu_list_cloned = [zu.clone() for zu in model_output['zu_list']]
    style_adv_loss = style_adversarial_loss_with_view_labels(
        zu_list_cloned,
        model.style_classifier,  # Shared classifier
        lambda_adv
    )

    # Style loss should decrease (z_u predicts views better); if it increases, check training

    # 5. Shared contrastive loss
    if len(model_output['zs_list']) == 2:
        zs_list_for_contrastive = model_output['zs_list']
        if not all(zs.requires_grad for zs in zs_list_for_contrastive):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("z_s does not require grad; contrastive loss cannot optimize the model.")

        # InfoNCE: noise_robust_loss_fn may be None
        if use_infonce or noise_robust_loss_fn is not None:
            contrastive_loss, contrastive_stats = shared_contrastive_loss_robust(
                zs_list_for_contrastive,
                noise_robust_loss_fn,
                margin=contrastive_margin,
                use_infonce=use_infonce,
                temperature=contrastive_temperature,
                num_neg_samples=num_neg_samples
            )

            if not contrastive_loss.requires_grad:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Contrastive loss does not require grad; cannot optimize the model.")
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Contrastive loss skipped: use_infonce=False and noise_robust_loss_fn=None")
            contrastive_loss = torch.tensor(0.0, device=views[0].device)
            contrastive_stats = {
                'pos_avg_dist': torch.tensor(0.0, device=views[0].device),
                'neg_avg_dist': torch.tensor(0.0, device=views[0].device),
                'pos_count': 0,
                'neg_count': 0
            }
    else:
        contrastive_loss = torch.tensor(0.0, device=views[0].device)
        contrastive_stats = {
            'pos_avg_dist': torch.tensor(0.0, device=views[0].device),
            'neg_avg_dist': torch.tensor(0.0, device=views[0].device),
            'pos_count': 0,
            'neg_count': 0
        }

    # 6. Total loss (no KL term)
    total_loss = (weight_recon * recon_loss +
                  weight_cluster * cluster_loss +
                  weight_decorr * decorr_loss +
                  weight_style_adv * style_adv_loss +
                  weight_contrastive * contrastive_loss)

    # Loss scale diagnostics (first call only)
    global _compute_total_loss_call_count
    _compute_total_loss_call_count += 1
    if _compute_total_loss_call_count == 1:
        import logging
        logger = logging.getLogger(__name__)
        weighted_contrastive = weight_contrastive * contrastive_loss.item()

        logger.info("Loss scale check (first call):")
        logger.info(f"  Recon={recon_loss.item():.4f} (weighted: {weight_recon * recon_loss.item():.4f})")
        logger.info(f"  Cluster={cluster_loss.item():.4f} (weighted: {weight_cluster * cluster_loss.item():.4f})")
        logger.info(f"  Contrastive={contrastive_loss.item():.4f} (weighted: {weighted_contrastive:.4f})")
        logger.info(f"  Decor={decorr_loss.item():.4f} (weighted: {weight_decorr * decorr_loss.item():.4f})")
        logger.info(f"  StyleAdv={style_adv_loss.item():.4f} (weighted: {weight_style_adv * style_adv_loss.item():.4f})")
        logger.info(f"  Total={total_loss.item():.4f}")

        contrastive_ratio = weighted_contrastive / (total_loss.item() + 1e-8) * 100
        if contrastive_ratio < 5:
            logger.warning(f"  Contrastive contribution too small ({contrastive_ratio:.1f}%), may not optimize effectively.")

    return {
        'recon_loss': recon_loss,
        'cluster_loss': cluster_loss,
        'decorr_zs_zp': decorr_zs_zp,
        'decorr_zu_sem': decorr_zu_sem,
        'decorr_loss': decorr_loss,
        'style_adv_loss': style_adv_loss,
        'contrastive_loss': contrastive_loss,
        'contrastive_stats': contrastive_stats,
        'total_loss': total_loss,
        'cluster_predictions': cluster_predictions,
        'dec_q': all_q,  # DEC soft assignments
        'dec_p': all_p,  # DEC target distribution
    }
