"""
Model architecture module v5.
Tri-way latent multi-view MLP + clustering head.
No VAE; MLP extracts feature representations directly.
v4 improvement: style adversarial loss predicts view labels to force z_style to learn view-specific style.
v5 improvement: integrates UOT semantic fusion for inference on misaligned data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ----------------------------
# Gradient Reversal Layer
# ----------------------------

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.lambd
        return grad_output.neg() * lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ----------------------------
# Model components
# ----------------------------

def create_encoder_backbone(input_dim: int,
                           hidden_dims: list = [1024, 1024, 1024],
                           output_dim: int = None,
                           dropout: float = 0.2,
                           use_batch_norm: bool = True,
                           activation: str = 'relu') -> nn.Sequential:
    """
    Create an encoder backbone network.

    Args:
        input_dim: Input dimension.
        hidden_dims: Hidden layer dimensions; default [1024, 1024, 1024].
        output_dim: Output dimension; if None, use the last hidden layer dimension.
        dropout: Dropout rate; default 0.2.
        use_batch_norm: Whether to use BatchNorm; default True.
        activation: Activation type ('relu', 'tanh', 'sigmoid'); default 'relu'.

    Returns:
        encoder: Encoder network.
    """
    layers = []
    current_dim = input_dim

    # Select activation function
    if activation.lower() == 'relu':
        activation_fn = nn.ReLU(True)
    elif activation.lower() == 'tanh':
        activation_fn = nn.Tanh()
    elif activation.lower() == 'sigmoid':
        activation_fn = nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    # Build hidden layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim

    # Output layer (if output_dim is specified)
    if output_dim is not None:
        layers.append(nn.Linear(current_dim, output_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(activation_fn)

    return nn.Sequential(*layers)


class ViewMLPEncoder(nn.Module):
    """
    Per-view MLP encoder:
    x^(v) -> [z_s^(v), z_p^(v), z_u^(v)]
    Outputs three feature representations directly (no VAE).
    """
    def __init__(self, in_dim, z_shared_dim, z_priv_dim, z_style_dim,
                 hidden_dims: list = [1024, 1024, 1024],
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        # Backbone network
        self.backbone = create_encoder_backbone(
            input_dim=in_dim,
            hidden_dims=hidden_dims,
            output_dim=None,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation
        )

        backbone_output_dim = hidden_dims[-1] if hidden_dims else in_dim

        # Independent MLP head for each latent
        # z_shared
        self.zs_head = nn.Linear(backbone_output_dim, z_shared_dim)

        # z_private
        self.zp_head = nn.Linear(backbone_output_dim, z_priv_dim)

        # z_style
        self.zu_head = nn.Linear(backbone_output_dim, z_style_dim)

        self.z_shared_dim = z_shared_dim
        self.z_priv_dim = z_priv_dim
        self.z_style_dim = z_style_dim

    def forward(self, x):
        h = self.backbone(x)

        # Output three feature representations directly
        zs = self.zs_head(h)
        zp = self.zp_head(h)
        zu = self.zu_head(h)

        return {
            'zs': zs,
            'zp': zp,
            'zu': zu,
        }


class ViewMLPDecoder(nn.Module):
    """
    MLP decoder:
    [z_s, z_p, z_u] -> x_hat^(v)
    """
    def __init__(self, out_dim, z_total_dim,
                 hidden_dims: list = [1024, 1024, 1024],
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        self.net = create_encoder_backbone(
            input_dim=z_total_dim,
            hidden_dims=hidden_dims,
            output_dim=out_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation
        )

    def forward(self, z_cat):
        return self.net(z_cat)


class TriDMVC_MLP(nn.Module):
    """
    Tri-way disentangled multi-view MLP model + clustering head:
    - Per-view MLP encoder/decoder
    - Three latents: z_s (shared), z_p (private), z_u (style)
    - Clustering head on semantic representation [z_s, z_p]
    """
    def __init__(self, view_dims,
                 z_shared_dim=16, z_priv_dim=16, z_style_dim=16,
                 n_clusters=10,
                 hidden_dims: list = [1024, 1024, 1024],
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        """
        Initialize TriDMVC_MLP.

        Args:
            view_dims: Input dimension for each view.
            z_shared_dim: Shared representation dimension.
            z_priv_dim: Private representation dimension.
            z_style_dim: Style representation dimension.
            n_clusters: Number of clusters.
            hidden_dims: Hidden layer dimensions; default [1024, 1024, 1024].
            dropout: Dropout rate; default 0.2.
            use_batch_norm: Whether to use BatchNorm; default True.
            activation: Activation type ('relu', 'tanh', 'sigmoid'); default 'relu'.
        """
        super().__init__()
        self.n_views = len(view_dims)
        self.z_shared_dim = z_shared_dim
        self.z_priv_dim = z_priv_dim
        self.z_style_dim = z_style_dim
        self.n_clusters = n_clusters

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.cluster_heads = nn.ModuleList()  # Clustering heads
        # Style classifier: shared across views to distinguish z_u from different views
        # so the classifier learns true style features rather than memorizing input source
        self.style_classifier = nn.Linear(z_style_dim, self.n_views)  # Shared classifier

        for d in view_dims:
            # MLP encoder
            self.encoders.append(
                ViewMLPEncoder(
                    in_dim=d,
                    z_shared_dim=z_shared_dim,
                    z_priv_dim=z_priv_dim,
                    z_style_dim=z_style_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation=activation
                )
            )
            # MLP decoder
            self.decoders.append(
                ViewMLPDecoder(
                    out_dim=d,
                    z_total_dim=z_shared_dim + z_priv_dim + z_style_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation=activation
                )
            )
            # Clustering head: predict cluster labels from semantic [z_s, z_p]
            self.cluster_heads.append(
                nn.Linear(z_shared_dim + z_priv_dim, n_clusters)
            )

    def encode_views(self, views):
        """
        Encode all views.

        Returns:
            zs_list, zp_list, zu_list: Lists of feature representations.
        """
        zs_list, zp_list, zu_list = [], [], []

        for v, enc in zip(views, self.encoders):
            enc_output = enc(v)
            zs_list.append(enc_output['zs'])
            zp_list.append(enc_output['zp'])
            zu_list.append(enc_output['zu'])

        return zs_list, zp_list, zu_list

    def decode_views(self, zs_list, zp_list, zu_list):
        """Decode all views."""
        recons = []
        for zs, zp, zu, dec in zip(zs_list, zp_list, zu_list, self.decoders):
            z_cat = torch.cat([zs, zp, zu], dim=-1)
            rec = dec(z_cat)
            recons.append(rec)
        return recons

    def semantic_embedding(self, zs_list, zp_list):
        """
        Build semantic representation:
        - shared: cross-view average
        - private: concatenation across views
        """
        zs_stack = torch.stack(zs_list, dim=0)   # [V, B, D_s]
        zs_mean = torch.mean(zs_stack, dim=0)    # [B, D_s]
        zp_cat = torch.cat(zp_list, dim=-1)      # [B, V*D_p]
        sem = torch.cat([zs_mean, zp_cat], dim=-1)
        return sem

    def get_semantic_embeddings_per_view(self, zs_list, zp_list):
        """
        Per-view semantic representations (for view-level losses).
        Returns: [sem_view1, sem_view2, ...]
        """
        sem_list = []
        for zs, zp in zip(zs_list, zp_list):
            sem_view = torch.cat([zs, zp], dim=-1)
            sem_list.append(sem_view)
        return sem_list

    def forward(self, views):
        """
        Forward pass.

        Returns:
            dict with latents, reconstructions, semantic embeddings, etc.
        """
        zs_list, zp_list, zu_list = self.encode_views(views)

        recons = self.decode_views(zs_list, zp_list, zu_list)
        sem = self.semantic_embedding(zs_list, zp_list)
        sem_per_view = self.get_semantic_embeddings_per_view(zs_list, zp_list)

        return {
            "zs_list": zs_list,
            "zp_list": zp_list,
            "zu_list": zu_list,
            "sem": sem,
            "sem_per_view": sem_per_view,
            "recons": recons,
        }
