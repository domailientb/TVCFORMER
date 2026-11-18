# -----------------------------------------------------------------------------
# Siamese Distribution-aware Edge Initializer (SDEI)
# Extended module for TVCFormer
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEdgeInit(nn.Module):
    """
    Siamese Distribution-aware Edge Initializer (SDEI)
    - Statistical features: eta = (normalized local mean, normalized local std) [B, C, M, 2]
    - Shape features: 6 invariant time-series features of local window [B, C, M, 6] (optional)
    - Shared Siamese MLP encoder
    - Cosine similarity between embeddings for local window pairs
    - Learnable / fixed gating mechanism
    - Contrastive alignment loss
    """

    def __init__(
        self,
        use_shape=True,
        hidden_dim=32,
        embedding_dim=16,
        gate_init=-3.0,
        gate_fixed=False,
    ):
        super().__init__()
        self.use_shape = bool(use_shape)
        self.gate_fixed = bool(gate_fixed)
        feature_dim = 8 if self.use_shape else 2

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.gate_param = nn.Parameter(torch.tensor(float(gate_init)))

    @staticmethod
    def extract_shape_features(x):
        """
        x: [B, C, M, W]
        returns: [B, C, M, 6]
        1. mean standardized first difference
        2. std standardized first difference
        3. mean absolute first difference
        4. lag-1 autocorrelation
        5. mean standardized value
        6. std standardized value
        """
        eps = 1e-5
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(eps)
        xn = (x - mu) / std

        dx = x[..., 1:] - x[..., :-1]
        dx_std = dx.std(dim=-1, keepdim=True).clamp_min(eps)
        dxn = dx / dx_std

        dx_mean = dxn.mean(dim=-1, keepdim=True)
        dx_std_feat = dxn.std(dim=-1, keepdim=True)
        dx_abs = dxn.abs().mean(dim=-1, keepdim=True)

        x1 = xn[..., :-1]
        x2 = xn[..., 1:]
        lag1 = ((x1 * x2).mean(dim=-1, keepdim=True)) / (
            x1.std(dim=-1, keepdim=True) * x2.std(dim=-1, keepdim=True) + eps
        )
        x_mean = xn.mean(dim=-1, keepdim=True)
        x_std = xn.std(dim=-1, keepdim=True)

        return torch.cat(
            [dx_mean, dx_std_feat, dx_abs, lag1, x_mean, x_std], dim=-1
        )

    def encode(self, eta, shape=None):
        if self.use_shape and shape is not None:
            features = torch.cat([eta, shape], dim=-1)
        else:
            features = eta
        z = self.encoder(features)
        z = F.normalize(z, dim=-1)
        return z

    def get_gate(self):
        if self.gate_fixed:
            return torch.tensor(1.0, device=self.gate_param.device)
        return torch.sigmoid(self.gate_param)

    def forward(self, eta, windows, index_vector):
        """
        eta: [B, C, M, 2]
        windows: [B, C, M, W]
        index_vector: [B, K_sub, 3] (col 1 is src_channel, col 2 is dst_channel)
        """
        B, C, M, _ = eta.shape
        shape = self.extract_shape_features(windows) if self.use_shape else None
        z = self.encode(eta, shape)  # [B, C, M, D]

        src_channel = index_vector[:, :, 1].long()
        dst_channel = index_vector[:, :, 2].long()
        K_sub = src_channel.shape[1]

        src_idx = src_channel[:, :, None, None].expand(B, K_sub, M, z.size(-1))
        dst_idx = dst_channel[:, :, None, None].expand(B, K_sub, M, z.size(-1))

        z_src = torch.gather(z, dim=1, index=src_idx)  # [B, K_sub, M, D]
        z_dst = torch.gather(z, dim=1, index=dst_idx)  # [B, K_sub, M, D]

        # Siamese cosine similarity for all window pairs (m, n)
        a_siam = torch.matmul(z_src, z_dst.transpose(-1, -2)).clamp(-1.0, 1.0)

        # Original statistical cosine similarity (Eq. 11-12)
        eta_src = torch.gather(
            eta, 1, src_channel[:, :, None, None].expand(B, K_sub, M, 2)
        )
        eta_dst = torch.gather(
            eta, 1, dst_channel[:, :, None, None].expand(B, K_sub, M, 2)
        )

        eta_src = F.normalize(eta_src, dim=-1)
        eta_dst = F.normalize(eta_dst, dim=-1)
        a_stat = torch.matmul(eta_src, eta_dst.transpose(-1, -2)).clamp(-1.0, 1.0)

        gate = self.get_gate()
        a_fused = (1.0 - gate) * a_stat + gate * a_siam

        # Contrastive Loss calculation:
        # Pull corresponding temporal windows (m == n) across selected channel pairs together,
        # push apart non-corresponding windows (m != n).
        tau = 0.1
        sim_matrix = torch.matmul(z_src, z_dst.transpose(-1, -2)) / tau
        sim_flat = sim_matrix.reshape(-1, M, M)
        labels = torch.arange(M, device=eta.device).unsqueeze(0).expand(sim_flat.size(0), -1)
        loss_contrast = F.cross_entropy(sim_flat.reshape(-1, M), labels.reshape(-1))

        return a_fused, gate, loss_contrast
