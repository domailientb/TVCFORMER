from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = REPOSITORY_ROOT / "official"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(OFFICIAL_ROOT) in sys.path:
    sys.path.remove(str(OFFICIAL_ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))

from FinalProject.Interval_TVC.official_bridge import TVCFormer  # noqa: E402
from utils.correlation_utils import (  # noqa: E402
    get_pearson_correlation_vector,
    top_k_channel_pairs_choose,
)
from utils.normal_utils import get_statistics_full  # noqa: E402
from utils.patch_utils import cut_detail_patches  # noqa: E402


def ordered_quantiles(raw: Tensor) -> Tensor:
    """Map [lower-gap, median, upper-gap] parameters to ordered quantiles."""
    if raw.shape[-1] != 3:
        raise ValueError("raw quantile parameters must have a final dimension of 3")
    lower_gap = F.softplus(raw[..., 0])
    median = raw[..., 1]
    upper_gap = F.softplus(raw[..., 2])
    return torch.stack((median - lower_gap, median, median + upper_gap), dim=-1)


class QuantileFlattenHead(nn.Module):
    def __init__(
        self,
        individual: bool,
        channels: int,
        input_features: int,
        prediction_length: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.channels = channels
        self.prediction_length = prediction_length
        output_features = 3 * prediction_length

        if individual:
            self.linears = nn.ModuleList(
                nn.Linear(input_features, output_features) for _ in range(channels)
            )
            self.dropouts = nn.ModuleList(
                nn.Dropout(head_dropout) for _ in range(channels)
            )
        else:
            self.linear = nn.Linear(input_features, output_features)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, features: Tensor) -> Tensor:
        flattened = torch.flatten(features, start_dim=-2)
        if self.individual:
            outputs = [
                self.dropouts[channel](self.linears[channel](flattened[:, channel]))
                for channel in range(self.channels)
            ]
            raw = torch.stack(outputs, dim=1)
        else:
            raw = self.dropout(self.linear(flattened))
        raw = raw.reshape(
            raw.shape[0], self.channels, 3, self.prediction_length
        ).permute(0, 1, 3, 2)
        return ordered_quantiles(raw)


class IntervalTVCFormer(nn.Module):
    """Quantile head over the unchanged official TVCFormer batch-dependent backbone."""

    def __init__(self, configs: object) -> None:
        super().__init__()
        self.point_model = TVCFormer.Model(configs)
        if self.point_model.decomposition:
            raise ValueError("interval mode currently requires decomposition=False")
        if self.point_model.na_head or self.point_model.last_value_residual:
            raise ValueError("interval mode requires the standard TVCFormer prediction path")
        if self.point_model.graph_gate:
            raise ValueError("interval mode currently requires graph_gate=0")

        backbone = self.point_model.model
        if backbone.pretrain_head or backbone.na_head:
            raise ValueError("interval mode requires the standard flatten decoder")
        backbone.detail_decoder = QuantileFlattenHead(
            individual=backbone.individual,
            channels=backbone.c_in,
            input_features=backbone.detail_patch_total_len,
            prediction_length=configs.pred_len,
            head_dropout=configs.head_dropout,
        )

    @property
    def backbone(self):
        return self.point_model.model

    def forward(self, x: Tensor) -> Tensor:
        """Return [batch, horizon, channel, 3] quantiles on the input scale."""
        backbone = self.backbone
        z = x.permute(0, 2, 1)

        if backbone.graph_selector is None:
            channel_relevance = get_pearson_correlation_vector(z)
            channel_indices = top_k_channel_pairs_choose(
                c_vector=channel_relevance,
                ch_num=backbone.c_in,
                keep_rate=backbone.subgraph_keep_rate,
                keep_self_cycles=backbone.graph_keep_self_cycles,
                device=backbone.device,
            )
        else:
            channel_indices = backbone.graph_selector.select(
                z,
                keep_rate=backbone.subgraph_keep_rate,
                keep_self_cycles=backbone.graph_keep_self_cycles,
            )

        z_statistics = z.permute(0, 2, 1)
        z_statistics = get_statistics_full(
            x_in=z_statistics,
            padding_len=backbone.x_padding_len,
            padding_mode=backbone.padding_patch,
            block_num=backbone.time_node_num,
            subtract_last=backbone.subtract_last,
            eps=backbone.eps,
        )
        z_statistics = backbone.statistic_norm(z_statistics.permute(0, 3, 1, 2))
        z_statistics = z_statistics.permute(0, 2, 3, 1)

        z = backbone.padding_patch_layer(z)
        if backbone.revin:
            z = backbone.revin_layer(z.permute(0, 2, 1), "norm").permute(0, 2, 1)

        windows = None
        if backbone.use_sdei:
            windows = torch.stack(
                torch.chunk(backbone.padding_patch_layer(z), backbone.time_node_num, dim=-1),
                dim=-2,
            )
        edge_indices, edge_weights, _ = backbone.graph_generator(
            z_statistics, channel_indices, windows=windows
        )
        backbone.contrastive_loss = getattr(
            backbone.graph_generator,
            "last_loss_contrast",
            torch.tensor(0.0, device=z.device),
        )
        backbone.gate_value = getattr(backbone.graph_generator, "last_gate", None)

        z = cut_detail_patches(z, backbone.time_node_num, backbone.patch_len, backbone.stride)
        _, encoded = backbone.backbone(
            z, edge_indices, edge_weights, disable_graph=backbone.a_ge
        )
        quantiles = backbone.detail_decoder(encoded.permute(0, 1, 3, 2))
        lower = quantiles[:, :, :, 0].permute(0, 2, 1)
        median = quantiles[:, :, :, 1].permute(0, 2, 1)
        upper = quantiles[:, :, :, 2].permute(0, 2, 1)

        if backbone.revin:
            lower = backbone.revin_layer(lower, "denorm")
            median = backbone.revin_layer(median, "denorm")
            upper = backbone.revin_layer(upper, "denorm")
        return torch.stack((lower, median, upper), dim=-1)
