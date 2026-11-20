"""Official forecasting architecture with a Siamese TVC graph generator."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / 'official'
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

from official.models.TVCFormer import Model as OfficialTVCFormer
from FinalProject.Siamese_TVC.models.graph_generator import (
    CandidateSupportSelector, GraphGeneratorSiamese,
)


class Model(OfficialTVCFormer):
    def __init__(self, configs, *args, **kwargs):
        # Build the ordinary backbone; only enable its local-window extraction
        # after replacing the unused statistical graph generator below.
        configs.sdei = 0
        super().__init__(configs, *args, **kwargs)
        backbones = [self.model] if not self.decomposition else [self.model_res, self.model_trend]
        for backbone in backbones:
            backbone.use_sdei = True
            backbone.graph_generator = GraphGeneratorSiamese(
                timenode_num=backbone.time_node_num,
                channel_num=backbone.c_in,
                statistics_dim=backbone.statistics_num,
                conv_hidden_size=configs.graph_conv_hidden_size,
                conv_kernel_size=configs.graph_conv_kernel_size,
                hidden_dim=getattr(configs, 'siamese_hidden_dim', 32),
                embedding_dim=getattr(configs, 'siamese_embedding_dim', 16),
                gate_init=getattr(configs, 'siamese_gate_init', -3.0),
                gate_mode=getattr(configs, 'siamese_gate_mode', 'learned'),
                candidate_mode=getattr(configs, 'candidate_mode', 'hybrid'),
                candidate_max_keep_rate=getattr(configs, 'candidate_max_keep_rate', 1.0),
                cnn_dim=getattr(configs, 'siamese_cnn_dim', 32),
                measure_graph_time=getattr(configs, 'measure_graph_time', False),
            ).to(configs.device)
            # The official backbone already supports a selector hook. Keep the
            # adapter plain (not an nn.Module) so parameters are registered only
            # once under graph_generator.
            object.__setattr__(backbone, 'graph_selector',
                               CandidateSupportSelector(backbone.graph_generator, backbone))
        self.gate_value = None

    def graph_diagnostics(self):
        backbones = [self.model] if not self.decomposition else [self.model_res, self.model_trend]
        result = []
        for backbone in backbones:
            generator = backbone.graph_generator
            if generator.last_graph is not None:
                result.append({
                    'adjacency': generator.last_graph,
                    'edge_index': generator.last_edge_index,
                    'embeddings': generator.last_embeddings,
                    'gate': generator.last_gate,
                    'pearson': generator.last_pearson,
                    'candidate_diagnostics': generator.last_candidate_diagnostics,
                    'graph_build_seconds': generator.last_graph_build_seconds,
                })
        return result

    def forward(self, x):
        output = super().forward(x)
        backbones = [self.model] if not self.decomposition else [self.model_res, self.model_trend]
        gates = [b.graph_generator.last_gate for b in backbones
                 if b.graph_generator.last_gate is not None]
        self.gate_value = sum(gates) / len(gates) if gates else None
        return output
