# ATVF: Adaptive Time-Variable Coupling TVCFormer.
# The original TVCFormer backbone and forecasting head are reused unchanged.

import torch
from torch import nn

from models import TVCFormer
from layers.ATVF_GraphGenerator import ATVFGraphGenerator


class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.base_model = TVCFormer.Model(configs, **kwargs)

        def replace_graph_generator(backbone):
            backbone.graph_generator = ATVFGraphGenerator(
                timenode_num=configs.time_node_num,
                channel_num=configs.enc_in,
                embeded_dim=2,
                hidden_dim=getattr(configs, "atvf_hidden_dim", 16),
                edge_keep_rate=getattr(configs, "edge_keep_rate", 0.35),
                lambda_stability=getattr(configs, "lambda_stability", 0.01),
                lambda_sparsity=getattr(configs, "lambda_sparsity", 0.001),
                lambda_smooth=getattr(configs, "lambda_smooth", 0.001),
            )

        if self.base_model.decomposition:
            replace_graph_generator(self.base_model.model_res)
            replace_graph_generator(self.base_model.model_trend)
        else:
            replace_graph_generator(self.base_model.model)

    def forward(self, x):
        return self.base_model(x)

    def get_aux_loss(self):
        if self.base_model.decomposition:
            generators = [
                self.base_model.model_res.graph_generator,
                self.base_model.model_trend.graph_generator,
            ]
        else:
            generators = [self.base_model.model.graph_generator]

        losses = [
            module.last_aux_loss
            for module in generators
            if module.last_aux_loss is not None
        ]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()
