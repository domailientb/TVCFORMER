# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# Description:
# Used in TVCFormer’s Graph Encoding Unit.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class dgl_graph_layer_for_trans(nn.Module):
    def __init__(self, d_model, hidden_size, n_layer, method_type, aggregator_type='gcn', feat_drop=0., norm=None, activation=None):
        super(dgl_graph_layer_for_trans, self).__init__()
        
        self.d_model = d_model
        
        if method_type == 'SAGE':
            if n_layer == 1:
                self.graph_layers_list = nn.ModuleList([dglnn.SAGEConv(d_model, d_model, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation)])
                
            elif n_layer == 2:
                self.graph_layers_list = nn.ModuleList([dglnn.SAGEConv(d_model, hidden_size, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation),
                                                        dglnn.SAGEConv(hidden_size, d_model, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation)
                                                        ])
            else:
                hidden_layer_num =  n_layer - 2
                rough_layers = [dglnn.SAGEConv(d_model, hidden_size, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation)]
                rough_layers.extend([dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation) for i in range(hidden_layer_num)])
                rough_layers.append(dglnn.SAGEConv(hidden_size, d_model, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation))
                self.graph_layers_list = nn.ModuleList(rough_layers)
        else:
            raise ValueError("do not have such gnn method type")
    
    def forward(self, g, x_in, weights):
        
        node_num = x_in.shape[-2]
        x_flatten = torch.flatten(x_in, start_dim=0, end_dim=-2)
        
        for graph_layer in self.graph_layers_list:
            x_flatten =  graph_layer(g, x_flatten, weights)
        
        x_out = x_flatten.reshape(-1, node_num, self.d_model)
        
        return x_out


class dgl_graph_for_trans(nn.Module):
    def __init__(self,
                    d_model,
                    graph_r,
                    graph_hidden_size=64,
                    graph_method_type='SAGE'):
        super(dgl_graph_for_trans, self).__init__()
        
        self.graph_layer = dgl_graph_layer_for_trans(d_model, graph_hidden_size, graph_r, method_type=graph_method_type)
        
    def forward(self, x_in, g, weights):
        
        temp_shape = x_in.shape
        x_out = torch.flatten(x_in, start_dim=1, end_dim=2)
        x_out = self.graph_layer(g, x_out, weights)
        x_out = x_out.reshape(temp_shape)

        return x_out
