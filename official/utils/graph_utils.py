# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# -----------------------------------------------------------------------------

import torch
from torch import Tensor
import dgl

def genearte_graph_from_coo(batch_index, batch_values, total_node_num):
    edge_index_mat = batch_index.T
    batch_graph = dgl.graph((edge_index_mat[0], edge_index_mat[-1]), num_nodes=total_node_num)
    weight_values = batch_values
    
    return batch_graph, weight_values

def subgraph_tensor_to_full_graph_sparse_tensor(batch_subgraph_weight, bath_subgraph_index, channel_num, time_node_num):
    total_node_num = channel_num * time_node_num
    subgraph_indexs = batch_subgraph_weight.nonzero().T
    src_local_add_values = torch.mul(bath_subgraph_index[subgraph_indexs[0], subgraph_indexs[1]][:,1], time_node_num)
    des_local_add_values = torch.mul(bath_subgraph_index[subgraph_indexs[0], subgraph_indexs[1]][:,2], time_node_num)
    full_graph_src_indexs = torch.add(src_local_add_values, subgraph_indexs[-2])
    full_graph_des_indexs = torch.add(des_local_add_values, subgraph_indexs[-1])
    full_graph_edge_indexs = torch.stack([full_graph_src_indexs, full_graph_des_indexs],dim=-1)
    batch_add_values = subgraph_indexs[0,:] * total_node_num
    sparse_indexs = torch.add(full_graph_edge_indexs, batch_add_values.unsqueeze(-1))
    sparse_values = batch_subgraph_weight[subgraph_indexs[0], subgraph_indexs[1], subgraph_indexs[2], subgraph_indexs[3]]
    return sparse_indexs, sparse_values