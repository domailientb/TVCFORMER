# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# Description:
# Used to generate the TVC Graph.
# -----------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.graph_utils import subgraph_tensor_to_full_graph_sparse_tensor


class graph_generator(nn.Module):
    def __init__(self, timenode_num, channel_num, embeded_dim, conv_hidden_size=4, conv_kernel_size=3):
        super(graph_generator, self).__init__()
        
        assert conv_kernel_size % 2 != 0, "conv_kernel_size must be an odd number"
        
        self.timenode_num = timenode_num
        self.channel_num = channel_num
        self.embeded_dim = embeded_dim
        
        padding_size = (conv_kernel_size - 1) // 2
        self.sparse_padding = nn.ReflectionPad2d(padding=(padding_size, padding_size, padding_size, padding_size))
        self.mode_capture_component = nn.Parameter(torch.empty((conv_hidden_size, 1, conv_kernel_size, conv_kernel_size)), requires_grad=True)
        self.sparse_capture_act = nn.ReLU()
        self.subgraph_fusion_conv = nn.Conv2d(in_channels=conv_hidden_size, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sparse_norm = nn.InstanceNorm2d(1)
        self.sparse_fusion_act = nn.ReLU()
        
        nn.init.kaiming_uniform_(self.mode_capture_component, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x_in, index_vector):
        x_graph = self.generate_chosen_subgraph_features(x_in=x_in, index_vector=index_vector)
        x_graph = self.sparse_module(x_graph)
        sparse_indexs, sparse_values = subgraph_tensor_to_full_graph_sparse_tensor(x_graph, index_vector, self.channel_num, self.timenode_num)
        batch_size = x_graph.shape[0]
        total_node_num = batch_size * self.channel_num * self.timenode_num
        adj = torch.sparse.FloatTensor(sparse_indexs.T, sparse_values, torch.Size([total_node_num,total_node_num]))
        return sparse_indexs, sparse_values, adj
    
    def get_conv_weights(self, mode):
        if mode == 'symmetry':
            conv_weight_half = torch.div(self.mode_capture_component, 2)
            conv_weight_half_T = conv_weight_half.permute(0,1,3,2)
            conv_weights = torch.add(conv_weight_half, conv_weight_half_T)
        else:
            conv_weights = self.mode_capture_component
        return conv_weights
    
    def generate_chosen_subgraph_features(self, x_in, index_vector):
        source_extend_index = index_vector[:,:,1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.timenode_num,self.embeded_dim)
        destination_extend_index = index_vector[:,:,2].unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.timenode_num,self.embeded_dim)
        select_source_mat = torch.gather(x_in, dim=1, index=source_extend_index)
        select_destination_mat = torch.gather(x_in, dim=1, index=destination_extend_index)
        extend_source_mat = select_source_mat.unsqueeze(-2).repeat(1, 1, 1, self.timenode_num, 1)
        extend_destination_mat = select_destination_mat.unsqueeze(-3).repeat(1, 1, self.timenode_num, 1, 1)
        x_out = torch.cosine_similarity(extend_source_mat, extend_destination_mat, dim=-1)
        return x_out
    
    def sparse_module(self, x_in):
        sub_graph_num = x_in.shape[1]
        x_flatten = x_in.flatten(start_dim=0, end_dim=1).unsqueeze(-1)
        x_flatten = x_flatten.permute(0,3,1,2)
        x_flatten = self.sparse_padding(x_flatten)
        conv_weights = self.get_conv_weights('symmetry')
        x_flatten = F.conv2d(x_flatten, conv_weights)
        x_flatten = self.sparse_capture_act(x_flatten)
        x_flatten = self.subgraph_fusion_conv(x_flatten)
        x_flatten = self.sparse_norm(x_flatten)
        x_flatten = self.sparse_fusion_act(x_flatten)
        x_out = x_flatten.squeeze(1).reshape(-1, sub_graph_num, self.timenode_num, self.timenode_num)
        return x_out
