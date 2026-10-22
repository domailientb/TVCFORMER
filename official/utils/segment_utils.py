# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import math

def generate_token_values(cover_tokens, num_output=1, generate_method='average'):
    if generate_method == 'average':
        output_token = torch.mean(cover_tokens, dim=-2, keepdim=True)
    elif generate_method == 'zeros':
        output_token = torch.mean(cover_tokens, dim=-2, keepdim=True) * 0
    else:
        raise ValueError('method not defined')
    if num_output > 1:
        output_tokens = output_token.repeat(1, 1, num_output, 1)
    else:
        output_tokens = output_token
    
    return output_tokens

def generate_local_tokens(x_in, time_node_num, generate_method='average'):
    segment_tuple = torch.chunk(x_in, time_node_num, dim=-2)
    local_token_list = []
    for segment_unit in segment_tuple:
        temp_tokens = generate_token_values(cover_tokens=segment_unit, num_output=1, generate_method=generate_method)
        local_token_list.append(temp_tokens)
    
    output_tokens = torch.cat(local_token_list, dim=-2)
    
    return output_tokens

def embedding_local_tokens(local_tokens, detail_tokens, time_node_num):
    splited_detail_token_list = torch.chunk(detail_tokens, time_node_num, dim=-2)
    splited_detail_tokens = torch.stack(splited_detail_token_list, dim=2)
    extended_local_tokens = local_tokens.unsqueeze(-2)
    cat_tokens = torch.cat((extended_local_tokens, splited_detail_tokens), dim=-2)
    embeded_tokens = torch.flatten(cat_tokens, start_dim=2, end_dim=-2)
    
    return embeded_tokens

def split_local_tokens_from_total_tokens(total_tokens, time_node_num):
    splited_total_token_list = torch.chunk(total_tokens, time_node_num, dim=-2)
    splited_total_tokens = torch.stack(splited_total_token_list, dim=2)
    local_tokens = splited_total_tokens[:, :, :, 0, :]
    
    detail_tokens = torch.flatten(splited_total_tokens[:, :, :, 1:, :], start_dim=2, end_dim=-2)
    
    return local_tokens, detail_tokens
