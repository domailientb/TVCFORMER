# -----------------------------------------------------------------------------
# This file is adapted from PatchTST (https://github.com/PatchTST/PatchTST)
# Original project: “PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”
# Original repository license: Apache-2.0
# Copyright (c) Nie et al.
#
# Modifications made by Yuan Feng et al., 2025:
# TVCFormer parameter adjustment and process optimization.
# -----------------------------------------------------------------------------

import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import yaml
import dgl
if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='config pareser')
    config_parser.add_argument('-c',
                        '--config_yaml',
                        default=
                        'train.yml',
                        type=str,
                        metavar='FILE',
                        help='YAML config file specifying default arguments')
    
    config_args, unknown_config = config_parser.parse_known_args()
    
    parser = argparse.ArgumentParser(description='TVCFormer model for Time Series Forecasting')
    
    # Random seed settings
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='datasets/Ett_Series')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=720)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='TVCFormer')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--padding_patch', default='start')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--time_node_num', type=int, default=8)
    parser.add_argument('--gnn_r_degree', type=int, default=1)
    parser.add_argument('--gnn_hidden_size', type=int, default=32)

    parser.add_argument('--graph_conv_hidden_size', type=int, default=4)
    parser.add_argument('--graph_conv_kernel_size', type=int, default=3)
    parser.add_argument('--revins', type=bool, default=True)
    parser.add_argument('--affine', type=bool, default=False)
    parser.add_argument('--subtract_last', type=bool, default=False)
    parser.add_argument('--decomposition', type=bool, default=False)
    parser.add_argument('--decomposition_kernel_size', type=int, default=25)
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--individual', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--test_flop', action='store_true', default=False)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--visual_samples_num', type=int, default=1)
    
    parser.add_argument('--eps', type=float, default=0.00001)
    parser.add_argument('--subgraph_keep_rate', type=float, default=0.5)
    parser.add_argument('--graph_keep_self_cycles', action='store_true', default=False)
    
    with open(config_args.config_yaml, 'r', encoding='utf-8') as f:
        default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)
    
    args, unknown_args = parser.parse_known_args()
    
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    dgl.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)
    
    Exp = Exp_Main
    
    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_dm{}_nh{}_el{}_dl{}_df{}_gch{}_gck{}_tn{}_kr{}_{}'.format(
                args.model_id,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.graph_conv_hidden_size,
                args.graph_conv_kernel_size,
                args.time_node_num,
                args.subgraph_keep_rate,
                ii)
            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            if args.do_predict:
                pass
            
            torch.cuda.empty_cache()
    
    elif args.visualize:
        ii = 0
        setting = '{}_dm{}_nh{}_el{}_dl{}_df{}_gch{}_gck{}_tn{}_kr{}_{}'.format(
                args.model_id,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.graph_conv_hidden_size,
                args.graph_conv_kernel_size,
                args.time_node_num,
                args.subgraph_keep_rate,
                ii)
        exp = Exp(args)
        print('>>>>>>>visualizing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.visualize(setting, args.visual_samples_num)
        
        torch.cuda.empty_cache()
        
    else:
        ii = 0
        setting = '{}_dm{}_nh{}_el{}_dl{}_df{}_gch{}_gck{}_tn{}_kr{}_{}'.format(
                args.model_id,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.graph_conv_hidden_size,
                args.graph_conv_kernel_size,
                args.time_node_num,
                args.subgraph_keep_rate,
                ii)
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()
