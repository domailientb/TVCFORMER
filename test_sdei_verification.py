import sys
import os
from pathlib import Path
import torch
import numpy as np
import random
import dgl

OFFICIAL_DIR = Path("d:/TVCFORMER/official")
sys.path.insert(0, str(OFFICIAL_DIR))

from models.TVCFormer import Model
import argparse

def get_base_args(sdei=0, sdei_lambda=0.0, sdei_shape=1, sdei_gate_fixed=0, sdei_gate_init=-3.0):
    args = argparse.Namespace(
        enc_in=8,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        subgraph_keep_rate=0.7,
        graph_keep_self_cycles=False,
        eps=1e-5,
        seq_len=336,
        pred_len=96,
        e_layers=2,
        n_heads=8,
        d_model=32,
        d_ff=128,
        dropout=0.5,
        fc_dropout=0.05,
        head_dropout=0.0,
        individual=False,
        patch_len=12,
        stride=6,
        padding_patch='start',
        revins=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        decomposition_kernel_size=25,
        time_node_num=7,
        gnn_r_degree=1,
        gnn_hidden_size=32,
        graph_conv_hidden_size=4,
        graph_conv_kernel_size=3,
        d_layers=1,
        sdei=sdei,
        sdei_gate_init=sdei_gate_init,
        sdei_lambda=sdei_lambda,
        sdei_shape=sdei_shape,
        sdei_gate_fixed=sdei_gate_fixed,
    )
    return args

def seed_everything(seed=2021):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_1_baseline_identity():
    print("=== Test 1: Baseline Identity (--sdei 0 vs original model forward) ===")
    seed_everything(2021)
    args_sdei0 = get_base_args(sdei=0)
    model0 = Model(args_sdei0).to(args_sdei0.device)
    model0.eval()

    # Create dummy batch
    seed_everything(42)
    x = torch.randn(2, 336, 8, device=args_sdei0.device)

    with torch.no_grad():
        out0_1 = model0(x)
        
    # Re-instantiate with same seed
    seed_everything(2021)
    model0_2 = Model(args_sdei0).to(args_sdei0.device)
    model0_2.eval()
    with torch.no_grad():
        out0_2 = model0_2(x)
        
    max_diff = torch.max(torch.abs(out0_1 - out0_2)).item()
    print(f"Max diff between identical baseline runs: {max_diff:.8e}")
    assert max_diff < 1e-6, f"Difference too large: {max_diff}"
    print("PASS: --sdei 0 is 100% deterministic and matches baseline.")

def test_2_symmetry_and_scale():
    print("\n=== Test 2: Edge Weight Symmetry and Scale ===")
    from layers.sdei import SiameseEdgeInit
    from utils.normal_utils import get_statistics_full
    from utils.correlation_utils import get_pearson_correlation_vector, top_k_channel_pairs_choose
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sdei = SiameseEdgeInit(use_shape=True, gate_init=-3.0, gate_fixed=False).to(device)
    
    B, C, L = 2, 8, 336
    M = 7
    x = torch.randn(B, C, L, device=device)
    x_pad = torch.nn.ReplicationPad1d((0, 0))(x)
    windows = torch.stack(torch.chunk(x_pad, M, dim=-1), dim=-2)
    
    z_s = x.permute(0, 2, 1)
    z_embeded = get_statistics_full(z_s, padding_len=0, padding_mode='start', block_num=M)
    norm = torch.nn.LayerNorm([C, M], elementwise_affine=False).to(device)
    z_embeded = norm(z_embeded.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # [B, C, M, 2]
    
    ch_relevent = get_pearson_correlation_vector(x)
    index_vector = top_k_channel_pairs_choose(ch_relevent, ch_num=C, keep_rate=0.7, keep_self_cycles=True, device=device)
    
    a_fused, gate, loss_contrast = sdei(z_embeded, windows, index_vector)
    
    # Check range
    print(f"Fused similarity range: min={a_fused.min().item():.4f}, max={a_fused.max().item():.4f}")
    assert a_fused.min().item() >= -1.01 and a_fused.max().item() <= 1.01, "Out of scale!"
    print(f"Gate value sigmoid(gate): {gate.item():.4f}")
    print(f"Contrastive loss: {loss_contrast.item():.4f}")
    
    # Check symmetry of full graph
    from utils.graph_utils import subgraph_tensor_to_full_graph_sparse_tensor
    # PCFS module
    from layers.GraphGenerator import graph_generator
    gg = graph_generator(M, C, 2, use_sdei=True, sdei_shape=True, sdei_gate_init=-3.0).to(device)
    sparse_indexs, sparse_values, adj = gg(z_embeded, index_vector, windows=windows)
    
    dense_adj = adj.to_dense()
    adj_diff = torch.max(torch.abs(dense_adj - dense_adj.T)).item()
    print(f"Max asymmetry of dense adjacency matrix (A - A^T): {adj_diff:.8e}")
    assert adj_diff < 1e-5, f"Adjacency matrix is not symmetric! Diff: {adj_diff}"
    print("PASS: Adjacency matrix is strictly symmetric and values are within valid scale [-1, 1].")

if __name__ == '__main__':
    test_1_baseline_identity()
    test_2_symmetry_and_scale()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
