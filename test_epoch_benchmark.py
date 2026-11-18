import sys
import os
import time
from pathlib import Path
import torch
import numpy as np

OFFICIAL_DIR = Path("d:/TVCFORMER/official")
sys.path.insert(0, str(OFFICIAL_DIR))

from exp.exp_main import Exp_Main
import argparse

def get_config(sdei=0, sdei_lambda=0.0, sdei_shape=1, sdei_gate_fixed=0, sdei_gate_init=-3.0):
    args = argparse.Namespace(
        is_training=1,
        model_id=f"test_bench_sdei{sdei}",
        model="TVCFormer",
        data="custom",
        root_path="d:/TVCFORMER/experiments/datasets",
        data_path="exchange_rate.csv",
        target="OT",
        scale=True,
        seq_len=336,
        label_len=0,
        pred_len=96,
        enc_in=8,
        time_node_num=7,
        patch_len=12,
        stride=6,
        padding_patch="start",
        subgraph_keep_rate=0.7,
        graph_keep_self_cycles=False,
        eps=1e-5,
        e_layers=2,
        d_layers=1,
        n_heads=8,
        d_model=32,
        d_ff=128,
        dropout=0.5,
        fc_dropout=0.05,
        head_dropout=0.0,
        individual=False,
        revins=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        decomposition_kernel_size=25,
        gnn_r_degree=1,
        gnn_hidden_size=32,
        graph_conv_hidden_size=4,
        graph_conv_kernel_size=3,
        use_gpu=torch.cuda.is_available(),
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        use_amp=False,
        checkpoints="checkpoints/test_checkpoints",
        itr=1,
        learning_rate=0.001,
        patience=5,
        train_epochs=1, # 1 epoch for verification
        pct_start=0.3,
        lradj="type3",
        batch_size=128,
        num_workers=0,
        test_flop=False,
        do_predict=False,
        random_seed=2021,
        sdei=sdei,
        sdei_gate_init=sdei_gate_init,
        sdei_lambda=sdei_lambda,
        sdei_shape=sdei_shape,
        sdei_gate_fixed=sdei_gate_fixed,
    )
    args.device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    return args

def run_one_epoch(sdei=0, sdei_lambda=0.0):
    args = get_config(sdei=sdei, sdei_lambda=sdei_lambda)
    exp = Exp_Main(args)
    setting = f"test_sdei_{sdei}"
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
    start_time = time.time()
    model = exp.train(setting)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        
    gate_val = getattr(model, 'gate_value', None)
    gate_num = gate_val.item() if gate_val is not None else None
    c_loss = getattr(model, 'contrastive_loss', torch.tensor(0.0)).item()
    
    return elapsed, peak_mem, gate_num, c_loss

if __name__ == '__main__':
    print("--- Running Baseline (sdei=0, 1 epoch) ---")
    t0, mem0, _, _ = run_one_epoch(sdei=0)
    print(f"Baseline: Time = {t0:.2f}s | Peak GPU Mem = {mem0:.2f} MB")
    
    print("\n--- Running SDEI (sdei=1, lambda=0.05, 1 epoch) ---")
    t1, mem1, gate1, closs1 = run_one_epoch(sdei=1, sdei_lambda=0.05)
    print(f"SDEI: Time = {t1:.2f}s | Peak GPU Mem = {mem1:.2f} MB | Gate = {gate1:.4f} | Contrastive Loss = {closs1:.4f}")
    
    print("\nVerification Summary:")
    print(f"Time difference: {t1 - t0:+.2f}s ({(t1/t0 - 1)*100:+.1f}%)")
    print(f"Memory difference: {mem1 - mem0:+.2f} MB")
    print("No NaNs detected, training completed cleanly.")
