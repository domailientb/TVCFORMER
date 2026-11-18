import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path

ROOT_DIR = Path("d:/TVCFORMER")
OFFICIAL_DIR = ROOT_DIR / "official"
sys.path.insert(0, str(OFFICIAL_DIR))

from exp.exp_main import Exp_Main
from utils.tools import EarlyStopping, adjust_learning_rate
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn

def get_args(pred_len, sdei, sdei_lambda, sdei_shape, seed=2021):
    args = argparse.Namespace(
        is_training=1,
        model_id=f"ExchangeRate_A_pl{pred_len}",
        model="TVCFormer",
        data="custom",
        root_path=str(ROOT_DIR / "experiments" / "datasets"),
        data_path="exchange_rate.csv",
        target="OT",
        scale=True,
        seq_len=336,
        label_len=0,
        pred_len=pred_len,
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
        checkpoints="checkpoints/Exchange_Rate",
        itr=1,
        learning_rate=0.001,
        patience=5,
        train_epochs=100,
        pct_start=0.3,
        lradj="type3",
        batch_size=128,
        num_workers=0,
        test_flop=False,
        do_predict=False,
        random_seed=seed,
        sdei=sdei,
        sdei_gate_init=-3.0,
        sdei_lambda=sdei_lambda,
        sdei_shape=sdei_shape,
        sdei_gate_fixed=0,
    )
    args.device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    return args

def evaluate_metrics(exp, model, flag):
    data, _ = exp._get_data(flag=flag)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=min(exp.args.batch_size, len(data)),
        shuffle=False,
        drop_last=False
    )
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(exp.device)
            output_y = model(batch_x)
            batch_y_y = batch_y[:, -exp.args.pred_len:, :].to(exp.device)
            preds.append(output_y.detach().cpu().numpy())
            trues.append(batch_y_y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    model.train()
    return mse, mae

def train_and_eval(args, setting):
    exp = Exp_Main(args)
    train_data, train_loader = exp._get_data(flag='train')
    vali_data, _ = exp._get_data(flag='val')
    vali_loader = torch.utils.data.DataLoader(
        vali_data,
        batch_size=min(args.batch_size, len(vali_data)),
        shuffle=False,
        drop_last=False
    )
    test_data, _ = exp._get_data(flag='test')
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=min(args.batch_size, len(test_data)),
        shuffle=False,
        drop_last=False
    )
    
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)
        
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    model_optim = exp._select_optimizer()
    criterion_y = nn.MSELoss()
    
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    t0 = time.time()
    
    for epoch in range(args.train_epochs):
        exp.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            
            output_y = exp.model(batch_x)
            batch_y_y = batch_y[:, -args.pred_len:, :].to(exp.device)
            loss_mse = criterion_y(output_y, batch_y_y)
            
            contrastive_loss = getattr(exp.model, 'contrastive_loss', torch.tensor(0.0, device=exp.device))
            sdei_lambda = getattr(args, 'sdei_lambda', 0.0)
            loss = loss_mse + sdei_lambda * contrastive_loss
            
            loss.backward()
            model_optim.step()
            
            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()
                
        vali_loss = exp.vali(vali_data, vali_loader, criterion_y=criterion_y, mode_flag='val')
        if vali_loss < best_val_loss:
            best_val_loss = vali_loss
            best_epoch = epoch + 1
            
        early_stopping(vali_loss, exp.model, path)
        if early_stopping.early_stop:
            break
            
        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
            
    train_time = time.time() - t0
    
    # Load best model
    best_model_path = path + '/checkpoint.pth'
    exp.model.load_state_dict(torch.load(best_model_path))
    
    val_mse, val_mae = evaluate_metrics(exp, exp.model, flag='val')
    test_mse, test_mae = evaluate_metrics(exp, exp.model, flag='test')
    
    gate_val = getattr(exp.model, 'gate_value', None)
    gate_num = float(gate_val.item()) if gate_val is not None else 0.0
    
    return {
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "best_epoch": best_epoch,
        "gate": gate_num,
        "train_time_s": train_time
    }

def run_experiment(pred_len, sdei, sdei_lambda, sdei_shape, seed=2021):
    args = get_args(pred_len, sdei, sdei_lambda, sdei_shape, seed=seed)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    cfg_name = "baseline" if sdei == 0 else f"sdei_sh{sdei_shape}_l{sdei_lambda}"
    setting = f"StepA_{cfg_name}_pl{pred_len}_seed{seed}"
    
    metrics = train_and_eval(args, setting)
    
    return {
        "config": cfg_name,
        "sdei": sdei,
        "lambda": sdei_lambda,
        "shape": sdei_shape,
        "horizon": pred_len,
        "seed": seed,
        "val_mse": metrics["val_mse"],
        "test_mse": metrics["test_mse"],
        "test_mae": metrics["test_mae"],
        "best_epoch": metrics["best_epoch"],
        "sigmoid_gate": metrics["gate"],
        "train_time_s": metrics["train_time_s"]
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=None)
    cmd_args = parser.parse_args()
    
    horizons = [cmd_args.horizon] if cmd_args.horizon is not None else [96, 192, 336, 720]
    
    configs = [
        {"name": "baseline", "sdei": 0, "lambda": 0.0, "shape": 0},
        {"name": "sdei_sh1_l0.0", "sdei": 1, "lambda": 0.0, "shape": 1},
        {"name": "sdei_sh1_l0.05", "sdei": 1, "lambda": 0.05, "shape": 1},
        {"name": "sdei_sh1_l0.1", "sdei": 1, "lambda": 0.1, "shape": 1},
        {"name": "sdei_sh1_l0.3", "sdei": 1, "lambda": 0.3, "shape": 1},
        {"name": "sdei_sh0_l0.0", "sdei": 1, "lambda": 0.0, "shape": 0},
        {"name": "sdei_sh0_l0.05", "sdei": 1, "lambda": 0.05, "shape": 0},
        {"name": "sdei_sh0_l0.1", "sdei": 1, "lambda": 0.1, "shape": 0},
        {"name": "sdei_sh0_l0.3", "sdei": 1, "lambda": 0.3, "shape": 0},
    ]
    
    out_dir = ROOT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    csv_file = out_dir / "step_a_screening.csv"
    
    results = []
    if csv_file.exists():
        existing_df = pd.read_csv(csv_file)
        results = existing_df.to_dict(orient='records')
        print(f"Loaded {len(results)} existing results from {csv_file}")
    
    for h in horizons:
        for c in configs:
            already_done = any(r['config'] == c['name'] and r['horizon'] == h for r in results)
            if already_done:
                print(f"[SKIP] Already computed: {c['name']} (horizon={h})")
                continue
                
            print(f"\n==========================================")
            print(f">>> Running: {c['name']} | Horizon={h} | Seed=2021")
            print(f"==========================================")
            
            res = run_experiment(
                pred_len=h,
                sdei=c['sdei'],
                sdei_lambda=c['lambda'],
                sdei_shape=c['shape'],
                seed=2021
            )
            results.append(res)
            
            pd.DataFrame(results).to_csv(csv_file, index=False)
            print(f"==> Done {c['name']} (h={h}) | Val MSE: {res['val_mse']:.6f} | Test MSE: {res['test_mse']:.6f} | Gate: {res['sigmoid_gate']:.4f} | Epoch: {res['best_epoch']}")
            
    print("\n" + "="*60)
    print("STEP A SCREENING COMPLETE! SUMMARY TABLE (Validation MSE):")
    print("="*60)
    df = pd.DataFrame(results)
    pivot = df.pivot(index='config', columns='horizon', values='val_mse')
    print(pivot.to_string())
