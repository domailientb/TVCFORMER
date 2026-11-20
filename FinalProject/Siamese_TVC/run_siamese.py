"""Run isolated Siamese_TVC experiment using official TVCFormer data pipeline."""
import argparse
import contextlib
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'official'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'FinalProject/Siamese_TVC/configs/ETTh1_pl96.yaml'))
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--data')
    parser.add_argument('--root_path')
    parser.add_argument('--data_path')
    parser.add_argument('--target')
    parser.add_argument('--pred_len', type=int)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--use_gpu', type=lambda x: str(x).lower() in ('1','true','yes','on'))
    parser.add_argument('--candidate_mode', choices=('pearson', 'siamese', 'hybrid'))
    parser.add_argument('--siamese_gate_mode', choices=('learned', 'statistical', 'siamese'))
    parser.add_argument('--train_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--measure_graph_time', action='store_true')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f) or {}
    for key in ('data', 'root_path', 'data_path', 'target', 'pred_len', 'random_seed', 'use_gpu',
                'candidate_mode', 'siamese_gate_mode', 'train_epochs', 'patience', 'batch_size'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.measure_graph_time:
        cfg['measure_graph_time'] = True
    cfg['model'] = 'TVCFormer_Siamese'
    cfg['device'] = torch.device(f"cuda:{cfg.get('gpu', 0)}" if cfg.get('use_gpu', True) and torch.cuda.is_available() else 'cpu')
    cfg['use_gpu'] = cfg['device'].type == 'cuda'
    cfg['is_training'] = 1
    cfg['checkpoints'] = str(ROOT / 'FinalProject/Siamese_TVC/checkpoints')
    cfg['results_dir'] = str(ROOT / 'FinalProject/Siamese_TVC/results' / str(cfg['data']))
    cfg['logs_dir'] = str(ROOT / 'FinalProject/Siamese_TVC/logs' / str(cfg['data']))
    cfg['root_path'] = str((ROOT / cfg['root_path']).resolve()) if not Path(cfg['root_path']).is_absolute() else cfg['root_path']
    os.makedirs(cfg['logs_dir'], exist_ok=True)
    seed = int(cfg.get('random_seed', 2021))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cfg['use_gpu']:
        torch.cuda.manual_seed_all(seed)
    try:
        import dgl
        dgl.random.seed(seed)
    except Exception:
        pass
    from FinalProject.Siamese_TVC.exp_siamese import Exp_Siamese
    from types import SimpleNamespace
    config = SimpleNamespace(**cfg)
    setting = (f"{cfg['model_id']}_pl{cfg['pred_len']}_seed{seed}_"
               f"{cfg.get('candidate_mode','hybrid')}_{cfg.get('siamese_gate_mode','learned')}")
    log_path = os.path.join(cfg['logs_dir'], setting + '.log')

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, value):
            for stream in self.streams: stream.write(value); stream.flush()
        def flush(self):
            for stream in self.streams: stream.flush()

    with open(log_path, 'a', encoding='utf-8') as log, contextlib.redirect_stdout(Tee(sys.__stdout__, log)):
        print(f"config={args.config} data={cfg['data']} horizon={cfg['pred_len']} device={cfg['device']}")
        exp = Exp_Siamese(config)
        exp.train(setting)
        if args.do_test:
            exp.test(setting, test=1)
    return log_path


if __name__ == '__main__':
    main()
