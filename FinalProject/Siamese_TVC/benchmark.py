"""Resumable paired E0/E1/E2 suite for the Clustering_TVC datasets."""
import os
os.environ.setdefault('DGLBACKEND', 'pytorch')

import argparse
import contextlib
import csv
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'official'
SIAMESE_DIR = ROOT / 'FinalProject' / 'Siamese_TVC'
for entry in (str(SIAMESE_DIR), str(OFFICIAL), str(ROOT)):
    while entry in sys.path:
        sys.path.remove(entry)
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(1, str(ROOT))

DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather')
SEEDS = (2021, 2023, 2025)
HORIZONS = (96, 192, 336, 720)
RESULTS = SIAMESE_DIR / 'results'
RUNS_CSV = RESULTS / 'benchmark_full_runs.csv'
SUMMARY_CSV = RESULTS / 'benchmark_full_summary.csv'
FIELDS = ('dataset', 'experiment', 'horizon', 'seed', 'status', 'run_id',
          'mse', 'mae', 'best_epoch', 'best_validation_loss', 'train_seconds',
          'inference_seconds', 'graph_build_seconds', 'peak_gpu_memory_mb',
          'graph_density', 'graph_degree_mean', 'graph_degree_std',
          'graph_weight_mean', 'graph_weight_std', 'graph_symmetry_error',
          'graph_self_loop_count', 'graph_gate', 'candidate_union_mean',
          'candidate_pearson_only_mean', 'candidate_siamese_only_mean',
          'candidate_shared_mean', 'pearson_missed_fraction',
          'embedding_std', 'embedding_cosine_std', 'embedding_near_duplicate_fraction',
          'device', 'epochs', 'batch_size', 'learning_rate', 'patience', 'error_msg')


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class FairBaselineExp:
    """Original official model using the same validation-only loop and loader."""
    def __new__(cls, args):
        from FinalProject.Siamese_TVC.exp_siamese import Exp_Siamese
        from models.TVCFormer import Model as OfficialModel

        class BaselineAdapter(Exp_Siamese):
            def _build_model(self):
                return OfficialModel(self.args).float().to(self.device)
        return BaselineAdapter(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--experiments', nargs='+', choices=('E0', 'E1', 'E2'), default=['E0', 'E1', 'E2'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--use_gpu', type=parse_bool)
    parser.add_argument('--root_path', help='Override root data directory for every selected dataset')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--results_csv', default=str(RUNS_CSV))
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.is_absolute():
        results_path = (ROOT / results_path).resolve()
    summary_path = (SUMMARY_CSV if results_path == RUNS_CSV.resolve()
                    else results_path.with_name(results_path.stem + '_summary.csv'))
    rows_by_key = read_completed(results_path)
    failure_count = 0
    run_count = len(args.datasets) * len(args.horizons) * len(args.seeds) * len(args.experiments)
    run_number = 0

    for dataset in args.datasets:
        dataset_cfg = read_yaml(SIAMESE_DIR / 'configs' / f'{dataset}.yaml')
        common_cfg = read_yaml(SIAMESE_DIR / 'configs' / 'common.yaml')
        base = {**common_cfg, **dataset_cfg}
        root_path = Path(args.root_path) if args.root_path else Path(base['root_path'])
        base['root_path'] = str((ROOT / root_path).resolve() if not root_path.is_absolute() else root_path.resolve())

        for seed in args.seeds:
            log_path = SIAMESE_DIR / 'logs' / f'{dataset}_seed{seed}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8', buffering=1) as log:
                tee = Tee(sys.__stdout__, log)
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    print(f'\n===== DATASET={dataset} SEED={seed} START={now()} =====', flush=True)
                    print(f'dataset_config={json_safe(dataset_cfg)}', flush=True)
                    for horizon in args.horizons:
                        for exp_name in args.experiments:
                            run_number += 1
                            cfg = dict(base)
                            experiment_cfg = read_yaml(SIAMESE_DIR / 'configs' / f'{exp_name}.yaml')
                            cfg.update(experiment_cfg)
                            cfg.update({'data': dataset.lower() if dataset == 'Weather' else dataset,
                                        'pred_len': horizon, 'random_seed': seed,
                                        'model_id': f'{dataset}_{exp_name}_SiameseTVC',
                                        'experiment': exp_name,
                                        'measure_graph_time': True})
                            if args.epochs is not None: cfg['train_epochs'] = args.epochs
                            if args.patience is not None: cfg['patience'] = args.patience
                            if args.batch_size is not None: cfg['batch_size'] = args.batch_size
                            if args.use_gpu is not None: cfg['use_gpu'] = args.use_gpu
                            if args.gpu is not None: cfg['gpu'] = args.gpu
                            # The benchmark runs one experiment process on one device.
                            # Exp_Basic requires these attributes even when multi-GPU is off.
                            cfg['use_multi_gpu'] = False
                            cfg['devices'] = str(cfg.get('gpu', 0))
                            cfg['device'] = torch.device(
                                f"cuda:{cfg.get('gpu', 0)}" if cfg.get('use_gpu', True) and torch.cuda.is_available() else 'cpu')
                            cfg['use_gpu'] = cfg['device'].type == 'cuda'
                            cfg['is_training'] = 1
                            cfg['checkpoints'] = str(SIAMESE_DIR / 'checkpoints')
                            cfg['results_dir'] = str(RESULTS / dataset)
                            cfg['logs_dir'] = str(SIAMESE_DIR / 'logs' / dataset)
                            run_id = f'{dataset}_{exp_name}_pl{horizon}_seed{seed}'
                            checkpoint = Path(cfg['checkpoints']) / run_id / 'checkpoint.pth'
                            key = (dataset, exp_name, str(horizon), str(seed))
                            if (not args.no_resume and not args.eval_only and key in rows_by_key
                                    and checkpoint.is_file()):
                                print(f'[{run_number}/{run_count}] RESUME skip completed {run_id}', flush=True)
                                continue
                            if args.eval_only and not checkpoint.is_file():
                                row = failure_row(dataset, exp_name, horizon, seed, run_id, cfg,
                                                  f'checkpoint not found: {checkpoint}')
                                write_run_row(results_path, row); failure_count += 1
                                continue

                            started_wall = now()
                            print(f'\n[{run_number}/{run_count}] START run_id={run_id} at={started_wall}', flush=True)
                            print(f'hyperparameters={json_safe(cfg)}', flush=True)
                            try:
                                seed_everything(seed, cfg['device'])
                                exp_args = SimpleNamespace(**cfg)
                                experiment = FairBaselineExp(exp_args) if exp_name == 'E0' else make_siamese_exp(exp_args)
                                graph_timer = attach_graph_timer(experiment.model)
                                if experiment.device.type == 'cuda':
                                    torch.cuda.reset_peak_memory_stats(experiment.device)
                                train_start = time.perf_counter()
                                if not args.eval_only:
                                    experiment.train(run_id)
                                train_seconds = time.perf_counter() - train_start
                                inference_start = time.perf_counter()
                                mse, mae = experiment.test(run_id, test=1)
                                inference_seconds = time.perf_counter() - inference_start
                                root_model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
                                graph_seconds = sum(getattr(module, 'graph_build_seconds_total', 0.0)
                                                    for module in root_model.modules())
                                if exp_name == 'E0':
                                    graph_seconds = graph_timer['seconds']
                                graph_path = Path(cfg['results_dir']) / f'{run_id}_graph.npz'
                                graph_quality = graph_diagnostics(graph_path, cfg['enc_in'], cfg['time_node_num'])
                                peak_memory = (torch.cuda.max_memory_allocated(experiment.device) / (1024 ** 2)
                                               if experiment.device.type == 'cuda' else float('nan'))
                                row = {'dataset': dataset, 'experiment': exp_name, 'horizon': horizon,
                                       'seed': seed, 'status': 'completed', 'run_id': run_id,
                                       'mse': mse, 'mae': mae,
                                       'best_epoch': getattr(experiment, 'best_epoch', ''),
                                       'best_validation_loss': getattr(experiment, 'best_validation_loss', ''),
                                       'train_seconds': train_seconds, 'inference_seconds': inference_seconds,
                                       'graph_build_seconds': graph_seconds, 'peak_gpu_memory_mb': peak_memory,
                                       **graph_quality,
                                       'device': str(experiment.device), 'epochs': cfg['train_epochs'],
                                       'batch_size': cfg['batch_size'], 'learning_rate': cfg['learning_rate'],
                                       'patience': cfg['patience'], 'error_msg': ''}
                                print(f'RUN_COMPLETED run_id={run_id} mse={mse:.8f} mae={mae:.8f} '
                                      f'best_epoch={row["best_epoch"]} train_seconds={train_seconds:.2f} '
                                      f'inference_seconds={inference_seconds:.2f} graph_seconds={graph_seconds:.2f} '
                                      f'peak_gpu_memory_mb={peak_memory} end={now()}', flush=True)
                            except KeyboardInterrupt:
                                row = failure_row(dataset, exp_name, horizon, seed, run_id, cfg, 'KeyboardInterrupt')
                                row['status'] = 'interrupted'
                                write_run_row(results_path, row)
                                print(f'RUN_INTERRUPTED run_id={run_id} end={now()}', flush=True)
                                raise
                            except Exception as error:
                                failure_count += 1
                                row = failure_row(dataset, exp_name, horizon, seed, run_id, cfg,
                                                  f'{type(error).__name__}: {error}')
                                print(f'RUN_FAILED run_id={run_id} reason={row["error_msg"]} end={now()}', flush=True)
                                traceback.print_exc()
                            write_run_row(results_path, row)
                            rows_by_key[key] = row
                    print(f'===== DATASET={dataset} SEED={seed} END={now()} =====', flush=True)

    all_rows = read_rows(results_path)
    write_summary(all_rows, summary_path)
    print(f'Raw results: {results_path}')
    print(f'Summary results: {summary_path}')
    if failure_count:
        raise SystemExit(f'{failure_count} run(s) failed; inspect per-dataset seed logs and status CSV')


def parse_bool(value):
    value = str(value).lower()
    if value in ('1', 'true', 'yes', 'on'): return True
    if value in ('0', 'false', 'no', 'off'): return False
    raise argparse.ArgumentTypeError(f'invalid boolean: {value}')


def read_yaml(path):
    with path.open(encoding='utf-8-sig') as stream:
        return yaml.safe_load(stream) or {}


def json_safe(value):
    import json
    return json.dumps(value, sort_keys=True, default=str)


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def seed_everything(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    try:
        import dgl
        dgl.random.seed(seed)
    except Exception:
        pass


def make_siamese_exp(args):
    from FinalProject.Siamese_TVC.exp_siamese import Exp_Siamese
    return Exp_Siamese(args)


def attach_graph_timer(model):
    root = model.module if hasattr(model, 'module') else model
    modules = [module for module in root.modules()
               if module.__class__.__name__ in ('graph_generator', 'GraphGeneratorSiamese')]
    timer = {'seconds': 0.0, 'started': None}
    handles = []
    def before(module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].is_cuda:
            torch.cuda.synchronize(inputs[0].device)
        timer['started'] = time.perf_counter()
    def after(module, inputs, output):
        if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].is_cuda:
            torch.cuda.synchronize(inputs[0].device)
        timer['seconds'] += time.perf_counter() - timer['started']
    for module in modules:
        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
    return timer


def failure_row(dataset, experiment, horizon, seed, run_id, cfg, reason):
    return {'dataset': dataset, 'experiment': experiment, 'horizon': horizon,
            'seed': seed, 'status': 'failed', 'run_id': run_id,
            'mse': '', 'mae': '', 'best_epoch': '', 'best_validation_loss': '',
            'train_seconds': '', 'inference_seconds': '', 'graph_build_seconds': '',
            'peak_gpu_memory_mb': '', 'device': str(cfg.get('device', 'unknown')),
            **{name: '' for name in FIELDS if name.startswith('graph_') or name.startswith('candidate_')
               or name.startswith('pearson_') or name.startswith('embedding_')},
            'epochs': cfg.get('train_epochs'), 'batch_size': cfg.get('batch_size'),
            'learning_rate': cfg.get('learning_rate'), 'patience': cfg.get('patience'),
            'error_msg': reason}


def graph_diagnostics(path, channels, time_nodes):
    """Read the test graph snapshot and return compact run-level diagnostics."""
    names = ('graph_density', 'graph_degree_mean', 'graph_degree_std',
             'graph_weight_mean', 'graph_weight_std', 'graph_symmetry_error',
             'graph_self_loop_count', 'graph_gate', 'candidate_union_mean',
             'candidate_pearson_only_mean', 'candidate_siamese_only_mean',
             'candidate_shared_mean', 'pearson_missed_fraction', 'embedding_std',
             'embedding_cosine_std', 'embedding_near_duplicate_fraction')
    empty = {name: '' for name in names}
    if not path.is_file():
        return empty
    with np.load(path) as data:
        indices = data['graph_indices']
        values = data['graph_values'].astype(np.float64, copy=False)
        nodes = int(data['graph_shape'][-1])
        offdiag = indices[0] != indices[1]
        batch_nodes = max(int(channels) * int(time_nodes), 1)
        sample_count = max(nodes // batch_nodes, 1)
        degrees = np.bincount(indices[0, offdiag], minlength=nodes).astype(np.float64)
        weights = values[values != 0]
        edge_keys = indices[0].astype(np.int64) * nodes + indices[1].astype(np.int64)
        reverse_keys = indices[1].astype(np.int64) * nodes + indices[0].astype(np.int64)
        order = np.argsort(edge_keys)
        sorted_keys = edge_keys[order]
        positions = np.searchsorted(sorted_keys, reverse_keys)
        present = positions < sorted_keys.size
        matched = np.zeros_like(present)
        matched[present] = sorted_keys[positions[present]] == reverse_keys[present]
        reverse_values = np.zeros_like(values)
        reverse_values[matched] = values[order[positions[matched]]]
        result = {
            'graph_density': float(offdiag.sum() / max(sample_count * batch_nodes * (batch_nodes - 1), 1)),
            'graph_degree_mean': float(degrees.mean()),
            'graph_degree_std': float(degrees.std()),
            'graph_weight_mean': float(weights.mean()) if weights.size else 0.0,
            'graph_weight_std': float(weights.std()) if weights.size else 0.0,
            'graph_symmetry_error': float(np.max(np.abs(values - reverse_values))) if values.size else 0.0,
            'graph_self_loop_count': int((indices[0] == indices[1]).sum()),
            'graph_gate': float(np.asarray(data['gate']).mean()) if 'gate' in data else '',
        }
        for source, output in (('candidate_union_candidates', 'candidate_union_mean'),
                               ('candidate_pearson_only', 'candidate_pearson_only_mean'),
                               ('candidate_siamese_only', 'candidate_siamese_only_mean'),
                               ('candidate_shared', 'candidate_shared_mean'),
                               ('candidate_pearson_missed_fraction', 'pearson_missed_fraction')):
            result[output] = float(np.asarray(data[source]).mean()) if source in data else ''
        if 'embeddings' in data:
            embeddings = data['embeddings'].astype(np.float32, copy=False)
            result['embedding_std'] = float(embeddings.std())
            sample_cosines = []
            for sample in embeddings:
                flat = sample.reshape(-1, sample.shape[-1])
                norm = flat / np.maximum(np.linalg.norm(flat, axis=-1, keepdims=True), 1e-12)
                cosine = norm @ norm.T
                upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
                sample_cosines.append(upper)
            cosines = np.concatenate(sample_cosines) if sample_cosines else np.empty(0)
            result['embedding_cosine_std'] = float(cosines.std()) if cosines.size else 0.0
            result['embedding_near_duplicate_fraction'] = float((cosines >= 0.98).mean()) if cosines.size else 0.0
        else:
            result.update({'embedding_std': '', 'embedding_cosine_std': '',
                           'embedding_near_duplicate_fraction': ''})
        return result


def read_rows(path):
    if not path.is_file(): return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def read_completed(path):
    return {(row['dataset'], row['experiment'], str(row['horizon']), str(row['seed'])): row
            for row in read_rows(path) if row.get('status') == 'completed'
            and _finite(row.get('mse')) and _finite(row.get('mae'))}


def write_run_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(path)
    key = (str(row['dataset']), str(row['experiment']), str(row['horizon']), str(row['seed']))
    rows = [old for old in rows if (str(old.get('dataset')), str(old.get('experiment')),
                                    str(old.get('horizon')), str(old.get('seed'))) != key]
    rows.append(row)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, path):
    from collections import defaultdict
    import math
    completed = [row for row in rows if row.get('status') == 'completed']
    groups = defaultdict(list)
    for row in completed:
        groups[(row['dataset'], row['horizon'], row['experiment'])].append(row)
    summary = []
    for (dataset, horizon, experiment), records in sorted(groups.items()):
        entry = {'dataset': dataset, 'horizon': horizon, 'experiment': experiment,
                 'completed_seeds': len(records)}
        for metric in ('mse', 'mae', 'train_seconds', 'inference_seconds',
                       'graph_build_seconds', 'peak_gpu_memory_mb', 'graph_density',
                       'graph_degree_mean', 'graph_weight_mean', 'graph_gate',
                       'candidate_siamese_only_mean', 'pearson_missed_fraction',
                       'embedding_std', 'embedding_cosine_std'):
            vals = [float(row[metric]) for row in records
                    if row.get(metric) not in ('', None) and _finite(row.get(metric))]
            entry[metric + '_mean'] = float(np.mean(vals)) if vals else ''
            entry[metric + '_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else ''
        baseline = groups.get((dataset, horizon, 'E0'), [])
        matched = {int(row['seed']): row for row in baseline}
        paired = [(row, matched[int(row['seed'])]) for row in records
                  if experiment != 'E0' and int(row['seed']) in matched
                  and _finite(row.get('mse')) and _finite(matched[int(row['seed'])].get('mse'))]
        if paired:
            entry['paired_mse_improvement_pct_mean'] = float(np.mean([
                (float(base['mse']) - float(proposed['mse'])) / max(abs(float(base['mse'])), 1e-12) * 100
                for proposed, base in paired]))
            entry['paired_mae_improvement_pct_mean'] = float(np.mean([
                (float(base['mae']) - float(proposed['mae'])) / max(abs(float(base['mae'])), 1e-12) * 100
                for proposed, base in paired]))
        summary.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in summary for key in row})
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)


def _finite(value):
    try: return np.isfinite(float(value))
    except (TypeError, ValueError): return False


if __name__ == '__main__':
    main()
