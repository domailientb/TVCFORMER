"""Train-split-only descriptive diagnostics for the five benchmark datasets."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather')


def load_rows(dataset):
    import yaml
    with (PROJECT / 'configs' / f'{dataset}.yaml').open(encoding='utf-8-sig') as stream:
        config = yaml.safe_load(stream)
    path = ROOT / config['root_path'] / config['data_path']
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    dates = [row.get('date', '') for row in rows]
    # Match official Dataset_ETTh/ETTm/Custom training boundaries.
    if dataset in ('ETTh1', 'ETTh2'):
        train_end = 12 * 30 * 24
    elif dataset in ('ETTm1', 'ETTm2'):
        train_end = 12 * 30 * 24 * 4
    else:
        train_end = int(len(rows) * 0.7)
    columns = [key for key in rows[0] if key not in ('date', config.get('target', 'OT'))]
    columns.append(config.get('target', 'OT'))
    values = np.asarray([[float(row[col]) for col in columns] for row in rows[:train_end]], dtype=np.float64)
    return config, path, dates, columns, values


def describe(dataset):
    config, path, dates, columns, values = load_rows(dataset)
    row_count, channel_count = values.shape
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95], axis=0)
    centered = values - means
    denom = np.sqrt((centered ** 2).sum(axis=0))
    corr = (centered.T @ centered) / np.maximum(denom[:, None] * denom[None, :], 1e-12)
    block = max(int(336 // 7), 1)
    blocks = values[:(row_count // block) * block].reshape(-1, block, channel_count)
    block_mean = blocks.mean(axis=1, keepdims=True)
    block_std = blocks.std(axis=1, keepdims=True).clip(min=1e-12)
    normalized_blocks = (blocks - block_mean) / block_std
    shape_cosine = np.einsum('twi,twj->tij', normalized_blocks, normalized_blocks) / block
    shape_cosine = shape_cosine.mean(axis=0)
    upper = corr[np.triu_indices(channel_count, k=1)]
    result = {
        'dataset': dataset, 'data_key': config['data'], 'file': str(path.relative_to(ROOT)),
        'rows_total': len(dates), 'rows_train_for_stats': row_count,
        'channels': channel_count, 'first_timestamp': dates[0], 'last_timestamp': dates[-1],
        'train_mean_channel_std': float(stds.mean()),
        'train_std_min': float(stds.min()), 'train_std_max': float(stds.max()),
        'train_scale_ratio_max_over_min': float(stds.max() / max(stds.min(), 1e-12)),
        'pearson_abs_mean': float(np.abs(upper).mean()) if upper.size else 0.0,
        'pearson_abs_median': float(np.median(np.abs(upper))) if upper.size else 0.0,
        'pearson_abs_below_0_2_fraction': float((np.abs(upper) < 0.2).mean()) if upper.size else 0.0,
        'pearson_min': float(upper.min()) if upper.size else 1.0,
        'pearson_max': float(upper.max()) if upper.size else 1.0,
        'low_abs_pearson_high_local_shape_fraction': float(np.mean([
            abs(corr[i, j]) < 0.2 and shape_cosine[i, j] > 0.8
            for i in range(channel_count) for j in range(i + 1, channel_count)])) if channel_count > 1 else 0.0,
    }
    channel_rows = []
    for index, name in enumerate(columns):
        channel_rows.append({
            'dataset': dataset, 'channel': name, 'train_mean': float(means[index]),
            'train_std': float(stds[index]), 'train_q05': float(q05[index]),
            'train_median': float(median[index]), 'train_q95': float(q95[index]),
            'train_min': float(values[:, index].min()), 'train_max': float(values[:, index].max()),
            'train_difference_std': float(np.diff(values[:, index]).std()),
        })
    pairs = []
    for i in range(channel_count):
        for j in range(i + 1, channel_count):
            pairs.append({'dataset': dataset, 'channel_i': columns[i], 'channel_j': columns[j],
                          'pearson': float(corr[i, j]),
                          'mean_aligned_local_shape_cosine': float(shape_cosine[i, j]),
                          'low_abs_pearson_high_shape': bool(abs(corr[i, j]) < 0.2 and shape_cosine[i, j] > 0.8)})
    return result, channel_rows, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--output_dir', default=str(PROJECT / 'analysis'))
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute(): output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    overview, channels, pairs = [], [], []
    for dataset in args.datasets:
        summary, channel_stats, pair_corr = describe(dataset)
        overview.append(summary); channels.extend(channel_stats); pairs.extend(pair_corr)
    _write(output / 'dataset_overview.csv', overview)
    _write(output / 'train_channel_statistics.csv', channels)
    _write(output / 'train_pairwise_pearson.csv', pairs)
    print(f'Dataset diagnostics written to {output.resolve()}')
    print('All distribution/correlation summaries use train rows only; do not tune from test data.')


def _write(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
