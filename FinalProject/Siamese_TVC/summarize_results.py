"""Rebuild benchmark mean/std and paired improvement CSV from completed runs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from FinalProject.Siamese_TVC.benchmark import read_rows, write_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(PROJECT / 'results' / 'benchmark_full_runs.csv'))
    parser.add_argument('--output', default=str(PROJECT / 'results' / 'benchmark_full_summary.csv'))
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_absolute(): source = ROOT / source
    if not destination.is_absolute(): destination = ROOT / destination
    rows = read_rows(source)
    write_summary(rows, destination)
    print(f'Read {len(rows)} run rows from {source}; summary written to {destination}')


if __name__ == '__main__':
    main()
