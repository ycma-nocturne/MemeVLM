"""Verify the chosen adapter and replay Task B scoring without models."""

import argparse
import json
from pathlib import Path

from meme_study.data import content_hash

import pair_constrained_study
import pair_study


PROJECT = Path(__file__).resolve().parent.parent


def verify_pairs(runner, project, folder, split, receipt):
    settings = runner.read_json(folder / 'run.json')
    jobs, gold, _ = runner.compile_matrix(project, split)
    index = runner.prediction_index(folder / 'records', jobs)
    if set(index.values()) != set((folder / 'records').glob('*.json')):
        raise ValueError(f'Incomplete record inventory: {folder}')
    if receipt is not None:
        frozen = runner.read_json(receipt)['manifest']
        if settings != {**frozen, 'freeze_sha256': runner.digest(receipt)}:
            raise ValueError(f'Run differs from held-out freeze: {folder}')
    summary, predictions = runner.summarize(jobs, gold, folder / 'records',
                                            content_hash(settings), split)
    for value, name in ((summary, 'summary.json'), (predictions, 'predictions.json')):
        if json.loads(json.dumps(value)) != runner.read_json(folder / name):
            raise ValueError(f'Replayed score differs: {folder / name}')
    return {'scored_calls': len(predictions), 'valid_predictions': summary['valid_predictions']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=PROJECT / 'results')
    args = parser.parse_args()
    directory = PROJECT / 'local-study'
    record = pair_study.read_json(directory / 'training/selected-adapter.json')
    selected = record['selection']
    if selected != min(record['epochs'], key=lambda epoch: epoch['validation_loss']):
        raise ValueError('Selected epoch differs from the lowest validation loss')
    if pair_study.digest(PROJECT / selected['adapter']) != selected['sha256']:
        raise ValueError('Selected adapter SHA-256 differs from training record')
    report = {'selected_epoch': selected['epoch'], 'validation_loss': selected['validation_loss']}
    for prefix, runner, freeze in (
        ('pair', pair_study, 'pair-unconstrained.freeze.json'),
        ('pair-constrained', pair_constrained_study, 'pair-constrained.freeze.json'),
    ):
        for split in ('dev', 'test'):
            name = f'{prefix}-{split}'
            report[name] = verify_pairs(runner, PROJECT, args.results / name, split,
                                        directory / freeze if split == 'test' else None)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
