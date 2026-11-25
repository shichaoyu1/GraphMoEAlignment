"""Separate frozen-token screening before expanding the MRI Pareto matrix."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.utsw_authority import save_json, seed_all
from glioma.eval.idea3_pareto_screen import choose_masks, predict_role, summarize, patient_errors, risks
from glioma.models.idea3_pareto import TokenReader


KIND = 'idea3_frozen_screen_v1'
ROLES = ('train', 'validation', 'test')


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[1]
    files = ['eval/idea3_pareto_screen.py', 'cli/idea3_pareto_screen.py',
             'models/idea3_pareto.py', 'data/idea3_pareto.py']
    return hashlib.sha256(b''.join((root / name).read_bytes() for name in files)).hexdigest()


def validate_archive(archive):
    seen = set()
    for role in ROLES:
        ids, tokens, targets = (archive[key][role] for key in ('subject_ids', 'tokens', 'targets'))
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Token patient IDs must be unique and split-disjoint')
        seen.update(ids)
        if tokens.ndim != 3 or tokens.shape[1:] != (64, 64) or len(tokens) != len(ids) or targets.shape != (len(ids), 4):
            raise ValueError('Token or target shape does not match this 64-token reader')
        if not torch.isfinite(tokens).all() or not torch.isfinite(targets).all():
            raise ValueError('Nonfinite tokens or targets')
        if not ((targets[:, :3] == -1) | (targets[:, :3] == 0) | (targets[:, :3] == 1)).all():
            raise ValueError('Molecular targets must be -1, 0, or 1')
        if not ((targets[:, 3] >= 0) & (targets[:, 3] <= 1)).all():
            raise ValueError('Phenotype targets must be complete ratios in [0,1]')
    if archive['signature']['k'] != 8:
        raise ValueError('Screening requires the original eight-token reader protocol')
    if not 0 <= archive['signature']['fold'] < 5:
        raise ValueError('Invalid token fold')


def evaluate(args):
    if min(args.proposals, args.random_repeats, args.bootstrap) < 1 or args.local_steps < 0:
        raise ValueError('Require positive proposal/repeat/bootstrap budgets and nonnegative local steps')
    tokens_path, output = args.tokens.resolve(), args.output.resolve()
    if output in tokens_path.parents:
        raise ValueError('Choose a screen output outside the input token archive directory')
    archive = torch.load(tokens_path, map_location='cpu', weights_only=True)
    validate_archive(archive)
    fold, training_seed = int(archive['signature']['fold']), int(archive['signature']['seed'])
    if getattr(args, 'fold', None) is not None and fold != args.fold:
        raise ValueError('Token fold does not match requested fold')
    if getattr(args, 'training_seed', None) is not None and training_seed != args.training_seed:
        raise ValueError('Token seed does not match requested training seed')
    split_hash = archive['signature'].get('splits')
    manifest_hash = archive['signature'].get('manifest')
    if getattr(args, 'authority_source', None) is not None:
        split_hash = fingerprint(args.authority_source / 'splits' / f'fold{fold}.json')
        if split_hash != archive['signature'].get('splits'):
            raise ValueError('Token authority split hash does not match current authority archive')
    if getattr(args, 'image_manifest', None) is not None:
        manifest_hash = fingerprint(args.image_manifest)
        if manifest_hash != archive['signature'].get('manifest'):
            raise ValueError('Token image manifest hash does not match current image archive')
    config = {'kind': KIND, 'tokens_hash': fingerprint(tokens_path), 'source_hash': source_hash(),
              'fold': fold, 'training_seed': training_seed, 'seed': args.seed,
              'device': str(args.device), 'proposals': args.proposals, 'local_steps': args.local_steps,
              'random_repeats': args.random_repeats, 'bootstrap': args.bootstrap, 'k': 8,
              'authority_split_hash': split_hash, 'image_manifest_hash': manifest_hash}
    output.mkdir(parents=True, exist_ok=True)
    existing = output / 'config.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')) != config:
        raise ValueError(f'Existing screening configuration differs at {existing}; choose a new output')
    save_json(existing, config)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        report = json.loads(summary_path.read_text(encoding='utf-8'))
        for name, expected in report['artifact_hashes'].items():
            if not (output / name).is_file() or fingerprint(output / name) != expected:
                raise ValueError(f'Completed screening artifact changed or missing: {output / name}')
        return report
    seed_all(args.seed)
    device = torch.device(args.device)
    reader = TokenReader().to(device).eval().requires_grad_(False)
    reader.load_state_dict(archive['reader'])
    tokens = {role: archive['tokens'][role].to(device) for role in ROLES}
    targets = {role: archive['targets'][role].to(device) for role in ROLES}
    started = perf_counter()
    masks = choose_masks(tokens['train'], targets['train'], reader, 8, args.proposals, args.local_steps, args.seed)
    mask_record = {'selection_role': 'train', 'objective': 'linear population risk, train scales only',
                   'masks': {str(weight): mask for weight, mask in masks.items()}}
    save_json(output / 'selected_masks.json', mask_record)
    frames = []
    for index, role in enumerate(ROLES):
        frame = predict_role(tokens[role], targets[role], archive['subject_ids'][role], reader, masks,
                             targets['train'], role, fold, training_seed, args.random_repeats, args.seed + index + 1)
        atomic_csv(frame, output / f'{role}_predictions.csv')
        frames.append(frame)
    curves, endpoints = summarize(pd.concat(frames, ignore_index=True), args.bootstrap, args.seed)
    atomic_csv(pd.DataFrame([{k: v for k, v in row.items() if k != 'molecular_counts'} for row in curves]), output / 'risk_curves.csv')
    left, right = set(masks[0.]), set(masks[1.])
    artifacts = ['selected_masks.json', 'risk_curves.csv', *(f'{role}_predictions.csv' for role in ROLES)]
    report = {'kind': KIND, 'status': 'diagnostic_only', 'config': config, 'curves': curves,
              'endpoints': endpoints, 'role_ids': archive['subject_ids'],
              'endpoint_mask_jaccard': len(left & right) / len(left | right),
              'reference_budget': {'fixed8_train_selected': 8, 'random8': 8, 'all_tokens_reference': 64,
                                   'train_frequency_median': 0, 'constant_half_train_median': 0},
              'all_tokens_reference_scope': 'Different evidence budget, reader reference only',
              'elapsed_seconds': perf_counter() - started,
              'artifact_hashes': {name: fingerprint(output / name) for name in artifacts}}
    save_json(summary_path, report)
    return report


def aggregate(args):
    if args.bootstrap < 1:
        raise ValueError('Bootstrap budget must be positive')
    configs = []
    for path in sorted(args.source.resolve().rglob('config.json')):
        if json.loads(path.read_text(encoding='utf-8')).get('kind') == KIND:
            configs.append(path)
    if not configs:
        raise ValueError('No frozen-reader screening runs found')
    frames, reports, seen = [], [], set()
    cohort = {role: set() for role in ('validation', 'test')}
    expected = set(args.expected_folds)
    shared = None
    for path in configs:
        report_path = path.parent / 'summary.json'
        if not report_path.is_file():
            raise ValueError(f'Incomplete screening run: {path.parent}')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        config = json.loads(path.read_text(encoding='utf-8'))
        if report['config'] != config:
            raise ValueError('Screening config and summary disagree')
        comparison = {k: v for k, v in config.items() if k not in ('fold', 'tokens_hash', 'authority_split_hash')}
        if shared is not None and shared != comparison:
            raise ValueError('Crossfold screening configurations differ')
        shared = comparison
        fold = config['fold']
        if fold in seen:
            raise ValueError('Duplicate fold screening runs')
        seen.add(fold)
        for role in cohort:
            ids = set(report['role_ids'][role])
            if cohort[role].intersection(ids):
                raise ValueError(f'OOF {role} patients repeat across folds')
            cohort[role].update(ids)
        for name, checksum in report['artifact_hashes'].items():
            if not (path.parent / name).is_file() or fingerprint(path.parent / name) != checksum:
                raise ValueError(f'Corrupt or incomplete screening artifact: {path.parent / name}')
        for role in ('validation', 'test'):
            frames.append(pd.read_csv(path.parent / f'{role}_predictions.csv'))
        reports.append(report)
    if seen != expected:
        raise ValueError(f'Expected folds {sorted(expected)}, found {sorted(seen)}')
    frame = pd.concat(frames, ignore_index=True)
    curves, endpoints = summarize(frame, args.bootstrap, args.seed)
    patients = patient_errors(frame)
    pooled_curves = [{'role': role, 'method': method, 'weight': float(weight), **risks(rows)}
                     for (role, method, weight), rows in patients.groupby(['role', 'method', 'weight'])]
    output = args.output.resolve()
    if output in [path.parent for path in configs]:
        raise ValueError('Aggregate output must differ from each fold output')
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame([{k: v for k, v in row.items() if k != 'molecular_counts'} for row in curves]), output / 'risk_curves.csv')
    result = {'kind': KIND + '_aggregate', 'status': 'diagnostic_only', 'folds': sorted(seen),
              'training_seed': shared['training_seed'], 'cohort_sizes': {role: len(ids) for role, ids in cohort.items()},
              'endpoints': endpoints, 'per_fold': [{'fold': r['config']['fold'], 'endpoints': r['endpoints'],
                                                  'endpoint_mask_jaccard': r['endpoint_mask_jaccard']} for r in reports],
              'validation_is_primary': True,
              'validation_scope': 'Exploratory: frozen readers were checkpoint-selected on their validation sets',
              'decision': 'No automatic approval or launch of full experiments; inspect directional replication and simple references',
              'aggregation': 'Pooled OOF per-field/class balanced molecular Brier and patient-weighted MAE; paired fold-stratified patient bootstrap; differs from old equal-fold analysis',
              'pooled_curves': pooled_curves,
              'test_scope': 'Exploratory for previously inspected fold0; do not select masks or checkpoints from these results',
              'input_hashes': {str(path.parent): fingerprint(path.parent / 'summary.json') for path in configs}}
    save_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    current = commands.add_parser('evaluate')
    current.add_argument('--tokens', type=Path, required=True)
    current.add_argument('--output', type=Path, required=True)
    current.add_argument('--device', default='cuda')
    current.add_argument('--authority-source', type=Path)
    current.add_argument('--image-manifest', type=Path)
    current.add_argument('--fold', type=int)
    current.add_argument('--training-seed', type=int)
    current.add_argument('--proposals', type=int, default=256)
    current.add_argument('--local-steps', type=int, default=128)
    current.add_argument('--random-repeats', type=int, default=256)
    current.add_argument('--seed', type=int, default=2026)
    current.add_argument('--bootstrap', type=int, default=10000)
    current = commands.add_parser('aggregate')
    current.add_argument('--source', type=Path, required=True)
    current.add_argument('--output', type=Path, required=True)
    current.add_argument('--expected-folds', type=int, nargs='+', default=list(range(5)))
    current.add_argument('--bootstrap', type=int, default=10000)
    current.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    report = evaluate(args) if args.command == 'evaluate' else aggregate(args)
    print(f'Screen complete: status={report["status"]}; summary={args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
