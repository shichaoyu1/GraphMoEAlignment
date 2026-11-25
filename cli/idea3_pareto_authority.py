"""Evaluate continuous token policies in the original authority patient folds."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.idea3_authority import metadata_splits, withdraw_report
from glioma.cli.idea3_pareto import configuration, fingerprint, save_checkpoint
from glioma.cli.utsw_authority import save_json, seed_all
from glioma.data.idea3 import fixed_view, image_sources
from glioma.data.utsw_authority import GradeEvidence, compile_grade, FIELDS
from glioma.eval.utsw_authority import prediction_frame
from glioma.models.idea3_pareto import EvidencePolicy, TokenGradeReadout, TokenReader
from glioma.models.utsw_authority import grade_loss


def execute(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_file = args.policy / 'best.pt'
    policy_file = torch.load(model_file, map_location=device, weights_only=True)
    cache = torch.load(args.tokens, map_location='cpu', weights_only=True)
    if policy_file['config']['fold'] != args.fold or cache['signature']['fold'] != args.fold:
        raise ValueError('Policy and token encoder must use the requested authority fold')
    if policy_file['config']['seed'] != args.seed or cache['signature']['seed'] != args.seed:
        raise ValueError('Policy and token encoder must use the requested seed')
    manifest = json.loads((args.authority_source / 'manifest.json').read_text(encoding='utf-8'))
    subsets = metadata_splits(args.authority_source, manifest, args.fold)
    if fingerprint(args.authority_source / 'splits' / f'fold{args.fold}.json') != cache['signature']['splits']:
        raise ValueError('Token encoder was fitted on different patient splits')
    configuration(output, {'policy_hash': fingerprint(model_file), 'tokens_hash': fingerprint(args.tokens),
                           'manifest_hash': fingerprint(args.authority_source / 'manifest.json'),
                           'epochs': args.epochs, 'seed': args.seed, 'fold': args.fold,
                           'repeats': args.repeats, 'data_root': str(args.data_root.resolve())})
    if (output / 'summary.json').exists():
        return
    reader = TokenReader().to(device).eval()
    reader.load_state_dict(cache['reader'])
    reader.requires_grad_(False)
    all_rows = sum(subsets.values(), [])
    encoded, rows, index = [], [], {}
    image_cache = args.image_cache or output / 'full_view_cache'
    image_cache.mkdir(parents=True, exist_ok=True)
    source_marker = image_cache / 'source.json'
    source_config = {'data_root': str(args.data_root.resolve()), 'size': 64}
    if source_marker.exists() and json.loads(source_marker.read_text(encoding='utf-8')) != source_config:
        raise ValueError('Whole-view cache belongs to a different MRI source')
    if not source_marker.exists():
        temporary_marker = image_cache / f'source.{os.getpid()}.json'
        save_json(temporary_marker, source_config)
        temporary_marker.replace(source_marker)
    with torch.no_grad():
        for row in all_rows:
            sid = row['subject_id']
            image_path = image_cache / (sid + '.npz')
            if image_path.exists():
                with np.load(image_path) as saved:
                    image = torch.tensor(saved['images'], dtype=torch.float32)
            else:
                _, volumes, _ = image_sources(args.data_root / sid)
                values = fixed_view(volumes)
                temporary = image_path.with_suffix(f'.{os.getpid()}.tmp.npz')
                np.savez(temporary, images=values)
                temporary.replace(image_path)
                image = torch.tensor(values, dtype=torch.float32)
            index[sid] = len(rows)
            rows.append(row)
            encoded.append(reader.encode(image[None].to(device))[0])
    tokens = torch.stack(encoded)
    context = tokens.mean(1)
    molecular = torch.tensor([row['features'] for row in rows], device=device)
    active = torch.tensor([row['active'] for row in rows], device=device)
    labels = torch.tensor([row['label'] for row in rows], device=device)
    available = torch.ones(len(rows), 5, dtype=torch.bool, device=device)
    available[:, 4] = torch.tensor([any(row['record'][key] in values for key, values in FIELDS.items())
                                     for row in rows], device=device)
    def evidence(ids):
        return GradeEvidence(tokens.new_empty((len(ids), 0)), molecular[ids],
                             available[ids], active[ids])
    assignments = {role: torch.tensor([index[row['subject_id']] for row in members], device=device)
                   for role, members in subsets.items()}
    state = policy_file['model']
    experts = len({key.split('.')[1] for key in state if key.startswith('experts.')})
    direct = policy_file['config']['method'].startswith('direct')
    policy = EvidencePolicy(64, 64, width=state['input.weight'].shape[0], experts=experts, direct=direct).to(device)
    policy.load_state_dict(state)
    policy.selection_k = policy_file['config']['k']
    policy.greedy = policy_file['config']['method'] == 'direct_sft'
    policy.eval().requires_grad_(False)
    heads = {}
    for variant in ('report', 'joint', 'readonly'):
        seed_all(args.seed)
        head = TokenGradeReadout(variant).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
        best, selected = float('inf'), None
        for _ in range(args.epochs):
            for ids in assignments['train'][torch.randperm(len(assignments['train']), device=device)].split(8):
                masks = torch.rand(len(ids), 64, device=device).topk(policy.selection_k, -1).indices
                loss = grade_loss(head(tokens[ids], masks, evidence(ids)), labels[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(2026)
                ids = assignments['validation']
                masks = torch.rand(len(ids), 64, device=device).topk(policy.selection_k, -1).indices
                scores = head(tokens[ids], masks, evidence(ids))['p_star']
                score = float((scores - torch.nn.functional.one_hot(labels[ids], 3)).square().mean())
            if score < best:
                best, selected = score, deepcopy(head.state_dict())
        head.load_state_dict(selected)
        heads[variant] = head.eval().requires_grad_(False)
        save_checkpoint(output / (variant + '.pt'), model=selected)
    frames = []
    test = assignments['test']
    eligible = test[compile_grade(evidence(test))['active']]
    if len(eligible) < 2:
        raise ValueError('Need at least two active same-fold donors')
    with torch.no_grad():
        for weight in np.linspace(0, 1, 21):
            modes = ['own', 'withdrawn'] + (['swapped'] if any(abs(weight - x) < 1e-8 for x in (0, .25, .5, .75, 1)) else [])
            for condition in modes:
                pairs = [(int(i), int(j)) for i in eligible for j in eligible if i != j] if condition == 'swapped' else [(int(i), int(i)) for i in test]
                for sample in range(args.repeats):
                    torch.manual_seed(2027 + sample)
                    for start in range(0, len(pairs), 8):
                        pair = pairs[start:start + 8]
                        recipient = torch.tensor([i for i, _ in pair], device=device)
                        donor = torch.tensor([j for _, j in pair], device=device)
                        prefs = context.new_full((len(pair),), float(weight))
                        logits = policy.sample(context[donor], prefs, policy_file['config']['steps'], policy_file['config']['noise'])[0]
                        mask = logits.topk(policy.selection_k, -1).indices
                        report = evidence(recipient)
                        if condition == 'withdrawn':
                            report = withdraw_report(report)
                        for variant, head in heads.items():
                            observed = head(tokens[donor], mask, report)
                            if (observed['p_star'][~observed['allowed']].abs() > 1e-6).any():
                                raise AssertionError('Rule projection failed')
                            frame = prediction_frame([rows[i]['subject_id'] for i, _ in pair], labels[recipient].cpu().numpy(),
                                                     observed['p0'].cpu().numpy(), observed['p_star'].cpu().numpy(),
                                                     observed['active'].cpu().numpy(), observed['status'].cpu().numpy(), args.fold, variant)
                            frame['weight'], frame['sample'], frame['condition'] = weight, sample, condition
                            frame['image_subject_id'] = [rows[j]['subject_id'] for _, j in pair]
                            frames.append(frame)
            print(f'Authority fold={args.fold} preference={weight:.2f} complete', flush=True)
    result = pd.concat(frames, ignore_index=True)
    if result.loc[result.condition == 'withdrawn', 'active'].any():
        raise AssertionError('Report withdrawal did not revoke the rule')
    protected = result[(result.model == 'readonly') & (result.condition != 'withdrawn') & result.active]
    grouped = protected.groupby('subject_id').raw_p4
    protected_change = float((grouped.max() - grouped.min()).max())
    if protected_change > 1e-6:
        raise AssertionError('MRI or preference changed the protected Grade 4 probability')
    report_only = result[(result.model == 'report') & (result.condition != 'withdrawn')]
    columns = ['raw_p2', 'raw_p3', 'raw_p4', 'p2', 'p3', 'p4']
    grouped_report = report_only.groupby('subject_id')[columns]
    if ((grouped_report.max() - grouped_report.min()) > 1e-6).any(axis=None):
        raise AssertionError('Report-only probabilities changed with MRI or preference')
    atomic_csv(result, output / 'predictions.csv')
    save_json(output / 'summary.json', {'fold': args.fold, 'n_patients': len(test), 'active_n': len(eligible),
                                       'n_predictions': len(result), 'protocol': 'whole-FOV token grading'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('tokens', 'policy', 'authority-source', 'data-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--image-cache', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.repeats < 1:
        parser.error('epochs and repeats must be positive')
    execute(args)


if __name__ == '__main__':
    main()
