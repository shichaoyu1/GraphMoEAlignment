"""Separate controlled, cached MRI training and evaluation entry points."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.utsw_authority import save_json, seed_all
from glioma.data.idea3_pareto import ControlledEnvironment, TokenEnvironment, balanced_weights, cohort_rows
from glioma.eval.idea3_pareto import exact_oracle, hypervolume
from glioma.models.idea3_pareto import EvidencePolicy, TokenReader
from glioma.training.idea3_pareto import METHODS, evaluate, optimize


@dataclass
class Configuration:
    method: str = 'moe_pg'
    warmup: int = 2000
    updates: int = 2000
    batch: int = 8
    group: int = 8
    steps: int = 16
    noise: float = .05
    k: int = 8
    lr: float = 1e-4
    proposals: int = 256
    local_steps: int = 128
    validate_every: int = 100
    eval_repeats: int = 8
    test_repeats: int = 32
    seed: int = 42
    fold: int = 0
    fixed_weight: float = None
    epsilon: float = None
    task: str = 'sensors'
    teacher_mode: str = 'top8'
    nft_objective: str = 'legacy'


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(path, **values):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(values, temporary)
    temporary.replace(path)


def prepare(args):
    if args.epochs < 1 or not 1 <= args.k <= 64:
        raise ValueError('Reader epochs must be positive and evidence budget must be in [1,64]')
    source, authority, output = args.prepared.resolve(), args.authority_source.resolve(), args.output.resolve()
    if output == source or source in output.parents or output == authority or authority in output.parents:
        raise ValueError('Choose a new prepared-token output outside both source archives')
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    splits = cohort_rows(source, authority, args.fold)
    targets, images, ids = {}, {}, {}
    for role, rows in splits.items():
        targets[role] = torch.tensor([[v if v is not None else -1 for v in r['anchors']] + [r['phenotype']]
                                     for r in rows], dtype=torch.float32)
        ids[role] = [r['subject_id'] for r in rows]
        images[role] = []
        for row in rows:
            with np.load(source / 'cache' / (row['subject_id'] + '.npz')) as cache:
                images[role].append(torch.tensor(cache['images'], dtype=torch.float32))
        images[role] = torch.stack(images[role])
    signature = {'manifest': fingerprint(source / 'manifest.json'),
                 'splits': fingerprint(authority / 'splits' / f'fold{args.fold}.json'),
                 'seed': args.seed, 'fold': args.fold, 'epochs': args.epochs, 'k': args.k}
    configuration(output, signature)
    if (output / 'tokens.pt').exists():
        print('Prepared tokens already complete', flush=True)
        return
    device = torch.device(args.device)
    reader = TokenReader().to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = {role: balanced_weights(target.to(device)) for role, target in targets.items()}
    best, state = float('inf'), None
    scale = max(float((targets['train'][:, 3] - targets['train'][:, 3].median()).abs().mean()), .05)
    for epoch in range(args.epochs):
        reader.train()
        for indices in torch.randperm(len(images['train'])).split(8):
            token = reader.encode(images['train'][indices].to(device))
            mask = torch.rand(len(indices), 64, device=device).topk(args.k, -1).indices
            prediction = reader(token, mask)
            target = targets['train'][indices].to(device)
            loss = (((prediction[:, :3] - target[:, :3].clamp_min(0)).square() *
                     weights['train'][indices]).sum(-1).mean() / .25 +
                    (prediction[:, 3] - target[:, 3]).abs().mean() / scale)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.)
            optimizer.step()
        reader.eval()
        losses = []
        with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(2026)
            for indices in torch.arange(len(images['validation'])).split(8):
                token = reader.encode(images['validation'][indices].to(device)).repeat_interleave(16, 0)
                mask = torch.rand(len(token), 64, device=device).topk(args.k, -1).indices
                prediction = reader(token, mask).reshape(len(indices), 16, 4)
                target = targets['validation'][indices].to(device)
                loss = ((prediction[:, :, :3] - target[:, None, :3].clamp_min(0)).square() *
                        weights['validation'][indices, None]).sum(-1).mean(-1) / .25
                loss += (prediction[:, :, 3] - target[:, None, 3]).abs().mean(-1) / scale
                losses.extend(loss.cpu().tolist())
        score = float(np.mean(losses))
        if score < best:
            best = score
            state = {key: value.detach().cpu().clone() for key, value in reader.state_dict().items()}
        print(f'Reader epoch={epoch+1} validation={score:.6f}', flush=True)
    reader.load_state_dict(state)
    tokens = {}
    with torch.no_grad():
        for role in images:
            tokens[role] = torch.cat([reader.encode(batch.to(device)).cpu() for batch in images[role].split(8)])
    save_checkpoint(output / 'tokens.pt', tokens=tokens, targets=targets, subject_ids=ids,
                    reader=state, signature=signature)


def configuration(output, config):
    root = Path(__file__).resolve().parents[1]
    files = ['models/idea3_pareto.py', 'data/idea3_pareto.py', 'training/idea3_pareto.py',
             'eval/idea3_pareto.py', 'cli/idea3_pareto.py']
    config = {**config, 'source_hash': hashlib.sha256(b''.join((root / f).read_bytes() for f in files)).hexdigest()}
    path = output / 'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError(f'Existing configuration differs at {path}. Choose a new output directory.')
    save_json(path, config)


def run(args):
    config = Configuration(**{key: getattr(args, key, field.default)
                              for key, field in Configuration.__dataclass_fields__.items()})
    initial_policy = getattr(args, 'init_policy', None)
    if config.method.endswith('epsilon'):
        config.fixed_weight = .5
        if config.epsilon is None or not 0 <= config.epsilon <= 1:
            raise ValueError('Epsilon training requires --epsilon in [0,1]')
    if config.group < 2 or config.noise <= 0 or config.steps < 1 or config.warmup < 0 or config.updates < 1:
        raise ValueError('Require positive budgets/noise and group >= 2')
    if config.warmup == 0 and initial_policy is None:
        raise ValueError('Zero warmup requires --init-policy pointing to a supervised checkpoint')
    if config.teacher_mode not in ('top8', 'best'):
        raise ValueError('teacher-mode must be top8 or best')
    if config.nft_objective not in ('legacy', 'population_stch'):
        raise ValueError('nft-objective must be legacy or population_stch')
    if config.nft_objective != 'legacy' and not config.method.endswith('nft'):
        raise ValueError('population_stch NFT objective requires an NFT method')
    if min(config.batch, config.k, config.proposals, config.validate_every, config.eval_repeats, config.test_repeats) < 1:
        raise ValueError('Batch size, evidence/proposal counts and evaluation budgets must be positive')
    if config.local_steps < 0 or config.lr <= 0 or not 0 <= config.fold < 5:
        raise ValueError('Require local_steps >= 0, lr > 0 and fold in [0,4]')
    if config.fixed_weight is not None and not 0 <= config.fixed_weight <= 1:
        raise ValueError('Fixed preference must be in [0,1]')
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature = asdict(config)
    # Keep legacy configuration fields stable; source changes still require a new run directory.
    if config.nft_objective == 'legacy':
        signature.pop('nft_objective')
    if initial_policy is not None:
        signature['initial_policy_hash'] = fingerprint(initial_policy)
    seed_all(config.seed)
    if args.command == 'controlled':
        train = ControlledEnvironment(config.task, device)
        validation = train
        test = train
        if config.k != 4:
            raise ValueError('Controlled tasks require k=4')
        save_json(output / 'oracle.json', exact_oracle(train, np.linspace(0, 1, 101)))
    else:
        signature['tokens_hash'] = fingerprint(args.tokens)
        archive = torch.load(args.tokens, map_location='cpu', weights_only=True)
        expected = {'fold': config.fold, 'seed': config.seed, 'k': config.k}
        if any(archive['signature'][key] != value for key, value in expected.items()):
            raise ValueError('Token archive fold/seed/evidence budget does not match this run')
        reader = TokenReader()
        reader.load_state_dict(archive['reader'])
        environments = {role: TokenEnvironment(archive['tokens'][role], archive['targets'][role], reader,
                                                device, archive['subject_ids'][role])
                        for role in ('train', 'validation', 'test')}
        train, validation, test = (environments[role] for role in ('train', 'validation', 'test'))
        validation.scales = test.scales = train.scales
    configuration(output, signature)
    if (output / 'summary.json').exists():
        print(f'Already complete: {output}', flush=True)
        return
    if config.k > train.items:
        raise ValueError('Evidence budget exceeds number of tokens')
    experts = 3 if config.method.startswith('moe') else 1
    context_width = train.context.shape[-1]
    width = 64
    if experts == 1:
        target = sum(p.numel() for p in EvidencePolicy(train.items, context_width, experts=3).parameters())
        width = min(range(32, 257), key=lambda w: abs(sum(p.numel() for p in
                    EvidencePolicy(train.items, context_width, width=w).parameters()) - target))
    # Parameter matching must not consume the policy's initialization RNG stream.
    seed_all(config.seed)
    policy = EvidencePolicy(train.items, context_width, width=width, experts=experts,
                            direct=config.method.startswith('direct')).to(device)
    policy.selection_k = config.k
    policy.greedy = config.method == 'direct_sft'
    if initial_policy is not None:
        initial = torch.load(initial_policy, map_location=device, weights_only=True)
        for key in ('seed', 'fold', 'k', 'steps', 'noise'):
            if initial['config'][key] != getattr(config, key):
                raise ValueError(f'Initial checkpoint has incompatible {key}')
        if initial['config']['method'].startswith('direct') != policy.direct:
            raise ValueError('Initial checkpoint uses a different policy family')
        if args.command == 'controlled' and initial['config']['task'] != config.task:
            raise ValueError('Initial checkpoint comes from a different controlled task')
        if args.command == 'train':
            previous = json.loads((initial_policy.parent / 'config.json').read_text(encoding='utf-8'))
            if previous.get('tokens_hash') != signature['tokens_hash']:
                raise ValueError('Initial checkpoint uses a different frozen MRI token archive')
        policy.load_state_dict(initial['model'])

    def checkpoint(update, model, optimizer, reference, history, best_score, best_state, dual):
        save_checkpoint(output / 'last.pt', update=update, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        reference=reference.state_dict(), history=history, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                        best_score=best_score, best_state=best_state, dual=dual)

    resume = torch.load(output / 'last.pt', map_location=device, weights_only=False) if (output / 'last.pt').is_file() else None
    start = perf_counter()
    history = optimize(policy, train, validation, config, checkpoint, resume)
    training_seconds = perf_counter() - start
    save_checkpoint(output / 'best.pt', model=policy.state_dict(), config=asdict(config))
    preferences = np.linspace(0, 1, 21) if config.fixed_weight is None else [config.fixed_weight]
    training_risk = None
    if config.fixed_weight is not None:
        _, training_points = evaluate(policy, train, [config.fixed_weight], config.eval_repeats,
                                       config.steps, config.noise, config.k, 2026)
        training_risk = training_points[0]
    records, points = evaluate(policy, test, preferences, config.test_repeats,
                               config.steps, config.noise, config.k, 2027)
    frame = pd.DataFrame(records)
    frame['method'], frame['fold'], frame['seed'] = ('fixed_' + config.method if config.fixed_weight is not None else config.method), config.fold, config.seed
    atomic_csv(frame, output / 'predictions.csv')
    save_json(output / 'summary.json', {'method': config.method, 'risk_points': points,
                                       'hypervolume': hypervolume(points), 'history': history,
                                       'n_patients': len(test.context), 'sampling': 'expected single-mask loss',
                                       'nft_objective': config.nft_objective if config.method.endswith('nft') else None,
                                       'implementation_profile': 'v1_fixed_teacher_random_kernel',
                                       'training_seconds_this_invocation': training_seconds,
                                       'width': width, 'experts': experts,
                                       'parameter_count': sum(p.numel() for p in policy.parameters()),
                                       'training_risk': training_risk, 'epsilon': config.epsilon,
                                       'test_constraint_satisfied': (all(p[1] <= config.epsilon + .005 for p in points)
                                                                     if config.epsilon is not None else None)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('controlled', 'train'):
        current = sub.add_parser(command)
        current.add_argument('--output', type=Path, required=True)
        current.add_argument('--device', default='cuda')
        current.add_argument('--tokens', type=Path)
        current.add_argument('--init-policy', type=Path)
        for key, field in Configuration.__dataclass_fields__.items():
            flag = '--' + key.replace('_', '-')
            value = field.default
            options = {'choices': ('legacy', 'population_stch')} if key == 'nft_objective' else {}
            current.add_argument(flag, default=value, type=float if key in ('fixed_weight', 'epsilon') else type(value), **options)
        current.set_defaults(k=4 if command == 'controlled' else 8)
    current = sub.add_parser('prepare')
    current.add_argument('--prepared', type=Path, required=True)
    current.add_argument('--authority-source', type=Path, required=True)
    current.add_argument('--output', type=Path, required=True)
    current.add_argument('--fold', type=int, default=0, choices=range(5))
    current.add_argument('--seed', type=int, default=42)
    current.add_argument('--epochs', type=int, default=30)
    current.add_argument('--k', type=int, default=8)
    current.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.method not in METHODS:
            parser.error('Unknown method. Choose ' + ', '.join(METHODS))
        if args.command == 'train' and args.tokens is None:
            parser.error('train requires --tokens')
        run(args)


if __name__ == '__main__':
    main()
