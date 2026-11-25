"""Aggregate paired OOF risk curves with fixed-reference hypervolume intervals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glioma.cli.idea3_anchor import atomic_csv
from glioma.cli.utsw_authority import save_json
from glioma.eval.idea3_pareto import hypervolume


def analyze(source, output, repetitions):
    files = sorted(source.glob('*/seed*/fold*/predictions.csv'))
    files += sorted(source.glob('*/seed*/fold*/fixed_*/predictions.csv'))
    if not files:
        raise ValueError('No completed main experiment predictions found')
    for file in files:
        if not (file.parent / 'summary.json').exists():
            raise ValueError(f'Incomplete run at {file.parent}')
    frame = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    keys = ['method', 'seed', 'fold', 'subject_id', 'weight']
    if frame.duplicated(keys + ['sample']).any():
        raise ValueError('Duplicate patient/rollout results')
    extra = []
    for field in ('idh', '1p19q', 'mgmt'):
        target, prediction = 'target_' + field, 'prediction_' + field
        if target in frame:
            error = 'square_error_' + field
            frame[error] = (frame[prediction] - frame[target].clip(lower=0)) ** 2
            extra.extend([target, error])
    patients = frame.groupby(keys, as_index=False)[['anchor_loss', 'phenotype_loss', *extra]].mean()
    rng = np.random.default_rng(2026)
    sets, distributions, curves = {}, {}, []
    common_draws = {}
    for (method, seed), rows in patients.groupby(['method', 'seed']):
        ordered = rows[['fold', 'subject_id']].drop_duplicates().sort_values(['fold', 'subject_id'])
        identifiers = tuple(ordered.itertuples(index=False, name=None))
        if sets and next(iter(sets.values())) != identifiers:
            raise ValueError('Training seeds must cover identical OOF patients')
        sets.setdefault(seed, identifiers)
        if sets[seed] != identifiers:
            raise ValueError('Methods must cover identical OOF patients for each seed')
        for fold, group in ordered.groupby('fold'):
            n = len(group)
            if fold not in common_draws:
                common_draws[fold] = rng.multinomial(n, np.full(n, 1 / n), repetitions) / n
        weights = sorted(rows.weight.unique())
        values = np.zeros((repetitions + 1, len(weights), 2))
        for column, weight in enumerate(weights):
            selected = rows[rows.weight == weight]
            if set(selected.fold) != set(ordered.fold):
                raise ValueError('Preference grid is incomplete')
            for fold, group in selected.groupby('fold'):
                group = group.sort_values('subject_id')
                if tuple(group.subject_id) != tuple(ordered.loc[ordered.fold == fold, 'subject_id']):
                    raise ValueError('Preference grid has missing patients')
                loss = group[['anchor_loss', 'phenotype_loss']].to_numpy()
                values[0, column] += loss.mean(0) / selected.fold.nunique()
                values[1:, column] += common_draws[fold] @ loss / selected.fold.nunique()
                if extra:
                    bootstrap_anchor = np.zeros(repetitions)
                    for field in ('idh', '1p19q', 'mgmt'):
                        for label in (0, 1):
                            observed = group['target_' + field].to_numpy() == label
                            denominator = common_draws[fold] @ observed
                            numerator = common_draws[fold] @ (group['square_error_' + field].to_numpy() * observed)
                            bootstrap_anchor += np.divide(numerator, denominator, out=np.full(repetitions, np.nan), where=denominator > 0) / 6
                    values[1:, column, 0] -= (common_draws[fold] @ loss[:, 0]) / selected.fold.nunique()
                    values[1:, column, 0] += bootstrap_anchor / selected.fold.nunique()
            curves.append({'method': method, 'seed': int(seed), 'weight': float(weight),
                           'anchor_brier': float(values[0, column, 0]), 'phenotype_mae': float(values[0, column, 1])})
        distributions[method, seed] = np.array([hypervolume(points) if np.isfinite(points).all() else np.nan for points in values])
    methods = sorted(patients.method.unique())
    report, average = {}, {}
    seeds = sorted(patients.seed.unique())
    for method in methods:
        if any((method, seed) not in distributions for seed in seeds):
            raise ValueError(f'Missing training seeds for {method}')
        average[method] = np.mean([distributions[method, seed] for seed in seeds], axis=0)
        result = average[method]
        valid = result[1:][np.isfinite(result[1:])]
        if not len(valid):
            raise ValueError('No valid paired bootstrap replicates')
        report[method] = {'hypervolume': float(result[0]), 'ci95': np.quantile(valid, [.025, .975]).tolist(),
                          'valid_bootstrap': len(valid), 'invalid_fraction': 1 - len(valid) / repetitions,
                          'per_seed': {str(seed): float(distributions[method, seed][0]) for seed in seeds}}
    contrasts = {}
    if 'moe_pg' in average:
        for method in methods:
            if method == 'moe_pg':
                continue
            delta = average['moe_pg'] - average[method]
            valid = delta[1:][np.isfinite(delta[1:])]
            contrasts[method] = {'estimate': float(delta[0]), 'ci95': np.quantile(valid, [.025, .975]).tolist()}
            contrasts[method]['centered_bootstrap_p'] = float((1 + np.sum(np.abs(valid - delta[0]) >= abs(delta[0]))) / (len(valid) + 1))
        family = ['dense_pg', 'moe_sft', 'moe_nft', 'direct_sft']
        ordered = sorted(family, key=lambda key: contrasts.get(key, {}).get('centered_bootstrap_p', 1.))
        adjusted = 0.
        for rank, method in enumerate(ordered):
            if method in contrasts:
                adjusted = max(adjusted, min(1., (len(family) - rank) * contrasts[method]['centered_bootstrap_p']))
                contrasts[method]['holm_p'] = adjusted
    output.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(curves)
    atomic_csv(curve, output / 'risk_curves.csv')
    atomic_csv(patients, output / 'patient_losses.csv')
    save_json(output / 'summary.json', {'methods': report, 'moe_pg_minus_control': contrasts,
                                      'bootstrap_repetitions': repetitions, 'seeds': list(map(int, seeds)),
                                      'folds': sorted(map(int, patients.fold.unique())),
                                      'status': 'complete_five_fold' if patients.fold.nunique() == 5 else 'partial_fold_diagnostic',
                                      'intervals': 'paired patients conditional on fitted models; raw unadjusted intervals'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, group in curve.groupby('method'):
        group = group.groupby('weight')[['anchor_brier', 'phenotype_mae']].mean()
        axis.plot(group.anchor_brier, group.phenotype_mae * 100, '.-', label=method)
    axis.set(xlabel='Molecular balanced Brier', ylabel='Phenotype MAE (percentage points)')
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / 'risk_curves.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('bootstrap must be positive')
    analyze(args.source, args.output, args.bootstrap)


if __name__ == '__main__':
    main()
