from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from glioma.cli.idea3_pareto_screen import evaluate, aggregate, fingerprint, KIND
from glioma.eval.idea3_pareto_screen import patient_errors, endpoint_contrast
from glioma.models.idea3_pareto import TokenReader


def make_archive(path, fold=0, validation_prefix=None):
    torch.manual_seed(42)
    reader = TokenReader()
    targets = torch.tensor([[0., 0., 0., .2], [1., 1., 1., .4],
                            [0., 0., 0., .6], [1., 1., 1., .8]])
    archive = {'reader': reader.state_dict(), 'tokens': {}, 'targets': {}, 'subject_ids': {},
               'signature': {'fold': fold, 'seed': 42, 'k': 8}}
    for role in ('train', 'validation', 'test'):
        archive['tokens'][role] = torch.randn(4, 64, 64)
        archive['targets'][role] = targets.clone()
        prefix = validation_prefix if role == 'validation' and validation_prefix else f'{role}{fold}'
        archive['subject_ids'][role] = [prefix + '_' + str(i) for i in range(4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(archive, path)
    return archive


def arguments(tokens, output):
    return Namespace(tokens=tokens, output=output, device='cpu', proposals=4, local_steps=2,
                     random_repeats=2, bootstrap=30, seed=2026)


def aggregate_args(source, output, folds=(0, 1)):
    return Namespace(source=source, output=output, expected_folds=folds, bootstrap=30, seed=2026)


def test_train_selection_independent_of_validation_and_test_labels(tmp_path):
    first = tmp_path / 'inputs' / 'first.pt'
    archive = make_archive(first)
    original = evaluate(arguments(first, tmp_path / 'first'))
    for role in ('validation', 'test'):
        archive['targets'][role][:, :3] = 1 - archive['targets'][role][:, :3]
        archive['targets'][role][:, 3] = 1 - archive['targets'][role][:, 3]
    second = tmp_path / 'inputs' / 'second.pt'
    torch.save(archive, second)
    changed = evaluate(arguments(second, tmp_path / 'second'))
    first_masks = json.loads((tmp_path / 'first/selected_masks.json').read_text())
    second_masks = json.loads((tmp_path / 'second/selected_masks.json').read_text())
    assert first_masks == second_masks
    for role in ('validation', 'test'):
        a, b = (pd.read_csv(tmp_path / output / f'{role}_predictions.csv') for output in ('first', 'second'))
        pd.testing.assert_frame_equal(a.filter(regex='^prediction_'), b.filter(regex='^prediction_'))
        assert a['indices'].equals(b['indices'])
    assert original['status'] == changed['status'] == 'diagnostic_only'


def test_durable_resume_and_provenance_rejection(tmp_path):
    tokens = tmp_path / 'inputs/tokens.pt'
    make_archive(tokens)
    args = arguments(tokens, tmp_path / 'screen')
    report = evaluate(args)
    assert evaluate(args) == report
    args.local_steps += 1
    with pytest.raises(ValueError, match='configuration differs'):
        evaluate(args)
    args.local_steps -= 1
    path = args.output / 'test_predictions.csv'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='artifact changed'):
        evaluate(args)


def test_tokens_must_match_requested_authority_and_cached_images(tmp_path):
    tokens = tmp_path / 'inputs/tokens.pt'
    archive = make_archive(tokens)
    authority = tmp_path / 'authority'
    (authority / 'splits').mkdir(parents=True)
    split = authority / 'splits/fold0.json'
    split.write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    archive['signature'].update(splits=fingerprint(split), manifest=fingerprint(manifest))
    torch.save(archive, tokens)
    args = arguments(tokens, tmp_path / 'screen')
    args.authority_source, args.image_manifest = authority, manifest
    args.fold, args.training_seed = 1, 42
    with pytest.raises(ValueError, match='requested fold'):
        evaluate(args)
    args.fold = 0
    verified = evaluate(args)
    # Optional current-source checks must not change the archived cohort identity.
    unchecked = evaluate(arguments(tokens, tmp_path / 'unchecked'))
    assert verified['config'] == unchecked['config']
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='manifest hash'):
        evaluate(args)


def test_crossfold_summary_is_pooled_and_diagnostic_only(tmp_path):
    source = tmp_path / 'screen'
    for fold in (0, 1):
        tokens = tmp_path / 'inputs' / f'fold{fold}.pt'
        make_archive(tokens, fold)
        evaluate(arguments(tokens, source / f'fold{fold}'))
    result = aggregate(aggregate_args(source, tmp_path / 'aggregate'))
    assert result['status'] == 'diagnostic_only'
    assert result['cohort_sizes'] == {'validation': 8, 'test': 8}
    assert result['validation_is_primary']
    assert result['endpoints']['validation']['patients'] == 8
    assert result['endpoints']['test']['patients'] == 8
    assert 'Pooled OOF' in result['aggregation']
    with pytest.raises(ValueError, match='Expected folds'):
        aggregate(aggregate_args(source, tmp_path / 'missing', (0, 1, 2)))


def test_aggregate_rejects_incomplete_and_duplicate_oof_patients(tmp_path):
    source = tmp_path / 'screen'
    for fold in (0, 1):
        tokens = tmp_path / 'inputs' / f'fold{fold}.pt'
        make_archive(tokens, fold, validation_prefix='same_validation')
        evaluate(arguments(tokens, source / f'fold{fold}'))
    with pytest.raises(ValueError, match='validation patients repeat'):
        aggregate(aggregate_args(source, tmp_path / 'aggregate'))
    (source / 'fold1/summary.json').unlink()
    with pytest.raises(ValueError, match='Incomplete screening run'):
        aggregate(aggregate_args(source, tmp_path / 'aggregate'))


def test_pooled_class_balancing_differs_from_equal_fold_average():
    rows = []
    # Fold0 has one positive and three negatives, fold1 the reverse. Errors differ by fold.
    for fold, labels in [(0, [0, 0, 0, 1]), (1, [0, 1, 1, 1])]:
        for i, label in enumerate(labels):
            for weight in (0., 1.):
                row = {'role': 'test', 'fold': fold, 'seed': 42, 'method': 'fixed8_train_selected',
                       'subject_id': f'f{fold}_{i}', 'weight': weight, 'sample': 0,
                       'target_phenotype': .5, 'phenotype_loss': weight * .1}
                for field in ('idh', '1p19q', 'mgmt'):
                    row['target_' + field] = label
                    row['prediction_' + field] = label if weight == 0 else (1 - label if fold == 0 and label == 0 else label)
                rows.append(row)
    result = endpoint_contrast(patient_errors(pd.DataFrame(rows)), 50, 42)
    assert result['anchor_delta'] == pytest.approx(.375)
    assert result['phenotype_delta_pp'] == pytest.approx(10.)
    assert len(result['folds']) == 2
    assert result['expected_direction_folds'] == 0


def test_missing_evaluation_class_is_reported_not_fabricated(tmp_path):
    tokens = tmp_path / 'inputs/tokens.pt'
    archive = make_archive(tokens)
    archive['targets']['test'][:, 1] = 0
    torch.save(archive, tokens)
    result = evaluate(arguments(tokens, tmp_path / 'screen'))
    endpoint = result['endpoints']['test']
    assert endpoint['anchor_delta'] is None
    assert endpoint['valid_bootstrap'] == 0
    assert endpoint['invalid_bootstrap'] == 30
    assert endpoint['anchor_ci95'] is None
    assert not endpoint['intervals_support_expected_direction']
