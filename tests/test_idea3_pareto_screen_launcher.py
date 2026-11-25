"""Exercise launcher planning with temporary archives and no GPU or training."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


def bash_binary():
    git = shutil.which('git')
    git_bash = Path(git).parent.parent / 'bin/bash.exe' if git else Path('bash.exe')
    binary = str(git_bash) if git_bash.exists() else shutil.which('bash')
    if not binary:
        pytest.skip('Bash not installed')
    return binary


def launch(tmp_path, mode='screen', missing_cache=False, **options):
    repo = tmp_path / 'repo'
    scripts = repo / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / 'scripts/run_idea3_pareto_mri_screen_autodl.sh'
    # Force LF even when testing a Windows checkout.
    (scripts / source.name).write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
    prior = repo / 'output/idea3_pareto_mri_pilot_poolfix'
    token = prior / 'prepared/seed42/fold0/tokens.pt'
    token.parent.mkdir(parents=True, exist_ok=True)
    token.touch()
    for family in ('dense', 'moe'):
        directory = prior / f'main/{family}_sft/seed42/fold0'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'best.pt').touch()
        (directory / 'config.json').write_text('{}')
    authority = repo / 'output/idea3_rule_mri_swap/utsw_authority_grade'
    (authority / 'splits').mkdir(parents=True, exist_ok=True)
    (authority / 'manifest.json').write_text('{}')
    for fold in range(5):
        (authority / f'splits/fold{fold}.json').write_text('{}')
    cache = repo / 'output/idea3_pareto_mri_pilot/prepared_images/experiment1'
    if not missing_cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / 'manifest.json').write_text('{}')
    # Remove inherited experimental overrides so tests model the documented default.
    env = {k: v for k, v in os.environ.items() if not k.startswith('IDEA3_')}
    env.update(IDEA3_DRY_RUN='1', **options)
    if missing_cache:
        raw = tmp_path / 'raw'
        raw.mkdir()
        metadata = tmp_path / 'metadata.tsv'
        metadata.touch()
        env.update(IDEA3_DATA_ROOT=str(raw), IDEA3_METADATA=str(metadata))
    result = subprocess.run([bash_binary(), str(scripts / source.name), mode],
                            env=env, capture_output=True, text=True, timeout=30)
    return result


def test_screen_reuses_fold0_and_prepares_only_other_readers(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('glioma.cli.idea3_pareto_screen evaluate') == 5
    assert result.stdout.count('prepare --epochs 30 --device cuda') == 4
    assert 'idea3_pareto_mri_pilot_poolfix/prepared/seed42/fold0/tokens.pt' in result.stdout
    assert 'glioma.cli.idea3_pareto_screen aggregate' in result.stdout
    assert '--expected-folds 0 1 2 3 4' in result.stdout
    assert '--init-policy' not in result.stdout
    assert 'import_root' not in result.stdout


def test_posttrain_shares_initialization_and_budget(tmp_path):
    result = launch(tmp_path, 'posttrain', IDEA3_POST_UPDATES='200')
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('glioma.cli.idea3_pareto train') == 6
    assert result.stdout.count('--warmup 0 --updates 200') == 6
    assert result.stdout.count('--nft-objective population_stch') == 2
    for family in ('dense', 'moe'):
        assert result.stdout.count(f'{family}_sft/seed42/fold0/best.pt') == 3
    assert 'glioma.cli.analyze_idea3_pareto' in result.stdout
    assert 'idea3_pareto_screen evaluate' not in result.stdout


def test_cached_tokens_are_checked_before_any_new_reader(tmp_path):
    result = launch(tmp_path, IDEA3_FOLDS='1 0')
    assert result.returncode == 0, result.stderr
    assert result.stdout.index('glioma.cli.idea3_pareto_screen evaluate') < result.stdout.index('prepare --epochs')


def test_deleted_old_image_cache_is_rebuilt_before_cached_token_checks(tmp_path):
    result = launch(tmp_path, missing_cache=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.index('glioma.cli.idea3 prepare --data-root') < result.stdout.index('glioma.cli.idea3_pareto_screen evaluate')
    assert result.stdout.count('prepare --epochs 30 --device cuda') == 4


def test_missing_authority_fails_before_work(tmp_path):
    result = launch(tmp_path, IDEA3_AUTHORITY=str(tmp_path / 'missing'))
    assert result.returncode == 1
    assert 'Missing authority manifest' in result.stderr
    assert 'glioma.cli.' not in result.stdout


@pytest.mark.parametrize('folds', ['0 0', '5', '0 x'])
def test_bad_folds_fail_before_work(tmp_path, folds):
    result = launch(tmp_path, IDEA3_FOLDS=folds)
    assert result.returncode == 2
    assert 'glioma.cli.' not in result.stdout
