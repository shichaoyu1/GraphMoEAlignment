"""Population-objective NFT alignment and backward-compatible opt-in behavior."""

from argparse import Namespace
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from glioma.cli.analyze_idea3_pareto import analyze
from glioma.cli.idea3_pareto import Configuration, run
from glioma.cli.utsw_authority import seed_all
from glioma.data.idea3_pareto import ControlledEnvironment
from glioma.models.idea3_pareto import EvidencePolicy
from glioma.training.idea3_pareto import optimize, population_nft_advantage, scalar_risk


def small_config(**values):
    fields = dict(method='moe_nft', warmup=1, updates=2, batch=2, group=4,
                  steps=2, k=4, proposals=4, local_steps=0, validate_every=1,
                  eval_repeats=2, test_repeats=2)
    fields.update(values)
    return Configuration(**fields)


def test_population_advantage_uses_one_scale_preserving_between_patient_magnitudes():
    values = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 2.]], requires_grad=True)
    advantage, scale = population_nft_advantage(values, torch.tensor([2., 1.]), 2, 2)
    torch.testing.assert_close(advantage * scale, torch.tensor([-1., 1., -2., 2.]))
    assert advantage[2] == 2 * advantage[0]
    assert not advantage.requires_grad and not scale.requires_grad
    scaled, scaled_norm = population_nft_advantage(values, torch.tensor([14., 7.]), 2, 2)
    torch.testing.assert_close(scaled, advantage)
    torch.testing.assert_close(scaled_norm, 7 * scale)
    assert torch.equal(advantage.argsort(), torch.tensor([2, 0, 1, 3]))


@pytest.mark.parametrize('weight,first_is_preferred', [(0., False), (1., True)])
def test_population_advantage_follows_the_preferred_objective(weight, first_is_preferred):
    risks = torch.tensor([.4, .5], requires_grad=True)
    alpha = torch.autograd.grad(scalar_risk(risks, torch.tensor(weight), torch.tensor([.25, .1])), risks)[0]
    advantage, _ = population_nft_advantage(torch.tensor([[.1, .9], [.9, .1]]), alpha, 1, 2)
    assert bool(advantage[0] > 0) == first_is_preferred
    assert bool(advantage[1] > 0) != first_is_preferred
    assert alpha[1 if first_is_preferred else 0] == 0


def test_population_advantage_handles_no_within_group_signal():
    advantage, scale = population_nft_advantage(torch.ones(8, 2), torch.tensor([2., 3.]), 2, 4)
    assert torch.equal(advantage, torch.zeros(8))
    assert torch.isfinite(scale) and scale > 0


def test_legacy_default_matches_a_configuration_without_the_new_option():
    config = small_config()
    old_config = Namespace(**{key: value for key, value in vars(config).items() if key != 'nft_objective'})
    states, histories = [], []
    for current in (old_config, config):
        seed_all(42)
        environment = ControlledEnvironment('sensors', 'cpu')
        policy = EvidencePolicy(8, 2, experts=3)
        policy.selection_k = 4
        histories.append(optimize(policy, environment, environment, current))
        states.append(deepcopy(policy.state_dict()))
    assert histories[0] == histories[1]
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


@pytest.mark.parametrize('method', ['dense_nft', 'moe_nft'])
def test_population_nft_updates_are_finite_and_diagnosed(method):
    seed_all(42)
    environment = ControlledEnvironment('sensors', 'cpu')
    policy = EvidencePolicy(8, 2, experts=3 if method.startswith('moe') else 1)
    policy.selection_k = 4
    config = small_config(method=method, nft_objective='population_stch')
    history = optimize(policy, environment, environment, config)
    assert all(np.isfinite(row['loss']) for row in history)
    diagnostics = history[-1]['diagnostics']
    assert diagnostics['nft_objective'] == 'population_stch'
    assert diagnostics['scalar_advantage_scale'] > 0
    assert 0 <= diagnostics['advantage_clip_fraction'] <= 1
    assert np.isfinite(diagnostics['risk_coefficients']).all()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in policy.parameters())


def test_matched_sft_pg_nft_forks_share_initialization_and_analyzer_contrasts(tmp_path):
    config = small_config(method='moe_sft')
    seed_all(42)
    initial = tmp_path / 'initial.pt'
    # An earlier archive has no nft_objective field, and remains loadable as an initializer.
    torch.save({'model': EvidencePolicy(8, 2, experts=3).state_dict(),
                'config': {key: value for key, value in vars(config).items() if key != 'nft_objective'}}, initial)
    hashes = []
    for method in ('moe_sft', 'moe_pg', 'moe_nft'):
        current = small_config(method=method, warmup=0, updates=2,
                               nft_objective='population_stch' if method.endswith('nft') else 'legacy')
        output = tmp_path / 'main' / method / 'seed42' / 'fold0'
        run(Namespace(**vars(current), command='controlled', output=output, device='cpu', init_policy=initial))
        config_json = json.loads((output / 'config.json').read_text())
        summary = json.loads((output / 'summary.json').read_text())
        hashes.append(config_json['initial_policy_hash'])
        assert summary['history'][0]['update'] == 0
        assert summary['history'][-1]['update'] == 2
        if method.endswith('nft'):
            assert config_json['nft_objective'] == 'population_stch'
            assert summary['nft_objective'] == 'population_stch'
        else:
            assert 'nft_objective' not in config_json
    assert len(set(hashes)) == 1
    analyze(tmp_path / 'main', tmp_path / 'analysis', 50)
    report = json.loads((tmp_path / 'analysis' / 'summary.json').read_text())
    for suffix in ('nft', 'pg'):
        contrast = report[suffix + '_minus_sft']['moe']
        estimate = report['methods']['moe_' + suffix]['hypervolume'] - report['methods']['moe_sft']['hypervolume']
        assert contrast['estimate'] == pytest.approx(estimate)
        assert len(contrast['ci95']) == 2
        assert 'holm_p' not in contrast


def test_population_nft_option_cannot_silently_change_a_pg_run(tmp_path):
    config = small_config(method='moe_pg', nft_objective='population_stch')
    with pytest.raises(ValueError, match='requires an NFT method'):
        run(Namespace(**vars(config), command='controlled', output=tmp_path / 'invalid', device='cpu'))
