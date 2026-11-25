import numpy as np
import torch
import pytest
from glioma.cli.idea3_pareto import Configuration
from glioma.training.idea3_pareto import METHODS, optimize, evaluate, clipped_trajectory_objective
from glioma.cli.idea3_pareto import prepare, run
from glioma.models.idea3_pareto import TokenGradeReadout
from glioma.data.utsw_authority import GradeEvidence, features, molecular_record
from glioma.cli.idea3_authority import withdraw_report
from glioma.cli.analyze_idea3_pareto import analyze
from argparse import Namespace
import json
import pandas as pd

from glioma.data.idea3_pareto import ControlledEnvironment, balanced_weights
from glioma.eval.idea3_pareto import exact_oracle, hypervolume
from glioma.models.idea3_pareto import EvidencePolicy
from glioma.training.idea3_pareto import scalar_risk


def test_controlled_oracle_and_common_optimum():
    environment = ControlledEnvironment('sensors', 'cpu')
    oracle = exact_oracle(environment, [.5])
    np.testing.assert_allclose(oracle['randomized_tchebycheff'], [[1 / 13, 1 / 13]])
    assert len(oracle['deterministic_front']) == 9
    common = ControlledEnvironment('common', 'cpu')
    assert common.losses(torch.tensor([0]), common.masks[:1]).tolist() == [[0., 0.]]
    assert hypervolume([[0., 0.]]) == 1.01 ** 2


def test_balanced_loss_masks_missing_values():
    targets = torch.tensor([[0., 0., 1.], [1., 1., 0.], [1., -1., 0.]])
    weights = balanced_weights(targets)
    assert weights[2, 1] == 0
    assert torch.allclose((weights * .25).sum(-1).mean(), torch.tensor(.25))


def test_gaussian_probability_has_gradient_and_matches_density():
    torch.manual_seed(42)
    model = EvidencePolicy(items=8, context=2)
    context = torch.eye(2)
    pref = torch.tensor([.2, .8])
    with torch.no_grad():
        _, states, actions, _ = model.sample(context, pref, steps=2, noise=.1)
    probability = model.log_probability(states, actions, context, pref, .1)
    means = states[:, 0] + model(states[:, 0], context, pref, torch.zeros(2))[0] / 2
    expected = torch.distributions.Normal(means, .1 / np.sqrt(2)).log_prob(actions[:, 0]).sum(-1)
    assert torch.allclose(probability[:, 0], expected)
    probability.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_population_scalarization_differs_from_patient_scalarization():
    risks = torch.tensor([[0., 1.], [1., 0.]])
    weight = torch.tensor(.5)
    assert scalar_risk(risks.mean(0), weight, torch.ones(2)) < scalar_risk(risks, weight, torch.ones(2)).mean()


@pytest.mark.parametrize('steps', [2, 8])
def test_terminal_reward_gradient_does_not_shrink_with_denoising_steps(steps):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = EvidencePolicy(items=1, context=1, width=4)
    for parameter in model.parameters():
        parameter.data.zero_()
    context = torch.zeros(2048, 1)
    preference = torch.full((len(context),), .5)
    with torch.no_grad():
        terminal, states, actions, _ = model.sample(context, preference, steps=steps, noise=.1)
        # Removing the initial noise is an action-independent baseline.
        reward = terminal[:, 0] - states[:, 0, 0]
    log_probability = model.log_probability(states, actions, context, preference, .1)
    objective = clipped_trajectory_objective(log_probability, log_probability.detach(), reward)
    gradient = torch.autograd.grad(objective, model.experts[0][-1].bias)[0].item()
    # E[terminal] equals the shared velocity bias for every trajectory length.
    assert gradient == pytest.approx(1., abs=.1)


def test_token_grade_readonly_and_evidence_revocation():
    record = molecular_record({'IDH': 'mutated', '1p19Q CODEL': 'co-deleted', 'IDH_Testing Method': 'NGS',
                               'Tumor Type': 'OLIGODENDROGLIOMA, NOS'})
    evidence = GradeEvidence(torch.empty(2, 0), torch.tensor([features(record)] * 2),
                             torch.ones(2, 5, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    model = TokenGradeReadout('readonly').eval()
    tokens = torch.randn(2, 64, 64)
    mask = torch.arange(8).expand(2, -1)
    first = model(tokens, mask, evidence)
    second = model(tokens + 1, mask.flip(1), evidence)
    assert torch.equal(first['p0'][:, 2], second['p0'][:, 2])
    assert (first['p_star'][:, 2] == 0).all()
    withdrawn = model(tokens, mask, withdraw_report(evidence))
    assert not withdrawn['active'].any() and withdrawn['allowed'].all()


@pytest.mark.parametrize('method', METHODS)
def test_each_update_family_runs_against_exact_environment(method):
    torch.set_num_threads(2)
    environment = ControlledEnvironment('sensors', 'cpu')
    policy = EvidencePolicy(8, 2, experts=3 if method.startswith('moe') else 1, direct=method.startswith('direct'))
    policy.selection_k = 4
    config = Configuration(method=method, warmup=1, updates=1, batch=2, group=2, steps=2, k=4,
                           proposals=4, local_steps=0, validate_every=2, eval_repeats=2,
                           epsilon=.5 if method.endswith('epsilon') else None,
                           fixed_weight=.5 if method.endswith('epsilon') else None)
    history = optimize(policy, environment, environment, config)
    assert len(history) == 1 and np.isfinite(history[0]['loss'])
    records, points = evaluate(policy, environment, [0., .5, 1.], 2, 2, .05, 4, 2026)
    assert len(records) == 12 and len(points) == 3
    assert all(len(set(r['indices'])) == 4 for r in records)


def test_mri_prepare_train_and_resume_entrypoint(tmp_path):
    torch.set_num_threads(2)
    source, authority = tmp_path / 'source', tmp_path / 'authority'
    (source / 'cache').mkdir(parents=True)
    (authority / 'splits').mkdir(parents=True)
    rows, split = [], {}
    rng = np.random.default_rng(42)
    for role in ('train', 'validation', 'test'):
        split[role] = []
        for index in range(4):
            sid = role + str(index)
            split[role].append(sid)
            rows.append({'subject_id': sid, 'anchors': [index % 2] * 3, 'phenotype': index / 4})
            np.savez(source / 'cache' / (sid + '.npz'), images=rng.random((4, 16, 16, 16), dtype=np.float32))
    (source / 'manifest.json').write_text(json.dumps({'rows': rows}))
    (authority / 'splits' / 'fold0.json').write_text(json.dumps(split))
    prepared = tmp_path / 'tokens'
    prepare(Namespace(prepared=source, authority_source=authority, output=prepared, fold=0, seed=42,
                      epochs=1, k=4, device='cpu'))
    output = tmp_path / 'main' / 'moe_pg' / 'seed42' / 'fold0'
    config = Configuration(method='moe_pg', warmup=1, updates=1, batch=2, group=2, steps=2, k=4,
                           proposals=4, local_steps=0, validate_every=2, eval_repeats=2, test_repeats=2)
    args = Namespace(**vars(config), command='train', output=output, tokens=prepared / 'tokens.pt', device='cpu')
    run(args)
    assert json.loads((output / 'summary.json').read_text())['n_patients'] == 4
    before_resume = pd.read_csv(output / 'predictions.csv')
    # Simulate interruption during export after the final durable training checkpoint.
    (output / 'summary.json').unlink()
    run(args)
    assert (output / 'summary.json').is_file()
    pd.testing.assert_frame_equal(before_resume, pd.read_csv(output / 'predictions.csv'))
    analyze(tmp_path / 'main', tmp_path / 'analysis', 100)
    assert (tmp_path / 'analysis' / 'risk_curves.pdf').is_file()
