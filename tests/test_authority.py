"""Executable invariants for the proposed authority mechanism (not clinical validation)."""

import dataclasses
import importlib.util
import unittest

import numpy as np
import torch
from scipy.optimize import minimize

from glioma.data.authority_benchmarks import AuthoritySynthetic, s0_check
from glioma.models.authority_fusion import AuthorityFusion, TRAINED_VARIANTS, authority_loss
from glioma.modules.authority_projection import (project_registered, project_allowed_simplex,
                                                 project_disjoint_orders, GeneralAuthorityProjection)
from glioma.modules.authority_rules import compile_authority, relation_residuals


class AuthorityContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_s0_and_no_labels_in_compiler_contract(self):
        self.assertEqual(s0_check()["policy_blind_bayes_error"], .25)
        batch = AuthoritySynthetic("s1", count=20).materialize()
        with self.assertRaises(TypeError):
            compile_authority(batch)
        c1 = compile_authority(batch.evidence)
        batch.labels = (batch.labels + 1) % 4
        c2 = compile_authority(batch.evidence)
        for name in ("B", "b", "allowed_dependencies", "allowed_classes", "status"):
            self.assertTrue(torch.equal(getattr(c1, name), getattr(c2, name)))

    def test_split_ids_and_nested_training_data(self):
        small = AuthoritySynthetic("s1", count=8).materialize()
        full = AuthoritySynthetic("s1", count=20).materialize()
        self.assertTrue(torch.equal(small.evidence.tokens, full.evidence.tokens[:8]))
        self.assertTrue(torch.equal(small.labels, full.labels[:8]))
        val = AuthoritySynthetic("s1", split="val", count=8).materialize()
        self.assertFalse(set(small.sample_ids) & set(val.sample_ids))

    def test_target_switch_preserves_observations(self):
        for task in ("s1", "s2"):
            generator = AuthoritySynthetic(task, count=80)
            before, after = generator.materialize(), generator.materialize(event="target_switch")
            self.assertTrue(torch.equal(before.evidence.tokens, after.evidence.tokens))
            self.assertTrue(torch.equal(before.evidence.statements, after.evidence.statements))
            self.assertTrue(torch.equal(1-before.evidence.target_id, after.evidence.target_id))
            self.assertTrue((before.labels != after.labels).any())

    def test_invalid_unknown_missing_and_conflict(self):
        for task in ("s1", "s2"):
            generator = AuthoritySynthetic(task, count=12)
            original = generator.materialize()
            for event in ("invalid", "unknown", "missing"):
                altered = generator.materialize(event=event)
                c = compile_authority(altered.evidence)
                self.assertTrue((c.status == 1).all())
                self.assertFalse(c.relation_active.any())
                self.assertTrue(torch.equal(original.evidence.tokens, altered.evidence.tokens))
                self.assertTrue(torch.equal(original.labels, altered.labels))
                owners = altered.evidence.authority_sources
                visible = c.node_available[:, 0].gather(1, owners)
                self.assertEqual(bool(visible.any()), event == "unknown")
            self.assertTrue((compile_authority(generator.materialize(event="conflict").evidence).status == 2).all())
            self.assertTrue((compile_authority(generator.materialize(event="all_missing").evidence).status == 3).all())

    def test_auxiliary_changes_cannot_modify_compiler(self):
        generator = AuthoritySynthetic("s1", count=25)
        before = generator.materialize()
        after = generator.materialize(event="aux_conflict", kappa=1, strength=8)
        c1, c2 = compile_authority(before.evidence), compile_authority(after.evidence)
        for name in ("B", "b", "allowed_dependencies", "status"):
            self.assertTrue(torch.equal(getattr(c1, name), getattr(c2, name)))
        owner = before.evidence.authority_sources[:, 0]
        self.assertTrue(torch.equal(before.evidence.tokens[torch.arange(25), 0, owner],
                                    after.evidence.tokens[torch.arange(25), 0, owner]))

    def test_s2_latent_probabilities_obey_valid_relations(self):
        batch = AuthoritySynthetic("s2", count=200).materialize()
        compiled = compile_authority(batch.evidence)
        self.assertLessEqual(float(relation_residuals(batch.latent_probabilities, compiled).max()), 1e-7)
        # Individual noisy binary labels need not satisfy the conditional-probability ordering.
        self.assertGreater(float(relation_residuals(batch.labels, compiled).sum()), 0)


class AuthorityProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(27)

    def test_analytic_projection_matches_independent_scipy_solver(self):
        for task in ("s1", "s2"):
            batch = AuthoritySynthetic(task, count=12).materialize()
            c = compile_authority(batch.evidence)
            p = torch.rand(12, 4 if task == "s1" else 6, dtype=torch.float64)
            if task == "s1":
                p = p / p.sum(-1, keepdim=True)
            projected = project_registered(p, c)
            self.assertLessEqual(float(relation_residuals(projected, c).max()), 1e-8)
            torch.testing.assert_close(project_registered(projected, c), projected, atol=1e-10, rtol=0)
            for i in range(len(p)):
                B, b = c.B[i, c.constraint_active[i]].double().numpy(), c.b[i, c.constraint_active[i]].double().numpy()
                u = p[i].numpy()
                constraints = [{"type": "ineq", "fun": lambda x: B @ x - b, "jac": lambda x: B}]
                if task == "s1":
                    constraints.append({"type": "eq", "fun": lambda x: x.sum()-1,
                                        "jac": lambda x: np.ones_like(x)})
                solved = minimize(lambda x: .5*np.square(x-u).sum(), np.full_like(u, 1/len(u)),
                                  jac=lambda x: x-u, bounds=[(0,1)]*len(u), constraints=constraints,
                                  method="SLSQP", options={"ftol": 1e-12, "maxiter": 300})
                self.assertTrue(solved.success, solved.message)
                np.testing.assert_allclose(projected[i].numpy(), solved.x, atol=2e-6, rtol=0)

    def test_projection_gradients(self):
        p = torch.tensor([[.08,.31,.22,.39]], dtype=torch.float64, requires_grad=True)
        allowed = torch.tensor([[True, True, False, False]])
        self.assertTrue(torch.autograd.gradcheck(lambda x: project_allowed_simplex(x, allowed), (p,), atol=1e-5))
        v = torch.tensor([[.1,.8,.7,.2,.4,.6]], dtype=torch.float64, requires_grad=True)
        directions, active = torch.tensor([[1,0]]), torch.tensor([[True,True]])
        self.assertTrue(torch.autograd.gradcheck(lambda x: project_disjoint_orders(x,directions,active), (v,), atol=1e-5))

    @unittest.skipUnless(importlib.util.find_spec("cvxpylayers"), "Optional general projection dependencies")
    def test_general_projection_gradient_and_failure_states(self):
        p = torch.tensor([[.9,.1],[.3,.7]], dtype=torch.float64, requires_grad=True)
        B = torch.tensor([[[-1.,0.],[0.,0.]], [[1.,0.],[-1.,0.]]], dtype=torch.float64)
        b = torch.tensor([[-.2,0.],[.8,-.2]], dtype=torch.float64)
        projected, status = GeneralAuthorityProjection()(p,B,b)
        self.assertEqual(status.tolist(), [0,2])
        torch.testing.assert_close(projected[0], torch.tensor([.2,.8], dtype=torch.float64), atol=1e-6, rtol=0)
        projected[0, 0].backward()
        self.assertTrue(torch.isfinite(p.grad).all())


class AuthorityModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_forward_backward_all_baselines_and_parameter_matching(self):
        for task in ("s1", "s2"):
            batch = AuthoritySynthetic(task, count=8).materialize()
            sizes = {}
            for variant in TRAINED_VARIANTS:
                model = AuthorityFusion(task, variant)
                output = model(batch.evidence)
                loss = authority_loss(output, batch.labels, task, variant)
                loss.backward()
                self.assertTrue(torch.isfinite(loss), variant)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), variant)
                sizes[variant] = sum(p.numel() for p in model.parameters())
            self.assertEqual(sizes["acf"], sizes["learned_joint"])
            self.assertEqual(sizes["acf"], sizes["graph_only"])

    def test_protected_states_and_positive_bypass_control(self):
        generator = AuthoritySynthetic("s1", count=30)
        base = generator.materialize()
        changed = generator.materialize(event="aux_conflict", kappa=1, strength=8)
        model = AuthorityFusion("s1", "acf").eval()
        with torch.no_grad():
            a, b = model(base.evidence), model(changed.evidence)
            torch.testing.assert_close(a.protected_states, b.protected_states, atol=1e-6, rtol=0)
            self.assertLessEqual(float(b.relation_residuals.max()), 1e-6)
            aa, bb = model(base.evidence,bypass=True), model(changed.evidence,bypass=True)
            self.assertGreater(float((aa.protected_states-bb.protected_states).abs().max()), 1e-4)
            eigenvalues = torch.linalg.eigvalsh(model.manifold_states(a))
            self.assertTrue((eigenvalues > 0).all())

    def test_per_relation_noninterference_in_multilabel_task(self):
        batch = AuthoritySynthetic("s2", count=18).materialize()
        model = AuthorityFusion("s2").eval()
        baseline = model(batch.evidence)
        for r in range(2):
            token = batch.evidence.tokens.clone()
            owner = batch.evidence.authority_sources[:, r]
            for source in range(4):
                token[owner != source, 0, source] += 8
            changed = model(dataclasses.replace(batch.evidence,tokens=token))
            torch.testing.assert_close(baseline.protected_states[:,r],changed.protected_states[:,r],atol=1e-6,rtol=0)

    def test_role_shift_preserves_clean_graph_degree_distribution(self):
        batch = AuthoritySynthetic("s2", count=10).materialize()
        model = AuthorityFusion("s2")
        c = compile_authority(batch.evidence)
        before = model._mask(c)
        model.mask_policy = "role_shift"
        after = model._mask(c)
        self.assertTrue(torch.equal(before.sum((-1,-2)),after.sum((-1,-2))))
        self.assertTrue(torch.equal(before.sum(-1).sort(-1).values,after.sum(-1).sort(-1).values))
        self.assertFalse(torch.equal(before,after))


if __name__ == "__main__":
    unittest.main()
