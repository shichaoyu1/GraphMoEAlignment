import torch

from glioma.cli.idea3_role_states import TARGET_FIELD, build_state, scopeless_models
from glioma.data.utsw_authority import compile_grade
from glioma.models.idea3 import ReadOnlyGradeResidual
from glioma.models.utsw_authority import UTSWAuthorityGrade
from glioma.modules.authority_projection import project_allowed_simplex


def report(**overrides):
    return {"IDH": "mutated", TARGET_FIELD: "co-deleted", "IDH_Testing Method": "NGS",
            "MGMT": "methylated", "scope_eligible": True, "curated_record_accepted": True,
            "assay_validity": "unreported", **overrides}


def batch(n):
    # Include deliberately stale archive inputs. Transformations must rebuild
    # from the permitted fields regardless of the old features or activation.
    return {"images": torch.ones(n, 4, 7, 16, 16), "available": torch.ones(n, 5, dtype=torch.bool),
            "molecular": torch.zeros(n, 13), "active": torch.zeros(n, dtype=torch.bool)}


def test_qualification_and_revocation_from_visible_evidence():
    records = [report(), report(scope_eligible=False), report(**{TARGET_FIELD: "non co-deleted"}),
               report(curated_record_accepted=False), report(assay_validity="invalid")]
    inputs = batch(len(records))
    outputs = [build_state(inputs, records, stage, "cpu") for stage in ("C", "A", "C", "O", "A")]
    c, a, restored_c, absent, restored_a = outputs
    assert c.molecular.equal(a.molecular)
    assert c.molecular.equal(restored_c.molecular)
    assert c.active.equal(restored_c.active)
    assert a.active.equal(restored_a.active)
    assert not c.active.any() and not absent.active.any()
    assert compile_grade(a)["active"].tolist() == [True, False, False, False, False]
    assert torch.equal(compile_grade(a)["allowed"],
                       compile_grade(build_state(inputs, records, "A", "cpu", hide_mgmt=True))["allowed"])
    assert c.molecular[-1, 3:6].tolist() == [0, 0, 1]  # Invalid data are unavailable.
    assert (absent.molecular[:, 5] == 1).all()
    assert absent.available.equal(c.available)
    assert records[0][TARGET_FIELD] == "co-deleted"
    assert not inputs["active"].any() and not inputs["molecular"].any()


def molecular_probe(evidence):
    rule = compile_grade(evidence)
    raw = torch.stack([evidence.molecular[:, 12] * 4, evidence.molecular[:, 4],
                       evidence.molecular[:, 1]], dim=1).softmax(1)
    return {"p0": raw, "p_star": project_allowed_simplex(raw, rule["allowed"]), **rule}


@torch.inference_mode()
def test_scope_cannot_reach_any_soft_readout_and_legacy_model_is_restored():
    records = [report(), report(scope_eligible=False), report(scope_eligible=False,
                                                              **{TARGET_FIELD: "non co-deleted"})]
    inputs = batch(len(records))
    models = {"molecular_projected": molecular_probe, "acf": UTSWAuthorityGrade("acf").eval(),
              "learned_joint": UTSWAuthorityGrade("learned_joint").eval(),
              "read_only_residual": ReadOnlyGradeResidual().eval()}
    report_weights = {name: (model.report[0] if name == "read_only_residual" else model.report).weight
                      for name, model in models.items() if name != "molecular_projected"}
    before = {name: weight.detach().clone() for name, weight in report_weights.items()}
    masked = scopeless_models(models)
    a, c, absent = [build_state(inputs, records, stage, "cpu") for stage in ("A", "C", "O")]
    original = models["acf"](c)["p0"]
    for predict in masked.values():
        for evidence, same_rows in ((c, 2), (absent, 3)):
            out = predict(evidence)
            assert not out["active"].any()
            for key in ("p0", "p_star"):
                torch.testing.assert_close(out[key][:same_rows], out[key][:1].expand(same_rows, -1))
        out = predict(a)
        assert out["active"].tolist() == [True, False, False]
        assert out["p_star"][0, 2].abs().item() <= 1e-6
    torch.testing.assert_close(models["acf"](c)["p0"], original, atol=0, rtol=0)
    for name, weight in report_weights.items():
        torch.testing.assert_close(weight, before[name], atol=0, rtol=0)
    # The target itself remains visible in C even while the scope proxy is hidden.
    c_raw = masked["molecular_projected"](c)["p0"]
    assert not torch.equal(c_raw[1], c_raw[2])
