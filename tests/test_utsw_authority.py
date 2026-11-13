from dataclasses import replace
import numpy as np
import torch
from glioma.data.utsw_authority import (GradeEvidence, molecular_record, features, rule_active,
                                        compile_grade, fixed_folds)
from glioma.models.utsw_authority import UTSWAuthorityGrade, grade_loss
from glioma.modules.authority_projection import project_allowed_simplex
from glioma.eval.utsw_authority import prediction_frame, metrics


def record(**updates):
    row = {"IDH": "mutated", "1p19Q CODEL": "co-deleted", "MGMT": "NA",
           "IDH_Testing Method": "NGS", "Tumor Type": "OLIGODENDROGLIOMA GRADE 2"}
    row.update(updates)
    return molecular_record(row)


def example():
    return GradeEvidence(torch.rand(2,4,7,96,96), torch.tensor([features(record())]*2),
                         torch.ones(2,5,dtype=torch.bool), torch.tensor([True,False]))


def test_label_and_raw_diagnosis_excluded():
    a = record(**{"Tumor Grade":"2", "Pathology Information":"grade 2"})
    b = record(**{"Tumor Grade":"4", "Pathology Information":"grade 4"})
    assert a == b and features(a) == features(b) and rule_active(a) == rule_active(b)
    assert "Tumor Type" not in a and "Tumor Grade" not in a
    assert features(record(**{"Tumor Type":"OLIGODENDROGLIOMA, ANAPLASTIC"})) == features(a)


def test_missing_uncertain_invalid_and_out_of_scope():
    assert rule_active(record())
    for key in ("IDH", "1p19Q CODEL", "IDH_Testing Method"):
        assert not rule_active(record(**{key:"NA"}))
        assert not rule_active(record(**{key:"uncertain"}))
    assert not rule_active(record(**{"Tumor Type":"OTHER"}))
    r = record()
    r["assay_validity"] = "invalid"
    assert not rule_active(r)
    r = record()
    r["curated_record_accepted"] = False
    assert not rule_active(r)
    e = example()
    assert compile_grade(e)["active"].tolist() == [True, False]
    withdrawn = e.molecular.clone()
    withdrawn[:, 9:12] = torch.tensor([0.,0.,1.])
    assert not compile_grade(replace(e, molecular=withdrawn))["active"].any()
    compiled = compile_grade(replace(e, available=torch.zeros_like(e.available)))
    assert (compiled["status"] == 3).all() and not compiled["active"].any()


def test_projection_and_reference_conflict_retained():
    p0 = torch.tensor([[.1,.2,.7], [.2,.3,.5]])
    allowed = torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    p = project_allowed_simplex(p0,allowed)
    assert torch.all(p>=0) and torch.allclose(p.sum(1),torch.ones(2)) and p[0,2] == 0
    f = prediction_frame(["a","b"], [2,1], p0.numpy(),p.numpy(),[1,0],[0,1],0,"test")
    m = metrics(f)
    assert m["n"] == 2 and m["reference_conflict_n"] == 1 and m["residual_n"] == 0
    assert m["avr"] == 0 and m["avr_raw"] == 1
    assert np.isclose(m["brier"], ((p.numpy()-np.eye(3)[[2,1]])**2).mean())
    f["active"] = False
    assert metrics(f)["avr"] is None


def test_fixed_patient_split():
    rows = [dict(subject_id=f"p{i}",label=i%3,active=bool(i%2)) for i in range(90)]
    folds = fixed_folds(rows)
    assert folds == fixed_folds(rows)
    for fold in range(5):
        test = {s for s,f in folds.items() if f == fold}
        val = {s for s,f in folds.items() if f == (fold+1)%5}
        train = set(folds)-test-val
        assert len(train|test|val)==90 and not test&val and not train&(test|val)


def test_protected_state_and_positive_leakage_control():
    torch.manual_seed(101)
    torch.set_num_threads(2)
    e = example()
    model = UTSWAuthorityGrade().eval()
    joint = UTSWAuthorityGrade("learned_joint").eval()
    joint.load_state_dict(model.state_dict())
    changed = replace(e, images=torch.rand_like(e.images))
    with torch.no_grad():
        a, b = model(e), model(changed)
        j, k = joint(e), joint(changed)
        bad, bad_changed = model(e,bypass=True), model(changed,bypass=True)
    assert torch.equal(a["designated_state"][0], b["designated_state"][0])
    assert (j["designated_state"][0]-k["designated_state"][0]).abs().max() > 1e-7
    assert (bad["designated_state"][0]-bad_changed["designated_state"][0]).abs().max() > 1e-7
    assert torch.equal(a["allowed"],j["allowed"]) and torch.equal(a["active"],j["active"])
    assert not a["mask"][0,0,4,:4].any() and j["mask"][0,0,4,:4].all()
    assert torch.equal(a["node_states"][:,:,5:], b["node_states"][:,:,5:])
    assert torch.allclose(a["p_star"].sum(1),torch.ones(2)) and a["p_star"][0,2]==0
    model.train()
    loss = grade_loss(model(e), torch.tensor([0,2]))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoders[0].net[0].weight.grad.abs().sum() > 0


def test_case_selection_ignores_baseline_advantage():
    from glioma.visualization.utsw_authority import choose_case
    import pandas as pd
    rows=[]
    for sid, brier in (("A",.01),("B",.03),("C",.07)):
        rows.append(dict(subject_id=sid,model="acf",active=True,image_complete=True,
                         predicted_grade=2,reference_grade=2,brier=brier))
        rows.append(dict(subject_id=sid,model="learned_joint",active=True,image_complete=True,
                         predicted_grade=4 if sid=="A" else 2,reference_grade=2,brier=.6 if sid=="A" else .01))
    assert choose_case(pd.DataFrame(rows)) == ("B",True)


def test_training_resume_and_complete_export_pipeline(tmp_path, monkeypatch):
    """Synthetic I/O fixture only: exercises actual training/checkpoints and final export gates."""
    import json
    from argparse import Namespace
    import pandas as pd
    import pytest
    from glioma.cli import utsw_authority as cli

    class TinyFixtureModel(torch.nn.Module):
        def __init__(self, variant):
            super().__init__()
            self.variant=variant
            self.scale=torch.nn.Parameter(torch.tensor(6.))
        def forward(self,evidence,bypass=False):
            c=compile_grade(evidence,self.variant)
            p0=(evidence.images[:,:3].mean((2,3,4))*self.scale).softmax(1)
            return dict(p0=p0,p_star=project_allowed_simplex(p0,c["allowed"]),
                        designated_state=evidence.molecular,adjacency=c["mask"].float(),**c)

    monkeypatch.setattr(cli,"UTSWAuthorityGrade",TinyFixtureModel)
    rows=[]
    (tmp_path/"cache").mkdir()
    for i in range(60):
        r=record(**{"IDH":"mutated" if i%2==0 and i%3!=2 else "wild type"})
        sid=f"SYNTHETIC{i:03d}"
        row=dict(subject_id=sid,label=i%3,record=r,features=features(r),active=rule_active(r))
        rows.append(row)
        images=np.zeros((4,7,8,8),np.float32)
        images[i%3]=1
        np.savez(tmp_path/"cache"/f"{sid}.npz",images=images,masks=np.zeros((3,7,8,8)),
                 provenance=json.dumps(dict(subject_id=sid,z_ids=list(range(7)),use="SYNTHETIC TEST ONLY")))
    folds=fixed_folds(rows)
    cli.save_json(tmp_path/"manifest.json",dict(rows=rows,folds=folds,source_hash=cli.source_hash()))
    cli.save_json(tmp_path/"image_audit.json",dict(complete=True,errors=[]))
    cli.save_json(tmp_path/"smoke.json",dict(passed=True,use="SYNTHETIC TEST FIXTURE"))
    for fold in range(5):
        cli.save_json(tmp_path/"splits"/f"fold{fold}.json",{
            "train":[s for s,f in folds.items() if f not in (fold,(fold+1)%5)],
            "validation":[s for s,f in folds.items() if f==(fold+1)%5],
            "test":[s for s,f in folds.items() if f==fold]})
    args=Namespace(output=str(tmp_path),fold=0,variant="acf",device="cpu",bootstrap=10)
    real_predict=cli.predict
    calls=[0]
    def interrupted(*a,**kw):
        calls[0]+=1
        if calls[0]==2:
            raise RuntimeError("simulated interruption")
        return real_predict(*a,**kw)
    monkeypatch.setattr(cli,"predict",interrupted)
    with pytest.raises(RuntimeError,match="simulated interruption"):
        cli.train(args)
    state=torch.load(tmp_path/"fold0"/"acf"/"last.pt",weights_only=False)
    assert state["epoch"]==0
    monkeypatch.setattr(cli,"predict",real_predict)
    for fold in range(5):
        for variant in ("acf","learned_joint"):
            args.fold,args.variant=fold,variant
            cli.train(args)
    assert len(json.loads((tmp_path/"fold0"/"acf"/"history.json").read_text()))==50
    cli.cheap_controls(args)
    cli.aggregate(args)
    frame=pd.read_csv(tmp_path/"oof_predictions.csv")
    assert len(frame)==60*5 and frame.groupby("model").subject_id.nunique().min()==60
    assert (tmp_path/"utsw_authority_case.pdf").stat().st_size>1000
    provenance=json.loads((tmp_path/"figure_provenance.json").read_text())
    assert provenance["subject_id"].startswith("SYNTHETIC")
    from glioma.cli.package_utsw_authority import package_results
    package_results(tmp_path)
    assert (tmp_path/"results_for_paper.zip").exists()
