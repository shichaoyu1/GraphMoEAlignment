import numpy as np
import pandas as pd
import pytest

from glioma.data.tcga_lgg_grade import prepare_tcga
from glioma.eval.idea3_rule_mri_swap import MODELS, patient_results
from glioma.eval.idea3_tcga_lgg_swap import tcga_diagnostics


def test_swaps_respect_site_and_remove_discordant_patient_from_both_roles():
    ids = ["a", "b", "c", "d", "TCGA-DU-8164", "singleton"]
    sites = ["CS", "CS", "DU", "DU", "DU", "HT"]
    labels = [1, 0, 1, 0, 1, 1]
    p3 = np.array([.2, .5, .7, .85, .15, .55])
    rows = [dict(subject_id=sid, tss=site, idh_source="IHC" if i == 4 else "research exome",
                 record={"IDH_Testing Method": "IHC" if i == 4 else "NGS"})
            for i, (sid, site) in enumerate(zip(ids, sites))]
    frame = pd.DataFrame([
        dict(fold="external", model=model, subject_id=sid, image_subject_id=donor, label=labels[i],
             reference_grade=labels[i]+2, p2=1-p3[j], p3=p3[j], p4=0., raw_p4=.3)
        for model in MODELS for i, sid in enumerate(ids) for j, donor in enumerate(ids)
    ])
    ensemble, patients = patient_results(frame)
    report = tcga_diagnostics(ensemble, patients, rows)
    same_site = report["sensitivity"]["within_tissue_source_site"]
    assert (same_site["n_patients"], same_site["n_directed_swaps"]) == (5, 8)
    reduced = report["sensitivity"]["without_DU_8164"]
    assert (reduced["n_patients"], reduced["n_directed_swaps"]) == (5, 20)
    errors = (p3[None, :] - np.asarray(labels)[:, None])**2
    retained = [0, 1, 2, 3, 5]
    for result, donors in [(same_site, {0: [1], 1: [0], 2: [3, 4], 3: [2, 4], 4: [2, 3]}),
                           (reduced, {i: [j for j in retained if j != i] for i in retained})]:
        expected = np.mean([errors[i, js].mean() - errors[i, i] for i, js in donors.items()])
        assert result["models"]["read_only_residual"]["estimate"] == pytest.approx(expected)
    clinical, = report["clinical_ihc_with_full_donor_pool"]
    assert (clinical["subject_id"], clinical["n_donors"]) == ("TCGA-DU-8164", 5)


def test_wrong_input_bundle_is_rejected_before_cache_creation(tmp_path):
    bundle = tmp_path / "tcga.zip"
    bundle.write_bytes(b"partial upload")
    with pytest.raises(ValueError, match="audited 13-case release"):
        prepare_tcga(bundle, tmp_path, smoke=False)
    assert not (tmp_path / "cache").exists()
