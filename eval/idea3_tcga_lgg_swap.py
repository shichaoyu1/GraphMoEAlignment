"""Prespecified TCGA sensitivities on the scored five-fit external ensemble."""

from collections import Counter

from glioma.data.tcga_lgg_grade import BUNDLE_SHA256
from glioma.eval.idea3_rule_mri_swap import patient_results, summarize


def subset_results(frame, strata):
    _, patients = patient_results(frame.assign(fold=frame.subject_id.map(strata)))
    report = summarize(patients)
    effects = {model: {key: metrics["own_advantage"][key] for key in ("n", "estimate", "ci95")}
               for model, metrics in report["models"].items()}
    return dict(n_patients=report["n_patients"],
                n_directed_swaps=report["models"]["read_only_residual"]["n_pairs"],
                models=effects, read_only_by_stratum=report["models"]["read_only_residual"]["own_advantage"]["by_fold"])


def tcga_diagnostics(ensemble, patients, rows):
    sites = {r["subject_id"]: r["tss"] for r in rows}
    eligible_sites = {site for site, count in Counter(sites.values()).items() if count > 1}
    site = ensemble.subject_id.map(sites)
    matched = ensemble[site.isin(eligible_sites) & site.eq(ensemble.image_subject_id.map(sites))]
    sensitivity = {"within_tissue_source_site": subset_results(matched, sites)}

    discordant = "TCGA-DU-8164"
    if discordant in sites:
        sub = ensemble[ensemble.subject_id.ne(discordant) & ensemble.image_subject_id.ne(discordant)]
        sensitivity["without_DU_8164"] = subset_results(sub, {i: "TCGA-LGG" for i in sites})
    clinical = {r["subject_id"] for r in rows if r["record"]["IDH_Testing Method"] == "IHC"}
    observed = patients[(patients.model == "read_only_residual") & patients.subject_id.isin(clinical)]
    return dict(input_sha256=BUNDLE_SHA256, tissue_source_sites=sites, sensitivity=sensitivity,
                idh_method_sources=dict(Counter(r["idh_source"] for r in rows)),
                clinical_ihc_with_full_donor_pool=observed[
                    ["subject_id", "n_donors", "own_brier", "donor_brier", "own_advantage"]
                ].to_dict("records"))
