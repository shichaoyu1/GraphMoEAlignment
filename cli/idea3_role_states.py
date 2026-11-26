"""Evidence interventions and frozen report readouts for the 1p/19q three-state study."""

from dataclasses import replace
from functools import partial

import torch
from torch.func import functional_call

from glioma.data.utsw_authority import FIELDS, GradeEvidence, compile_grade, features, rule_active
from glioma.modules.authority_projection import project_allowed_simplex


TARGET_FIELD = "1p19Q CODEL"
SCOPE_INDEX = 12
STATES = ("O", "C", "A")


def build_state(batch, records, state, device, hide_mgmt=False):
    """Build fresh evidence from original records ordered as in the image batch.

    O hides the target, C exposes it as a hint and A permits the legacy rule.
    The final molecular slot carries scope to the compiler. Callers must use
    scopeless_models for study predictions, including A and its donor swaps.
    """
    if state not in STATES:
        raise ValueError(f"Unknown 1p/19q role {state!r}; expected one of {STATES}")
    if len(records) != len(batch["images"]) or not records:
        raise ValueError("Expected one original report per image")

    molecular, active = [], []
    for original in records:
        visible = dict(original)
        if state == "O" or visible["assay_validity"] == "invalid":
            visible[TARGET_FIELD] = ""
        if hide_mgmt:
            visible["MGMT"] = ""
        molecular.append(features(visible))
        active.append(state == "A" and rule_active(visible))
    # Keep the original report container available in O/C even if the target
    # was its only measured field. Availability conveys no target value.
    available = batch["available"].to(device)
    archived_report_available = [
        any(record[field] in levels for field, levels in FIELDS.items()) for record in records
    ]
    if not torch.equal(available[:, 4], torch.tensor(archived_report_available, device=device)):
        raise ValueError("Report availability differs between original records and image batch")
    return GradeEvidence(
        images=batch["images"].to(device),
        molecular=torch.tensor(molecular, dtype=torch.float32, device=device),
        available=available,
        active=torch.tensor(active, dtype=torch.bool, device=device),
    )


def _molecular_prediction(predict, evidence):
    # The archived logistic predictor consumes all 13 features. Its raw
    # probabilities are independent of the rule. Project the masked readout
    # with the genuine eligibility of the state under evaluation.
    readout_input = evidence.molecular.clone()
    readout_input[:, SCOPE_INDEX] = 0
    raw = predict(replace(evidence, molecular=readout_input))["p0"]
    rule = compile_grade(evidence)
    return {"p0": raw, "p_star": project_allowed_simplex(raw, rule["allowed"]), **rule}


def scopeless_models(models):
    """Keep the scope bit in the compiler only across all three interventions.

    In the archived neural models the report's first affine map is the only
    learned use of the molecular vector. A zero scope column is equivalent
    to supplying zero at that input; all other weights and the original rule
    and graph-update implementations are reused. functional_call applies
    this substitution only during a study prediction.
    """
    names = {"learned_joint": "report.weight", "acf": "report.weight",
             "read_only_residual": "report.0.weight"}
    if set(models) != set(names) | {"molecular_projected"}:
        raise ValueError("The three-state study requires all four archived grading models")
    result = {"molecular_projected": partial(_molecular_prediction, models["molecular_projected"])}
    for key, report_weight in names.items():
        model = models[key]
        if model.training:
            raise ValueError(f"Expected a frozen model in evaluation mode for {key}")
        masked = model.get_parameter(report_weight).detach().clone()
        if masked.ndim != 2 or masked.shape[1] != SCOPE_INDEX + 1:
            raise ValueError(f"Unexpected first report layer for {key}")
        masked[:, SCOPE_INDEX] = 0
        result[key] = partial(_predict_with_weights, model, {report_weight: masked})
    return result


def _predict_with_weights(model, weights, evidence):
    return functional_call(model, weights, (evidence,))
