"""Paired authority evaluation. Events are observations, never independent runs."""

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from glioma.data.authority_benchmarks import intervention_grid
from glioma.models.authority_fusion import STATUS_NAMES
from glioma.modules.authority_rules import relation_residuals


ARTIFACT_LEVELS = ("summary", "audit", "full")


def _atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _require_free_space(path, minimum_free_gb):
    if minimum_free_gb <= 0:
        return
    free = shutil.disk_usage(path).free
    required = int(minimum_free_gb * 1024 ** 3)
    if free < required:
        raise OSError(f"Disk guard: {free / 1024**3:.2f} GiB free below required {minimum_free_gb:g} GiB at {path}")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, batch, batch_size=256, device="cpu", bypass=False):
    """CPU arrays retain raw scores on abstained rows; use status to select predictions."""
    model.eval()
    collected = {}
    synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(batch.labels), batch_size):
        part = batch.take(list(range(start, min(start + batch_size, len(batch.labels))))).to(device)
        out = model(part.evidence, bypass=bypass)
        c = out.compiled
        arrays = dict(p0=out.p0, p_star=out.p_star, probabilities=out.probabilities,
                      status=out.status, protected=out.protected_states, protected_active=out.protected_active,
                      residual=out.relation_residuals, residual_p0=relation_residuals(out.p0, c),
                      residual_projected=relation_residuals(out.p_star, c), active=c.relation_active,
                      B=c.B, b=c.b, constraint_active=c.constraint_active,
                      authorized_sources=c.authorized_sources, directions=c.directions,
                      # Degree diagnostics explicitly include changes at each node.
                      in_degree=model._mask(c).sum(-1), out_degree=model._mask(c).sum(-2))
        for key, value in arrays.items():
            collected.setdefault(key, []).append(value.detach().cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = {key: np.concatenate(values) for key, values in collected.items()}
    result["seconds_per_sample"] = elapsed / len(batch.labels)
    return result


def metrics(result, batch, projection=False):
    p = result["p_star"] if projection else result["probabilities"]
    residual = result["residual_projected"] if projection else result["residual"]
    status, active = result["status"], result["active"]
    accepted = status < 2
    active_evaluated = active & accepted[:, None]
    labels = batch.labels.numpy()
    target = np.eye(4)[labels] if batch.evidence.task == "s1" else labels
    errors = ((p - target)**2).mean(-1)
    pred_n, denominator = int(accepted.sum()), int(active_evaluated.sum())
    violation_n = int(((residual > 1e-6) & active_evaluated).sum())
    eligible_samples = accepted & active.any(-1)
    sample_denominator = int(eligible_samples.sum())
    out = dict(n=len(p), predicted_n=pred_n, prediction_coverage=float(accepted.mean()),
               active_relation_n=int(active.sum()), avr_denominator=denominator, avr_numerator=violation_n,
               avr=violation_n / denominator if denominator else None,
               sample_avr_denominator=sample_denominator,
               sample_avr=float((residual[eligible_samples].max(-1) > 1e-6).mean()) if sample_denominator else None,
               brier=float(errors[accepted].mean()) if pred_n else None,
               max_constraint_residual=float(residual[active_evaluated].max()) if denominator else None,
               projection_mse=float(((result["p_star"][accepted] - result["p0"][accepted])**2).mean()) if pred_n else None,
               inference_ms_per_sample=float(result["seconds_per_sample"]*1000),
               status_counts={name: int((status == code).sum()) for code, name in STATUS_NAMES.items()})
    if batch.evidence.task == "s1":
        fine_p = p[:, 1] + p[:, 3]
        fine_y = labels % 2
        out.update(accuracy=float((p[accepted].argmax(-1) == labels[accepted]).mean()) if pred_n else None,
                   residual_accuracy=float(((fine_p[accepted] >= .5) == fine_y[accepted]).mean()) if pred_n else None,
                   residual_brier=float(((fine_p[accepted] - fine_y[accepted])**2).mean()) if pred_n else None)
        confidence = p.max(-1)
    else:
        out.update(accuracy=float(((p[accepted] >= .5) == labels[accepted]).mean()) if pred_n else None,
                   residual_accuracy=float(((p[accepted, 4:] >= .5) == labels[accepted, 4:]).mean()) if pred_n else None,
                   residual_brier=float(((p[accepted, 4:] - labels[accepted, 4:])**2).mean()) if pred_n else None,
                   latent_probability_brier=float(((p[accepted] - batch.latent_probabilities.numpy()[accepted])**2).mean()) if pred_n else None)
        confidence = np.abs(p - .5).mean(-1)
    ranked = np.flatnonzero(accepted)[np.argsort(-confidence[accepted], kind="stable")]
    # Coverage is relative to all inputs, not just the non-abstained subset.
    for coverage in (.5, .8, .95, 1.):
        take = max(1, int(np.ceil(coverage * len(p))))
        out[f"risk_at_coverage_{coverage:g}"] = float(errors[ranked[:take]].mean()) if len(ranked) >= take else None
    return out


def paired_shift(before, after, base_batch, changed_batch):
    same_policy = ((before["authorized_sources"] == after["authorized_sources"]) &
                   (before["directions"] == after["directions"]))
    same_target = (base_batch.evidence.target_id == changed_batch.evidence.target_id).numpy()
    eligible = (same_policy & same_target[:, None] & before["protected_active"] & after["protected_active"] &
                (before["status"] < 2)[:, None] & (after["status"] < 2)[:, None])
    delta = np.abs(after["protected"] - before["protected"]).max(-1)
    return delta, eligible


def save_event(path, before, after, base, changed, config, parameters):
    """Compact lossless arrays plus JSON metadata; row order shares stable sample/event IDs."""
    path = Path(path)
    delta, eligible = paired_shift(before, after, base, changed)
    arrays = {"before_" + key: value for key, value in before.items() if isinstance(value, np.ndarray)}
    arrays.update({"after_" + key: value for key, value in after.items() if isinstance(value, np.ndarray)})
    arrays.update(sample_ids=np.asarray(changed.sample_ids), event_ids=np.asarray(changed.event_ids),
                  labels_before=base.labels.numpy(), labels_after=changed.labels.numpy(),
                  latent_probabilities=changed.latent_probabilities.numpy(), protected_shift=delta, shift_eligible=eligible,
                  target_before=base.evidence.target_id.numpy(), target_after=changed.evidence.target_id.numpy())
    for name in ("source_available", "verified_validity", "source_quality", "statements"):
        arrays["before_" + name] = getattr(base.evidence, name).numpy()
        arrays["after_" + name] = getattr(changed.evidence, name).numpy()
    np.savez_compressed(str(path) + ".npz", **arrays)
    metadata = dict(data_seed=config["data_seed"], model_seed=config["model_seed"], intervention_seed=config["data_seed"],
                    source_ids=[0, 1, 2, 3],
                    intervention_rng="Coupled named SeedSequence streams 0..7, split=test; no independent resampling",
                    rule_version=changed.evidence.rule_version, event=parameters, config_hash=config.get("config_hash"),
                    source_hash=config.get("source_hash"),
                    before_seconds_per_sample=before["seconds_per_sample"], after_seconds_per_sample=after["seconds_per_sample"],
                    status_names=STATUS_NAMES, abstention="status >= 2; raw scores are not returned predictions",
                    shift_eligibility="Same task, authorized owner and relation direction, both active and predictable")
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_audit_base(path, before, base):
    """Save clean predictions once; intervention files only contain changed outputs."""
    arrays = {key: before[key] for key in
              ("p0", "p_star", "probabilities", "status", "protected", "protected_active", "residual", "active")}
    arrays.update(sample_ids=np.asarray(base.sample_ids), labels=base.labels.numpy(),
                  target=base.evidence.target_id.numpy())
    np.savez_compressed(path, **arrays)


def save_audit_event(path, before, after, base, changed, include_reference=False):
    delta, eligible = paired_shift(before, after, base, changed)
    arrays = {"after_" + key: after[key] for key in
              ("p0", "p_star", "probabilities", "status", "protected", "protected_active", "residual", "active")}
    arrays.update(event_ids=np.asarray(changed.event_ids), labels_after=changed.labels.numpy(),
                  target_after=changed.evidence.target_id.numpy(), protected_shift=delta, shift_eligible=eligible)
    if include_reference:
        arrays.update({"reference_" + key: before[key] for key in
                       ("probabilities", "status", "protected", "protected_active")})
    np.savez_compressed(path, **arrays)


def evaluate_suite(model, generator, directory, config, batch_size=256, device="cpu", full=True,
                   artifact_level="summary", minimum_free_gb=0):
    if artifact_level not in ARTIFACT_LEVELS:
        raise ValueError(f"Unknown artifact level: {artifact_level}; expected one of {ARTIFACT_LEVELS}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _require_free_space(directory, minimum_free_gb)
    base = generator.materialize()
    before = predict(model, base, batch_size, device)
    if artifact_level == "audit":
        save_audit_base(directory / "base.npz", before, base)
    summary = {}
    events = [("clean", {})] + (intervention_grid(generator.task) if full else [])
    for name, parameters in events:
        changed = base if name == "clean" else generator.materialize(**parameters)
        reference, reference_batch = before, base
        if name == "bypass":
            # Both forwards use the same deliberately leaky architecture and fixed parameters.
            reference = predict(model, base, batch_size, device, bypass=True)
            if generator.task == "s1":
                changed = generator.materialize(event="bypass", kappa=1, strength=8)
            else:
                tokens = changed.evidence.tokens.clone()
                owners = changed.evidence.authority_sources
                for source in range(4):
                    auxiliary = (owners != source).all(-1)
                    tokens[auxiliary, 0, source] += 8
                changed = replace(changed, evidence=replace(changed.evidence, tokens=tokens))
            after = predict(model, changed, batch_size, device, bypass=True)
        else:
            after = before if name == "clean" else predict(model, changed, batch_size, device)
        delta, eligible = paired_shift(reference, after, reference_batch, changed)
        report = metrics(after, changed)
        report.update(protected_shift_n=int(eligible.sum()),
                      protected_shift_max=float(delta[eligible].max()) if eligible.any() else None,
                      protected_shift_mean=float(delta[eligible].mean()) if eligible.any() else None,
                      intervention_parameters=parameters)
        if model.variant in ("learned", "transformer", "graph_only"):
            report["posthoc_projection"] = metrics(after, changed, projection=True)
        summary[name] = report
        if artifact_level != "summary":
            _require_free_space(directory, minimum_free_gb)
        if artifact_level == "full":
            save_event(directory / name, reference, after, reference_batch, changed, config, parameters)
        elif artifact_level == "audit" and name != "clean":
            save_audit_event(directory / f"{name}.npz", reference, after, reference_batch, changed,
                             include_reference=name == "bypass")
    _atomic_json(directory / "summary.json", summary)
    return summary
