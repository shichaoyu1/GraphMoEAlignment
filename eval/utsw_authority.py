"""Recomputable patient-level classification and authority metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def metrics(frame):
    n = len(frame)
    if not n:
        return {"n": 0}
    accepted = frame.status.to_numpy() < 2
    f = frame.loc[accepted]
    y = f.label.to_numpy(dtype=int)
    p = f[[f"p{g}" for g in (2, 3, 4)]].to_numpy(float)
    p0 = f[[f"raw_p{g}" for g in (2, 3, 4)]].to_numpy(float)
    active = f.active.to_numpy(bool)
    pred = p.argmax(1) if len(p) else np.array([], dtype=int)
    result = {"n": n, "prediction_n": len(f), "prediction_coverage": float(accepted.mean()),
              "active_n": int(active.sum()), "activation_coverage": float(frame.active.mean()),
              "reference_conflict_n": int((active & (y == 2)).sum())}
    if not len(f):
        return {**result, "brier": None, "accuracy": None, "avr": None}
    recalls = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    result.update(brier=float(((p - np.eye(3)[y]) ** 2).mean()), accuracy=float((pred == y).mean()),
                  macro_f1=float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                  balanced_accuracy=float(recalls[np.unique(y)].mean()),
                  recall_grade2=float(recalls[0]), recall_grade3=float(recalls[1]), recall_grade4=float(recalls[2]),
                  projection_mse=float(((p-p0)**2).mean()),
                  avr=float((p[active, 2] > 1e-6).mean()) if active.any() else None,
                  avr_raw=float((p0[active, 2] > 1e-6).mean()) if active.any() else None,
                  confusion_matrix=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())
    residual = active & (y < 2)
    result["residual_n"] = int(residual.sum())
    q = p[residual, :2]
    q = q / q.sum(1, keepdims=True).clip(1e-12)
    result["residual_brier"] = float(((q-np.eye(2)[y[residual]])**2).mean()) if residual.any() else None
    result["residual_accuracy"] = float((q.argmax(1)==y[residual]).mean()) if residual.any() else None
    for intervention in ("noise", "occlusion"):
        key = f"{intervention}_shift"
        if key in f:
            shifts = f.loc[active, key].dropna().to_numpy()
            result[f"{key}_n"] = len(shifts)
            result[f"{key}_mean"] = float(shifts.mean()) if len(shifts) else None
            result[f"{key}_max"] = float(shifts.max()) if len(shifts) else None
    return result


def paired_bootstrap(left, right, repetitions=2000):
    """Patient bootstrap of ACF minus comparator; no repeated training-seed claim."""
    a, b = left.sort_values("subject_id"), right.sort_values("subject_id")
    if a.subject_id.tolist() != b.subject_id.tolist() or a.label.tolist() != b.label.tolist():
        raise ValueError("Unmatched paired predictions")
    rng = np.random.default_rng(2026)
    keys = ["brier", "accuracy", "residual_brier", "residual_accuracy", "avr"]
    samples = {key: [] for key in keys}
    for _ in range(repetitions):
        indices = rng.integers(0, len(a), len(a))
        ma, mb = metrics(a.iloc[indices]), metrics(b.iloc[indices])
        for key in keys:
            if ma.get(key) is not None and mb.get(key) is not None:
                samples[key].append(ma[key] - mb[key])
    ma, mb = metrics(a), metrics(b)
    return {key: dict(difference=ma[key]-mb[key], ci95=np.quantile(values, [.025, .975]).tolist(),
                      bootstrap_n=len(values)) for key, values in samples.items() if values}


def prediction_frame(ids, labels, p0, p, active, status, fold, variant):
    frame = pd.DataFrame(dict(subject_id=ids, label=np.asarray(labels), reference_grade=np.asarray(labels)+2,
                              active=np.asarray(active).astype(bool), status=np.asarray(status), fold=fold,
                              model=variant, image_complete=True))
    for index, grade in enumerate((2, 3, 4)):
        frame[f"raw_p{grade}"] = p0[:, index]
        frame[f"p{grade}"] = p[:, index]
    frame["predicted_grade"] = p.argmax(1) + 2
    frame["brier"] = ((p-np.eye(3)[np.asarray(labels)])**2).mean(1)
    frame["allowed_grades"] = np.where(frame.active, "2|3", "2|3|4")
    frame["reference_conflict"] = frame.active & (frame.label == 2)
    return frame
