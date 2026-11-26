"""Audit codeletion proxies and frozen-model input support before role interventions."""

from collections import Counter, defaultdict

from glioma.data.utsw_authority import FIELDS, features, rule_active


TARGET = "1p19Q CODEL"
OTHER_FIELDS = tuple(field for field in FIELDS if field != TARGET)


def _target_table(rows, field_value):
    table = defaultdict(lambda: {"co-deleted": 0, "non co-deleted": 0})
    for row in rows:
        value = field_value(row)
        key = str(value).lower() if isinstance(value, bool) else str(value)
        table[key][row["record"][TARGET]] += 1
    return dict(sorted(table.items()))


def _support_by_fold(labeled, measured, folds):
    """Count held-out reports whose ablated vector and policy occurred during training."""
    result = {}
    for fold in range(5):
        train = [r for r in labeled if folds[r["subject_id"]] not in (fold, (fold + 1) % 5)]
        test = [r for r in measured if folds[r["subject_id"]] == fold]
        archive_inputs = Counter()
        for row in train:
            record = row["record"]
            available = any(record[k] in values for k, values in FIELDS.items())
            archive_inputs[tuple(row["features"]), available, row["active"]] += 1
        by_state = {}
        for state in ("absent", "complementary"):
            counts = []
            for row in test:
                record = dict(row["record"], scope_eligible=False)
                if state == "absent":
                    record[TARGET] = ""
                # Measured subjects have a visible report in both paired conditions.
                key = tuple(features(record)), True, False
                counts.append(archive_inputs[key])
            by_state[state] = {
                "n_test": len(test),
                "with_exact_train_match": sum(n > 0 for n in counts),
                "without_exact_train_match": counts.count(0),
                "min_train_matches": min(counts, default=None),
            }
        result[str(fold)] = {"n_original_train": len(train), **by_state}
    return result


def audit_leakage(manifest):
    """Verify archived duplicate encodings and return aggregate counts without patient IDs."""
    rows = manifest["rows"]
    if len({r["subject_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate subject in authority manifest")
    for row in rows:
        record = row["record"]
        positive = record[TARGET] == "co-deleted"
        candidate = positive and record["IDH"] == "mutated"
        active = rule_active(record)
        if (row["features"] != features(record) or active != row["active"]
                or candidate != row["molecular_candidate"]
                or (active and row["label"] == 2) != row["reference_conflict"]):
            raise ValueError(f"Archived evidence disagrees with the source record for {row['subject_id']}")
    labeled = [r for r in rows if r["label"] is not None]
    folds = manifest["folds"]
    if (set(folds) != {r["subject_id"] for r in labeled}
            or any(f not in range(5) for f in folds.values())
            or any(r["label"] not in (0, 1, 2) for r in labeled)):
        raise ValueError("Invalid patient folds or labels in authority manifest")
    measured = [r for r in labeled if r["record"][TARGET] in FIELDS[TARGET]]
    proxies = {
        "scope_eligible": _target_table(measured, lambda r: r["record"]["scope_eligible"]),
        **{k: _target_table(measured, lambda r: r[k])
           for k in ("active", "molecular_candidate", "reference_conflict")},
    }
    return {
        "n_records": len(rows),
        "n_labeled": len(labeled),
        "n_target_measured": len(measured),
        "n_target_unknown": len(labeled) - len(measured),
        "n_with_other_report_field_when_target_hidden": sum(
            any(r["record"][k] in FIELDS[k] for k in OTHER_FIELDS) for r in measured
        ),
        "proxy_by_measured_target": proxies,
        "other_fields_by_measured_target": {
            k: _target_table(measured, lambda r: r["record"][k]) for k in OTHER_FIELDS
        },
        "accepted_on_labeled": dict(Counter(str(r["record"]["curated_record_accepted"]).lower()
                                           for r in labeled)),
        "assay_validity_on_labeled": dict(Counter(r["record"]["assay_validity"] for r in labeled)),
        "fold_input_support": _support_by_fold(labeled, measured, folds),
        "fold_input_support_definition": (
            "Exact original training-fold molecular feature vector, report availability and inactive "
            "rule, after masking scope for prediction. Observed across subjects; this does not "
            "establish clinical support for an unaccepted or downgraded report."
        ),
        "mitigation": [
            "Re-encode visible fields; replace all three target channels by the unknown category in absent.",
            "Suppress the historical diagnosis-derived scope bit in the predictive report. Reserve it for "
            "the authorized rule and the separate unmodified archive replay.",
            "Recompile from visible evidence and qualification. Exclude archived activation, candidate, "
            "conflict flag, reference grade, original diagnosis text and patient identifiers from predictors.",
            "Keep IDH, MGMT, testing method, their missingness and MRI fixed within patient. Their observed "
            "association with codeletion can reduce the conditional increment of the measured target.",
        ],
    }
