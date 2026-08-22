"""Attribute accuracy differences between two mobile-actions result files.

The input files are the per-row JSON arrays written by mobile_actions.py.  The
report separates call-count, tool-name, argument-key, and argument-value errors
and uses paired wins/losses so a headline accuracy delta is not mistaken for a
single-engine error count.
"""
import argparse
import json
from collections import Counter


def load_rows(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("rows", []) if isinstance(payload, dict) else payload


def arg_error_flags(row):
    if not row.get("name_ok"):
        return False, False
    key_error = False
    value_error = False
    for got, want in zip(row.get("got") or [], row.get("want") or []):
        got_args = got.get("arguments") or {}
        want_args = want.get("arguments") or {}
        if set(got_args) != set(want_args):
            key_error = True
        if any(got_args.get(key) != value for key, value in want_args.items()):
            value_error = True
    return key_error, value_error


def error_class(row):
    got_count = len(row.get("got") or [])
    want_count = len(row.get("want") or [])
    if row.get("exact_ok"):
        return "exact"
    if got_count < want_count:
        return "under_call"
    if got_count > want_count:
        return "over_call"
    if not row.get("name_ok"):
        return "wrong_name"
    key_error, value_error = arg_error_flags(row)
    if key_error and value_error:
        return "argument_keys_and_values"
    if key_error:
        return "argument_keys"
    return "argument_values"


def engine_profile(rows):
    classes = Counter(error_class(row) for row in rows)
    field_total = Counter()
    field_mismatch = Counter()
    missing_keys = Counter()
    extra_keys = Counter()
    field_examples = {}
    for row_index, row in enumerate(rows):
        if not row.get("name_ok"):
            continue
        for got, want in zip(row.get("got") or [], row.get("want") or []):
            tool = want.get("name", "<unknown>")
            got_args = got.get("arguments") or {}
            want_args = want.get("arguments") or {}
            for key, value in want_args.items():
                label = f"{tool}.{key}"
                field_total[label] += 1
                if key not in got_args:
                    missing_keys[label] += 1
                if got_args.get(key) != value:
                    field_mismatch[label] += 1
                    examples = field_examples.setdefault(label, [])
                    if len(examples) < 5:
                        examples.append({
                            "row": row_index,
                            "expected": value,
                            "got": got_args.get(key),
                        })
            for key in got_args.keys() - want_args.keys():
                extra_keys[f"{tool}.{key}"] += 1
    buckets = {}
    for expected_count in sorted({row.get("n_expected", len(row.get("want") or []))
                                  for row in rows}):
        selected = [row for row in rows
                    if row.get("n_expected", len(row.get("want") or [])) == expected_count]
        buckets[str(expected_count)] = {
            "total": len(selected),
            "exact": sum(bool(row.get("exact_ok")) for row in selected),
            "names": sum(bool(row.get("name_ok")) for row in selected),
            "under_call": sum(len(row.get("got") or []) < len(row.get("want") or [])
                              for row in selected),
            "over_call": sum(len(row.get("got") or []) > len(row.get("want") or [])
                             for row in selected),
        }
    return {
        "total": len(rows),
        "exact": classes["exact"],
        "names": sum(bool(row.get("name_ok")) for row in rows),
        "error_classes": dict(classes),
        "call_buckets": buckets,
        "argument_fields": {
            key: {
                "total": field_total[key],
                "mismatches": field_mismatch[key],
                "error_rate": field_mismatch[key] / field_total[key],
            }
            for key in sorted(field_total, key=lambda item: (-field_mismatch[item], item))
        },
        "missing_argument_keys": dict(missing_keys.most_common()),
        "extra_argument_keys": dict(extra_keys.most_common()),
        "argument_field_examples": field_examples,
    }


def paired_profile(candidate, reference):
    wins = Counter()
    losses = Counter()
    examples = {}
    for index, (cand, ref) in enumerate(zip(candidate, reference)):
        if cand.get("query") != ref.get("query") or cand.get("want") != ref.get("want"):
            raise ValueError(f"result files are not aligned at row {index}")
        cand_ok = bool(cand.get("exact_ok"))
        ref_ok = bool(ref.get("exact_ok"))
        if ref_ok and not cand_ok:
            category = error_class(cand)
            losses[category] += 1
            examples.setdefault("reference_only_" + category, []).append(index)
        elif cand_ok and not ref_ok:
            category = error_class(ref)
            wins[category] += 1
            examples.setdefault("candidate_only_" + category, []).append(index)
    return {
        "reference_only": sum(losses.values()),
        "candidate_only": sum(wins.values()),
        "net_reference_advantage": sum(losses.values()) - sum(wins.values()),
        "reference_only_by_candidate_error": dict(losses),
        "candidate_only_by_reference_error": dict(wins),
        "net_by_error_class": {
            key: losses[key] - wins[key] for key in sorted(set(losses) | set(wins))
        },
        "example_indices": {key: value[:12] for key, value in sorted(examples.items())},
    }


def pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


def print_engine(label, profile):
    total = profile["total"]
    print(f"{label}: exact {profile['exact']}/{total} ({pct(profile['exact'], total):.1f}%), "
          f"names {profile['names']}/{total} ({pct(profile['names'], total):.1f}%)")
    print("  errors:", ", ".join(
        f"{key}={value}" for key, value in profile["error_classes"].items()
        if key != "exact"))
    for count, bucket in profile["call_buckets"].items():
        print(f"  {count}-call: exact {bucket['exact']}/{bucket['total']}, "
              f"names {bucket['names']}/{bucket['total']}, "
              f"under={bucket['under_call']}, over={bucket['over_call']}")
    worst = list(profile["argument_fields"].items())[:8]
    if worst:
        print("  worst fields:", ", ".join(
            f"{key}={value['mismatches']}/{value['total']}"
            for key, value in worst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="result file for the engine under investigation")
    parser.add_argument("reference", help="paired reference result file")
    parser.add_argument("--out", help="optional machine-readable JSON report")
    args = parser.parse_args()

    candidate = load_rows(args.candidate)
    reference = load_rows(args.reference)
    if len(candidate) != len(reference):
        raise ValueError(f"row-count mismatch: {len(candidate)} != {len(reference)}")

    report = {
        "candidate": engine_profile(candidate),
        "reference": engine_profile(reference),
        "paired": paired_profile(candidate, reference),
    }
    print_engine("candidate", report["candidate"])
    print_engine("reference", report["reference"])
    paired = report["paired"]
    print(f"paired: reference-only={paired['reference_only']}, "
          f"candidate-only={paired['candidate_only']}, "
          f"net={paired['net_reference_advantage']}")
    print("  net attribution:", ", ".join(
        f"{key}={value}" for key, value in paired["net_by_error_class"].items()))

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print("report ->", args.out)


if __name__ == "__main__":
    main()
