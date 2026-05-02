#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def read_name_value_csv(path):
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {row["name"]: float(row["value"]) for row in rows}


def read_field_stats(path):
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        out = {}
        for row in rows:
            field = row.pop("field")
            out[field] = {k: float(v) for k, v in row.items()}
        return out


def read_spectrum(path):
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return [
            {
                "normalized_radius": float(row["normalized_radius"]),
                "shell_sum_energy": float(row["shell_sum_energy"]),
                "count": int(row["count"]),
            }
            for row in rows
        ]


def allowed_diff(a, b, rtol, atol):
    return atol + rtol * max(abs(a), abs(b))


def check_value(label, a, b, rtol, atol, failures, maxima):
    diff = abs(a - b)
    limit = allowed_diff(a, b, rtol, atol)
    rel = diff / max(abs(a), abs(b), 1.0e-300)
    if diff > maxima[0]:
        maxima[:] = [diff, rel, label, a, b, limit]
    if diff > limit:
        failures.append((label, a, b, diff, rel, limit))


def check_limit(label, value, limit, failures, maxima):
    if value > maxima[0]:
        maxima[:] = [value, value, label, value, 0.0, limit]
    if value > limit:
        failures.append((label, value, 0.0, value, value, limit))


def main():
    parser = argparse.ArgumentParser(
        description="Compare two VALIDATION_DIR outputs with numeric tolerances."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scalar-rtol", type=float, default=1.0e-5)
    parser.add_argument("--scalar-atol", type=float, default=1.0e-8)
    parser.add_argument("--field-rtol", type=float, default=1.0e-4)
    parser.add_argument("--field-atol", type=float, default=1.0e-6)
    parser.add_argument("--spectrum-total-rtol", type=float, default=1.0e-5)
    parser.add_argument("--spectrum-l1-rtol", type=float, default=1.0e-4)
    parser.add_argument("--spectrum-l2-rtol", type=float, default=1.0e-4)
    parser.add_argument("--spectrum-max-total-rtol", type=float, default=1.0e-5)
    args = parser.parse_args()

    failures = []
    maxima = [0.0, 0.0, "", 0.0, 0.0, 0.0]

    base_scalars = read_name_value_csv(args.baseline / "scalars.csv")
    cand_scalars = read_name_value_csv(args.candidate / "scalars.csv")
    if base_scalars.keys() != cand_scalars.keys():
        failures.append(("scalar keys", sorted(base_scalars), sorted(cand_scalars), math.inf, math.inf, 0.0))
    else:
        for key in sorted(base_scalars):
            check_value(
                f"scalar.{key}",
                base_scalars[key],
                cand_scalars[key],
                args.scalar_rtol,
                args.scalar_atol,
                failures,
                maxima,
            )

    base_fields = read_field_stats(args.baseline / "field_stats.csv")
    cand_fields = read_field_stats(args.candidate / "field_stats.csv")
    if base_fields.keys() != cand_fields.keys():
        failures.append(("field keys", sorted(base_fields), sorted(cand_fields), math.inf, math.inf, 0.0))
    else:
        for field in sorted(base_fields):
            if base_fields[field].keys() != cand_fields[field].keys():
                failures.append((f"field.{field}.keys", sorted(base_fields[field]), sorted(cand_fields[field]), math.inf, math.inf, 0.0))
                continue
            for key in sorted(base_fields[field]):
                check_value(
                    f"field.{field}.{key}",
                    base_fields[field][key],
                    cand_fields[field][key],
                    args.field_rtol,
                    args.field_atol,
                    failures,
                    maxima,
                )

    base_spectrum = read_spectrum(args.baseline / "energy_spectrum.csv")
    cand_spectrum = read_spectrum(args.candidate / "energy_spectrum.csv")
    if len(base_spectrum) != len(cand_spectrum):
        failures.append(("spectrum length", len(base_spectrum), len(cand_spectrum), math.inf, math.inf, 0.0))
    else:
        base_energy = []
        cand_energy = []
        for i, (a, b) in enumerate(zip(base_spectrum, cand_spectrum)):
            if a["count"] != b["count"]:
                failures.append((f"spectrum[{i}].count", a["count"], b["count"], math.inf, math.inf, 0.0))
            check_value(
                f"spectrum[{i}].radius",
                a["normalized_radius"],
                b["normalized_radius"],
                0.0,
                1.0e-15,
                failures,
                maxima,
            )
            base_energy.append(a["shell_sum_energy"])
            cand_energy.append(b["shell_sum_energy"])

        base_total = sum(abs(v) for v in base_energy)
        cand_total = sum(abs(v) for v in cand_energy)
        total_scale = max(base_total, cand_total, 1.0e-300)
        base_l2 = math.sqrt(sum(v * v for v in base_energy))
        diff_l1 = sum(abs(a - b) for a, b in zip(base_energy, cand_energy))
        diff_l2 = math.sqrt(sum((a - b) * (a - b) for a, b in zip(base_energy, cand_energy)))
        diff_max = max((abs(a - b) for a, b in zip(base_energy, cand_energy)), default=0.0)

        check_limit(
            "spectrum.total_rel",
            abs(base_total - cand_total) / total_scale,
            args.spectrum_total_rtol,
            failures,
            maxima,
        )
        check_limit(
            "spectrum.l1_rel",
            diff_l1 / total_scale,
            args.spectrum_l1_rtol,
            failures,
            maxima,
        )
        check_limit(
            "spectrum.l2_rel",
            diff_l2 / max(base_l2, 1.0e-300),
            args.spectrum_l2_rtol,
            failures,
            maxima,
        )
        check_limit(
            "spectrum.max_abs_over_total",
            diff_max / total_scale,
            args.spectrum_max_total_rtol,
            failures,
            maxima,
        )

    print(
        "max_abs_diff={:.9g} max_rel_diff={:.9g} at={} baseline={:.17g} candidate={:.17g} allowed={:.9g}".format(
            maxima[0], maxima[1], maxima[2], maxima[3], maxima[4], maxima[5]
        )
    )
    if failures:
        print(f"FAIL: {len(failures)} values outside tolerance")
        for label, a, b, diff, rel, limit in failures[:20]:
            print(f"{label}: baseline={a} candidate={b} abs={diff} rel={rel} allowed={limit}")
        return 1

    print("PASS: validation outputs are within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
