"""Collect failed and near-threshold regression cases for visual review."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from batch_test_52 import STRICT_THRESHOLDS


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def copy_case(row: dict[str, str], destination: Path) -> None:
    source = Path(row["contact_sheet"]).parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def metric_triggers(
    row: dict[str, str],
    *,
    ratio_limit: float,
) -> list[dict[str, float]]:
    triggers: list[dict[str, float]] = []
    for metric, threshold in STRICT_THRESHOLDS.items():
        try:
            actual = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        ratio = actual / threshold if threshold > 0 else 0.0
        if ratio >= ratio_limit:
            triggers.append(
                {
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                    "threshold_ratio": ratio,
                }
            )
    return triggers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-ratio", type=float, default=0.80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []

    if args.prior_dir:
        for row in read_rows(args.prior_dir / "cases.csv"):
            if row["passed_python"].lower() == "true":
                continue
            destination = (
                args.output_dir
                / "首轮失败_算法改进前"
                / row["card"]
                / f"layout_{int(row['layout_index']):02d}"
            )
            copy_case(row, destination)
            index_rows.append(
                {
                    "category": "首轮失败_算法改进前",
                    "case_id": row["case_id"],
                    "card": row["card"],
                    "layout_index": int(row["layout_index"]),
                    "trigger_metrics": row["failed_checks_python"],
                    "trigger_details": row["error"],
                    "source_folder": str(Path(row["contact_sheet"]).parent),
                    "archive_folder": str(destination.resolve()),
                }
            )

    near_count = 0
    final_failed_count = 0
    for row in read_rows(args.final_dir / "cases.csv"):
        passed = row["passed_python"].lower() == "true"
        triggers = metric_triggers(row, ratio_limit=args.near_ratio)
        if passed and not triggers:
            continue
        category = "最终失败" if not passed else "最终临界通过_阈值80pct"
        if passed:
            near_count += 1
        else:
            final_failed_count += 1
        destination = (
            args.output_dir
            / category
            / row["card"]
            / f"layout_{int(row['layout_index']):02d}"
        )
        copy_case(row, destination)
        index_rows.append(
            {
                "category": category,
                "case_id": row["case_id"],
                "card": row["card"],
                "layout_index": int(row["layout_index"]),
                "trigger_metrics": "; ".join(
                    trigger["metric"] for trigger in triggers
                )
                or row["failed_checks_python"],
                "trigger_details": json.dumps(
                    triggers,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "source_folder": str(Path(row["contact_sheet"]).parent),
                "archive_folder": str(destination.resolve()),
            }
        )

    columns = [
        "category",
        "case_id",
        "card",
        "layout_index",
        "trigger_metrics",
        "trigger_details",
        "source_folder",
        "archive_folder",
    ]
    with (args.output_dir / "异常样本索引.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "near_threshold_ratio": args.near_ratio,
        "prior_failed_cases": sum(
            row["category"] == "首轮失败_算法改进前" for row in index_rows
        ),
        "final_failed_cases": final_failed_count,
        "final_near_threshold_cases": near_count,
        "archived_cases_total": len(index_rows),
    }
    (args.output_dir / "异常样本汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
