"""
Cleanup baseline audit reports and generate per-agency routing CSVs.

a) Removes all *-audit-report-baseline.json files from every agency's htmls/ folder.
b) Reads each remaining *-audit-report.json, extracts routing_label, and writes
   a CSV per agency to cabinet-agencies/{agency}.csv with columns:
     document_id, routing_label
"""

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

AGENCIES = [
    "commerce", "deq", "dmva", "doa", "dpi", "it",
    "labor", "ncagr", "ncdcr", "ncdhhs", "ncdoi",
    "ncdor", "ncdps", "nctreasurer",
]


def remove_baseline_reports(htmls_dir: Path) -> int:
    """Delete all *-audit-report-baseline.json files. Returns count removed."""
    removed = 0
    for f in htmls_dir.rglob("*-audit-report-baseline.json"):
        f.unlink()
        removed += 1
    return removed


def collect_routing_labels(htmls_dir: Path) -> list[tuple[str, str]]:
    """Return (document_id, routing_label) for each audit report found."""
    rows = []
    for report in sorted(htmls_dir.rglob("*-audit-report.json")):
        # Skip baselines that somehow remain
        if "baseline" in report.name:
            continue
        # document_id = the parent folder name (the doc folder inside htmls/)
        doc_id = report.parent.name
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            label = data.get("routing_label", "")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not read {report}: {exc}")
            label = "ERROR"
        rows.append((doc_id, label))
    return rows


def main():
    output_dir = ROOT / "cabinet-agencies"
    output_dir.mkdir(exist_ok=True)

    for agency in AGENCIES:
        htmls_dir = ROOT / agency / "htmls"
        if not htmls_dir.is_dir():
            print(f"[{agency}] no htmls/ directory — skipping")
            continue

        # (a) Remove baselines
        removed = remove_baseline_reports(htmls_dir)
        print(f"[{agency}] removed {removed} baseline report(s)")

        # (b) Build CSV
        rows = collect_routing_labels(htmls_dir)
        csv_path = output_dir / f"{agency}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["document_id", "routing_label"])
            writer.writerows(rows)
        print(f"[{agency}] wrote {len(rows)} row(s) to {csv_path.name}")


if __name__ == "__main__":
    main()
