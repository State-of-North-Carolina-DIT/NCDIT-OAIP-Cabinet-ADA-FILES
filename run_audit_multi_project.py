#!/usr/bin/env python3
"""Run auditor across 10 GCP projects in parallel.

Distributes document folders round-robin across projects and runs
one auditor subprocess per project (each handling ~10 docs sequentially).

Usage:
    # Run on ncdit-audit-04-17 (100 docs across 10 projects)
    python run_audit_multi_project.py \
        ../../google/backend/e2e-results/ncdit-audit-04-17/

    # Custom projects and workers
    python run_audit_multi_project.py \
        ../../google/backend/e2e-results/ncdit-audit-04-17/ \
        --projects ada-compliance-engine-1,ada-compliance-engine-2 \
        --workers 2
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECTS = [f"ada-compliance-engine-{i}" for i in range(1, 11)]


def discover_doc_folders(input_dir: Path) -> list[Path]:
    """Find all document subfolders with source.pdf + .json."""
    folders = []
    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "source.pdf").exists():
            continue
        json_files = [f for f in d.glob("*.json") if "audit" not in f.stem.lower()]
        if json_files:
            folders.append(d)
    return folders


def run_project_batch(project_id: str, doc_folders: list[Path], skip_llm: bool = False) -> list[dict]:
    """Run the auditor on a batch of docs using one GCP project."""
    results = []
    auditor_path = Path(__file__).parent / "auditor.py"
    python = Path(__file__).resolve().parent.parent.parent / "backend" / ".venv" / "bin" / "python"

    if not python.exists():
        python = Path(sys.executable)

    for folder in doc_folders:
        slug = folder.name

        # Pass folder path — auditor auto-discovers source.pdf + .json + .html
        cmd = [str(python), str(auditor_path), str(folder)]
        if skip_llm:
            cmd += ["--skip-llm"]

        env = os.environ.copy()
        env["PROJECT_ID"] = project_id
        env["GEMINI_LOCATION"] = "global"

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env
            )
            elapsed = time.time() - start

            # Read the audit report
            report_path = folder / f"{slug}-audit-report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                routing = report.get("routing", "unknown")
                score = report.get("quality_score", "?")
                n_concerns = len(report.get("concerns", []))
                method = report.get("decision_method", "?")
                print(f"  {'OK':>6}  {slug} (score={score}, routing={routing}, "
                      f"method={method}, concerns={n_concerns}, {elapsed:.1f}s) [{project_id}]")
                results.append({
                    "slug": slug,
                    "project": project_id,
                    "score": score,
                    "routing": routing,
                    "method": method,
                    "concerns": n_concerns,
                    "elapsed": round(elapsed, 1),
                    "status": "ok",
                })
            else:
                # Check for errors in output
                stderr_snippet = (proc.stderr or "")[-200:]
                print(f"  {'FAIL':>6}  {slug} ({elapsed:.1f}s) [{project_id}] — no report generated")
                if stderr_snippet:
                    print(f"         stderr: {stderr_snippet}")
                results.append({
                    "slug": slug,
                    "project": project_id,
                    "status": "fail",
                    "error": stderr_snippet,
                    "elapsed": round(elapsed, 1),
                })

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  {'TIMEOUT':>6}  {slug} ({elapsed:.1f}s) [{project_id}]")
            results.append({
                "slug": slug,
                "project": project_id,
                "status": "timeout",
                "elapsed": round(elapsed, 1),
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"  {'ERROR':>6}  {slug} ({elapsed:.1f}s) [{project_id}] — {e}")
            results.append({
                "slug": slug,
                "project": project_id,
                "status": "error",
                "error": str(e),
                "elapsed": round(elapsed, 1),
            })

    return results


def generate_report(results: list[dict], input_dir: Path, elapsed_total: float) -> str:
    """Generate AUDITOR-REPORT.md content."""
    ok_results = [r for r in results if r["status"] == "ok"]
    fail_results = [r for r in results if r["status"] != "ok"]

    # Routing distribution
    routing_counts = {}
    for r in ok_results:
        routing = r["routing"]
        routing_counts[routing] = routing_counts.get(routing, 0) + 1

    # Score distribution
    scores = [r["score"] for r in ok_results if isinstance(r["score"], (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Grade distribution
    grade_counts = {"Good": 0, "Fair": 0, "Poor": 0, "Critical": 0}
    for s in scores:
        if s >= 90:
            grade_counts["Good"] += 1
        elif s >= 70:
            grade_counts["Fair"] += 1
        elif s >= 50:
            grade_counts["Poor"] += 1
        else:
            grade_counts["Critical"] += 1

    # Concern statistics
    total_concerns = sum(r.get("concerns", 0) for r in ok_results)

    lines = []
    lines.append("# Auditor Report")
    lines.append("")
    lines.append(f"**Batch**: `{input_dir.name}`")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total documents**: {len(results)}")
    lines.append(f"**Successful**: {len(ok_results)}")
    lines.append(f"**Failed/Timeout**: {len(fail_results)}")
    lines.append(f"**Total time**: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    lines.append("")

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Average Score | {avg_score:.1f} |")
    lines.append(f"| Total Concerns | {total_concerns} |")
    lines.append(f"| Avg Concerns/Doc | {total_concerns/len(ok_results):.1f} |" if ok_results else "")
    lines.append("")

    lines.append("## Grade Distribution")
    lines.append("")
    lines.append("| Grade | Score Range | Count | % |")
    lines.append("|-------|-----------|-------|---|")
    for grade, range_str in [("Good", "90-100"), ("Fair", "70-89"), ("Poor", "50-69"), ("Critical", "0-49")]:
        count = grade_counts[grade]
        pct = (count / len(ok_results) * 100) if ok_results else 0
        lines.append(f"| {grade} | {range_str} | {count} | {pct:.0f}% |")
    lines.append("")

    lines.append("## Routing Distribution")
    lines.append("")
    lines.append("| Routing | Count | % |")
    lines.append("|---------|-------|---|")
    for routing in ["auto_approve", "human_review", "reject"]:
        count = routing_counts.get(routing, 0)
        pct = (count / len(ok_results) * 100) if ok_results else 0
        lines.append(f"| {routing} | {count} | {pct:.0f}% |")
    lines.append("")

    # Problem documents (reject + low scores)
    problem_docs = sorted(
        [r for r in ok_results if r["routing"] == "reject" or (isinstance(r["score"], (int, float)) and r["score"] < 70)],
        key=lambda r: r.get("score", 0)
    )
    if problem_docs:
        lines.append("## Problem Documents (Reject or Score < 70)")
        lines.append("")
        lines.append("| Document | Score | Routing | Concerns |")
        lines.append("|----------|-------|---------|----------|")
        for r in problem_docs:
            lines.append(f"| {r['slug']} | {r['score']} | {r['routing']} | {r['concerns']} |")
        lines.append("")

    # All results table
    lines.append("## All Results")
    lines.append("")
    lines.append("| # | Document | Score | Routing | Method | Concerns | Time |")
    lines.append("|---|----------|-------|---------|--------|----------|------|")
    for i, r in enumerate(sorted(ok_results, key=lambda x: x["slug"]), 1):
        lines.append(
            f"| {i} | {r['slug']} | {r['score']} | {r['routing']} | "
            f"{r['method']} | {r['concerns']} | {r['elapsed']}s |"
        )
    lines.append("")

    if fail_results:
        lines.append("## Failed Documents")
        lines.append("")
        lines.append("| Document | Status | Error |")
        lines.append("|----------|--------|-------|")
        for r in fail_results:
            error = r.get("error", "")[:100]
            lines.append(f"| {r['slug']} | {r['status']} | {error} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run auditor across multiple GCP projects")
    parser.add_argument("input_dir", type=Path, help="Directory with doc subfolders")
    parser.add_argument(
        "--projects", type=str, default=None,
        help="Comma-separated GCP project IDs (default: ada-compliance-engine-1..10)"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM calls (programmatic only)"
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Path for AUDITOR-REPORT.md (default: input_dir/AUDITOR-REPORT.md)"
    )
    args = parser.parse_args()

    projects = args.projects.split(",") if args.projects else PROJECTS
    doc_folders = discover_doc_folders(args.input_dir)

    print(f"Discovered {len(doc_folders)} documents in {args.input_dir}")
    print(f"Distributing across {len(projects)} GCP projects")
    print(f"LLM calls: {'SKIP' if args.skip_llm else 'ENABLED'}")
    print()

    # Round-robin distribute docs across projects
    project_batches: dict[str, list[Path]] = {p: [] for p in projects}
    for i, folder in enumerate(doc_folders):
        project = projects[i % len(projects)]
        project_batches[project].append(folder)

    for p, folders in project_batches.items():
        print(f"  {p}: {len(folders)} docs")
    print()

    start_time = time.time()

    # Run all projects in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=len(projects)) as pool:
        futures = {}
        for project, folders in project_batches.items():
            if not folders:
                continue
            future = pool.submit(run_project_batch, project, folders, args.skip_llm)
            futures[future] = project

        for future in as_completed(futures):
            project = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"ERROR: Project {project} failed: {e}")

    elapsed_total = time.time() - start_time

    # Print summary
    ok = [r for r in all_results if r["status"] == "ok"]
    fail = [r for r in all_results if r["status"] != "ok"]
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(ok)} OK, {len(fail)} failed/timeout in {elapsed_total:.0f}s")

    if ok:
        routing_counts = {}
        for r in ok:
            routing_counts[r["routing"]] = routing_counts.get(r["routing"], 0) + 1
        print(f"Routing: {routing_counts}")

        scores = [r["score"] for r in ok if isinstance(r["score"], (int, float))]
        if scores:
            print(f"Scores: avg={sum(scores)/len(scores):.1f}, min={min(scores)}, max={max(scores)}")

    # Generate and write report
    report_path = args.report or (args.input_dir / "AUDITOR-REPORT.md")
    report_content = generate_report(all_results, args.input_dir, elapsed_total)
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nReport written to: {report_path}")

    # Also save raw results as JSON
    results_json_path = args.input_dir / "audit-batch-results.json"
    with open(results_json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_dir": str(args.input_dir),
            "projects": projects,
            "total_docs": len(all_results),
            "ok": len(ok),
            "failed": len(fail),
            "elapsed_seconds": round(elapsed_total, 1),
            "results": all_results,
        }, f, indent=2)
    print(f"Raw results: {results_json_path}")


if __name__ == "__main__":
    main()
