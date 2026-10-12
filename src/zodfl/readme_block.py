"""Insert generated result tables into README.md between markers. No number in the README is typed by hand.

    python -m zodfl.readme_block --results results_local --readme README.md

Headline choice (fixed in docs/BLUEPRINT.md before any grid result existed): primary task `vru_present`
(pedestrian or vulnerable vehicle visible), primary metric worst-group accuracy over road condition x light.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import (aggregator_pairs_md, gap_closed_md, load_records, paired_diff, retention_table, reverse_md,
                     split_records, stage2_pairs_md, table)

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PRIMARY = "vru_present"
REGIMES = ("", "@10pct")          # full-label probe, low-label probe
BASE_TAG = "fedavg_random20"


def build_block(results_dir: str | Path) -> str:
    recs = load_records(results_dir)
    if not recs:
        raise FileNotFoundError(f"no result files in {results_dir}")
    fam = split_records(recs)
    stage1, stage2, reverse = fam["stage1"], fam["stage2"], fam["reverse"]
    out: list[str] = []
    summary = Path(results_dir) / "data_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        out += [f"Data: {s['n_frames']:,} ZOD frames, {len(s['stream_by_country'])} countries; "
                f"stream {s['split_counts'].get('stream', 0):,}, probe-train {s['split_counts'].get('probe_train', 0):,}, "
                f"probe-test {s['split_counts'].get('probe_test', 0):,}.", "",
                "| condition group | frames | probe-test frames |", "|---|---|---|"]
        for c, n in sorted(s["frames_by_cond"].items(), key=lambda kv: -kv[1]):
            out.append(f"| {c} | {n:,} | {s['probe_test_by_cond'].get(c, 0):,} |")
        out.append("")
    if stage1:
        for reg in REGIMES:
            task = PRIMARY + reg
            label = "full-label probe" if not reg else "low-label probe (10% of probe-train)"
            out += [f"#### Stage 1: `{task}`, {label}, groups = road condition x light", "",
                    table(stage1, task, "by_cond"), ""]
            tags = {r["tag"] for r in stage1}
            if BASE_TAG in tags:
                out.append(f"Worst-group accuracy, difference against `{BASE_TAG}` (paired by seed):")
                out.append("")
                for tag in sorted(tags - {BASE_TAG}):
                    out.append(f"- `{tag}`: {paired_diff(stage1, task, 'by_cond', 'worst_group', BASE_TAG, tag)}")
                out.append("")
                if any(t.startswith("balanced_") for t in tags):
                    out += [aggregator_pairs_md(stage1, task), ""]
    if stage2:
        out += [f"#### Stage 2 (generation gap): `{PRIMARY}`, groups = road condition x light", "",
                retention_table(stage2, PRIMARY, "by_cond"), "", stage2_pairs_md(stage2, PRIMARY), "",
                gap_closed_md(stage2, PRIMARY), ""]
    if reverse:
        out += [f"#### Reverse transfer: `{PRIMARY}`", "", reverse_md(reverse, PRIMARY), ""]
    out.append("All tasks, per-country tables and every paired difference: [results/REPORT.md](results/REPORT.md).")
    return "\n".join(out)


def insert(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{readme} has no {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()
    insert(Path(a.readme), build_block(a.results))
    print("README results block updated")


if __name__ == "__main__":
    main()
