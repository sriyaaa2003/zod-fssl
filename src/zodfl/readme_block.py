"""Insert generated result tables into README.md between markers. No number in the README is typed by hand.

    python -m zodfl.readme_block --results results_local --readme README.md

Headline choice (fixed in docs/BLUEPRINT.md before any grid result existed): primary task `vru_present`
(pedestrian or vulnerable vehicle visible), primary metric worst-group accuracy over road condition x light.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .det_report import det_block
from .report import (aggregator_pairs_md, gap_closed_md, load_records, paired_diff, retention_table, reverse_md,
                     split_records, stage2_pairs_md, table)

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PRIMARY = "vru_present"
REGIMES = ("", "@10pct")          # full-label probe, low-label probe
BASE_TAG = "fedavg_random20"


KEY_STAGE1 = ["init", "fedavg_all", "fedavg_random20", "balanced_random20", "central_all", "ldawa_random20",
              "fedema_random20", "layer4only_random20", "anchor10_random20", "tax_random20+interp0.25"]
KEY_STAGE2 = ["s2_A_teacher", "s2_B_floor", "s2_B_oracle", "s2_B_kd_incar", "s2_C_kd_incar"]


def _keep(text: str, starts: tuple[str, ...], drop: tuple[str, ...] = ()) -> str:
    """Keep only the lines that start with one of `starts` (and contain none of `drop`); the README shows key rows only."""
    kept = "\n".join(l for l in text.splitlines() if l.startswith(starts) and not any(d in l for d in drop))
    return re.sub(r"(\|[^\n]*)\n(- )", r"\1\n\n\2", kept)           # a list right after a table needs a blank line


def build_block(results_dir: str | Path) -> str:
    """The short block shown in the README: key rows only. Every variant and task is in results/REPORT.md."""
    recs = load_records(results_dir)
    if not recs:
        raise FileNotFoundError(f"no result files in {results_dir}")
    fam = split_records(recs)
    stage1, stage2, reverse = fam["stage1"], fam["stage2"], fam["reverse"]
    out: list[str] = []
    summary = Path(results_dir) / "data_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        out += [f"Data: {s['n_frames']:,} ZOD frames, {len(s['stream_by_country'])} countries; stream {s['split_counts'].get('stream', 0):,}, "
                f"probe-train {s['split_counts'].get('probe_train', 0):,}, probe-test {s['split_counts'].get('probe_test', 0):,}. "
                f"Rarest groups: wet/night {s['frames_by_cond'].get('wet/night', 0):,} and snow/night "
                f"{s['frames_by_cond'].get('snow/night', 0):,} frames.", ""]
    if stage1:
        task = PRIMARY + "@10pct"
        keep = [r for r in stage1 if r["tag"] in KEY_STAGE1]
        out += [f"**Stage 1** (`{task}`: low-label probe, 5 seeds; `init` = the pretrained model with no federated training). "
                "Worst-group = the weakest road-condition x light group.", "", table(keep, task, "by_cond"), ""]
    if stage2:
        keep2 = [r for r in stage2 if r["tag"] in KEY_STAGE2]
        out += [f"**Stage 2, generation gap** (`{PRIMARY}`, 5 seeds).", "", retention_table(keep2, PRIMARY, "by_cond"), "",
                _keep(gap_closed_md(stage2, PRIMARY), ("Gap closed", "- s2_B_kd_incar:")), ""]
    if reverse:
        out += [f"**Reverse transfer, B back to A** (`{PRIMARY}`).", "", _keep(reverse_md(reverse, PRIMARY), ("|", "- rev_A_from_ensemble", "- rev_A_from_control, gen2"), drop=("rev_B_", "rev_A_from_oracle")), ""]
    det = det_block(results_dir)
    if det:
        out += [det, ""]
    out.append("Every variant, task, per-country table and paired difference: [results/REPORT.md](results/REPORT.md).")
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
