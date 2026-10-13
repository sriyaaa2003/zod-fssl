"""Tables for the detection generation-gap study, generated from results/detection/*.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ORDER = ["det_A_teacher", "det_B_floor", "det_B_kd_floor", "det_B_kd_incar", "det_B_oracle"]
CLASS_NAMES = {"1": "Vehicle", "2": "VulnerableVehicle", "3": "Pedestrian"}


def load_det(results_dir: str | Path) -> list[dict]:
    d = Path(results_dir) / "detection"
    recs = [json.loads(p.read_text()) for p in sorted(d.glob("det_*__seed*.json"))] if d.exists() else []
    return [r for r in recs if r.get("split") == "test" and not r.get("variant")]


def _by_tag(recs: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in recs:
        out.setdefault(r["tag"], {})[r["seed"]] = r
    return out


def _ms(vals: list[float]) -> str:
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main_table(recs: list[dict]) -> str:
    t = _by_tag(recs)
    rows = ["| model | seeds | AP | AP50 | worst-group AP | mean-group AP | Vehicle AP | VulnerableVehicle AP | Pedestrian AP |",
            "|---|---|---|---|---|---|---|---|---|"]
    for tag in ORDER:
        if tag not in t:
            continue
        rs = list(t[tag].values())
        pc = [_ms([r["eval"]["per_class"].get(c) for r in rs]) for c in ("1", "2", "3")]
        rows.append(f"| {tag} | {len(rs)} | {_ms([r['eval']['ap'] for r in rs])} | {_ms([r['eval']['ap50'] for r in rs])} | "
                    f"{_ms([r['eval']['worst_group_ap'] for r in rs])} | {_ms([r['eval']['mean_group_ap'] for r in rs])} | "
                    f"{pc[0]} | {pc[1]} | {pc[2]} |")
    return "\n".join(rows)


def group_table(recs: list[dict], metric: str = "ap") -> str:
    t = _by_tag(recs)
    groups = sorted({g for r in recs for g in r["eval"]["groups"]})
    n = {g: next(r["eval"]["groups"][g]["n"] for r in recs if g in r["eval"]["groups"]) for g in groups}
    rows = ["| model | " + " | ".join(f"{g} (n={n[g]})" for g in groups) + " |", "|---|" + "---|" * len(groups)]
    for tag in ORDER:
        if tag in t:
            rs = list(t[tag].values())
            rows.append(f"| {tag} | " + " | ".join(_ms([r["eval"]["groups"].get(g, {}).get(metric) for r in rs]) for g in groups) + " |")
    return "\n".join(rows)


def _paired(t: dict, a: str, b: str, getter) -> str:
    if a not in t or b not in t:
        return "n/a"
    seeds = sorted(set(t[a]) & set(t[b]))
    if len(seeds) < 2:
        return "n/a (need at least 2 shared seeds)"
    d = np.array([getter(t[b][s]) - getter(t[a][s]) for s in seeds])
    from scipy import stats
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.3f} (95% CI {d.mean() - half:+.3f} to {d.mean() + half:+.3f}, n={len(d)} seeds)"


def pairs_md(recs: list[dict]) -> str:
    t = _by_tag(recs)
    lines = []
    for label, getter in (("AP", lambda r: r["eval"]["ap"]), ("worst-group AP", lambda r: r["eval"]["worst_group_ap"])):
        lines.append(f"Paired by seed, {label}:")
        for a, b in (("det_B_floor", "det_B_kd_incar"), ("det_B_floor", "det_B_kd_floor"), ("det_B_kd_incar", "det_B_oracle"),
                     ("det_B_kd_incar", "det_B_kd_floor")):
            lines.append(f"- `{b}` minus `{a}`: {_paired(t, a, b, getter)}")
        lines.append("")
    return "\n".join(lines)


def gap_closed_det(recs: list[dict]) -> str:
    """(student - floor) / (oracle - floor) on AP, per seed; seeds with a floor-oracle gap below 0.01 AP are skipped."""
    t = _by_tag(recs)
    lines = ["Share of the floor-to-oracle AP gap closed, (student - floor) / (oracle - floor):"]
    for tag in ("det_B_kd_incar", "det_B_kd_floor"):
        vals = []
        for s in sorted(set(t.get(tag, {})) & set(t.get("det_B_floor", {})) & set(t.get("det_B_oracle", {}))):
            f, o, k = (t[x][s]["eval"]["ap"] for x in ("det_B_floor", "det_B_oracle", tag))
            if o - f >= 0.01:
                vals.append((k - f) / (o - f))
        if vals:
            lines.append(f"- {tag}: {_ms(vals)} (n={len(vals)} seeds)")
    return "\n".join(lines)


def det_block(results_dir: str | Path) -> str:
    recs = load_det(results_dir)
    if not recs:
        return ""
    first_pairs = pairs_md(recs).split("\n\n")[0]
    head = ("**Detection** (Faster R-CNN, teacher A = ResNet-18, student B = MobileNetV3; ZOD 2D boxes; COCO-style AP on "
            "held-out test frames, 3 seeds).")
    return "\n".join([head, "", main_table(recs), "", first_pairs, ""])
