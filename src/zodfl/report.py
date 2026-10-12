"""Render results/*.json to REPORT.md (+ optional figure). Contains no numbers of its own."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import forgetting


def load_records(results_dir: str | Path, exclude_prefixes: tuple[str, ...] = ("pilot",)) -> list[dict]:
    """All run files except pilots (short health-check runs on a separate seed are never part of the results)."""
    recs = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*__seed*.json"))]
    return [r for r in recs if not r["tag"].startswith(exclude_prefixes)]


PREFIXES = {"s2_": "stage2", "rev_": "reverse", "det_": "detection"}


def family(tag: str) -> str:
    for pre, name in PREFIXES.items():
        if tag.startswith(pre):
            return name
    return "stage1"


def split_records(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"stage1": [], "stage2": [], "reverse": [], "detection": []}
    for r in recs:
        out[family(r["tag"])].append(r)
    return out


def _final(rec: dict) -> dict:
    return rec["evals"][-1]


def _task_names(recs: list[dict]) -> list[str]:
    names: list[str] = []
    for r in recs:
        for e in r["evals"]:
            for t in e["tasks"]:
                if t not in names:
                    names.append(t)
    return names


def summarize(rec: dict, task: str, group_key: str = "by_cond") -> dict | None:
    t = _final(rec)["tasks"].get(task)
    if not t or group_key not in t:
        return None
    g = t[group_key]
    hist = [{k: v["acc"] for k, v in e["tasks"][task][group_key]["groups"].items() if v["n"] >= g["min_n"]}
            for e in rec["evals"] if task in e["tasks"] and group_key in e["tasks"][task]]
    forg = forgetting(hist)
    return {"overall": g["overall_acc"], "mean_group": g["mean_group_acc"], "worst_group": g["worst_group_acc"],
            "worst_name": g["worst_group"], "max_forgetting": max(forg.values()) if forg else None,
            "n_groups": g["n_groups_eval"]}


def _ms(vals: list[float]) -> str:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return "n/a"
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def table(recs: list[dict], task: str, group_key: str) -> str:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        s = summarize(r, task, group_key)
        if s:
            by_tag[r["tag"]].append({**s, "totals": r["totals"]})
    lines = ["| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |",
             "|---|---|---|---|---|---|---|"]
    for tag in sorted(by_tag):
        rows = by_tag[tag]
        lines.append(
            f"| {tag} | {len(rows)} | {_ms([x['overall'] for x in rows])} | {_ms([x['mean_group'] for x in rows])} | "
            f"{_ms([x['worst_group'] for x in rows])} | {_ms([x['max_forgetting'] for x in rows])} | "
            f"{np.mean([x['totals'].get('frames_trained_on', 0) for x in rows]):.0f} |")
    return "\n".join(lines)


def paired_diff(recs: list[dict], task: str, group_key: str, metric: str, base: str, other: str) -> str:
    """Mean per-seed difference other-base with a t-based 95% interval (few seeds: wide, read with care)."""
    def pick(tag):
        out = {}
        for r in recs:
            if r["tag"] == tag:
                s = summarize(r, task, group_key)
                if s and s[metric] is not None and not np.isnan(s[metric]):
                    out[r["seed"]] = s[metric]
        return out
    a, b = pick(base), pick(other)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return "n/a (need at least 2 shared seeds)"
    d = np.array([b[s] - a[s] for s in seeds])
    from scipy import stats
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+.3f} (95% CI {d.mean() - half:+.3f} to {d.mean() + half:+.3f}, n={len(d)} seeds)"


GEN_ORDER = ["s2_A_teacher", "s2_B_floor", "s2_B_oracle", "s2_B_kd_public", "s2_B_kd_incar", "s2_B_kd_incar_rarity",
             "s2_C_oracle", "s2_C_kd_incar", "s2_C_kd_incar_rarity"]


def retention_table(recs: list[dict], task: str, group_key: str, metric: str = "worst_group") -> str:
    """Per generation variant: the metric, and the metric as a share of the teacher A's on the same seed."""
    s2 = [r for r in recs if r["tag"].startswith("s2_")]
    teacher = {r["seed"]: summarize(r, task, group_key) for r in s2 if r["tag"] == "s2_A_teacher"}
    lines = [f"| model | seeds | {metric.replace('_', '-')} acc | share of teacher A | mean-group acc | overall acc |",
             "|---|---|---|---|---|---|"]
    for tag in [t for t in GEN_ORDER if any(r["tag"] == t for r in s2)]:
        vals, rel, mg, ov = [], [], [], []
        for r in s2:
            if r["tag"] != tag:
                continue
            sm = summarize(r, task, group_key)
            if not sm or sm[metric] is None or np.isnan(sm[metric]):
                continue
            vals.append(sm[metric])
            mg.append(sm["mean_group"])
            ov.append(sm["overall"])
            t = teacher.get(r["seed"])
            if t and t[metric] and not np.isnan(t[metric]):
                rel.append(sm[metric] / t[metric])
        lines.append(f"| {tag} | {len(vals)} | {_ms(vals)} | {_ms(rel)} | {_ms(mg)} | {_ms(ov)} |")
    return "\n".join(lines)


def render(results_dir: str | Path, out_path: str | Path, base_tag: str | None = None,
           group_keys: tuple[str, ...] = ("by_cond", "by_country")) -> Path:
    recs = load_records(results_dir)
    if not recs:
        raise FileNotFoundError(f"no result files in {results_dir}")
    parts = ["# Results (generated; do not edit by hand)", "",
             f"Source: {len(recs)} run files in `{results_dir}`. Groups smaller than the configured minimum test size are excluded.",
             ""]
    fam = split_records(recs)
    stage1, stage2, reverse = fam["stage1"], fam["stage2"], fam["reverse"]
    tasks = _task_names(stage1 or recs)
    tags = sorted({r["tag"] for r in stage1})
    for task in tasks:
        for gk in group_keys if stage1 else ():
            parts += [f"## Stage 1, task `{task}`, groups `{gk}`", "", table(stage1, task, gk), ""]
            if base_tag and base_tag in tags:
                parts += ["Paired differences against `" + base_tag + "` (worst-group accuracy):", ""]
                for tag in tags:
                    if tag != base_tag:
                        parts.append(f"- `{tag}`: {paired_diff(recs, task, gk, 'worst_group', base_tag, tag)}")
                parts.append("")
    if stage1:
        for task in tasks:
            for gk in group_keys[:1]:
                if any(r["tag"].startswith("balanced_") for r in stage1):
                    parts += [f"## Stage 1, matched aggregator comparison, task `{task}`", "",
                              aggregator_pairs_md(stage1, task, gk), ""]
    for task in _task_names(stage2):
        for gk in group_keys:
            parts += [f"## Stage 2 (generation gap), task `{task}`, groups `{gk}`", "",
                      retention_table(stage2, task, gk), ""]
        gk = "by_light" if task == "road_condition" else "by_cond"
        parts += [f"### Stage 2 paired comparisons, task `{task}`", "", stage2_pairs_md(stage2, task, gk), "",
                  gap_closed_md(stage2, task, gk), ""]
    for task in _task_names(reverse):
        parts += [f"## Reverse transfer, task `{task}`", "", reverse_md(reverse, task), ""]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


# ------------------------------------------------------------------ matched comparisons (all generated)
SELECTOR_TAGS = ("random20", "loss20", "novelty20", "rarity20", "all")
STAGE2_PAIRS = [
    ("s2_B_kd_incar", "s2_B_kd_incar_rarity", "B: rarity-aware minus plain in-car distillation"),
    ("s2_C_kd_incar", "s2_C_kd_incar_rarity", "C: rarity-aware minus plain in-car distillation"),
    ("s2_B_kd_public", "s2_B_kd_incar", "B: in-car minus public-set distillation"),
    ("s2_A_teacher", "s2_B_kd_incar", "B (in-car) minus teacher A"),
    ("s2_A_teacher", "s2_C_kd_incar", "C (in-car) minus teacher A"),
    ("s2_B_kd_incar", "s2_C_kd_incar", "C minus B, one more generation (in-car)"),
    ("s2_B_oracle", "s2_B_kd_incar", "B in-car distillation minus true labels"),
    ("s2_C_oracle", "s2_C_kd_incar", "C in-car distillation minus true labels"),
    ("s2_B_floor", "s2_B_kd_incar", "B in-car distillation minus the 1%-label floor"),
]


def aggregator_pairs_md(recs: list[dict], task: str, gk: str = "by_cond") -> str:
    lines = [f"Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`{task}`):", ""]
    for sel in SELECTOR_TAGS:
        lines.append(f"- {sel}: {paired_diff(recs, task, gk, 'worst_group', f'fedavg_{sel}', f'balanced_{sel}')}")
    return "\n".join(lines)


def stage2_pairs_md(recs: list[dict], task: str, gk: str = "by_cond") -> str:
    lines = [f"Worst-group accuracy, paired by seed (`{task}`):", ""]
    for base, other, label in STAGE2_PAIRS:
        lines.append(f"- {label}: {paired_diff(recs, task, gk, 'worst_group', base, other)}")
    return "\n".join(lines)


def gap_closed_md(recs: list[dict], task: str, gk: str = "by_cond") -> str:
    """Share of the gap between the 1%-label floor and the all-labels oracle that distillation closes, per seed."""
    def get(tag):
        return {r["seed"]: summarize(r, task, gk)["worst_group"] for r in recs if r["tag"] == tag and summarize(r, task, gk)}
    floor, oracle = get("s2_B_floor"), get("s2_B_oracle")
    lines = [f"Gap closed on worst-group accuracy, (student - floor) / (oracle - floor), generation B (`{task}`):", ""]
    for tag in ("s2_B_kd_public", "s2_B_kd_incar", "s2_B_kd_incar_rarity"):
        stu = get(tag)
        vals = [(stu[s] - floor[s]) / (oracle[s] - floor[s]) for s in sorted(set(stu) & set(floor) & set(oracle))
                if oracle[s] - floor[s] > 0.05]
        lines.append(f"- {tag}: {_ms(vals)} (n={len(vals)} seeds with a floor-oracle gap above 0.05)")
    return "\n".join(lines)


# ------------------------------------------------------------------ reverse transfer
REVERSE_ORDER = ["rev_A_teacher", "rev_A_from_control", "rev_A_from_oracle", "rev_A_from_ensemble",
                 "rev_B_oracle", "rev_B_control"]


def _acc_by_generation(rec: dict, task: str) -> dict[str, float]:
    """Frame-weighted accuracy over the countries of each generation, from the per-country test groups."""
    t = _final(rec)["tasks"].get(task)
    if not t or "by_country" not in t:
        return {}
    groups = t["by_country"]["groups"]
    out = {}
    for i, countries in enumerate(rec["config"]["stage2"]["gen_countries"]):
        have = [(groups[c]["acc"], groups[c]["n"]) for c in countries if c in groups]
        n = sum(x[1] for x in have)
        if n:
            out[f"gen{i + 1}"] = sum(a * m for a, m in have) / n
    return out


def reverse_md(recs: list[dict], task: str) -> str:
    """Accuracy on held-out frames from generation-1 (retention) and generation-2/3 (new-market) countries."""
    from scipy import stats

    by_tag: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for r in recs:
        by_tag[r["tag"]][r["seed"]] = _acc_by_generation(r, task)
    lines = [f"Accuracy by country group of the test frames (`{task}`; gen1 = the old cars' own countries):", "",
             "| model | seeds | gen1 countries | gen2 countries | gen3 countries |", "|---|---|---|---|---|"]
    for tag in [t for t in REVERSE_ORDER if t in by_tag]:
        cols = [_ms([v.get(g) for v in by_tag[tag].values() if g in v]) for g in ("gen1", "gen2", "gen3")]
        lines.append(f"| {tag} | {len(by_tag[tag])} | " + " | ".join(cols) + " |")
    lines += ["", "Paired by seed, reverse-transferred A' minus the original A (95% CI):", ""]
    base = by_tag.get("rev_A_teacher", {})
    for tag in [t for t in REVERSE_ORDER if t.startswith("rev_A_from")]:
        for g in ("gen1", "gen2", "gen3"):
            seeds = sorted(s for s in by_tag.get(tag, {}) if s in base and g in by_tag[tag][s] and g in base[s])
            if len(seeds) < 2:
                continue
            d = np.array([by_tag[tag][s][g] - base[s][g] for s in seeds])
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            lines.append(f"- {tag}, {g}: {d.mean():+.3f} (95% CI {d.mean() - half:+.3f} to {d.mean() + half:+.3f}, n={len(d)})")
    return "\n".join(lines)
