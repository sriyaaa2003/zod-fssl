"""Load the experiment grid (config/grid.toml) and apply a run's overrides to the base config."""
from __future__ import annotations

import dataclasses as dc
import tomllib
from pathlib import Path

from . import config as C


@dc.dataclass
class RunSpec:
    tag: str
    overrides: dict = dc.field(default_factory=dict)
    train: bool = True
    centralized: bool = False
    seeds: list[int] | None = None


def load_grid(path: str | Path) -> list[RunSpec]:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    specs = [RunSpec(**r) for r in raw.get("run", [])]
    tags = [s.tag for s in specs]
    if len(set(tags)) != len(tags):
        raise ValueError(f"duplicate run tags: {sorted({t for t in tags if tags.count(t) > 1})}")
    return specs


def config_for(base: C.Config, spec: RunSpec) -> C.Config:
    """Validated config for one run; unknown override keys raise."""
    for section in spec.overrides:
        if not hasattr(base, section):
            raise ValueError(f"run {spec.tag!r}: unknown config section {section!r}")
    return C.override(base, **spec.overrides) if spec.overrides else base
