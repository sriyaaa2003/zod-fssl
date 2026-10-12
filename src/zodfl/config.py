"""Typed, validated configuration loaded from TOML. No secrets live here."""
from __future__ import annotations

import dataclasses as dc
import tomllib
from pathlib import Path
from typing import Any

SELECTORS = ("all", "random", "loss", "novelty", "rarity")
AGGREGATORS = ("fedavg", "balanced", "ldawa", "ldawa_norm", "fedema")


@dc.dataclass
class DataCfg:
    root: str = "/data"
    raw_dir: str = "/data/zod_raw/frames"
    shards_dir: str = "/data/shards"
    index_path: str = "/data/index.parquet"
    image_hw: tuple[int, int] = (144, 256)
    shard_size: int = 2000
    max_frames: int = 0
    max_corrupt_fraction: float = 0.002
    probe_fraction: float = 0.20
    probe_seed: int = 7


@dc.dataclass
class IndexCfg:
    night_values: list[str] = dc.field(default_factory=lambda: ["night", "twilight"])
    condition_columns: list[str] = dc.field(default_factory=lambda: ["road_condition", "light"])
    vru_columns: list[str] = dc.field(default_factory=lambda: ["num_vulnerable_vehicles", "num_pedestrians"])
    density_column: str = "num_vehicles"
    density_bins: int = 3


@dc.dataclass
class FederationCfg:
    n_rounds: int = 24
    client_key: str = "country"
    min_client_frames: int = 200
    arrival_batch: int = 128
    train_batch: int = 32
    min_final_batch: int = 8
    max_buffer: int = 63
    local_lr: float = 0.02
    backbone_lr_mult: float = 0.1      # pretrained layer3/layer4 learn at local_lr * this; heads at local_lr
    momentum: float = 0.9
    weight_decay: float = 1e-4
    pretrained: bool = True
    rarity_floor: float = 1e-4
    train_layers: list[str] = dc.field(default_factory=lambda: ["layer3", "layer4"])   # pretrained blocks that learn
    anchor_mu: float = 0.0    # L2 pull of the trained pretrained layers toward the shipped weights (0 = off)


@dc.dataclass
class SelectionCfg:
    kind: str = "random"
    budget: float = 0.2
    novelty_ema: float = 0.99


@dc.dataclass
class AggregationCfg:
    kind: str = "fedavg"
    tau: float = 0.7          # FedEMA lambda autoscaler (Zhuang et al. 2022 default)


@dc.dataclass
class EvalCfg:
    eval_every: int = 6
    probe_c: float = 1.0
    probe_fractions: list[float] = dc.field(default_factory=lambda: [1.0, 0.1])   # share of probe_train used
    interp_alphas: list[float] = dc.field(default_factory=list)   # also score (1-a)*shipped + a*trained weights
    min_group_size: int = 30
    bootstrap: int = 200
    tasks: list[dict[str, Any]] = dc.field(default_factory=list)


@dc.dataclass
class Stage2Cfg:
    """Generation gap: fleet generations with different models/sensors/hardware (docs/BLUEPRINT.md stage 2)."""
    gen_countries: list[list[str]] = dc.field(default_factory=lambda: [["PL", "DE", "FR"], ["SE", "GB", "IE", "CZ"],
                                                                      ["IT", "NO", "HU", "NL", "LU"]])
    archs: list[str] = dc.field(default_factory=lambda: ["resnet18", "mobilenet_v3_small", "mobilenet_v3_small"])
    sensor_hw: list[list[int]] = dc.field(default_factory=lambda: [[144, 256], [108, 192], [72, 128]])
    sensor_fov: list[float] = dc.field(default_factory=lambda: [1.0, 0.9, 0.8])   # centre-crop fraction
    quant_bits: list[int] = dc.field(default_factory=lambda: [0, 0, 8])           # weight bits at deployment, 0 = float
    n_rounds: int = 8
    passes: int = 2
    batch: int = 64
    lr: float = 0.01
    backbone_lr_mult: float = 0.1       # pretrained body learns at lr * this, heads at lr
    grad_clip: float = 5.0              # max gradient norm; 0 = off
    max_skipped_fraction: float = 0.02  # fail if more steps than this were skipped for non-finite loss
    momentum: float = 0.9
    weight_decay: float = 1e-4
    temperature: float = 2.0
    rarity_alpha: float = 0.5            # distillation weight per frame ~ (1/q(cond))^alpha, 0 = off
    floor_label_fraction: float = 0.01   # labelled share of the next generation's frames in the "floor" run
    n_public: int = 3000                 # frames in the shared public set for public-set distillation
    teacher_sees: str = "old"            # "old": teacher reads the previous sensor view; "new": the student's view
    pretrained: bool = True


@dc.dataclass
class RunCfg:
    seeds: list[int] = dc.field(default_factory=lambda: [0, 1, 2])
    device: str = "cuda"
    results_dir: str = "/data/results"


@dc.dataclass
class Config:
    data: DataCfg = dc.field(default_factory=DataCfg)
    index: IndexCfg = dc.field(default_factory=IndexCfg)
    federation: FederationCfg = dc.field(default_factory=FederationCfg)
    selection: SelectionCfg = dc.field(default_factory=SelectionCfg)
    aggregation: AggregationCfg = dc.field(default_factory=AggregationCfg)
    eval: EvalCfg = dc.field(default_factory=EvalCfg)
    stage2: Stage2Cfg = dc.field(default_factory=Stage2Cfg)
    run: RunCfg = dc.field(default_factory=RunCfg)

    def validate(self) -> "Config":
        f, s = self.federation, self.selection
        if s.kind not in SELECTORS:
            raise ValueError(f"selection.kind must be one of {SELECTORS}, got {s.kind!r}")
        if self.aggregation.kind not in AGGREGATORS:
            raise ValueError(f"aggregation.kind must be one of {AGGREGATORS}, got {self.aggregation.kind!r}")
        if not 0 < s.budget <= 1:
            raise ValueError("selection.budget must be in (0, 1]")
        if s.kind == "all" and s.budget != 1.0:
            raise ValueError("selection.kind='all' requires budget = 1.0")
        if f.train_batch < 2 or f.min_final_batch < 2:
            raise ValueError("SimSiam needs at least 2 frames per step")
        if f.max_buffer < f.train_batch:
            raise ValueError("federation.max_buffer must be >= train_batch")
        if f.max_buffer >= 2 * f.train_batch:
            raise ValueError("federation.max_buffer must be < 2*train_batch so the buffer stays bounded")
        if not f.train_layers or not set(f.train_layers) <= {"layer3", "layer4"}:
            raise ValueError("federation.train_layers must be a non-empty subset of ['layer3', 'layer4']")
        if f.anchor_mu < 0:
            raise ValueError("federation.anchor_mu must be >= 0")
        if not all(0.0 <= a <= 1.0 for a in self.eval.interp_alphas):
            raise ValueError("eval.interp_alphas must lie in [0, 1]")
        if f.n_rounds < 1:
            raise ValueError("federation.n_rounds must be >= 1")
        if not 0 < self.data.probe_fraction < 0.5:
            raise ValueError("data.probe_fraction must be in (0, 0.5)")
        g = self.stage2
        n = len(g.gen_countries)
        if not (len(g.archs) == len(g.sensor_hw) == len(g.sensor_fov) == len(g.quant_bits) == n):
            raise ValueError("stage2: gen_countries, archs, sensor_hw, sensor_fov, quant_bits need one entry per generation")
        flat = [c for gen in g.gen_countries for c in gen]
        if len(set(flat)) != len(flat):
            raise ValueError("stage2: a country may belong to only one generation")
        if g.teacher_sees not in ("old", "new"):
            raise ValueError("stage2.teacher_sees must be 'old' or 'new'")
        if not all(0 < f <= 1 for f in g.sensor_fov):
            raise ValueError("stage2.sensor_fov values must be in (0, 1]")
        return self


def _fill(cls, raw: dict[str, Any]):
    names = {fld.name for fld in dc.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(raw)
    if "image_hw" in kwargs:
        kwargs["image_hw"] = tuple(kwargs["image_hw"])
    return cls(**kwargs)


def from_dict(raw: dict[str, Any]) -> Config:
    parts = {
        "data": DataCfg, "index": IndexCfg, "federation": FederationCfg, "selection": SelectionCfg,
        "aggregation": AggregationCfg, "eval": EvalCfg, "stage2": Stage2Cfg, "run": RunCfg,
    }
    unknown = set(raw) - set(parts)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return Config(**{k: _fill(c, raw.get(k, {})) for k, c in parts.items()}).validate()


def load(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        return from_dict(tomllib.load(fh))


def override(cfg: Config, **sections: dict[str, Any]) -> Config:
    """Return a validated copy with section fields replaced: override(cfg, selection={'kind': 'loss'})."""
    new = dc.replace(cfg, **{name: dc.replace(getattr(cfg, name), **vals) for name, vals in sections.items()})
    return new.validate()
