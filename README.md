# Do the rare cases survive? Federated learning on the Zenseact Open Dataset

Cars in different countries learn together without sharing images. Averaging favours whoever has the most data, and most
driving is ordinary daytime driving. This repo asks whether rare, safety-relevant situations (night, wet road, snow) get lost
when a fleet learns together, and when a new car generation with a different model and camera inherits what the fleet knows.
It runs on all 100,000 ZOD frames (14 countries, 2020-2022).

It is a small-scale study of known methods, not a new framework. Results are mixed and the negative ones are kept in.

## What was done

1. **Stage 1, learning together.** 12 countries are clients. Each sees its frames once, in time order, and picks a share of them.
   Learning is self-supervised (SimSiam on a pretrained ResNet-18). The server averages with FedAvg, condition-balanced
   weights, L-DAWA or FedEMA. Quality is measured with frozen-feature probes, reported for the average and the **worst group**
   (road condition x light). Follow-ups test ways to adapt less: train only the last block, pull toward the starting weights,
   mix adapted and starting weights.
2. **Stage 2, generation gap.** A big model (generation A) cannot hand its weights to a smaller one with a different camera
   (B, then C with 8-bit weights). Knowledge crosses as predictions: the old model labels the new car's own frames. Also tested:
   sending knowledge back from the small models to A.
3. **Detection.** The same generation-gap idea on ZOD's 2D boxes with Faster R-CNN (ResNet-18 teacher, MobileNetV3 student).

The headline task is `vru_present` (a pedestrian or vulnerable road user is visible) and the headline metric is worst-group
accuracy; both were fixed before any result existed (`docs/BLUEPRINT.md`).

## Results

<!-- RESULTS:START -->
Data: 100,000 ZOD frames, 14 countries; stream 80,026, probe-train 9,987, probe-test 9,987. Rarest groups: wet/night 3,009 and snow/night 2,635 frames.

**Stage 1** (`vru_present@10pct`: low-label probe, 5 seeds; `init` = the pretrained model with no federated training). Worst-group = the weakest road-condition x light group.

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| anchor10_random20 | 5 | 0.805 ± 0.001 | 0.777 ± 0.001 | 0.657 ± 0.007 | 0.015 ± 0.006 | 16207 |
| balanced_random20 | 5 | 0.804 ± 0.001 | 0.776 ± 0.002 | 0.654 ± 0.002 | 0.016 ± 0.003 | 16207 |
| central_all | 5 | 0.808 ± 0.001 | 0.773 ± 0.002 | 0.642 ± 0.004 | 0.028 ± 0.003 | 79845 |
| fedavg_all | 5 | 0.807 ± 0.002 | 0.770 ± 0.004 | 0.633 ± 0.007 | 0.035 ± 0.007 | 79845 |
| fedavg_random20 | 5 | 0.804 ± 0.001 | 0.776 ± 0.002 | 0.649 ± 0.005 | 0.022 ± 0.006 | 16207 |
| fedema_random20 | 5 | 0.805 ± 0.001 | 0.775 ± 0.002 | 0.649 ± 0.005 | 0.022 ± 0.005 | 16207 |
| init | 5 | 0.807 ± 0.000 | 0.779 ± 0.000 | 0.669 ± 0.000 | n/a | 0 |
| layer4only_random20 | 5 | 0.807 ± 0.001 | 0.780 ± 0.001 | 0.666 ± 0.003 | 0.006 ± 0.001 | 16207 |
| ldawa_random20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.002 | 0.649 ± 0.004 | 0.021 ± 0.005 | 16207 |
| tax_random20+interp0.25 | 5 | 0.806 ± 0.001 | 0.779 ± 0.001 | 0.666 ± 0.003 | 0.004 ± 0.002 | 16207 |

**Stage 2, generation gap** (`vru_present`, 5 seeds).

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.586 ± 0.034 | 1.000 ± 0.000 | 0.780 ± 0.010 | 0.847 ± 0.006 |
| s2_B_floor | 5 | 0.447 ± 0.134 | 0.756 ± 0.191 | 0.649 ± 0.071 | 0.753 ± 0.028 |
| s2_B_oracle | 5 | 0.685 ± 0.006 | 1.173 ± 0.073 | 0.787 ± 0.004 | 0.787 ± 0.007 |
| s2_B_kd_incar | 5 | 0.612 ± 0.032 | 1.045 ± 0.012 | 0.769 ± 0.011 | 0.833 ± 0.005 |
| s2_C_kd_incar | 5 | 0.634 ± 0.035 | 1.082 ± 0.040 | 0.764 ± 0.005 | 0.800 ± 0.011 |

Gap closed on worst-group accuracy, (student - floor) / (oracle - floor), generation B (`vru_present`):
- s2_B_kd_incar: 0.659 ± 0.094 (n=5 seeds with a floor-oracle gap above 0.05)

**Reverse transfer, B back to A** (`vru_present`).

| model | seeds | gen1 countries | gen2 countries | gen3 countries |
|---|---|---|---|---|
| rev_A_teacher | 5 | 0.843 ± 0.004 | 0.846 ± 0.007 | 0.862 ± 0.011 |
| rev_A_from_control | 5 | 0.834 ± 0.006 | 0.838 ± 0.013 | 0.846 ± 0.010 |
| rev_A_from_ensemble | 5 | 0.847 ± 0.002 | 0.886 ± 0.005 | 0.871 ± 0.005 |

- rev_A_from_control, gen2: -0.008 (95% CI -0.017 to +0.000, n=5)
- rev_A_from_ensemble, gen1: +0.004 (95% CI +0.000 to +0.008, n=5)
- rev_A_from_ensemble, gen2: +0.040 (95% CI +0.028 to +0.052, n=5)
- rev_A_from_ensemble, gen3: +0.009 (95% CI -0.010 to +0.028, n=5)

**Detection** (Faster R-CNN, teacher A = ResNet-18, student B = MobileNetV3; ZOD 2D boxes; COCO-style AP on held-out test frames, 3 seeds).

| model | seeds | AP | AP50 | worst-group AP | mean-group AP | Vehicle AP | VulnerableVehicle AP | Pedestrian AP |
|---|---|---|---|---|---|---|---|---|
| det_A_teacher | 3 | 0.136 ± 0.002 | 0.294 ± 0.004 | 0.112 ± 0.003 | 0.129 ± 0.003 | 0.285 ± 0.002 | 0.052 ± 0.003 | 0.070 ± 0.002 |
| det_B_floor | 3 | 0.009 ± 0.000 | 0.032 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.000 | 0.026 ± 0.001 | 0.001 ± 0.002 | 0.000 ± 0.000 |
| det_B_kd_floor | 3 | 0.069 ± 0.001 | 0.165 ± 0.001 | 0.049 ± 0.003 | 0.069 ± 0.002 | 0.153 ± 0.001 | 0.022 ± 0.001 | 0.031 ± 0.003 |
| det_B_kd_incar | 3 | 0.069 ± 0.001 | 0.166 ± 0.002 | 0.048 ± 0.001 | 0.069 ± 0.001 | 0.154 ± 0.001 | 0.020 ± 0.000 | 0.032 ± 0.002 |
| det_B_oracle | 3 | 0.062 ± 0.001 | 0.155 ± 0.003 | 0.042 ± 0.001 | 0.065 ± 0.003 | 0.139 ± 0.002 | 0.018 ± 0.001 | 0.027 ± 0.002 |

Paired by seed, AP:
- `det_B_kd_incar` minus `det_B_floor`: +0.060 (95% CI +0.058 to +0.062, n=3 seeds)
- `det_B_kd_floor` minus `det_B_floor`: +0.060 (95% CI +0.058 to +0.062, n=3 seeds)
- `det_B_oracle` minus `det_B_kd_incar`: -0.007 (95% CI -0.012 to -0.002, n=3 seeds)
- `det_B_kd_floor` minus `det_B_kd_incar`: -0.000 (95% CI -0.003 to +0.003, n=3 seeds)


Every variant, task, per-country table and paired difference: [results/REPORT.md](results/REPORT.md).
<!-- RESULTS:END -->

## What the results say

**Stage 1.**
- Federated self-supervised training did **not** beat the pretrained model. With the low-label probe, the untrained backbone
  (`init`) has the best worst-group score (0.669), and the more a run adapts, the lower it gets: all frames 0.633, 20% of
  frames 0.649. More adaptation moved the features away from the pretrained ones, and the rare groups paid.
- Smarter averaging did not help: L-DAWA and FedEMA match FedAvg to within 0.003. Condition-balanced averaging and rarity-based
  frame selection gave no reliable gain, and pooling all data centrally was not reliably different from federating.
- Adapting less is what helps: training only the last block (0.666) or mixing 25% adapted with 75% starting weights (0.666)
  gets back to about `init`. Neither beats it (the mixing weight was the best of three tried).

**Stage 2 (classification).**
- A student that only sees the teacher's predictions on its own frames closes about two thirds of the gap between "1% of
  labels" and "all labels" with no labels. Its worst-group accuracy (0.612) is at or above the teacher's (0.586), at a small
  cost in average accuracy.
- Knowledge did not decay over a second generation on the headline task, but it did on others (traffic density), so "it
  survives" depends on the task. On road condition all generation-C models score identically, which points to a limit of C's
  cheap camera rather than of distillation (inferred, not checked).
- Reverse transfer: distilling one small model back into A makes A slightly worse. Distilling the ensemble of the later
  students into A helps in the middle generation's countries (+0.040, 95% CI +0.028 to +0.052) and is about neutral elsewhere.

**Detection.**
- A student trained only on the teacher's confident boxes (AP 0.069) beat the same student trained on all true boxes
  (0.062; paired difference +0.007, 95% CI +0.002 to +0.012). I did not test why. Teacher AP is 0.136.
- Rare condition groups are lower for the teacher and the student alike (wet/night about 0.05 against 0.08 for normal/day),
  and I saw no sign that the transfer loses them more; with 3 seeds this is weak evidence.

## Limitations

- Stage 1 and the classification stage 2 use image-level labels from ZOD's metadata, not boxes. The backbone starts from
  ImageNet weights, so this is adaptation of a shipped model, not training from scratch.
- Detection is small: one teacher-to-student step, 3 seeds, 11,659 frames at 960x540. Absolute AP is low because ZOD objects
  are tiny (a pedestrian is about 5 px wide at that size). Federated pretraining was not run for detection. The "10% labels"
  student got the same epochs as the others, so it had about 10x fewer steps and is nearly untrained (AP 0.009); the
  gap-closed ratio there is inflated and should not be read as "better than labels".
- Five seeds (three for detection); intervals are t-based on few numbers and many comparisons were made, so intervals that
  barely exclude zero should not be over-read.
- Recipe choices (BatchNorm heads, small backbone learning rate, stage-2 and detection settings) came from pilots on a separate
  seed and validation frames, not from test results. The first stage-2 attempt diverged; it is archived and not reported
  (`docs/BLUEPRINT.md`, ADR-8 to ADR-18).
- Quantization is simulated. Not done: LiDAR, Sequences, soft distillation for detection, a third detection generation.
- The last runs (phase-2 grid, reverse transfer, detection) ran on Kaggle GPUs, the rest on Modal T4s; each result file records
  its hardware.

## Reproduce

Needs a Modal account and the ZOD files (apply at zod.zenseact.com). Nothing large is stored on the laptop.

```bash
modal secret create zod-download ZOD_URL="<your link>"          # link from Zenseact's approval email
modal run modal_download.py::download --version full
modal run modal_index.py::explore
modal run modal_app.py::index && modal run modal_app.py::prep   # frame index, 100k images -> uint8 shards
modal run --detach modal_app.py::grid                           # stage 1: config/grid.toml x seeds
modal run --detach modal_app.py::stage2 --seeds 0,1,2,3,4       # stage 2
modal run modal_det_prep.py::select && modal run modal_det_prep.py::parts && modal run modal_det_prep.py::finalize  # detection images
python -m zodfl.readme_block --results results --readme README.md
```

The phase-2 grid, reverse transfer and detection can also run on free Kaggle GPUs: `python -m zodfl.kaggle_bundle --username
<you>` builds the bundles, `kaggle_push.py` uploads data from Modal, and `kaggle/worker.py` / `kaggle/det_worker.py` run the
job lists. Tests (synthetic data, no GPU): `python -m pytest`.

## Layout

`config/` settings and grids. `src/zodfl/` library. `modal_*.py`, `kaggle*` pipeline entry points. `tests/`. `docs/BLUEPRINT.md`
design and decisions. `results/` raw result files, `REPORT.md` (all tables), `detection/`.

## Data and attribution

Zenseact Open Dataset (ZOD), CC BY-SA 4.0: Alibeigi et al., *Zenseact Open Dataset: A large-scale and diverse multimodal
dataset for autonomous driving*, ICCV 2023. Images used are the face- and plate-blurred version. Frames are not redistributed.
