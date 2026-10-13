# Do the rare cases survive? Federated learning on the Zenseact Open Dataset

Cars in different countries learn together without sharing camera images. Averaging favors whoever has the most data, and most
driving is ordinary daytime driving. This repo asks whether the rare, safety-relevant situations (night, wet road, snow) get
lost, first when the fleet learns, and then when a new generation of cars with a different model, camera and chip inherits
what the fleet knows. Everything runs on the real data: all 100,000 ZOD Frames, 14 countries, 2020-2022.

It is a small-scale study of known methods with a few new twists, not a new framework. The results are mixed, and the
negative ones are kept in.

## The two stages

**Stage 1, learning together.** Twelve countries are the clients. Each sees its frames once, in time order, keeps no dataset, and
picks a share of them to learn from. Learning is self-supervised (SimSiam on a pretrained ResNet-18), no labels needed. The
server averages the updates, either plainly (FedAvg) or *condition-balanced*, where every road-condition x light group counts
equally however few frames it has. Four ways of picking frames are compared: random, high loss ("surprising"), novelty, and
rarity. Quality is measured with frozen-feature probes on held-out frames the clients never saw, reported for the average and
for the **worst group**.

**Stage 2, the generation gap.** Generation A (ResNet-18, full camera) is trained federated with labels. Generation B (a smaller
MobileNet, lower resolution, narrower view) and generation C (smaller still, 8-bit weights) cannot take A's weights. Knowledge
crosses as predictions: the old model runs inside the new car on its own frames (teacher-in-the-car), or on a small shared
public set. A rarity-aware version weights rare conditions up. Chains A -> B -> C play the telephone game, and knowledge
is also sent back from the small models to A (reverse transfer).

**Follow-ups on stage 1.** Two other aggregation rules (L-DAWA, FedEMA) and four ways of adapting less (train only the last
block, pull towards the starting weights, mix adapted and starting weights) test whether the cost of adapting can be avoided.

## Results

Generated from `results/*.json` by `python -m zodfl.readme_block`; nothing below is typed by hand.
Groups are road condition (normal, wet, snow) x light (day, night). The headline task is `vru_present` (a pedestrian or
vulnerable road user is visible) and the headline metric is worst-group accuracy; both were fixed before any result existed.

<!-- RESULTS:START -->
Data: 100,000 ZOD frames, 14 countries; stream 80,026, probe-train 9,987, probe-test 9,987.

| condition group | frames | probe-test frames |
|---|---|---|
| normal/day | 60,764 | 6,071 |
| normal/night | 17,010 | 1,700 |
| wet/day | 12,015 | 1,199 |
| snow/day | 4,567 | 456 |
| wet/night | 3,009 | 299 |
| snow/night | 2,635 | 262 |

#### Stage 1: `vru_present`, full-label probe, groups = road condition x light

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| anchor01_all | 5 | 0.858 ± 0.001 | 0.821 ± 0.002 | 0.745 ± 0.009 | 0.016 ± 0.007 | 79845 |
| anchor01_random20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.744 ± 0.004 | 0.006 ± 0.002 | 16207 |
| anchor10_all | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.741 ± 0.002 | 0.004 ± 0.001 | 79845 |
| anchor10_random20 | 5 | 0.857 ± 0.000 | 0.820 ± 0.000 | 0.738 ± 0.001 | 0.005 ± 0.002 | 16207 |
| anchor1_all | 5 | 0.859 ± 0.000 | 0.824 ± 0.001 | 0.751 ± 0.006 | 0.007 ± 0.003 | 79845 |
| anchor1_random20 | 5 | 0.857 ± 0.001 | 0.822 ± 0.002 | 0.745 ± 0.006 | 0.005 ± 0.002 | 16207 |
| balanced_all | 5 | 0.858 ± 0.000 | 0.821 ± 0.001 | 0.744 ± 0.006 | 0.014 ± 0.002 | 79845 |
| balanced_loss20 | 5 | 0.857 ± 0.001 | 0.822 ± 0.001 | 0.742 ± 0.007 | 0.006 ± 0.003 | 16207 |
| balanced_novelty20 | 5 | 0.857 ± 0.001 | 0.822 ± 0.002 | 0.743 ± 0.008 | 0.006 ± 0.002 | 16207 |
| balanced_random20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.743 ± 0.005 | 0.004 ± 0.001 | 16207 |
| balanced_rarity20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.744 ± 0.005 | 0.005 ± 0.002 | 16207 |
| central_all | 5 | 0.858 ± 0.001 | 0.820 ± 0.001 | 0.738 ± 0.004 | 0.021 ± 0.004 | 79845 |
| fedavg_all | 5 | 0.858 ± 0.001 | 0.820 ± 0.002 | 0.743 ± 0.009 | 0.016 ± 0.003 | 79845 |
| fedavg_loss20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.742 ± 0.004 | 0.005 ± 0.002 | 16207 |
| fedavg_novelty20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.002 | 0.743 ± 0.007 | 0.005 ± 0.002 | 16207 |
| fedavg_random20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.745 ± 0.004 | 0.004 ± 0.002 | 16207 |
| fedavg_rarity20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.745 ± 0.005 | 0.005 ± 0.001 | 16207 |
| fedema_all | 5 | 0.858 ± 0.000 | 0.821 ± 0.001 | 0.745 ± 0.008 | 0.015 ± 0.005 | 79845 |
| fedema_random20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.746 ± 0.006 | 0.005 ± 0.002 | 16207 |
| init | 5 | 0.856 ± 0.000 | 0.820 ± 0.000 | 0.742 ± 0.000 | n/a | 0 |
| layer4only_all | 5 | 0.857 ± 0.001 | 0.824 ± 0.001 | 0.755 ± 0.007 | 0.005 ± 0.003 | 79845 |
| layer4only_random20 | 5 | 0.857 ± 0.000 | 0.819 ± 0.000 | 0.740 ± 0.002 | 0.005 ± 0.000 | 16207 |
| ldawa_all | 5 | 0.858 ± 0.001 | 0.821 ± 0.002 | 0.746 ± 0.009 | 0.014 ± 0.003 | 79845 |
| ldawa_norm_all | 5 | 0.858 ± 0.001 | 0.821 ± 0.003 | 0.748 ± 0.010 | 0.013 ± 0.003 | 79845 |
| ldawa_norm_random20 | 5 | 0.857 ± 0.001 | 0.822 ± 0.001 | 0.744 ± 0.004 | 0.005 ± 0.002 | 16207 |
| ldawa_random20 | 5 | 0.857 ± 0.000 | 0.822 ± 0.001 | 0.744 ± 0.004 | 0.005 ± 0.003 | 16207 |
| tax_all | 5 | 0.858 ± 0.001 | 0.821 ± 0.002 | 0.745 ± 0.007 | 0.013 ± 0.002 | 79845 |
| tax_all+interp0.25 | 5 | 0.857 ± 0.000 | 0.823 ± 0.001 | 0.744 ± 0.002 | 0.001 ± 0.000 | 79845 |
| tax_all+interp0.5 | 5 | 0.858 ± 0.001 | 0.823 ± 0.002 | 0.749 ± 0.005 | 0.004 ± 0.002 | 79845 |
| tax_all+interp0.75 | 5 | 0.859 ± 0.001 | 0.824 ± 0.002 | 0.748 ± 0.006 | 0.006 ± 0.004 | 79845 |
| tax_random20 | 5 | 0.857 ± 0.001 | 0.822 ± 0.002 | 0.745 ± 0.006 | 0.005 ± 0.001 | 16207 |
| tax_random20+interp0.25 | 5 | 0.857 ± 0.000 | 0.820 ± 0.001 | 0.740 ± 0.002 | 0.002 ± 0.002 | 16207 |
| tax_random20+interp0.5 | 5 | 0.857 ± 0.000 | 0.820 ± 0.001 | 0.739 ± 0.002 | 0.004 ± 0.002 | 16207 |
| tax_random20+interp0.75 | 5 | 0.857 ± 0.000 | 0.821 ± 0.001 | 0.740 ± 0.004 | 0.004 ± 0.002 | 16207 |

Worst-group accuracy, difference against `fedavg_random20` (paired by seed):

- `anchor01_all`: +0.000 (95% CI -0.009 to +0.009, n=5 seeds)
- `anchor01_random20`: -0.001 (95% CI -0.003 to +0.001, n=5 seeds)
- `anchor10_all`: -0.004 (95% CI -0.007 to -0.001, n=5 seeds)
- `anchor10_random20`: -0.007 (95% CI -0.011 to -0.003, n=5 seeds)
- `anchor1_all`: +0.005 (95% CI -0.004 to +0.014, n=5 seeds)
- `anchor1_random20`: +0.000 (95% CI -0.005 to +0.005, n=5 seeds)
- `balanced_all`: -0.001 (95% CI -0.008 to +0.005, n=5 seeds)
- `balanced_loss20`: -0.003 (95% CI -0.009 to +0.003, n=5 seeds)
- `balanced_novelty20`: -0.002 (95% CI -0.008 to +0.004, n=5 seeds)
- `balanced_random20`: -0.002 (95% CI -0.004 to +0.000, n=5 seeds)
- `balanced_rarity20`: -0.001 (95% CI -0.004 to +0.001, n=5 seeds)
- `central_all`: -0.007 (95% CI -0.015 to +0.000, n=5 seeds)
- `fedavg_all`: -0.002 (95% CI -0.010 to +0.006, n=5 seeds)
- `fedavg_loss20`: -0.003 (95% CI -0.011 to +0.005, n=5 seeds)
- `fedavg_novelty20`: -0.002 (95% CI -0.007 to +0.003, n=5 seeds)
- `fedavg_rarity20`: +0.000 (95% CI -0.003 to +0.003, n=5 seeds)
- `fedema_all`: +0.000 (95% CI -0.007 to +0.007, n=5 seeds)
- `fedema_random20`: +0.001 (95% CI -0.003 to +0.004, n=5 seeds)
- `init`: -0.003 (95% CI -0.007 to +0.002, n=5 seeds)
- `layer4only_all`: +0.009 (95% CI +0.004 to +0.015, n=5 seeds)
- `layer4only_random20`: -0.005 (95% CI -0.011 to +0.002, n=5 seeds)
- `ldawa_all`: +0.001 (95% CI -0.006 to +0.009, n=5 seeds)
- `ldawa_norm_all`: +0.003 (95% CI -0.006 to +0.012, n=5 seeds)
- `ldawa_norm_random20`: -0.001 (95% CI -0.004 to +0.001, n=5 seeds)
- `ldawa_random20`: -0.001 (95% CI -0.003 to +0.001, n=5 seeds)
- `tax_all`: +0.000 (95% CI -0.007 to +0.007, n=5 seeds)
- `tax_all+interp0.25`: -0.001 (95% CI -0.006 to +0.003, n=5 seeds)
- `tax_all+interp0.5`: +0.004 (95% CI -0.003 to +0.011, n=5 seeds)
- `tax_all+interp0.75`: +0.003 (95% CI -0.005 to +0.012, n=5 seeds)
- `tax_random20`: +0.000 (95% CI -0.005 to +0.005, n=5 seeds)
- `tax_random20+interp0.25`: -0.005 (95% CI -0.009 to +0.000, n=5 seeds)
- `tax_random20+interp0.5`: -0.006 (95% CI -0.009 to -0.003, n=5 seeds)
- `tax_random20+interp0.75`: -0.005 (95% CI -0.007 to -0.002, n=5 seeds)

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`vru_present`):

- random20: -0.002 (95% CI -0.004 to +0.000, n=5 seeds)
- loss20: +0.000 (95% CI -0.009 to +0.009, n=5 seeds)
- novelty20: +0.000 (95% CI -0.005 to +0.005, n=5 seeds)
- rarity20: -0.001 (95% CI -0.004 to +0.001, n=5 seeds)
- all: +0.001 (95% CI -0.005 to +0.007, n=5 seeds)

#### Stage 1: `vru_present@10pct`, low-label probe (10% of probe-train), groups = road condition x light

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| anchor01_all | 5 | 0.807 ± 0.001 | 0.770 ± 0.003 | 0.633 ± 0.006 | 0.036 ± 0.006 | 79845 |
| anchor01_random20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.002 | 0.650 ± 0.004 | 0.020 ± 0.004 | 16207 |
| anchor10_all | 5 | 0.805 ± 0.000 | 0.775 ± 0.001 | 0.656 ± 0.005 | 0.016 ± 0.006 | 79845 |
| anchor10_random20 | 5 | 0.805 ± 0.001 | 0.777 ± 0.001 | 0.657 ± 0.007 | 0.015 ± 0.006 | 16207 |
| anchor1_all | 5 | 0.807 ± 0.002 | 0.770 ± 0.003 | 0.628 ± 0.006 | 0.041 ± 0.006 | 79845 |
| anchor1_random20 | 5 | 0.804 ± 0.001 | 0.776 ± 0.003 | 0.653 ± 0.003 | 0.018 ± 0.004 | 16207 |
| balanced_all | 5 | 0.807 ± 0.001 | 0.770 ± 0.003 | 0.632 ± 0.006 | 0.037 ± 0.006 | 79845 |
| balanced_loss20 | 5 | 0.804 ± 0.001 | 0.776 ± 0.001 | 0.656 ± 0.004 | 0.016 ± 0.006 | 16207 |
| balanced_novelty20 | 5 | 0.804 ± 0.001 | 0.775 ± 0.001 | 0.655 ± 0.003 | 0.017 ± 0.002 | 16207 |
| balanced_random20 | 5 | 0.804 ± 0.001 | 0.776 ± 0.002 | 0.654 ± 0.002 | 0.016 ± 0.003 | 16207 |
| balanced_rarity20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.002 | 0.654 ± 0.006 | 0.017 ± 0.006 | 16207 |
| central_all | 5 | 0.808 ± 0.001 | 0.773 ± 0.002 | 0.642 ± 0.004 | 0.028 ± 0.003 | 79845 |
| fedavg_all | 5 | 0.807 ± 0.002 | 0.770 ± 0.004 | 0.633 ± 0.007 | 0.035 ± 0.007 | 79845 |
| fedavg_loss20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.001 | 0.651 ± 0.004 | 0.021 ± 0.006 | 16207 |
| fedavg_novelty20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.001 | 0.653 ± 0.003 | 0.019 ± 0.001 | 16207 |
| fedavg_random20 | 5 | 0.804 ± 0.001 | 0.776 ± 0.002 | 0.649 ± 0.005 | 0.022 ± 0.006 | 16207 |
| fedavg_rarity20 | 5 | 0.804 ± 0.001 | 0.775 ± 0.001 | 0.648 ± 0.004 | 0.023 ± 0.006 | 16207 |
| fedema_all | 5 | 0.807 ± 0.001 | 0.771 ± 0.002 | 0.633 ± 0.013 | 0.036 ± 0.013 | 79845 |
| fedema_random20 | 5 | 0.805 ± 0.001 | 0.775 ± 0.002 | 0.649 ± 0.005 | 0.022 ± 0.005 | 16207 |
| init | 5 | 0.807 ± 0.000 | 0.779 ± 0.000 | 0.669 ± 0.000 | n/a | 0 |
| layer4only_all | 5 | 0.808 ± 0.001 | 0.778 ± 0.001 | 0.650 ± 0.003 | 0.019 ± 0.003 | 79845 |
| layer4only_random20 | 5 | 0.807 ± 0.001 | 0.780 ± 0.001 | 0.666 ± 0.003 | 0.006 ± 0.001 | 16207 |
| ldawa_all | 5 | 0.807 ± 0.002 | 0.770 ± 0.003 | 0.630 ± 0.004 | 0.039 ± 0.004 | 79845 |
| ldawa_norm_all | 5 | 0.807 ± 0.002 | 0.771 ± 0.003 | 0.634 ± 0.008 | 0.035 ± 0.008 | 79845 |
| ldawa_norm_random20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.002 | 0.649 ± 0.004 | 0.022 ± 0.006 | 16207 |
| ldawa_random20 | 5 | 0.805 ± 0.001 | 0.776 ± 0.002 | 0.649 ± 0.004 | 0.021 ± 0.005 | 16207 |
| tax_all | 5 | 0.807 ± 0.002 | 0.771 ± 0.003 | 0.634 ± 0.008 | 0.035 ± 0.008 | 79845 |
| tax_all+interp0.25 | 5 | 0.805 ± 0.000 | 0.773 ± 0.001 | 0.639 ± 0.008 | 0.029 ± 0.008 | 79845 |
| tax_all+interp0.5 | 5 | 0.806 ± 0.002 | 0.770 ± 0.004 | 0.629 ± 0.007 | 0.040 ± 0.007 | 79845 |
| tax_all+interp0.75 | 5 | 0.806 ± 0.001 | 0.770 ± 0.002 | 0.635 ± 0.008 | 0.033 ± 0.008 | 79845 |
| tax_random20 | 5 | 0.804 ± 0.001 | 0.775 ± 0.003 | 0.649 ± 0.005 | 0.022 ± 0.006 | 16207 |
| tax_random20+interp0.25 | 5 | 0.806 ± 0.001 | 0.779 ± 0.001 | 0.666 ± 0.003 | 0.004 ± 0.002 | 16207 |
| tax_random20+interp0.5 | 5 | 0.805 ± 0.001 | 0.777 ± 0.002 | 0.662 ± 0.006 | 0.008 ± 0.006 | 16207 |
| tax_random20+interp0.75 | 5 | 0.805 ± 0.001 | 0.776 ± 0.001 | 0.655 ± 0.003 | 0.014 ± 0.003 | 16207 |

Worst-group accuracy, difference against `fedavg_random20` (paired by seed):

- `anchor01_all`: -0.016 (95% CI -0.026 to -0.006, n=5 seeds)
- `anchor01_random20`: +0.001 (95% CI -0.001 to +0.004, n=5 seeds)
- `anchor10_all`: +0.007 (95% CI -0.000 to +0.015, n=5 seeds)
- `anchor10_random20`: +0.008 (95% CI +0.000 to +0.016, n=5 seeds)
- `anchor1_all`: -0.021 (95% CI -0.026 to -0.015, n=5 seeds)
- `anchor1_random20`: +0.004 (95% CI +0.001 to +0.007, n=5 seeds)
- `balanced_all`: -0.017 (95% CI -0.023 to -0.010, n=5 seeds)
- `balanced_loss20`: +0.007 (95% CI -0.001 to +0.015, n=5 seeds)
- `balanced_novelty20`: +0.006 (95% CI +0.001 to +0.011, n=5 seeds)
- `balanced_random20`: +0.005 (95% CI -0.001 to +0.012, n=5 seeds)
- `balanced_rarity20`: +0.005 (95% CI -0.002 to +0.013, n=5 seeds)
- `central_all`: -0.007 (95% CI -0.014 to +0.001, n=5 seeds)
- `fedavg_all`: -0.015 (95% CI -0.023 to -0.007, n=5 seeds)
- `fedavg_loss20`: +0.002 (95% CI -0.000 to +0.004, n=5 seeds)
- `fedavg_novelty20`: +0.004 (95% CI -0.001 to +0.009, n=5 seeds)
- `fedavg_rarity20`: -0.001 (95% CI -0.006 to +0.005, n=5 seeds)
- `fedema_all`: -0.016 (95% CI -0.031 to -0.001, n=5 seeds)
- `fedema_random20`: +0.000 (95% CI -0.004 to +0.004, n=5 seeds)
- `init`: +0.020 (95% CI +0.014 to +0.026, n=5 seeds)
- `layer4only_all`: +0.001 (95% CI -0.008 to +0.010, n=5 seeds)
- `layer4only_random20`: +0.017 (95% CI +0.011 to +0.023, n=5 seeds)
- `ldawa_all`: -0.019 (95% CI -0.027 to -0.011, n=5 seeds)
- `ldawa_norm_all`: -0.015 (95% CI -0.023 to -0.006, n=5 seeds)
- `ldawa_norm_random20`: +0.001 (95% CI -0.004 to +0.005, n=5 seeds)
- `ldawa_random20`: +0.001 (95% CI -0.001 to +0.003, n=5 seeds)
- `tax_all`: -0.015 (95% CI -0.027 to -0.003, n=5 seeds)
- `tax_all+interp0.25`: -0.009 (95% CI -0.019 to +0.001, n=5 seeds)
- `tax_all+interp0.5`: -0.020 (95% CI -0.029 to -0.011, n=5 seeds)
- `tax_all+interp0.75`: -0.013 (95% CI -0.019 to -0.008, n=5 seeds)
- `tax_random20`: +0.000 (95% CI -0.005 to +0.005, n=5 seeds)
- `tax_random20+interp0.25`: +0.017 (95% CI +0.011 to +0.023, n=5 seeds)
- `tax_random20+interp0.5`: +0.013 (95% CI +0.008 to +0.017, n=5 seeds)
- `tax_random20+interp0.75`: +0.006 (95% CI +0.001 to +0.011, n=5 seeds)

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`vru_present@10pct`):

- random20: +0.005 (95% CI -0.001 to +0.012, n=5 seeds)
- loss20: +0.005 (95% CI -0.003 to +0.013, n=5 seeds)
- novelty20: +0.002 (95% CI -0.000 to +0.004, n=5 seeds)
- rarity20: +0.006 (95% CI +0.001 to +0.011, n=5 seeds)
- all: -0.001 (95% CI -0.006 to +0.003, n=5 seeds)

#### Stage 2 (generation gap): `vru_present`, groups = road condition x light

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.586 ± 0.034 | 1.000 ± 0.000 | 0.780 ± 0.010 | 0.847 ± 0.006 |
| s2_B_floor | 5 | 0.447 ± 0.134 | 0.756 ± 0.191 | 0.649 ± 0.071 | 0.753 ± 0.028 |
| s2_B_oracle | 5 | 0.685 ± 0.006 | 1.173 ± 0.073 | 0.787 ± 0.004 | 0.787 ± 0.007 |
| s2_B_kd_public | 5 | 0.593 ± 0.063 | 1.010 ± 0.058 | 0.769 ± 0.018 | 0.834 ± 0.009 |
| s2_B_kd_incar | 5 | 0.612 ± 0.032 | 1.045 ± 0.012 | 0.769 ± 0.011 | 0.833 ± 0.005 |
| s2_B_kd_incar_rarity | 5 | 0.614 ± 0.039 | 1.048 ± 0.026 | 0.770 ± 0.011 | 0.834 ± 0.005 |
| s2_C_oracle | 5 | 0.701 ± 0.010 | 1.200 ± 0.063 | 0.757 ± 0.007 | 0.723 ± 0.018 |
| s2_C_kd_incar | 5 | 0.634 ± 0.035 | 1.082 ± 0.040 | 0.764 ± 0.005 | 0.800 ± 0.011 |
| s2_C_kd_incar_rarity | 5 | 0.657 ± 0.032 | 1.123 ± 0.038 | 0.766 ± 0.006 | 0.798 ± 0.009 |

Worst-group accuracy, paired by seed (`vru_present`):

- B: rarity-aware minus plain in-car distillation: +0.002 (95% CI -0.015 to +0.018, n=5 seeds)
- C: rarity-aware minus plain in-car distillation: +0.024 (95% CI -0.003 to +0.051, n=5 seeds)
- B: in-car minus public-set distillation: +0.019 (95% CI -0.023 to +0.061, n=5 seeds)
- B (in-car) minus teacher A: +0.026 (95% CI +0.019 to +0.034, n=5 seeds)
- C (in-car) minus teacher A: +0.048 (95% CI +0.020 to +0.076, n=5 seeds)
- C minus B, one more generation (in-car): +0.021 (95% CI -0.001 to +0.044, n=5 seeds)
- B in-car distillation minus true labels: -0.073 (95% CI -0.113 to -0.033, n=5 seeds)
- C in-car distillation minus true labels: -0.068 (95% CI -0.102 to -0.033, n=5 seeds)
- B in-car distillation minus the 1%-label floor: +0.165 (95% CI +0.035 to +0.295, n=5 seeds)

Gap closed on worst-group accuracy, (student - floor) / (oracle - floor), generation B (`vru_present`):

- s2_B_kd_public: 0.625 ± 0.193 (n=5 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar: 0.659 ± 0.094 (n=5 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar_rarity: 0.682 ± 0.101 (n=5 seeds with a floor-oracle gap above 0.05)

#### Reverse transfer: `vru_present`

Accuracy by country group of the test frames (`vru_present`; gen1 = the old cars' own countries):

| model | seeds | gen1 countries | gen2 countries | gen3 countries |
|---|---|---|---|---|
| rev_A_teacher | 5 | 0.843 ± 0.004 | 0.846 ± 0.007 | 0.862 ± 0.011 |
| rev_A_from_control | 5 | 0.834 ± 0.006 | 0.838 ± 0.013 | 0.846 ± 0.010 |
| rev_A_from_oracle | 5 | 0.775 ± 0.007 | 0.815 ± 0.009 | 0.839 ± 0.009 |
| rev_A_from_ensemble | 5 | 0.847 ± 0.002 | 0.886 ± 0.005 | 0.871 ± 0.005 |
| rev_B_oracle | 5 | 0.762 ± 0.008 | 0.831 ± 0.010 | 0.833 ± 0.004 |
| rev_B_control | 5 | 0.827 ± 0.003 | 0.842 ± 0.007 | 0.848 ± 0.010 |

Paired by seed, reverse-transferred A' minus the original A (95% CI):

- rev_A_from_control, gen1: -0.009 (95% CI -0.012 to -0.006, n=5)
- rev_A_from_control, gen2: -0.008 (95% CI -0.017 to +0.000, n=5)
- rev_A_from_control, gen3: -0.016 (95% CI -0.020 to -0.011, n=5)
- rev_A_from_oracle, gen1: -0.068 (95% CI -0.074 to -0.062, n=5)
- rev_A_from_oracle, gen2: -0.031 (95% CI -0.039 to -0.023, n=5)
- rev_A_from_oracle, gen3: -0.023 (95% CI -0.033 to -0.013, n=5)
- rev_A_from_ensemble, gen1: +0.004 (95% CI +0.000 to +0.008, n=5)
- rev_A_from_ensemble, gen2: +0.040 (95% CI +0.028 to +0.052, n=5)
- rev_A_from_ensemble, gen3: +0.009 (95% CI -0.010 to +0.028, n=5)

All tasks, per-country tables and every paired difference: [results/REPORT.md](results/REPORT.md).
<!-- RESULTS:END -->

## What the results say

**Stage 1.**
- Adapting the pretrained backbone with federated self-supervised learning did **not** beat leaving it alone. With the
  full-label probe every variant, including the untrained backbone (`init`), is within noise of the others.
- With the low-label probe, `init` has the best worst-group score, and the more a run adapts, the lower it gets. Using every
  frame is worse than using 20% of them. That is a dose-response: more adaptation moved the representation away from the
  pretrained features, and the rare groups paid.
- Condition-balanced averaging and rarity selection did not give a reliable gain. The matched comparisons are small and their
  intervals mostly include zero. I do not claim an effect.
- Surprise-based selection (`loss`, `novelty`) did not increase forgetting compared with random selection.
- Pooling all the data in one place was not reliably different from federating it (nominally a little better in the
  low-label probe, with an interval that includes zero). Both were clearly below the untrained backbone there.
- **Better aggregation did not help.** L-DAWA (layer-wise cosine weighting) and FedEMA matched plain FedAvg to within 0.003 on
  the low-label worst group, with and without frame selection. A possible reason (not tested) is that every client moves the
  shared weights in a similar direction, so how the updates are averaged hardly matters.
- **What did help was adapting less.** Low-label worst-group accuracy (5 seeds, `init` = 0.669): FedAvg on all frames 0.633,
  on 20% of frames 0.649. Training only the last block (`layer4`) on 20% of frames gave 0.666, nearly back to `init`. An
  L2 pull towards the starting weights helped a little and monotonically with its strength (mu 0.1 / 1 / 10 on 20% of frames:
  0.650 / 0.653 / 0.657). Averaging the adapted weights with the starting weights (WiSE-FT style, 25% adapted) on 20% of
  frames gave 0.666, again about `init`. The same interpolation on all frames did not help (0.639 / 0.629 / 0.635 at 25 / 50 /
  75%). The best of these only recovers the untrained backbone; none beats it. Three interpolation weights were tried, so
  that 0.666 is the best of three, not a pre-registered test.

**Stage 2.**
- On the headline task, a student that only sees the teacher's predictions on its own frames closed most of the gap between
  "1% of labels" and "all labels" with no labels at all: about two thirds of it (see the gap-closed lines above).
- The students' worst-group accuracy is at or above the teacher's, while their overall accuracy is a little lower. They trade
  a little average accuracy for balance. A plausible reason, not tested here: the students see frames from countries the
  teacher never trained on.
- Knowledge did not decay over a second generation on the headline task, but it did on other tasks (see `results/REPORT.md`,
  traffic density), so "it survives" is task-dependent.
- On road condition, all three generation-C models, including the one trained on true labels, score identically to three
  decimals in every seed. That is consistent with a constant prediction, and since the model trained on true labels does the
  same, it points to a limit of C's cheap camera (resolution, narrower view, 8-bit weights) and not to a failure of
  distillation. I did not check the predicted class counts, so this is an inference.
- Rarity-aware distillation gave no consistent benefit.
- **Reverse transfer (B back to A).** Distilling a single small student (B) back into the big model (A) did not improve A: on
  `vru_present` it was 0.009 to 0.016 lower in accuracy in the old cars' countries and the newest countries (intervals exclude
  zero). Distilling the *ensemble* of the later students into A was better: +0.040 (95% CI +0.028 to +0.052) in the
  middle generation's countries, +0.004 and +0.009 elsewhere (intervals touch or include zero). So the small models
  collectively carry something the old model lacks, only for the countries they were built for; a single student does
  not. On other tasks the picture is mixed (road type improved in some country groups and fell in others; traffic density
  fell in two of three), so I read this as a task-dependent result with one clear positive. A model distilled from true labels
  of the new data (`rev_A_from_oracle`) forgot the old countries most (-0.068), which is the usual catastrophic-forgetting pattern.

## Limitations, stated plainly
- The probes and stage-2 targets are image-level labels taken from ZOD's metadata (pedestrian visible, road type, traffic
  density, road condition). This is not object detection, which the RISE thesis topic is about. Detection with ZOD's 2D boxes is
  the obvious next stage.
- The backbone starts from ImageNet weights; this is adaptation of a shipped model, not training from scratch.
- Five seeds. Seeds 3 and 4 were added after seeds 0-2 to tighten the intervals; all five are reported. Intervals are t-based
  on five numbers, and many comparisons were made, so single intervals that barely exclude zero should not be over-read.
- Recipe choices (BatchNorm in the SimSiam heads, a small learning rate for the pretrained layers, the stage-2 recipe) came
  from pilot runs on a separate seed and from label-free or validation-split checks, not from the test results. The first
  stage-2 attempt had a diverged teacher and weak teachers; its results are archived on the Modal volume and are not
  reported. Details and numbers: `docs/BLUEPRINT.md`, ADR-8 to ADR-16.
- Quantization is simulated (8-bit weights), and the "transition cars" carry both cameras so the teacher can read the old view.
- Not done: object detection, an ensemble of per-country teachers as the first teacher, LiDAR, Sequences.
- The last 31 runs (phase-2 grid and reverse transfer) ran on Kaggle GPUs, the rest on Modal T4s; each result file records
  its hardware. The same code and seeds were used, but small numeric differences between GPUs are possible.
- Compared with published federated self-supervised work, this study uses real country-level non-IID data on 8 MP driving
  images, but at a small scale. I did not search for every related paper; see `literature_review.md`.

## Reproduce
Needs a Modal account with GPU access and the ZOD files (apply at zod.zenseact.com). Heavy compute runs on Modal; nothing
large is stored on the laptop.

```bash
modal secret create zod-download ZOD_URL="<your link>"         # link from Zenseact's approval email
modal run modal_download.py::download --version full            # camera + annotations + infos, no LiDAR
modal run modal_index.py::explore                               # per-frame metadata -> volume
modal run modal_app.py::index && modal run modal_app.py::prep   # frame index, 100k images -> uint8 shards
modal run --detach modal_app.py::grid                           # stage 1: config/grid.toml x seeds
modal run --detach modal_app.py::stage2 --seeds 0,1,2,3,4       # stage 2
modal volume get zod-data /results results_local
python -m zodfl.readme_block --results results_local/results --readme README.md
```
The phase-2 grid (`config/grid_phase2.toml`) and reverse transfer can also run on Kaggle (free GPUs, private datasets):
`python -m zodfl.kaggle_bundle --username <you>` builds the code bundle and kernel; `kaggle_push.py` uploads the shards from
Modal so nothing passes through a laptop; `kaggle/worker.py` runs the job list in `kaggle/jobs.json` and writes result files.
Tests (synthetic data only, no GPU): `python -m pytest`.

## Layout
`config/` all settings and the experiment grid. `src/zodfl/` library (clients, selectors, aggregation, SimSiam, probes,
stage-2 generations, report). `modal_*.py` pipeline entry points. `tests/` unit and end-to-end tests. `docs/BLUEPRINT.md`
design, decisions and evidence. `results/` raw result files per run and `REPORT.md`; `results/pilots/` health-check runs.

## Data and attribution
Zenseact Open Dataset (ZOD), released under CC BY-SA 4.0: Alibeigi et al., *Zenseact Open Dataset: A large-scale and diverse
multimodal dataset for autonomous driving*, ICCV 2023. Images used are the face- and plate-blurred version. Frames are not
redistributed here.
