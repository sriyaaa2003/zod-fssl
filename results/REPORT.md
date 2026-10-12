# Results (generated; do not edit by hand)

Source: 105 run files in `results`. Groups smaller than the configured minimum test size are excluded.

## Stage 1, task `vru_present`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
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
| init | 5 | 0.856 ± 0.000 | 0.820 ± 0.000 | 0.742 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

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
- `init`: -0.003 (95% CI -0.007 to +0.002, n=5 seeds)

## Stage 1, task `vru_present`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.858 ± 0.000 | 0.852 ± 0.001 | 0.766 ± 0.005 | 0.026 ± 0.005 | 79845 |
| balanced_loss20 | 5 | 0.857 ± 0.001 | 0.842 ± 0.002 | 0.751 ± 0.005 | 0.021 ± 0.001 | 16207 |
| balanced_novelty20 | 5 | 0.857 ± 0.001 | 0.843 ± 0.003 | 0.757 ± 0.011 | 0.013 ± 0.011 | 16207 |
| balanced_random20 | 5 | 0.857 ± 0.000 | 0.842 ± 0.003 | 0.753 ± 0.008 | 0.019 ± 0.008 | 16207 |
| balanced_rarity20 | 5 | 0.857 ± 0.000 | 0.842 ± 0.001 | 0.751 ± 0.005 | 0.020 ± 0.000 | 16207 |
| central_all | 5 | 0.858 ± 0.001 | 0.854 ± 0.004 | 0.771 ± 0.005 | 0.023 ± 0.011 | 79845 |
| fedavg_all | 5 | 0.858 ± 0.001 | 0.853 ± 0.001 | 0.767 ± 0.005 | 0.026 ± 0.005 | 79845 |
| fedavg_loss20 | 5 | 0.857 ± 0.000 | 0.842 ± 0.002 | 0.753 ± 0.008 | 0.019 ± 0.003 | 16207 |
| fedavg_novelty20 | 5 | 0.857 ± 0.000 | 0.842 ± 0.004 | 0.757 ± 0.011 | 0.015 ± 0.009 | 16207 |
| fedavg_random20 | 5 | 0.857 ± 0.000 | 0.842 ± 0.003 | 0.753 ± 0.008 | 0.019 ± 0.008 | 16207 |
| fedavg_rarity20 | 5 | 0.857 ± 0.000 | 0.842 ± 0.001 | 0.751 ± 0.005 | 0.020 ± 0.000 | 16207 |
| init | 5 | 0.856 ± 0.000 | 0.843 ± 0.000 | 0.745 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: +0.013 (95% CI +0.004 to +0.022, n=5 seeds)
- `balanced_loss20`: -0.002 (95% CI -0.007 to +0.003, n=5 seeds)
- `balanced_novelty20`: +0.004 (95% CI -0.003 to +0.010, n=5 seeds)
- `balanced_random20`: +0.000 (95% CI +0.000 to +0.000, n=5 seeds)
- `balanced_rarity20`: -0.002 (95% CI -0.007 to +0.003, n=5 seeds)
- `central_all`: +0.018 (95% CI +0.002 to +0.033, n=5 seeds)
- `fedavg_all`: +0.014 (95% CI +0.004 to +0.023, n=5 seeds)
- `fedavg_loss20`: +0.000 (95% CI -0.008 to +0.008, n=5 seeds)
- `fedavg_novelty20`: +0.004 (95% CI -0.003 to +0.010, n=5 seeds)
- `fedavg_rarity20`: -0.002 (95% CI -0.007 to +0.003, n=5 seeds)
- `init`: -0.008 (95% CI -0.018 to +0.002, n=5 seeds)

## Stage 1, task `vru_present@10pct`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
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
| init | 5 | 0.807 ± 0.000 | 0.779 ± 0.000 | 0.669 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

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
- `init`: +0.020 (95% CI +0.014 to +0.026, n=5 seeds)

## Stage 1, task `vru_present@10pct`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.807 ± 0.001 | 0.786 ± 0.001 | 0.631 ± 0.009 | 0.057 ± 0.017 | 79845 |
| balanced_loss20 | 5 | 0.804 ± 0.001 | 0.788 ± 0.005 | 0.647 ± 0.020 | 0.030 ± 0.012 | 16207 |
| balanced_novelty20 | 5 | 0.804 ± 0.001 | 0.789 ± 0.002 | 0.655 ± 0.011 | 0.029 ± 0.011 | 16207 |
| balanced_random20 | 5 | 0.804 ± 0.001 | 0.789 ± 0.003 | 0.651 ± 0.009 | 0.039 ± 0.015 | 16207 |
| balanced_rarity20 | 5 | 0.805 ± 0.001 | 0.790 ± 0.002 | 0.651 ± 0.016 | 0.034 ± 0.010 | 16207 |
| central_all | 5 | 0.808 ± 0.001 | 0.787 ± 0.003 | 0.655 ± 0.018 | 0.073 ± 0.018 | 79845 |
| fedavg_all | 5 | 0.807 ± 0.002 | 0.786 ± 0.002 | 0.639 ± 0.011 | 0.078 ± 0.017 | 79845 |
| fedavg_loss20 | 5 | 0.805 ± 0.001 | 0.789 ± 0.003 | 0.651 ± 0.021 | 0.037 ± 0.009 | 16207 |
| fedavg_novelty20 | 5 | 0.805 ± 0.001 | 0.791 ± 0.002 | 0.659 ± 0.011 | 0.028 ± 0.011 | 16207 |
| fedavg_random20 | 5 | 0.804 ± 0.001 | 0.789 ± 0.002 | 0.651 ± 0.009 | 0.048 ± 0.019 | 16207 |
| fedavg_rarity20 | 5 | 0.804 ± 0.001 | 0.789 ± 0.002 | 0.651 ± 0.016 | 0.035 ± 0.009 | 16207 |
| init | 5 | 0.807 ± 0.000 | 0.789 ± 0.000 | 0.608 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: -0.020 (95% CI -0.020 to -0.020, n=5 seeds)
- `balanced_loss20`: -0.004 (95% CI -0.031 to +0.023, n=5 seeds)
- `balanced_novelty20`: +0.004 (95% CI -0.007 to +0.015, n=5 seeds)
- `balanced_random20`: +0.000 (95% CI +0.000 to +0.000, n=5 seeds)
- `balanced_rarity20`: -0.000 (95% CI -0.017 to +0.017, n=5 seeds)
- `central_all`: +0.004 (95% CI -0.023 to +0.031, n=5 seeds)
- `fedavg_all`: -0.012 (95% CI -0.025 to +0.002, n=5 seeds)
- `fedavg_loss20`: -0.000 (95% CI -0.024 to +0.024, n=5 seeds)
- `fedavg_novelty20`: +0.008 (95% CI -0.005 to +0.021, n=5 seeds)
- `fedavg_rarity20`: -0.000 (95% CI -0.017 to +0.017, n=5 seeds)
- `init`: -0.043 (95% CI -0.054 to -0.032, n=5 seeds)

## Stage 1, task `road_type`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.664 ± 0.002 | 0.630 ± 0.003 | 0.501 ± 0.010 | 0.019 ± 0.009 | 79845 |
| balanced_loss20 | 5 | 0.665 ± 0.001 | 0.630 ± 0.001 | 0.497 ± 0.008 | 0.008 ± 0.002 | 16207 |
| balanced_novelty20 | 5 | 0.664 ± 0.000 | 0.628 ± 0.001 | 0.488 ± 0.006 | 0.009 ± 0.002 | 16207 |
| balanced_random20 | 5 | 0.665 ± 0.000 | 0.628 ± 0.002 | 0.492 ± 0.009 | 0.010 ± 0.004 | 16207 |
| balanced_rarity20 | 5 | 0.664 ± 0.001 | 0.627 ± 0.002 | 0.488 ± 0.006 | 0.009 ± 0.003 | 16207 |
| central_all | 5 | 0.664 ± 0.002 | 0.631 ± 0.002 | 0.503 ± 0.010 | 0.021 ± 0.009 | 79845 |
| fedavg_all | 5 | 0.664 ± 0.002 | 0.630 ± 0.002 | 0.500 ± 0.012 | 0.019 ± 0.006 | 79845 |
| fedavg_loss20 | 5 | 0.665 ± 0.001 | 0.629 ± 0.002 | 0.494 ± 0.007 | 0.009 ± 0.002 | 16207 |
| fedavg_novelty20 | 5 | 0.664 ± 0.001 | 0.628 ± 0.002 | 0.491 ± 0.006 | 0.009 ± 0.002 | 16207 |
| fedavg_random20 | 5 | 0.665 ± 0.000 | 0.629 ± 0.001 | 0.491 ± 0.007 | 0.009 ± 0.003 | 16207 |
| fedavg_rarity20 | 5 | 0.665 ± 0.000 | 0.628 ± 0.001 | 0.488 ± 0.004 | 0.010 ± 0.003 | 16207 |
| init | 5 | 0.665 ± 0.000 | 0.627 ± 0.000 | 0.492 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: +0.010 (95% CI -0.001 to +0.021, n=5 seeds)
- `balanced_loss20`: +0.006 (95% CI +0.003 to +0.009, n=5 seeds)
- `balanced_novelty20`: -0.003 (95% CI -0.011 to +0.005, n=5 seeds)
- `balanced_random20`: +0.001 (95% CI -0.005 to +0.007, n=5 seeds)
- `balanced_rarity20`: -0.003 (95% CI -0.007 to +0.002, n=5 seeds)
- `central_all`: +0.012 (95% CI +0.002 to +0.022, n=5 seeds)
- `fedavg_all`: +0.009 (95% CI -0.009 to +0.026, n=5 seeds)
- `fedavg_loss20`: +0.003 (95% CI -0.003 to +0.009, n=5 seeds)
- `fedavg_novelty20`: +0.000 (95% CI -0.009 to +0.009, n=5 seeds)
- `fedavg_rarity20`: -0.003 (95% CI -0.007 to +0.001, n=5 seeds)
- `init`: +0.001 (95% CI -0.008 to +0.010, n=5 seeds)

## Stage 1, task `road_type`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.664 ± 0.002 | 0.658 ± 0.004 | 0.389 ± 0.023 | 0.073 ± 0.019 | 79845 |
| balanced_loss20 | 5 | 0.665 ± 0.001 | 0.661 ± 0.001 | 0.429 ± 0.010 | 0.030 ± 0.008 | 16207 |
| balanced_novelty20 | 5 | 0.664 ± 0.000 | 0.661 ± 0.002 | 0.420 ± 0.008 | 0.037 ± 0.008 | 16207 |
| balanced_random20 | 5 | 0.665 ± 0.000 | 0.661 ± 0.002 | 0.429 ± 0.000 | 0.029 ± 0.000 | 16207 |
| balanced_rarity20 | 5 | 0.664 ± 0.001 | 0.660 ± 0.001 | 0.420 ± 0.013 | 0.037 ± 0.013 | 16207 |
| central_all | 5 | 0.664 ± 0.002 | 0.661 ± 0.006 | 0.394 ± 0.028 | 0.063 ± 0.028 | 79845 |
| fedavg_all | 5 | 0.664 ± 0.002 | 0.656 ± 0.004 | 0.371 ± 0.023 | 0.093 ± 0.023 | 79845 |
| fedavg_loss20 | 5 | 0.665 ± 0.001 | 0.663 ± 0.001 | 0.431 ± 0.012 | 0.029 ± 0.009 | 16207 |
| fedavg_novelty20 | 5 | 0.664 ± 0.001 | 0.660 ± 0.002 | 0.414 ± 0.010 | 0.043 ± 0.010 | 16207 |
| fedavg_random20 | 5 | 0.665 ± 0.000 | 0.661 ± 0.002 | 0.426 ± 0.006 | 0.031 ± 0.006 | 16207 |
| fedavg_rarity20 | 5 | 0.665 ± 0.000 | 0.661 ± 0.001 | 0.426 ± 0.012 | 0.033 ± 0.010 | 16207 |
| init | 5 | 0.665 ± 0.000 | 0.666 ± 0.000 | 0.457 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: -0.037 (95% CI -0.064 to -0.010, n=5 seeds)
- `balanced_loss20`: +0.003 (95% CI -0.005 to +0.011, n=5 seeds)
- `balanced_novelty20`: -0.006 (95% CI -0.015 to +0.004, n=5 seeds)
- `balanced_random20`: +0.003 (95% CI -0.005 to +0.011, n=5 seeds)
- `balanced_rarity20`: -0.006 (95% CI -0.015 to +0.004, n=5 seeds)
- `central_all`: -0.031 (95% CI -0.061 to -0.002, n=5 seeds)
- `fedavg_all`: -0.054 (95% CI -0.077 to -0.031, n=5 seeds)
- `fedavg_loss20`: +0.006 (95% CI -0.010 to +0.022, n=5 seeds)
- `fedavg_novelty20`: -0.011 (95% CI -0.019 to -0.003, n=5 seeds)
- `fedavg_rarity20`: +0.000 (95% CI -0.013 to +0.013, n=5 seeds)
- `init`: +0.031 (95% CI +0.023 to +0.039, n=5 seeds)

## Stage 1, task `road_type@10pct`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.578 ± 0.003 | 0.550 ± 0.002 | 0.423 ± 0.011 | 0.019 ± 0.005 | 79845 |
| balanced_loss20 | 5 | 0.581 ± 0.001 | 0.554 ± 0.001 | 0.444 ± 0.005 | 0.013 ± 0.004 | 16207 |
| balanced_novelty20 | 5 | 0.582 ± 0.001 | 0.555 ± 0.001 | 0.449 ± 0.008 | 0.012 ± 0.004 | 16207 |
| balanced_random20 | 5 | 0.582 ± 0.002 | 0.553 ± 0.001 | 0.438 ± 0.009 | 0.013 ± 0.002 | 16207 |
| balanced_rarity20 | 5 | 0.582 ± 0.002 | 0.554 ± 0.001 | 0.440 ± 0.003 | 0.014 ± 0.005 | 16207 |
| central_all | 5 | 0.577 ± 0.004 | 0.549 ± 0.005 | 0.420 ± 0.016 | 0.021 ± 0.003 | 79845 |
| fedavg_all | 5 | 0.578 ± 0.003 | 0.550 ± 0.002 | 0.422 ± 0.012 | 0.018 ± 0.003 | 79845 |
| fedavg_loss20 | 5 | 0.581 ± 0.001 | 0.555 ± 0.001 | 0.447 ± 0.006 | 0.013 ± 0.003 | 16207 |
| fedavg_novelty20 | 5 | 0.581 ± 0.002 | 0.555 ± 0.001 | 0.447 ± 0.009 | 0.014 ± 0.006 | 16207 |
| fedavg_random20 | 5 | 0.582 ± 0.002 | 0.553 ± 0.001 | 0.436 ± 0.006 | 0.013 ± 0.002 | 16207 |
| fedavg_rarity20 | 5 | 0.582 ± 0.002 | 0.554 ± 0.001 | 0.439 ± 0.004 | 0.011 ± 0.005 | 16207 |
| init | 5 | 0.579 ± 0.000 | 0.551 ± 0.000 | 0.425 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: -0.013 (95% CI -0.030 to +0.004, n=5 seeds)
- `balanced_loss20`: +0.008 (95% CI +0.003 to +0.013, n=5 seeds)
- `balanced_novelty20`: +0.013 (95% CI +0.006 to +0.020, n=5 seeds)
- `balanced_random20`: +0.002 (95% CI -0.003 to +0.007, n=5 seeds)
- `balanced_rarity20`: +0.004 (95% CI -0.001 to +0.009, n=5 seeds)
- `central_all`: -0.016 (95% CI -0.037 to +0.005, n=5 seeds)
- `fedavg_all`: -0.014 (95% CI -0.031 to +0.002, n=5 seeds)
- `fedavg_loss20`: +0.011 (95% CI +0.004 to +0.019, n=5 seeds)
- `fedavg_novelty20`: +0.011 (95% CI +0.007 to +0.016, n=5 seeds)
- `fedavg_rarity20`: +0.003 (95% CI +0.000 to +0.006, n=5 seeds)
- `init`: -0.011 (95% CI -0.019 to -0.004, n=5 seeds)

## Stage 1, task `road_type@10pct`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.578 ± 0.003 | 0.575 ± 0.005 | 0.334 ± 0.019 | 0.044 ± 0.019 | 79845 |
| balanced_loss20 | 5 | 0.581 ± 0.001 | 0.580 ± 0.002 | 0.366 ± 0.016 | 0.030 ± 0.006 | 16207 |
| balanced_novelty20 | 5 | 0.582 ± 0.001 | 0.580 ± 0.002 | 0.366 ± 0.008 | 0.037 ± 0.015 | 16207 |
| balanced_random20 | 5 | 0.582 ± 0.002 | 0.581 ± 0.002 | 0.363 ± 0.008 | 0.033 ± 0.008 | 16207 |
| balanced_rarity20 | 5 | 0.582 ± 0.002 | 0.581 ± 0.001 | 0.369 ± 0.012 | 0.035 ± 0.016 | 16207 |
| central_all | 5 | 0.577 ± 0.004 | 0.572 ± 0.004 | 0.311 ± 0.019 | 0.061 ± 0.026 | 79845 |
| fedavg_all | 5 | 0.578 ± 0.003 | 0.574 ± 0.004 | 0.331 ± 0.023 | 0.044 ± 0.013 | 79845 |
| fedavg_loss20 | 5 | 0.581 ± 0.001 | 0.581 ± 0.002 | 0.374 ± 0.019 | 0.030 ± 0.006 | 16207 |
| fedavg_novelty20 | 5 | 0.581 ± 0.002 | 0.580 ± 0.003 | 0.366 ± 0.022 | 0.033 ± 0.008 | 16207 |
| fedavg_random20 | 5 | 0.582 ± 0.002 | 0.581 ± 0.002 | 0.363 ± 0.008 | 0.035 ± 0.008 | 16207 |
| fedavg_rarity20 | 5 | 0.582 ± 0.002 | 0.581 ± 0.002 | 0.366 ± 0.008 | 0.030 ± 0.006 | 16207 |
| init | 5 | 0.579 ± 0.000 | 0.578 ± 0.000 | 0.343 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: -0.029 (95% CI -0.050 to -0.007, n=5 seeds)
- `balanced_loss20`: +0.003 (95% CI -0.012 to +0.018, n=5 seeds)
- `balanced_novelty20`: +0.003 (95% CI -0.012 to +0.018, n=5 seeds)
- `balanced_random20`: +0.000 (95% CI +0.000 to +0.000, n=5 seeds)
- `balanced_rarity20`: +0.006 (95% CI -0.010 to +0.022, n=5 seeds)
- `central_all`: -0.051 (95% CI -0.078 to -0.025, n=5 seeds)
- `fedavg_all`: -0.031 (95% CI -0.055 to -0.008, n=5 seeds)
- `fedavg_loss20`: +0.011 (95% CI -0.012 to +0.035, n=5 seeds)
- `fedavg_novelty20`: +0.003 (95% CI -0.023 to +0.029, n=5 seeds)
- `fedavg_rarity20`: +0.003 (95% CI -0.005 to +0.011, n=5 seeds)
- `init`: -0.020 (95% CI -0.030 to -0.010, n=5 seeds)

## Stage 1, task `traffic_density`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.572 ± 0.002 | 0.682 ± 0.001 | 0.523 ± 0.002 | 0.008 ± 0.002 | 79845 |
| balanced_loss20 | 5 | 0.571 ± 0.000 | 0.678 ± 0.001 | 0.525 ± 0.000 | 0.009 ± 0.003 | 16207 |
| balanced_novelty20 | 5 | 0.572 ± 0.001 | 0.678 ± 0.001 | 0.525 ± 0.002 | 0.008 ± 0.003 | 16207 |
| balanced_random20 | 5 | 0.572 ± 0.001 | 0.679 ± 0.001 | 0.525 ± 0.001 | 0.008 ± 0.003 | 16207 |
| balanced_rarity20 | 5 | 0.571 ± 0.001 | 0.678 ± 0.001 | 0.525 ± 0.001 | 0.009 ± 0.003 | 16207 |
| central_all | 5 | 0.572 ± 0.001 | 0.681 ± 0.002 | 0.524 ± 0.001 | 0.016 ± 0.009 | 79845 |
| fedavg_all | 5 | 0.572 ± 0.002 | 0.683 ± 0.001 | 0.524 ± 0.002 | 0.008 ± 0.002 | 79845 |
| fedavg_loss20 | 5 | 0.572 ± 0.001 | 0.679 ± 0.001 | 0.525 ± 0.001 | 0.010 ± 0.002 | 16207 |
| fedavg_novelty20 | 5 | 0.571 ± 0.001 | 0.678 ± 0.001 | 0.524 ± 0.001 | 0.009 ± 0.002 | 16207 |
| fedavg_random20 | 5 | 0.572 ± 0.001 | 0.679 ± 0.001 | 0.525 ± 0.001 | 0.009 ± 0.003 | 16207 |
| fedavg_rarity20 | 5 | 0.571 ± 0.001 | 0.678 ± 0.001 | 0.525 ± 0.001 | 0.009 ± 0.003 | 16207 |
| init | 5 | 0.571 ± 0.000 | 0.678 ± 0.000 | 0.525 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: -0.001 (95% CI -0.004 to +0.001, n=5 seeds)
- `balanced_loss20`: +0.001 (95% CI -0.001 to +0.002, n=5 seeds)
- `balanced_novelty20`: +0.000 (95% CI -0.002 to +0.003, n=5 seeds)
- `balanced_random20`: +0.000 (95% CI -0.000 to +0.001, n=5 seeds)
- `balanced_rarity20`: +0.000 (95% CI -0.001 to +0.001, n=5 seeds)
- `central_all`: -0.001 (95% CI -0.002 to +0.000, n=5 seeds)
- `fedavg_all`: -0.001 (95% CI -0.004 to +0.002, n=5 seeds)
- `fedavg_loss20`: +0.000 (95% CI -0.001 to +0.002, n=5 seeds)
- `fedavg_novelty20`: -0.001 (95% CI -0.002 to +0.001, n=5 seeds)
- `fedavg_rarity20`: +0.000 (95% CI -0.001 to +0.001, n=5 seeds)
- `init`: +0.000 (95% CI -0.001 to +0.001, n=5 seeds)

## Stage 1, task `traffic_density`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.572 ± 0.002 | 0.604 ± 0.003 | 0.508 ± 0.003 | 0.026 ± 0.003 | 79845 |
| balanced_loss20 | 5 | 0.571 ± 0.000 | 0.599 ± 0.002 | 0.501 ± 0.010 | 0.012 ± 0.008 | 16207 |
| balanced_novelty20 | 5 | 0.572 ± 0.001 | 0.598 ± 0.002 | 0.493 ± 0.008 | 0.020 ± 0.000 | 16207 |
| balanced_random20 | 5 | 0.572 ± 0.001 | 0.599 ± 0.001 | 0.503 ± 0.010 | 0.010 ± 0.006 | 16207 |
| balanced_rarity20 | 5 | 0.571 ± 0.001 | 0.598 ± 0.002 | 0.498 ± 0.011 | 0.017 ± 0.006 | 16207 |
| central_all | 5 | 0.572 ± 0.001 | 0.601 ± 0.005 | 0.490 ± 0.009 | 0.047 ± 0.015 | 79845 |
| fedavg_all | 5 | 0.572 ± 0.002 | 0.604 ± 0.003 | 0.504 ± 0.006 | 0.031 ± 0.013 | 79845 |
| fedavg_loss20 | 5 | 0.572 ± 0.001 | 0.599 ± 0.002 | 0.504 ± 0.008 | 0.010 ± 0.006 | 16207 |
| fedavg_novelty20 | 5 | 0.571 ± 0.001 | 0.597 ± 0.002 | 0.493 ± 0.007 | 0.020 ± 0.000 | 16207 |
| fedavg_random20 | 5 | 0.572 ± 0.001 | 0.599 ± 0.001 | 0.498 ± 0.010 | 0.013 ± 0.007 | 16207 |
| fedavg_rarity20 | 5 | 0.571 ± 0.001 | 0.598 ± 0.003 | 0.498 ± 0.011 | 0.016 ± 0.006 | 16207 |
| init | 5 | 0.571 ± 0.000 | 0.596 ± 0.000 | 0.505 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: +0.010 (95% CI -0.005 to +0.024, n=5 seeds)
- `balanced_loss20`: +0.003 (95% CI -0.022 to +0.027, n=5 seeds)
- `balanced_novelty20`: -0.005 (95% CI -0.024 to +0.014, n=5 seeds)
- `balanced_random20`: +0.005 (95% CI -0.007 to +0.016, n=5 seeds)
- `balanced_rarity20`: -0.001 (95% CI -0.017 to +0.016, n=5 seeds)
- `central_all`: -0.009 (95% CI -0.029 to +0.012, n=5 seeds)
- `fedavg_all`: +0.006 (95% CI -0.013 to +0.025, n=5 seeds)
- `fedavg_loss20`: +0.006 (95% CI -0.016 to +0.027, n=5 seeds)
- `fedavg_novelty20`: -0.005 (95% CI -0.024 to +0.013, n=5 seeds)
- `fedavg_rarity20`: -0.000 (95% CI -0.018 to +0.017, n=5 seeds)
- `init`: +0.007 (95% CI -0.005 to +0.019, n=5 seeds)

## Stage 1, task `traffic_density@10pct`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.500 ± 0.005 | 0.596 ± 0.006 | 0.463 ± 0.005 | 0.023 ± 0.008 | 79845 |
| balanced_loss20 | 5 | 0.496 ± 0.001 | 0.597 ± 0.002 | 0.458 ± 0.001 | 0.015 ± 0.004 | 16207 |
| balanced_novelty20 | 5 | 0.496 ± 0.001 | 0.597 ± 0.002 | 0.457 ± 0.002 | 0.012 ± 0.003 | 16207 |
| balanced_random20 | 5 | 0.497 ± 0.001 | 0.597 ± 0.001 | 0.458 ± 0.001 | 0.010 ± 0.005 | 16207 |
| balanced_rarity20 | 5 | 0.497 ± 0.001 | 0.597 ± 0.002 | 0.459 ± 0.001 | 0.009 ± 0.003 | 16207 |
| central_all | 5 | 0.498 ± 0.003 | 0.593 ± 0.003 | 0.460 ± 0.003 | 0.031 ± 0.013 | 79845 |
| fedavg_all | 5 | 0.499 ± 0.004 | 0.596 ± 0.006 | 0.463 ± 0.005 | 0.020 ± 0.010 | 79845 |
| fedavg_loss20 | 5 | 0.497 ± 0.001 | 0.598 ± 0.002 | 0.458 ± 0.002 | 0.011 ± 0.005 | 16207 |
| fedavg_novelty20 | 5 | 0.496 ± 0.001 | 0.597 ± 0.002 | 0.457 ± 0.002 | 0.012 ± 0.004 | 16207 |
| fedavg_random20 | 5 | 0.497 ± 0.001 | 0.597 ± 0.001 | 0.458 ± 0.001 | 0.012 ± 0.005 | 16207 |
| fedavg_rarity20 | 5 | 0.497 ± 0.001 | 0.597 ± 0.003 | 0.459 ± 0.001 | 0.012 ± 0.004 | 16207 |
| init | 5 | 0.494 ± 0.000 | 0.596 ± 0.000 | 0.453 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: +0.005 (95% CI -0.001 to +0.012, n=5 seeds)
- `balanced_loss20`: -0.001 (95% CI -0.003 to +0.002, n=5 seeds)
- `balanced_novelty20`: -0.001 (95% CI -0.003 to +0.001, n=5 seeds)
- `balanced_random20`: -0.000 (95% CI -0.001 to +0.001, n=5 seeds)
- `balanced_rarity20`: +0.001 (95% CI -0.000 to +0.002, n=5 seeds)
- `central_all`: +0.002 (95% CI -0.003 to +0.007, n=5 seeds)
- `fedavg_all`: +0.005 (95% CI -0.001 to +0.010, n=5 seeds)
- `fedavg_loss20`: -0.000 (95% CI -0.003 to +0.002, n=5 seeds)
- `fedavg_novelty20`: -0.001 (95% CI -0.003 to +0.002, n=5 seeds)
- `fedavg_rarity20`: +0.001 (95% CI -0.001 to +0.002, n=5 seeds)
- `init`: -0.005 (95% CI -0.006 to -0.003, n=5 seeds)

## Stage 1, task `traffic_density@10pct`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.500 ± 0.005 | 0.528 ± 0.002 | 0.419 ± 0.011 | 0.089 ± 0.021 | 79845 |
| balanced_loss20 | 5 | 0.496 ± 0.001 | 0.522 ± 0.002 | 0.382 ± 0.009 | 0.019 ± 0.014 | 16207 |
| balanced_novelty20 | 5 | 0.496 ± 0.001 | 0.521 ± 0.003 | 0.380 ± 0.013 | 0.023 ± 0.017 | 16207 |
| balanced_random20 | 5 | 0.497 ± 0.001 | 0.523 ± 0.001 | 0.393 ± 0.005 | 0.013 ± 0.009 | 16207 |
| balanced_rarity20 | 5 | 0.497 ± 0.001 | 0.522 ± 0.001 | 0.393 ± 0.009 | 0.015 ± 0.009 | 16207 |
| central_all | 5 | 0.498 ± 0.003 | 0.517 ± 0.004 | 0.392 ± 0.024 | 0.098 ± 0.024 | 79845 |
| fedavg_all | 5 | 0.499 ± 0.004 | 0.526 ± 0.003 | 0.412 ± 0.014 | 0.097 ± 0.019 | 79845 |
| fedavg_loss20 | 5 | 0.497 ± 0.001 | 0.523 ± 0.003 | 0.388 ± 0.009 | 0.024 ± 0.011 | 16207 |
| fedavg_novelty20 | 5 | 0.496 ± 0.001 | 0.521 ± 0.003 | 0.380 ± 0.015 | 0.018 ± 0.014 | 16207 |
| fedavg_random20 | 5 | 0.497 ± 0.001 | 0.522 ± 0.001 | 0.389 ± 0.006 | 0.011 ± 0.005 | 16207 |
| fedavg_rarity20 | 5 | 0.497 ± 0.001 | 0.522 ± 0.001 | 0.393 ± 0.009 | 0.016 ± 0.008 | 16207 |
| init | 5 | 0.494 ± 0.000 | 0.512 ± 0.000 | 0.374 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: +0.030 (95% CI +0.010 to +0.050, n=5 seeds)
- `balanced_loss20`: -0.007 (95% CI -0.025 to +0.012, n=5 seeds)
- `balanced_novelty20`: -0.009 (95% CI -0.027 to +0.009, n=5 seeds)
- `balanced_random20`: +0.004 (95% CI -0.003 to +0.012, n=5 seeds)
- `balanced_rarity20`: +0.004 (95% CI -0.011 to +0.020, n=5 seeds)
- `central_all`: +0.003 (95% CI -0.033 to +0.039, n=5 seeds)
- `fedavg_all`: +0.023 (95% CI -0.001 to +0.046, n=5 seeds)
- `fedavg_loss20`: -0.001 (95% CI -0.018 to +0.016, n=5 seeds)
- `fedavg_novelty20`: -0.009 (95% CI -0.029 to +0.011, n=5 seeds)
- `fedavg_rarity20`: +0.004 (95% CI -0.014 to +0.023, n=5 seeds)
- `init`: -0.015 (95% CI -0.023 to -0.008, n=5 seeds)

## Stage 1, task `road_condition`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: n/a (need at least 2 shared seeds)
- `balanced_loss20`: n/a (need at least 2 shared seeds)
- `balanced_novelty20`: n/a (need at least 2 shared seeds)
- `balanced_random20`: n/a (need at least 2 shared seeds)
- `balanced_rarity20`: n/a (need at least 2 shared seeds)
- `central_all`: n/a (need at least 2 shared seeds)
- `fedavg_all`: n/a (need at least 2 shared seeds)
- `fedavg_loss20`: n/a (need at least 2 shared seeds)
- `fedavg_novelty20`: n/a (need at least 2 shared seeds)
- `fedavg_rarity20`: n/a (need at least 2 shared seeds)
- `init`: n/a (need at least 2 shared seeds)

## Stage 1, task `road_condition`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.889 ± 0.001 | 0.886 ± 0.003 | 0.650 ± 0.006 | 0.026 ± 0.011 | 79845 |
| balanced_loss20 | 5 | 0.886 ± 0.000 | 0.875 ± 0.002 | 0.634 ± 0.005 | 0.012 ± 0.003 | 16207 |
| balanced_novelty20 | 5 | 0.886 ± 0.000 | 0.875 ± 0.003 | 0.636 ± 0.002 | 0.012 ± 0.003 | 16207 |
| balanced_random20 | 5 | 0.886 ± 0.001 | 0.875 ± 0.002 | 0.636 ± 0.006 | 0.011 ± 0.003 | 16207 |
| balanced_rarity20 | 5 | 0.887 ± 0.000 | 0.875 ± 0.003 | 0.636 ± 0.002 | 0.010 ± 0.003 | 16207 |
| central_all | 5 | 0.888 ± 0.001 | 0.885 ± 0.002 | 0.652 ± 0.014 | 0.031 ± 0.005 | 79845 |
| fedavg_all | 5 | 0.888 ± 0.001 | 0.885 ± 0.002 | 0.644 ± 0.007 | 0.027 ± 0.010 | 79845 |
| fedavg_loss20 | 5 | 0.887 ± 0.001 | 0.875 ± 0.002 | 0.636 ± 0.002 | 0.012 ± 0.007 | 16207 |
| fedavg_novelty20 | 5 | 0.887 ± 0.000 | 0.876 ± 0.003 | 0.634 ± 0.005 | 0.011 ± 0.003 | 16207 |
| fedavg_random20 | 5 | 0.886 ± 0.001 | 0.874 ± 0.002 | 0.634 ± 0.002 | 0.010 ± 0.004 | 16207 |
| fedavg_rarity20 | 5 | 0.886 ± 0.000 | 0.875 ± 0.002 | 0.634 ± 0.005 | 0.013 ± 0.005 | 16207 |
| init | 5 | 0.886 ± 0.000 | 0.871 ± 0.000 | 0.635 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: +0.015 (95% CI +0.008 to +0.023, n=5 seeds)
- `balanced_loss20`: -0.000 (95% CI -0.005 to +0.005, n=5 seeds)
- `balanced_novelty20`: +0.002 (95% CI -0.004 to +0.008, n=5 seeds)
- `balanced_random20`: +0.002 (95% CI -0.006 to +0.010, n=5 seeds)
- `balanced_rarity20`: +0.002 (95% CI -0.002 to +0.006, n=5 seeds)
- `central_all`: +0.018 (95% CI -0.001 to +0.036, n=5 seeds)
- `fedavg_all`: +0.010 (95% CI +0.001 to +0.019, n=5 seeds)
- `fedavg_loss20`: +0.002 (95% CI -0.002 to +0.006, n=5 seeds)
- `fedavg_novelty20`: -0.000 (95% CI -0.008 to +0.008, n=5 seeds)
- `fedavg_rarity20`: -0.000 (95% CI -0.007 to +0.007, n=5 seeds)
- `init`: +0.001 (95% CI -0.002 to +0.004, n=5 seeds)

## Stage 1, task `road_condition@10pct`, groups `by_cond`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: n/a (need at least 2 shared seeds)
- `balanced_loss20`: n/a (need at least 2 shared seeds)
- `balanced_novelty20`: n/a (need at least 2 shared seeds)
- `balanced_random20`: n/a (need at least 2 shared seeds)
- `balanced_rarity20`: n/a (need at least 2 shared seeds)
- `central_all`: n/a (need at least 2 shared seeds)
- `fedavg_all`: n/a (need at least 2 shared seeds)
- `fedavg_loss20`: n/a (need at least 2 shared seeds)
- `fedavg_novelty20`: n/a (need at least 2 shared seeds)
- `fedavg_rarity20`: n/a (need at least 2 shared seeds)
- `init`: n/a (need at least 2 shared seeds)

## Stage 1, task `road_condition@10pct`, groups `by_country`

| run | seeds | overall acc | mean-group acc | worst-group acc | max forgetting | frames trained on |
|---|---|---|---|---|---|---|
| balanced_all | 5 | 0.852 ± 0.002 | 0.845 ± 0.004 | 0.565 ± 0.008 | 0.056 ± 0.005 | 79845 |
| balanced_loss20 | 5 | 0.856 ± 0.000 | 0.855 ± 0.004 | 0.615 ± 0.008 | 0.022 ± 0.021 | 16207 |
| balanced_novelty20 | 5 | 0.856 ± 0.000 | 0.853 ± 0.001 | 0.612 ± 0.008 | 0.013 ± 0.005 | 16207 |
| balanced_random20 | 5 | 0.856 ± 0.001 | 0.854 ± 0.003 | 0.618 ± 0.007 | 0.018 ± 0.009 | 16207 |
| balanced_rarity20 | 5 | 0.856 ± 0.001 | 0.853 ± 0.003 | 0.618 ± 0.006 | 0.020 ± 0.011 | 16207 |
| central_all | 5 | 0.853 ± 0.001 | 0.843 ± 0.003 | 0.564 ± 0.016 | 0.056 ± 0.017 | 79845 |
| fedavg_all | 5 | 0.852 ± 0.002 | 0.845 ± 0.004 | 0.559 ± 0.006 | 0.061 ± 0.006 | 79845 |
| fedavg_loss20 | 5 | 0.856 ± 0.001 | 0.854 ± 0.003 | 0.613 ± 0.008 | 0.026 ± 0.019 | 16207 |
| fedavg_novelty20 | 5 | 0.856 ± 0.000 | 0.854 ± 0.001 | 0.613 ± 0.008 | 0.015 ± 0.003 | 16207 |
| fedavg_random20 | 5 | 0.856 ± 0.001 | 0.855 ± 0.003 | 0.622 ± 0.005 | 0.016 ± 0.005 | 16207 |
| fedavg_rarity20 | 5 | 0.856 ± 0.001 | 0.853 ± 0.002 | 0.619 ± 0.007 | 0.022 ± 0.016 | 16207 |
| init | 5 | 0.855 ± 0.000 | 0.853 ± 0.000 | 0.608 ± 0.000 | n/a | 0 |

Paired differences against `fedavg_random20` (worst-group accuracy):

- `balanced_all`: -0.057 (95% CI -0.069 to -0.046, n=5 seeds)
- `balanced_loss20`: -0.007 (95% CI -0.016 to +0.002, n=5 seeds)
- `balanced_novelty20`: -0.010 (95% CI -0.020 to +0.000, n=5 seeds)
- `balanced_random20`: -0.004 (95% CI -0.010 to +0.001, n=5 seeds)
- `balanced_rarity20`: -0.004 (95% CI -0.007 to -0.001, n=5 seeds)
- `central_all`: -0.059 (95% CI -0.082 to -0.036, n=5 seeds)
- `fedavg_all`: -0.063 (95% CI -0.072 to -0.054, n=5 seeds)
- `fedavg_loss20`: -0.009 (95% CI -0.018 to +0.000, n=5 seeds)
- `fedavg_novelty20`: -0.009 (95% CI -0.020 to +0.003, n=5 seeds)
- `fedavg_rarity20`: -0.003 (95% CI -0.007 to +0.000, n=5 seeds)
- `init`: -0.014 (95% CI -0.021 to -0.008, n=5 seeds)

## Stage 1, matched aggregator comparison, task `vru_present`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`vru_present`):

- random20: -0.002 (95% CI -0.004 to +0.000, n=5 seeds)
- loss20: +0.000 (95% CI -0.009 to +0.009, n=5 seeds)
- novelty20: +0.000 (95% CI -0.005 to +0.005, n=5 seeds)
- rarity20: -0.001 (95% CI -0.004 to +0.001, n=5 seeds)
- all: +0.001 (95% CI -0.005 to +0.007, n=5 seeds)

## Stage 1, matched aggregator comparison, task `vru_present@10pct`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`vru_present@10pct`):

- random20: +0.005 (95% CI -0.001 to +0.012, n=5 seeds)
- loss20: +0.005 (95% CI -0.003 to +0.013, n=5 seeds)
- novelty20: +0.002 (95% CI -0.000 to +0.004, n=5 seeds)
- rarity20: +0.006 (95% CI +0.001 to +0.011, n=5 seeds)
- all: -0.001 (95% CI -0.006 to +0.003, n=5 seeds)

## Stage 1, matched aggregator comparison, task `road_type`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`road_type`):

- random20: +0.001 (95% CI -0.005 to +0.007, n=5 seeds)
- loss20: +0.003 (95% CI -0.001 to +0.006, n=5 seeds)
- novelty20: -0.003 (95% CI -0.006 to +0.001, n=5 seeds)
- rarity20: +0.001 (95% CI -0.005 to +0.006, n=5 seeds)
- all: +0.001 (95% CI -0.008 to +0.011, n=5 seeds)

## Stage 1, matched aggregator comparison, task `road_type@10pct`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`road_type@10pct`):

- random20: +0.002 (95% CI -0.003 to +0.007, n=5 seeds)
- loss20: -0.003 (95% CI -0.012 to +0.005, n=5 seeds)
- novelty20: +0.001 (95% CI -0.006 to +0.009, n=5 seeds)
- rarity20: +0.001 (95% CI -0.003 to +0.004, n=5 seeds)
- all: +0.001 (95% CI -0.005 to +0.008, n=5 seeds)

## Stage 1, matched aggregator comparison, task `traffic_density`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`traffic_density`):

- random20: +0.000 (95% CI -0.000 to +0.001, n=5 seeds)
- loss20: +0.000 (95% CI -0.000 to +0.001, n=5 seeds)
- novelty20: +0.001 (95% CI +0.000 to +0.002, n=5 seeds)
- rarity20: +0.000 (95% CI -0.000 to +0.001, n=5 seeds)
- all: -0.000 (95% CI -0.002 to +0.001, n=5 seeds)

## Stage 1, matched aggregator comparison, task `traffic_density@10pct`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`traffic_density@10pct`):

- random20: -0.000 (95% CI -0.001 to +0.001, n=5 seeds)
- loss20: -0.000 (95% CI -0.002 to +0.001, n=5 seeds)
- novelty20: -0.000 (95% CI -0.001 to +0.000, n=5 seeds)
- rarity20: +0.000 (95% CI -0.001 to +0.001, n=5 seeds)
- all: +0.001 (95% CI -0.001 to +0.002, n=5 seeds)

## Stage 1, matched aggregator comparison, task `road_condition`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`road_condition`):

- random20: n/a (need at least 2 shared seeds)
- loss20: n/a (need at least 2 shared seeds)
- novelty20: n/a (need at least 2 shared seeds)
- rarity20: n/a (need at least 2 shared seeds)
- all: n/a (need at least 2 shared seeds)

## Stage 1, matched aggregator comparison, task `road_condition@10pct`

Worst-group accuracy, `balanced` minus `fedavg` with the same frame selector (`road_condition@10pct`):

- random20: n/a (need at least 2 shared seeds)
- loss20: n/a (need at least 2 shared seeds)
- novelty20: n/a (need at least 2 shared seeds)
- rarity20: n/a (need at least 2 shared seeds)
- all: n/a (need at least 2 shared seeds)

## Stage 2 (generation gap), task `vru_present`, groups `by_cond`

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

## Stage 2 (generation gap), task `vru_present`, groups `by_country`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.637 ± 0.074 | 1.000 ± 0.000 | 0.809 ± 0.023 | 0.847 ± 0.006 |
| s2_B_floor | 5 | 0.267 ± 0.235 | 0.397 ± 0.314 | 0.592 ± 0.125 | 0.753 ± 0.028 |
| s2_B_oracle | 5 | 0.706 ± 0.011 | 1.122 ± 0.137 | 0.855 ± 0.002 | 0.787 ± 0.007 |
| s2_B_kd_public | 5 | 0.547 ± 0.131 | 0.849 ± 0.134 | 0.774 ± 0.044 | 0.834 ± 0.009 |
| s2_B_kd_incar | 5 | 0.690 ± 0.030 | 1.093 ± 0.090 | 0.807 ± 0.018 | 0.833 ± 0.005 |
| s2_B_kd_incar_rarity | 5 | 0.696 ± 0.023 | 1.103 ± 0.106 | 0.807 ± 0.017 | 0.834 ± 0.005 |
| s2_C_oracle | 5 | 0.610 ± 0.018 | 0.968 ± 0.101 | 0.835 ± 0.005 | 0.723 ± 0.018 |
| s2_C_kd_incar | 5 | 0.745 ± 0.020 | 1.185 ± 0.168 | 0.845 ± 0.003 | 0.800 ± 0.011 |
| s2_C_kd_incar_rarity | 5 | 0.740 ± 0.017 | 1.178 ± 0.165 | 0.845 ± 0.002 | 0.798 ± 0.009 |

### Stage 2 paired comparisons, task `vru_present`

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

## Stage 2 (generation gap), task `road_type`, groups `by_cond`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.191 ± 0.019 | 1.000 ± 0.000 | 0.455 ± 0.008 | 0.589 ± 0.005 |
| s2_B_floor | 5 | 0.060 ± 0.037 | 0.312 ± 0.195 | 0.348 ± 0.023 | 0.498 ± 0.003 |
| s2_B_oracle | 5 | 0.239 ± 0.014 | 1.256 ± 0.104 | 0.492 ± 0.007 | 0.484 ± 0.008 |
| s2_B_kd_public | 5 | 0.205 ± 0.026 | 1.077 ± 0.130 | 0.454 ± 0.017 | 0.565 ± 0.011 |
| s2_B_kd_incar | 5 | 0.215 ± 0.026 | 1.127 ± 0.101 | 0.428 ± 0.019 | 0.544 ± 0.011 |
| s2_B_kd_incar_rarity | 5 | 0.210 ± 0.027 | 1.096 ± 0.075 | 0.427 ± 0.018 | 0.544 ± 0.010 |
| s2_C_oracle | 5 | 0.049 ± 0.006 | 0.257 ± 0.035 | 0.351 ± 0.010 | 0.515 ± 0.016 |
| s2_C_kd_incar | 5 | 0.232 ± 0.018 | 1.225 ± 0.140 | 0.433 ± 0.018 | 0.525 ± 0.009 |
| s2_C_kd_incar_rarity | 5 | 0.223 ± 0.015 | 1.173 ± 0.117 | 0.431 ± 0.019 | 0.525 ± 0.008 |

## Stage 2 (generation gap), task `road_type`, groups `by_country`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.166 ± 0.049 | 1.000 ± 0.000 | 0.432 ± 0.024 | 0.589 ± 0.005 |
| s2_B_floor | 5 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.256 ± 0.006 | 0.498 ± 0.003 |
| s2_B_oracle | 5 | 0.076 ± 0.031 | 0.468 ± 0.200 | 0.381 ± 0.017 | 0.484 ± 0.008 |
| s2_B_kd_public | 5 | 0.139 ± 0.044 | 0.843 ± 0.182 | 0.406 ± 0.035 | 0.565 ± 0.011 |
| s2_B_kd_incar | 5 | 0.069 ± 0.051 | 0.376 ± 0.196 | 0.347 ± 0.031 | 0.544 ± 0.011 |
| s2_B_kd_incar_rarity | 5 | 0.065 ± 0.056 | 0.348 ± 0.243 | 0.345 ± 0.032 | 0.544 ± 0.010 |
| s2_C_oracle | 5 | 0.061 ± 0.022 | 0.379 ± 0.160 | 0.559 ± 0.003 | 0.515 ± 0.016 |
| s2_C_kd_incar | 5 | 0.027 ± 0.019 | 0.162 ± 0.111 | 0.333 ± 0.022 | 0.525 ± 0.009 |
| s2_C_kd_incar_rarity | 5 | 0.022 ± 0.023 | 0.118 ± 0.125 | 0.326 ± 0.024 | 0.525 ± 0.008 |

### Stage 2 paired comparisons, task `road_type`

Worst-group accuracy, paired by seed (`road_type`):

- B: rarity-aware minus plain in-car distillation: -0.005 (95% CI -0.015 to +0.004, n=5 seeds)
- C: rarity-aware minus plain in-car distillation: -0.010 (95% CI -0.018 to -0.001, n=5 seeds)
- B: in-car minus public-set distillation: +0.010 (95% CI -0.023 to +0.043, n=5 seeds)
- B (in-car) minus teacher A: +0.024 (95% CI -0.002 to +0.050, n=5 seeds)
- C (in-car) minus teacher A: +0.041 (95% CI +0.011 to +0.072, n=5 seeds)
- C minus B, one more generation (in-car): +0.017 (95% CI -0.003 to +0.038, n=5 seeds)
- B in-car distillation minus true labels: -0.024 (95% CI -0.041 to -0.007, n=5 seeds)
- C in-car distillation minus true labels: +0.184 (95% CI +0.162 to +0.205, n=5 seeds)
- B in-car distillation minus the 1%-label floor: +0.155 (95% CI +0.104 to +0.206, n=5 seeds)

Gap closed on worst-group accuracy, (student - floor) / (oracle - floor), generation B (`road_type`):

- s2_B_kd_public: 0.791 ± 0.170 (n=5 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar: 0.857 ± 0.095 (n=5 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar_rarity: 0.827 ± 0.109 (n=5 seeds with a floor-oracle gap above 0.05)

## Stage 2 (generation gap), task `traffic_density`, groups `by_cond`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.500 ± 0.056 | 1.000 ± 0.000 | 0.637 ± 0.014 | 0.580 ± 0.005 |
| s2_B_floor | 5 | 0.225 ± 0.036 | 0.453 ± 0.071 | 0.562 ± 0.009 | 0.342 ± 0.026 |
| s2_B_oracle | 5 | 0.274 ± 0.004 | 0.553 ± 0.066 | 0.587 ± 0.002 | 0.388 ± 0.005 |
| s2_B_kd_public | 5 | 0.473 ± 0.019 | 0.953 ± 0.082 | 0.607 ± 0.016 | 0.526 ± 0.007 |
| s2_B_kd_incar | 5 | 0.457 ± 0.015 | 0.922 ± 0.105 | 0.624 ± 0.010 | 0.512 ± 0.013 |
| s2_B_kd_incar_rarity | 5 | 0.456 ± 0.016 | 0.920 ± 0.104 | 0.625 ± 0.009 | 0.510 ± 0.012 |
| s2_C_oracle | 5 | 0.431 ± 0.009 | 0.870 ± 0.099 | 0.562 ± 0.014 | 0.480 ± 0.007 |
| s2_C_kd_incar | 5 | 0.337 ± 0.014 | 0.684 ± 0.103 | 0.601 ± 0.005 | 0.424 ± 0.009 |
| s2_C_kd_incar_rarity | 5 | 0.331 ± 0.011 | 0.669 ± 0.094 | 0.598 ± 0.005 | 0.419 ± 0.008 |

## Stage 2 (generation gap), task `traffic_density`, groups `by_country`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.413 ± 0.078 | 1.000 ± 0.000 | 0.593 ± 0.014 | 0.580 ± 0.005 |
| s2_B_floor | 5 | 0.233 ± 0.039 | 0.591 ± 0.208 | 0.454 ± 0.013 | 0.342 ± 0.026 |
| s2_B_oracle | 5 | 0.288 ± 0.008 | 0.720 ± 0.161 | 0.484 ± 0.003 | 0.388 ± 0.005 |
| s2_B_kd_public | 5 | 0.354 ± 0.075 | 0.861 ± 0.112 | 0.539 ± 0.011 | 0.526 ± 0.007 |
| s2_B_kd_incar | 5 | 0.316 ± 0.053 | 0.772 ± 0.079 | 0.541 ± 0.011 | 0.512 ± 0.013 |
| s2_B_kd_incar_rarity | 5 | 0.312 ± 0.056 | 0.760 ± 0.087 | 0.540 ± 0.009 | 0.510 ± 0.012 |
| s2_C_oracle | 5 | 0.349 ± 0.036 | 0.862 ± 0.129 | 0.524 ± 0.006 | 0.480 ± 0.007 |
| s2_C_kd_incar | 5 | 0.308 ± 0.031 | 0.759 ± 0.112 | 0.500 ± 0.010 | 0.424 ± 0.009 |
| s2_C_kd_incar_rarity | 5 | 0.312 ± 0.021 | 0.773 ± 0.121 | 0.498 ± 0.008 | 0.419 ± 0.008 |

### Stage 2 paired comparisons, task `traffic_density`

Worst-group accuracy, paired by seed (`traffic_density`):

- B: rarity-aware minus plain in-car distillation: -0.001 (95% CI -0.005 to +0.004, n=5 seeds)
- C: rarity-aware minus plain in-car distillation: -0.007 (95% CI -0.012 to -0.002, n=5 seeds)
- B: in-car minus public-set distillation: -0.016 (95% CI -0.048 to +0.016, n=5 seeds)
- B (in-car) minus teacher A: -0.043 (95% CI -0.109 to +0.022, n=5 seeds)
- C (in-car) minus teacher A: -0.163 (95% CI -0.245 to -0.080, n=5 seeds)
- C minus B, one more generation (in-car): -0.119 (95% CI -0.141 to -0.098, n=5 seeds)
- B in-car distillation minus true labels: +0.183 (95% CI +0.164 to +0.203, n=5 seeds)
- C in-car distillation minus true labels: -0.093 (95% CI -0.113 to -0.074, n=5 seeds)
- B in-car distillation minus the 1%-label floor: +0.231 (95% CI +0.184 to +0.279, n=5 seeds)

Gap closed on worst-group accuracy, (student - floor) / (oracle - floor), generation B (`traffic_density`):

- s2_B_kd_public: 4.169 ± 0.351 (n=4 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar: 3.938 ± 0.443 (n=4 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar_rarity: 3.902 ± 0.453 (n=4 seeds with a floor-oracle gap above 0.05)

## Stage 2 (generation gap), task `road_condition`, groups `by_cond`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 0 | n/a | n/a | n/a | n/a |
| s2_B_floor | 0 | n/a | n/a | n/a | n/a |
| s2_B_oracle | 0 | n/a | n/a | n/a | n/a |
| s2_B_kd_public | 0 | n/a | n/a | n/a | n/a |
| s2_B_kd_incar | 0 | n/a | n/a | n/a | n/a |
| s2_B_kd_incar_rarity | 0 | n/a | n/a | n/a | n/a |
| s2_C_oracle | 0 | n/a | n/a | n/a | n/a |
| s2_C_kd_incar | 0 | n/a | n/a | n/a | n/a |
| s2_C_kd_incar_rarity | 0 | n/a | n/a | n/a | n/a |

## Stage 2 (generation gap), task `road_condition`, groups `by_country`

| model | seeds | worst-group acc | share of teacher A | mean-group acc | overall acc |
|---|---|---|---|---|---|
| s2_A_teacher | 5 | 0.356 ± 0.018 | 1.000 ± 0.000 | 0.854 ± 0.005 | 0.850 ± 0.001 |
| s2_B_floor | 5 | 0.390 ± 0.039 | 1.101 ± 0.159 | 0.795 ± 0.045 | 0.769 ± 0.009 |
| s2_B_oracle | 5 | 0.597 ± 0.024 | 1.678 ± 0.035 | 0.850 ± 0.002 | 0.809 ± 0.005 |
| s2_B_kd_public | 5 | 0.367 ± 0.009 | 1.033 ± 0.060 | 0.839 ± 0.008 | 0.829 ± 0.003 |
| s2_B_kd_incar | 5 | 0.362 ± 0.006 | 1.021 ± 0.061 | 0.831 ± 0.001 | 0.791 ± 0.006 |
| s2_B_kd_incar_rarity | 5 | 0.364 ± 0.005 | 1.024 ± 0.053 | 0.833 ± 0.001 | 0.792 ± 0.006 |
| s2_C_oracle | 5 | 0.365 ± 0.000 | 1.027 ± 0.052 | 0.827 ± 0.000 | 0.778 ± 0.000 |
| s2_C_kd_incar | 5 | 0.365 ± 0.000 | 1.027 ± 0.052 | 0.827 ± 0.000 | 0.778 ± 0.000 |
| s2_C_kd_incar_rarity | 5 | 0.365 ± 0.000 | 1.027 ± 0.052 | 0.827 ± 0.000 | 0.778 ± 0.000 |

### Stage 2 paired comparisons, task `road_condition`

Worst-group accuracy, paired by seed (`road_condition`):

- B: rarity-aware minus plain in-car distillation: +0.008 (95% CI +0.006 to +0.011, n=5 seeds)
- C: rarity-aware minus plain in-car distillation: +0.000 (95% CI +0.000 to +0.000, n=5 seeds)
- B: in-car minus public-set distillation: -0.011 (95% CI -0.016 to -0.006, n=5 seeds)
- B (in-car) minus teacher A: -0.025 (95% CI -0.030 to -0.020, n=5 seeds)
- C (in-car) minus teacher A: -0.032 (95% CI -0.039 to -0.025, n=5 seeds)
- C minus B, one more generation (in-car): -0.007 (95% CI -0.012 to -0.003, n=5 seeds)
- B in-car distillation minus true labels: -0.047 (95% CI -0.054 to -0.040, n=5 seeds)
- C in-car distillation minus true labels: +0.000 (95% CI +0.000 to +0.000, n=5 seeds)
- B in-car distillation minus the 1%-label floor: +0.038 (95% CI -0.002 to +0.079, n=5 seeds)

Gap closed on worst-group accuracy, (student - floor) / (oracle - floor), generation B (`road_condition`):

- s2_B_kd_public: 0.592 ± 0.093 (n=4 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar: 0.459 ± 0.158 (n=4 seeds with a floor-oracle gap above 0.05)
- s2_B_kd_incar_rarity: 0.562 ± 0.087 (n=4 seeds with a floor-oracle gap above 0.05)
