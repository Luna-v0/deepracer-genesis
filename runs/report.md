# Experiment report

5 run record(s) under `runs/`.

## Combination table

| modality | render | algorithm | asymmetry | encoder | dr_profile | runs | completion_rate | lap_time_s | mean_progress_m | offtrack_rate | mean_return | mean_cost | cost_violation_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| camera | madrona | ppo | asymmetric | none | full | 1 | 0 | - | 1.95 | 1 | 11.5 | - | - |
| feature | none | ppo | symmetric | none | none | 2 | 0.987 ± 7.76e-05 | 10.5 ± 3.36 | 52.4 ± 16.8 | 0.0204 ± 0.00971 | 580 ± 149 | - | - |
| feature | none | ppo_lagrangian | symmetric | none | none | 1 | 1 | 8.98 | 59.2 | 0 | 626 | 69.7 | 0.949 |
| camera | madrona | ppo_lagrangian | symmetric | frozen_cnn | obs+action | 1 | 0 | - | 2.56 | 1 | 27.3 | 24 | 0.207 |

## Before/after (ablation pairs)

### safety  (baseline: `feature_lagrangian`)

| variant | Δ completion_rate | Δ cost_violation_rate | Δ lap_time_s | Δ mean_cost | Δ mean_progress_m | Δ mean_return | Δ offtrack_rate |
|---|---|---|---|---|---|---|---|
| transfer_madrona | -1 ± 0 | -0.742 ± 0 | - | -45.6 ± 0 | -56.6 ± 0 | -599 ± 0 | +1 ± 0 |

