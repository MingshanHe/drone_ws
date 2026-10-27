## Training check (nano world model, 80 epochs, rollout-level split)

| dataset | train/val samples | latent MSE@32 (no-op) | AUC@4/8/16/32 | now-AUC | veer-ranking (pillar rollouts) |
|---|---|---|---|---|---|
| `wm_dataset_mixed_tello40.npz` | 2447/584 | 2.119 (2.860) | 0.88/0.91/0.91/0.84 | 0.86 | 0.42 (n=24) |
| `wm_dataset_mixed_60body.npz` | 2447/584 | 2.037 (2.949) | 0.94/0.96/0.94/0.90 | 0.89 | 0.75 (n=24) |
| `wm_dataset_hyperdrone.npz` | 2444/580 | 0.872 (3.765) | 0.98/0.98/0.97/0.97 | 0.94 | 0.85 (n=20) |
| `wm_dataset_hyperdrone_tello40.npz` | 2444/580 | 0.748 (2.276) | 0.98/0.98/0.98/0.98 | 0.92 | 1.00 (n=20) |
| `wm_dataset_hyperdrone_tello40_rand.npz` | 2444/566 | 1.206 (2.122) | 0.91/0.94/0.94/0.94 | 0.89 | 1.00 (n=126) |

Reference: the same trainer on the original pybullet `wm_dataset.npz` gives MSE@32 1.603 (no-op 2.835), AUC 0.93/0.91/0.92/0.93, now-AUC 0.84, veer-ranking 1.00 (n=45). Metrics are on each dataset's own held-out rollouts; they show the data trains a model of the usual quality, not real-vehicle performance.
