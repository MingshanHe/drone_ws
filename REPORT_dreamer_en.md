# nano-Dreamer: a DreamerV3-style world model for the drone (first-round report)

Branch `dreamerv3-world-model`. Plan: [PLAN_dreamerv3.md](PLAN_dreamerv3.md). Full result tables: [dreamer_results.md](src/output/report/dreamer_results.md). The one-frame JEPA baseline report: [REPORT_en.md](REPORT_en.md). Chinese version: [REPORT_dreamer.md](REPORT_dreamer.md).

**Summary.** Replacing the one-frame JEPA predictor with a DreamerV3-style RSSM makes the world model clearly stronger: it has memory, imagines any number of steps, and its imagination can be decoded into images. Held-out collision foresight AUC@32 rises from 0.93 / 0.57 / 0.76 to 0.96–0.99 / 0.81 / 0.84, and on unseen houses from 0.69 to 0.85–0.90 (0.87 open-loop with the imagination-focused model of section 3.5). The closed loop has not benefited yet: the pybullet pillar crash rate is 34% (one-frame model 11%) and indoor survival stays at 16–19%. The bottleneck is the calibration of the MPC read-out and the indoor action menu, not the model.

---

## 1. How it works

![pipeline](src/output/report/pipeline_dreamer.png)

Every image, latent grid, probability and the decision shown are real: one window of dataset rollout 2 (pillar straight ahead, flying forward at 2× speed) and the actual forward pass of model A.

**(1) Learning the world model from the dataset.** At every 12 Hz step a four-layer conv encoder embeds the 64×64 image; together with the memory $h_t$ it gives the posterior latent $z_t$ (16 groups × 16 classes; cell colour = chosen class). A GRU maps $(h_t, z_t, a_t)$ to $h_{t+1}$; the prior guesses $\hat z_{t+1}$ from $h_{t+1}$ alone, and a KL term pulls the two together (dyn 1.0 / rep 0.1, free 1 nat). A decoder reconstructs the image from $(h_t, z_t)$, and three heads read the clearance $\hat d_t$ (symlog two-hot), P(< 0.7 m) / P(< 0.35 m), and continue. One loss is specific to this project: from the belief, each of the six commands is imagined for 8 steps, and the "inside the ring within k" probabilities must match the dataset's counterfactual oracle labels (the two grids on the right).

**(2) Planning in imagination at test time.** At every decision (12 Hz) the belief is first updated with the current image; the belief is kept across decisions, which is the memory. Each of the 5 menu commands is then imagined for 8 steps (= +32 frames), 8 samples each, averaged, giving P(< 0.7) / P(< 0.35) within 8 steps and $\hat d$. The trigger and cost are exactly those of the one-frame model's `WMPolicy` (relative trigger ≥ 0.4, 170 ms backstop, cost = 6·danger + heading + 1.5·|y| − 1.2·progress), so the A/B/C comparison compares world models only. The decision shown: forward has P(< 0.7) = 1.00 within 8 steps, veer_right 0.00, margin 0.54 → trigger, veer_right held 0.5 s. The decoded imagination (+8 / +16 / +32 frames) shows forward = the pillar fills the view, veer_right = the pillar slides out to the left.

**(3) Closed loop.** 48 Hz dataset sequences → 20k GPU training steps (≈ 30 min) → 1.26M-parameter checkpoint → policy at 3.5 ms per decision on CPU → simulator with 48 Hz velocity control (300 pybullet pillar courses; 300 starts in unseen ProcTHOR houses).

## 2. Model and training details

| Item | Value |
|---|---|
| Encoder | Conv k4 s2 ×4, channels 16/32/64/64, LayerNorm + SiLU, 1024-d output |
| RSSM | deter 256 (LayerNorm GRU), stoch 16 groups × 16 classes, hidden 128, unimix 1%, straight-through sampling |
| Decoder | Linear 512 → 4×4×32, 4 transposed convs → 64×64×3 |
| Heads | clearance: MLP 512→128→128→64 bins (symlog 0–4 m); collision 2 logits; continue 1 logit |
| Parameters | 1.26M (encoder 0.11M, RSSM 0.59M, decoder 0.34M, heads 0.22M) ≈ 1.2 MB int8 |
| Time scale | data at 48 Hz, one model step = 4 frames (12 Hz), 4 phase offsets as augmentation; 8 imagined steps = 667 ms = the old model's +32 |
| Losses | reconstruction MSE (sum over pixels) + clearance two-hot + collision / continue BCE + KL dyn 1.0 + KL rep 0.1 + counterfactual BCE 1.0 |
| Optimiser | Adam 3e-4, eps 1e-8, warmup 1k, grad-norm clip 100, bf16, batch 16 × 24 steps, 20k steps |
| Split | by rollout 80/20, stratified by in_path × passive (as the old model); every metric on held-out rollouts |
| Data | A pybullet only, 64 rollouts (7,680 frames); B indoor only, 256 (30,720); C merged, 320 (38,400); same hyper-parameters |
| Code | `src/dreamer/{nets,rssm,data}.py`, `src/train_dreamer.py`, `src/dreamer_policy.py`, `src/draw_dreamer_pipeline.py`, `src/dreamer_dream_video.py`, `src/eval_imagination.py`, `src/dreamer_summary.py` |

**Size compared with DreamerV3.** The model is **not** DreamerV3-sized. It sits between DreamerV3's two smallest presets (values from `dreamerv3/configs.yaml`; DreamerV3 always uses 32 stochastic groups):

| Model | deter | hidden | stoch × classes | CNN depth | MLP units | Parameters |
|---|---|---|---|---|---|---|
| DreamerV3 size1m | 512 | 64 | 32 × 4 | 4 | 64 | ≈ 1M |
| **this work, nano** | 256 | 128 | 16 × 16 | 16 | 128 | **1.26M** |
| this work, wide | 512 | 256 | 32 × 32 | 32 | 256 | 6.33M |
| DreamerV3 size12m | 2048 | 256 | 32 × 16 | 16 | 256 | ≈ 12M |
| DreamerV3 size50m | 4096 | 512 | 32 × 32 | 32 | 512 | ≈ 50M |
| DreamerV3 size200m (default) | 8192 | 1024 | 32 × 64 | 64 | 1024 | ≈ 200M |

Kept from DreamerV3: RSSM structure, discrete latents + unimix, KL balancing + free bits, symlog / two-hot, LayerNorm + SiLU, reconstruction plus reward-style heads. Changed: offline dataset only (no replay, no interaction); the reward head is replaced by clearance and collision heads; a counterfactual imagination loss is added; no actor-critic yet (route B2), planning is an imagination MPC (route B1). Not ported: the block-diagonal GRU (`blocks: 8`), the LaProp/AGC optimiser (we use Adam + norm clipping), bf16 weights everywhere.

## 3. Results

### 3.1 Held-out within the training set

The old model's AUC reads its head on one frame; Dreamer's is **open-loop**: observe 667 ms, then imagine 8 steps under the flown commands and read the head (a harder setting). Decoded gain = 1 − MSE(predicted, true) / MSE(current, true).

| Training set | Model | AUC@32 | Decoded gain @8 / @32 | Clearance MAE, imagined @32 | Veer ranking |
|---|---|---|---|---|---|
| A | one-frame JEPA (83k) | 0.93 | ≈ +0.4 @32 (decoder fitted afterwards) | — | 1.00 (n=45) |
| A | nano-Dreamer, 20k steps | **0.96** | +0.29 / +0.53 | 0.27 m | 0.95 (n=19) |
| A | nano-Dreamer, no FOV mask on L_cf, 8k steps | **0.99** | +0.33 / +0.56 | 0.27 m | 0.89 (n=19) |
| B | one-frame JEPA | 0.57 | ≤ 0 | — | nan (n=0) |
| B | nano-Dreamer, 20k steps | **0.76** | +0.03 / +0.05 | 0.46 m | 0.45 (n=76) |
| B | nano-Dreamer, early stop at 6k | **0.81** | +0.16 / +0.18 | 0.41 m | 0.46 (n=76) |
| C | one-frame JEPA | 0.76 | — | — | 0.87 (n=45) |
| C | nano-Dreamer, 20k steps | **0.84** | +0.11 / +0.18 | 0.38 m | 0.78 (n=78) |
| C | nano-Dreamer, early stop at 6k | **0.83** | +0.18 / +0.14 | 0.40 m | 0.82 (n=78) |

Training curves ([loss plot](src/output/report/dreamer_losses.png)): A plateaus at 4k steps (0.99 / +0.55); B is best at 4k (0.81 / +0.17) and drops to 0.74 / +0.06 by 16k: 256 indoor rollouts are over-fitted within 20k steps, hence the 6k early-stop variants.

### 3.2 Offline generalisation (held-out frames, \* = domain seen in training)

Collision foresight AUC@32:

| Training set | Model | pybullet pillars | hyperdrone pillars, random look (unseen) | houses 0–15 | houses 16–23 (unseen) |
|---|---|---|---|---|---|
| A | one-frame JEPA | 0.95\* | 0.85 | 0.51 | 0.46 |
| A | nano-Dreamer | 1.00\* | 0.73 | 0.64 | 0.75 |
| A | nano-Dreamer, no FOV mask | 1.00\* | 0.81 | 0.68 | 0.72 |
| B | one-frame JEPA | 0.44 | 0.57 | 0.76\* | 0.69 |
| B | nano-Dreamer | 0.59 | 0.76 | 0.84\* | **0.90** |
| B | nano-Dreamer 6k | 0.64 | 0.83 | 0.91\* | 0.89 |
| C | one-frame JEPA | 0.96\* | 0.81 | 0.77\* | 0.69 |
| C | nano-Dreamer | 1.00\* | 0.71 | 0.85\* | 0.85 |
| C | nano-Dreamer 6k | 0.99\* | **0.83** | 0.83\* | 0.85 |

Decoded future-frame gain @32 (0 = no better than copying the current frame):

| Training set | Model | pybullet pillars | random-look pillars | houses 0–15 | houses 16–23 (unseen) |
|---|---|---|---|---|---|
| A | one-frame JEPA | +0.64\* | +0.33 | −0.58 | −0.54 |
| A | nano-Dreamer | +0.91\* | −0.74 | −2.64 | −1.44 |
| B | one-frame JEPA | +0.28 | +0.27 | +0.12\* | −0.06 |
| B | nano-Dreamer | −2.55 | −0.24 | +0.16\* | **+0.10** |
| C | one-frame JEPA | +0.57\* | +0.37 | +0.10\* | +0.04 |
| C | nano-Dreamer | +0.85\* | −0.13 | +0.07\* | **+0.25** |
| C | nano-Dreamer 6k | +0.74\* | −0.01 | +0.20\* | +0.15 |

Reading: in **houses it has never seen**, the indoor-trained Dreamer reaches collision foresight 0.85–0.90, and its decoded future frames beat "copy the current frame" for the first time (the one-frame models are all ≤ 0). The price is that reconstruction ties the latent to appearance: A drops from 0.85 to 0.73 on random-look pillars (0.81 without the FOV mask, 0.83 for C 6k), and out of domain its decoded gain is strongly negative: it confidently draws the wrong thing.

### 3.3 Closed loop

| Training set | Model | pybullet crash rate (wm) | Trigger lead | Indoor survival (wm) | Indoor, menu with hover |
|---|---|---|---|---|---|
| A | one-frame JEPA | **11%** | +335 ms | 17% | 23% |
| A | nano-Dreamer, first read-out | 41% | +225 ms | 17% | 17% |
| A | nano-Dreamer, 8-sample mean, future only | 34% | +234 ms | 17% | 17% |
| A | nano-Dreamer, no FOV mask | 34% | +292 ms | not run | not run |
| B | one-frame JEPA | 82% | — | 16% | 19% |
| B | nano-Dreamer / 6k | 82% / not run | — | 17% / 12% | 16% / 14% |
| C | one-frame JEPA | 20% | — | 16% | 19% |
| C | nano-Dreamer / 6k | 20% / 27% | — | 15% / 19% | 16% / 18% |

Reference: a privileged oracle on the same menu survives 32% of indoor starts, 89% if it may hover. "Menu with hover" = hover 0.5 s when both veers have P(< 0.35 m) > 0.5 within 170 ms, the same rule for both model families. Decision latency: Dreamer 3.5 ms (CPU), one-frame model 0.1 ms.

### 3.4 Imagination quality

![imagination](src/output/report/imagination_eval.png)

`src/eval_imagination.py`. After 8 observed steps (667 ms) the model sees no further image and imagines up to +88 frames from memory and the commands actually flown. Curves average every held-out / unseen sequence (48 pybullet, 256 unseen-house sequences); green is the imagination-focused indoor model of section 3.5.

- **(a) Decoded images.** On pybullet the per-pixel MSE of the imagination stays at 0.02–0.03 for 88 frames, versus 0.06–0.08 for copying the last frame. In unseen houses the old C 6k model is at 0.048 vs 0.069 at +32 frames and falls back to the copy baseline by +88 (0.12 vs 0.14); the green model stays below it the whole way (0.085 at +88).
- **(b) Clearance.** On pybullet the distance head's error at +32 frames is 0.16 m (baseline "clearance unchanged" 0.37 m). In unseen houses C 6k only matches the baseline (0.3–0.4 m); the green model beats it (0.27 m at +32).
- **(c) Collision foresight.** Open-loop AUC is 0.97–0.98 on pybullet (one-frame model 0.91–0.93), 0.81–0.85 in unseen houses for C 6k and 0.87–0.89 for the green model (one-frame model 0.67–0.69), with almost no decay over the horizon.
- **(d) Value of memory.** Imagining after a single observed step, C 6k is worse than copying in unseen houses (−0.19); after 4 steps it reaches +0.53, and the green model +0.71. Beyond 4 steps the gain drops partly because later hand-offs are closer to furniture and harder to predict.

Strips: on a held-out pybullet sequence the imagination draws the pillar approaching, filling the view and sliding away; in an unseen house it only draws walls, floor and a blob of furniture, but $\hat d$ and P(< 0.7) follow the true values.

### 3.5 Training for imagination (indoor)

**Why indoor imagination is much worse than DreamerV3's.** Compared with [danijar/dreamerv3](https://github.com/danijar/dreamerv3), in order of impact:

1. **Data.** DreamerV3 learns online with millions of replayed frames and is evaluated in the same environment. Indoors we have 256 offline rollouts (16 houses, 30,720 frames); held-out quality starts to degrade after 4k steps ([loss curves](src/output/report/dreamer_losses.png)).
2. **One-step objective.** DreamerV3 uses only the one-step posterior / prior KL. Indoors the KL stays at 8–11 nats, i.e. the prior cannot predict the posterior, and errors compound over multi-step imagination.
3. **Model size.** DreamerV3's smallest standard preset is size12m (deter 2048); ours is 1.26M. The ablation below shows this is not the main cause.
4. **Other.** Images are the only observation (attitude is not fed in, although the airframe pitches up to 30° and turns the camera); rollouts are only 2.5 s long and the command changes every 1 s.

**What was done.** A new `--overshoot K` option (PlaNet-style latent / pixel overshooting; DreamerV3 has none): from random posterior states the prior imagines K = 8 steps, and the imagined states must decode the true future frames, match the posterior at that step (KL), and read the true clearance and collision labels, directly optimising what open-loop evaluation measures. In addition, 1024 new rollouts were generated in the same 16 training houses with a new seed and merged into 1280 rollouts (`wm_dataset_rooms_train60_1280.npz`).

| Indoor model (held-out training houses, open-loop) | Params | AUC@32 | Decoded gain @8 / @32 | Clearance error at +32 frames |
|---|---|---|---|---|
| B, 256 rollouts, 20k steps | 1.26M | 0.76 | +0.03 / +0.05 | 0.46 m |
| B, 256 rollouts, early stop at 6k | 1.26M | 0.81 | +0.16 / +0.18 | 0.41 m |
| wide, 256 rollouts, 8k steps | 6.33M | 0.75 | +0.09 / +0.10 | 0.46 m |
| wide + appearance augmentation, 256 rollouts | 6.33M | 0.76 | −0.13 / −0.07 | 0.49 m |
| wide + overshoot, 256 rollouts, 6k steps | 6.33M | 0.76 | +0.14 / +0.13 | 0.46 m |
| **nano + overshoot, 1280 rollouts, 12k steps** | 1.26M | **0.87** | **+0.30 / +0.39** | **0.32 m** |

On **unseen houses 16–23** (256 sequences, `eval_imagination.py`, the green curves in section 3.4):

| Unseen houses | Old C 6k | 1280 rollouts + overshoot |
|---|---|---|
| Open-loop AUC @4 / @32 | 0.85 / 0.81 | **0.89 / 0.87** |
| Decoded MSE at +32 frames (copy baseline 0.069) | 0.048 | **0.033** |
| Decoded MSE at +88 frames (copy baseline 0.140) | 0.121 | **0.085** |
| Clearance error at +32 frames ("unchanged" baseline 0.37 m) | 0.39 m | **0.27 m** |
| Decoded gain at +32 after 1 / 4 observed steps | −0.19 / +0.53 | **+0.25 / +0.71** |

Conclusion: **the bottleneck is data and the training objective, not capacity.** Widening to 6.3M did not help and appearance augmentation made it worse; 5× data plus overshooting makes the imagination beat "copy the current frame" over the whole horizon in unseen houses for the first time, and the clearance estimate beat the "unchanged" baseline.

![unseen houses, 1280 + overshoot](src/output/report/imagination_examples_rooms_unseen_1280os.png)

Compare with [the same 8 sequences for the old C 6k model](src/output/report/imagination_examples_rooms_unseen_C6k.png) (and [held-out training-house examples](src/output/report/imagination_examples_rooms_heldout_1280os.png)). The imagination is still blurry: the colour blocks and positions of walls, floor and large furniture match, while texture and small objects are lost; but P(< 0.7) rises at the right time when the view darkens towards a wall. Reaching DreamerV3's sharpness needs more data (more houses, not just more seeds in the same houses), attitude as an input, and only then a bigger model.

## 4. Demo videos

**4.1 "Dreams": real camera vs imagination.** Left: what the camera saw (12 Hz). Right: posterior reconstruction for the first 8 steps; after that the model **sees no image** and imagines from memory and the flown commands; every frame is decoded from an imagined state. Blue text: clearance and P(< 0.7 m) read from the imagined state.

pybullet, held-out rollouts, model A — [mp4](src/output/report/dream_pybullet.mp4)

![dream pybullet](src/output/report/dream_pybullet.gif)

Unseen houses 16–23, model C 6k — [mp4](src/output/report/dream_rooms_unseen.mp4)

![dream rooms](src/output/report/dream_rooms_unseen.gif)

**4.2 Flying a pillar forest by imagination** (pybullet, model A, denser and longer than the training courses) — [top-down mp4](src/output/report/forest_dreamer_pb_wm-mpc.mp4) · [onboard mp4](src/output/report/forest_dreamer_pb_wm-mpc_onboard.mp4)

![forest top-down](src/output/report/forest_dreamer_pb_wm-mpc.gif)

![forest onboard](src/output/report/forest_dreamer_pb_wm-mpc_onboard.gif)

This run reaches the goal but grazes three pillars (minimum clearance 0.04 m), the same problem as the 34% crash rate of section 3.3.

**4.3 Endurance in an unseen house** (ProcTHOR house 17, model C 6k, left top-down, right onboard) — [mp4](src/output/report/hd_room17_dreamer.mp4)

![room 17](src/output/report/hd_room17_dreamer.gif)

Contact after 4.6 s (minimum clearance 0.21 m): with only left / right veers there is no way out when furniture boxes the drone in.

**4.4 Decoded open-loop imagination (static):** [A pybullet](src/output/report/dreamer_openloop_pybullet.png) · [B indoor](src/output/report/dreamer_openloop_rooms.png) · [C merged](src/output/report/dreamer_openloop_pb_plus_rooms.png).

## 5. Why the closed loop has not improved, and next steps

1. **Read-out noise.** One sampled imagination flips a course between crash and no crash with the random stream; averaging 8 samples took the crash rate from 41% to 34%.
2. **"Within k" saturation.** The trained "within k" target includes the current state, so inside the 0.7 m ring every candidate reads 1.0, the margin vanishes and only the centring prior chooses the direction. The read-out now compares future steps only, but the veers' critical-ring predictions are still low and triggers come ≈ 100 ms later than with the one-frame model.
3. **Counterfactual FOV mask.** The one-frame model's "no supervision when the threat is outside the FOV" mask tells a model with memory not to remember out-of-view pillars; removing it improves held-out (0.99, random look 0.81) but not the closed loop (34%).
4. **The indoor ceiling is the menu.** The oracle on the same menu only survives 32%; the hover rule almost never fires for Dreamer because its veer critical predictions are low.
5. **Over-fitting.** B is best at 4k steps and clearly worse at 20k.

Next, in order of expected gain: (a) MPC cost on the imagined clearance (distance head) instead of thresholded probabilities, hover decided the same way; (b) retrain with held-out early stopping, more houses and the imagination-focused objective; (c) recalibrate the closed-loop thresholds for Dreamer or switch to a critic-style cost; (d) then the actor-critic in imagination (B2).

## 6. Reproduce

```bash
python src/train_dreamer.py --data src/output/wm_dataset.npz --out src/output/world_model_dreamer_pybullet.pth --tag pybullet --steps 20000
python src/train_dreamer.py --data src/output/wm_dataset_rooms_train60.npz --out src/output/world_model_dreamer_rooms6k.pth --tag rooms6k --steps 6000
python src/train_dreamer.py --data src/output/wm_dataset_pb_plus_rooms.npz --out src/output/world_model_dreamer_pb_plus_rooms6k.pth --tag pb_plus_rooms6k --steps 6000
python src/gen_wm_dataset_hyperdrone.py --scene rooms --rollouts 1024 --rooms 1024 --houses 16 --house-first 0 --len 120 --fov-deg 60 --camera body --seed 1 --out src/output/wm_dataset_rooms_train60_big.npz
python src/merge_datasets.py src/output/wm_dataset_rooms_train60.npz src/output/wm_dataset_rooms_train60_big.npz --out src/output/wm_dataset_rooms_train60_1280.npz
python src/train_dreamer.py --data src/output/wm_dataset_rooms_train60_1280.npz --out src/output/world_model_dreamer_rooms1280_os.pth --tag rooms1280_os --overshoot 8 --steps 12000 --eval-every 3000
python src/eval_offline.py --models dreamer_pybullet dreamer_rooms dreamer_pb_plus_rooms --domains pybullet rand_pillars rooms_train rooms_unseen --out dreamer_three
python src/eval_world_model_policy.py --seeds 300 --model src/output/world_model_dreamer_pybullet.pth        # conda env drone_python
python src/hd_closed_loop.py --scene rooms --houses 8 --house-first 16 --seeds 300 --sweep-seeds 0 --model src/output/world_model_dreamer_pybullet.pth [--allow-stop]
python src/eval_imagination.py
python src/plot_dreamer_losses.py --tags pybullet rooms rooms6k rooms_wide rooms_wide_os rooms1280_os
python src/imagination_examples.py --model src/output/world_model_dreamer_rooms1280_os.pth --data src/output/wm_dataset_rooms_unseen60.npz --all-rollouts --tag rooms_unseen_1280os --n 8
python src/draw_dreamer_pipeline.py
python src/dreamer_dream_video.py --model src/output/world_model_dreamer_pybullet.pth --data src/output/wm_dataset.npz --tag pybullet
```
