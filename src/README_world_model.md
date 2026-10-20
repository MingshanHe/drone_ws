# Lesson 29 — Nano world model: latent-space prediction for proactive avoidance

🌐 **English** (below) · [跳到繁體中文](#中文)

---

> **In one line:** Train a small model that predicts how the scene will change, so the drone starts dodging before an obstacle gets close. · **Builds on:** L17, L18, L19, L21, L4

> 🔭 **This lesson is the course's research preview.** It teaches the concepts —
> latent prediction, collision heads, proactive avoidance — with measured
> results. The deep dive (harder worlds, model-side memory, metric grounding,
> real hardware) continues in
> [**microdrone-world-model**](https://github.com/csinghans/microdrone-world-model).

## Why

Every model so far has been *reactive*: the depth net (Lesson 17) and the RL
policy (Lesson 19) both answer "what should I do about the obstacle that is close
**now**?" But at speed, "now" is already too late — by the time a pillar fills the
frame, a 27 g drone with bounded acceleration cannot turn hard enough to miss it.

A **world model** buys you the missing ingredient: *anticipation*. It learns how
the world will change — and how it changes *because of what you do* — so the
drone can act on what is **about to** happen. This is the frontier the course has
been pointing at; it is the same problem self-driving research attacks with world
models.

The trap is to build it by predicting the **next image** (diffusion / pixel
generation). That is slow and it hallucinates detail a controller can't use. We
take the **V-JEPA** route instead — *predict the next latent embedding, never the
next pixels*. No hallucinated frames, no diffusion latency, a fixed compute cost.

And the signature move returns, at the frontier this time: the real V-JEPA is a
billion-parameter model that needs an Orin-class GPU — it will **never** fit on a
GAP8. So you don't download it; you **train your own nano version** under the
512 KB int8 budget — and then you **close the loop**: a tiny latent MPC flies
from the camera alone, sees ~667 ms ahead, and triggers its dodge measurably
earlier than a reactive controller (+243 ms mean over 70 courses, ~500 ms in
the demo below).

## Concept

Four tiny networks, reusing the course's conv stack, all int8-able:

- **Encoder** `f_θ`: image → a 64-d latent `z`. Lesson 3's `TinyDronet` conv
  stack — but its global average pool is swapped for a **bearing-aware pooling**
  (four horizontal strips + a small projection). Global pooling averages *where*
  away: it can say "pillar close", never "pillar on the left" — and dodging
  left-vs-right is precisely a "which side" question. (Measured below: with
  global pooling the veer-ranking check sits at chance no matter how it is
  supervised; with strips it reaches 1.00.)
- **Predictor** `g_φ`: `(z_t, action) → ẑ_{t+k}` at **four horizons**
  `k ∈ {4, 8, 16, 32}` steps (~83 / 167 / 333 / 667 ms at 48 Hz) — one shared
  trunk, one tiny residual head per horizon (`ẑ = z_t + Δ_k`, so "nothing
  changes" is the free baseline). "Proactive" is a claim about *time*: to buy
  back even a few hundred milliseconds of reaction, the model has to see all
  the way out to ~667 ms, not one fixed 167 ms hop. And every training rollout scales the whole
  command set by a cruise-speed factor (0.6–1.6 m/s), so the heads learn
  danger *as a function of commanded speed* — that is what powers step 4c.
- **Collision heads**: `ẑ_k → P(within 0.7 m within k steps)` *and*
  `P(within 0.35 m within k steps)` — a **warn ring** and a **critical ring**
  per horizon. Two rings, because one ring is a region test, not a gradient:
  inside 0.7 m *every* action "warns" (passing a pillar crosses 0.7 m briefly
  — a correct prediction), and only the critical ring still separates "grazes
  past" from "about to hit". (Measured with one ring: the planner froze
  mid-course, hovering forever at the boundary.)
- **Danger-now head**: `z_t → P(too close right now)` — the *reactive* signal,
  kept on purpose: it is the honest baseline the anticipation must beat with
  the sensor held equal.

The trick that makes latent prediction work without a pixel loss (and without
collapsing to a constant) is a **target encoder**: an EMA copy of `f_θ` with a
stop-gradient, à la V-JEPA / BYOL, plus a VICReg-style variance guard:

```
Σ_k ‖ g_φ(f_θ(x_t), a_t)_k − sg(f_EMA(x_{t+k})) ‖²  +  variance-guard  +  danger BCEs
```

**The dataset is an experiment, not footage.** Each rollout resets the sim
(fresh trial), cruises forward, then flies a chain of **held segments**: every
~1 s a random high-level command — forward / slow / veer_left / veer_right /
climb / hover — held long enough to cover the longest horizon. We record the
**commanded setpoint**, not the measured velocity: a controller can only feed
the model a command, so that is what it must condition on. Labels stay free
(the simulator's signature move) and go further: a **counterfactual oracle**
labels *every* frame × candidate × horizon by rolling the command forward
kinematically through the known pillar layout. Executed rollouts teach "what
happened"; a planner needs "what would happen if". One honesty rule: a positive
label caused by a pillar **outside the camera's 60° FOV** is unanswerable from
a single frame, so it is masked out of both training and grading.

**What it took to close the loop (all measured on this repo):** the first
version had collision AUC 0.9 — and ranked "veer left vs veer right" at
chance. AUC is a *distance* question; planning is a *choice* question. Fixing
the choice took, in order: intervention segments (counterfactual contrasts),
the counterfactual oracle (dense ranking supervision), FOV masking (37 of 45
failing probe frames had the threat at 66–84° bearing — invisible), and
finally the bearing-aware pooling above (the root cause). The lesson: **a high
AUC does not mean a model can rank actions — test the decision, not the
detection.** That test ships here as the `veer-ranking` metric: on held-out
frames where geometry says one veer is truly safer, does the model rank it
safer? Chance is 0.5; this model measures 1.00.

Closing the *planner* took the same discipline, one measured failure at a
time: a flat max-over-horizons danger paralyses at the ring (→ urgency
weights); one ring has no gradient inside itself (→ the critical ring); an
absolute trigger threshold inherits the heads' course-dependent probability
floor and false-triggers on every clear course (→ the *relative* margin
trigger); slowing down is a fake evasion that creeps into the ring (→ veers
only); long committed maneuvers fly blind (→ short commits + a
corridor-centering prior from the drone's own odometry). The meta-lesson:
**with learned heads, every planner assumption is a hypothesis — measure it.**

**Why latent (V-JEPA), not explicit 4D geometry (4D-GS)?** Both are world models;
they sit at opposite ends of the trade-off.

| | **V-JEPA (this lesson)** | **4D Gaussian Splatting** |
|---|---|---|
| Represents | an implicit *latent* manifold | explicit metric 3D + time geometry |
| Predicts | the future *embedding* | renders the future *scene* |
| Strength | anticipation, abstraction, **bounded latency** | metric precision, photorealism |
| Cost | fixed FLOPs, **distils to int8 < 512 KB** | scales with #Gaussians → Orin-class |
| Hallucination | none (no pixels committed) | geometric "floaters" when extrapolating |

4D-GS is the heavyweight explicit-geometry cousin of the **2D occupancy grid you
built in Lesson 14a** — extended to 3D and time. It is wonderful for metric
precision but it does not fit on a GAP8. V-JEPA is the latent path that does,
which is why it is our on-board backbone.

## Hands-on

```bash
conda activate nanodrone-ai
python lessons/29_world_model/gen_wm_dataset.py --rollouts 64    # 1. intervention trials
python lessons/29_world_model/train_world_model.py --epochs 80   # 2. train the nano V-JEPA
python lessons/29_world_model/proactive_avoid.py                 # 3. the timing schematic
python lessons/29_world_model/wm_closed_loop.py                  # 4. closed loop from vision
python lessons/29_world_model/eval_world_model_policy.py         # 4b. the cluttered scoreboard
python lessons/29_world_model/speed_sweep.py                     # 4c. crash rate vs speed
python lessons/29_world_model/eval_robustness.py                 # 5. the sim-to-real trap, priced
python lessons/29_world_model/learn_policy.py                    # 6. learn the policy (~16 min)
```

Step 3 (`proactive_avoid.py`) isolates decision *timing* with privileged
geometry — a schematic. Step 4 removes the crutch: the danger signal in the
control loop comes from the **camera alone** (encoder → predictor → collision
heads), while pillar positions only stage the course and score the flight. The
planner is a tiny **latent MPC** at 12 Hz: encode the frame *once*, imagine
every candidate through the small MLPs (the expensive encoder is shared, so on
a GAP8 the whole deliberation is nearly free), evade when going straight is
predicted meaningfully more dangerous than the better veer — a *relative*
trigger, because absolute thresholds inherit the heads' course-dependent
probability floor — and choose the veer with the cheapest crit-weighted
future. The reactive baseline gets the same encoder — and, deliberately, a
*privileged* evasion direction. It can only lose on timing, so the comparison
isolates anticipation. Step 4c then sweeps the cruise speed on single-pillar
courses (the threat both policies can see): reaction triggers at a fixed
*distance*, anticipation at a fixed *time* — raise the speed and only one of
those budgets survives.

Step 6 acts on the planner verdict: instead of tuning the cost function, PPO
(Lesson 19's exact recipe — progress reward, crash penalty, nothing hand-shaped
about danger) learns the policy from the world model's own outputs, with the
last second of them stacked as memory — so a pillar that slides out of the 60°
FOV stays in the observation. Same encoder, same heads, same harnesses; only
the decision-maker is learned. Two memory flavours ship (`--recurrent` swaps
the stack for sb3-contrib's LSTM — the lesson's one optional dependency),
`--randomize` trains inside step 5's storm, `--edge-bias` re-weights the
per-episode training speeds toward the fast edge of the envelope, and
`--curriculum` runs the same budget through three diets in sequence
(measured below — it backfires); every trained variant joins the
scoreboards automatically.

Each script self-generates what it needs and runs on its own with `--selftest`.

## Checkpoint ✅

Step 1 (measured locally, full run):

```
WM-DATA OK: 64 rollouts x 120 steps @ 48 Hz, 87 held intervention segments, labels [k=4: n=7076 pos=0.25, k=8: n=6472 pos=0.27, k=16: n=5266 pos=0.30, k=32: n=3020 pos=0.39], saved .../output/wm_dataset.npz
```

Step 2 asserts the predictor beats "future == present", per-horizon AUC (warn
ring), the danger-now head, **and the veer ranking** — the number that says
the model can choose, not just detect (rollout-level split, so nothing leaks):

```
WORLD-MODEL OK: 2442 train seqs, latent MSE@32=1.200 (no-op 2.538), AUC@4/8/16/32=0.96/0.96/0.96/0.96, now-AUC=0.81, veer-ranking=1.00 (n=25), int8 weights=81.3 KB (<512 fits), saved .../output/world_model.pth
```

Step 4 flies the same course twice from vision alone and asserts the earlier
trigger, the clearance gain, no crash, goal reached — no privileged look-ahead
anywhere in control:

```
WM-CLOSED-LOOP OK: reactive min-clear=0.38 m (trigger@60), wm min-clear=0.48 m (trigger@36), lead=+24 steps (~500 ms earlier), crashes reactive/wm = 0/0, goal steps = 187/198 — danger signal from camera alone (no privileged look-ahead in control)
```

Step 4b is the cluttered scoreboard — 100 random multi-pillar courses (70
threatened, 30 clear for false-positive probing) plus the on-board bill an
embedded engineer asks for (weights are not the whole story: activations and
DMA workspace share the same 512 KB):

```
WORLD-POLICY OK: seeds=100 (70 in-path / 30 clear)
  crash_rate:        reactive 1% -> wm 16%
  mean_min_clearance: 0.41 m -> 0.38 m
  mean_trigger_lead: +243 ms (n=70 both triggered)
  false_positive:    reactive 0% -> wm 0% of clear runs
  goal_time:         wm +2% vs reactive (n=58 clean)
  decision latency:  0.4 ms measured (this CPU) | ~8 ms est @ GAP8 0.5 GMAC/s (3.9 M MACs, encoder shared across 6 candidates)
ONBOARD-BUDGET OK: weights=81.3 KB + peak_activation=28.0 KB + workspace(dbl-buf)=28.0 KB = 137.3 KB < 512 KB
```

Step 4c is the mechanism, measured — 30 single-pillar courses per cruise
speed, same seeds at every speed, every available policy (crash rates):

| cruise | reactive | wm (hand MPC) | **learned (stacked)** | learned (LSTM) | LSTM, edge-biased | LSTM, curriculum |
|---|---|---|---|---|---|---|
| 0.8 m/s | 0 % | 10 % | **0 %** | 0 % | 0 % | 7 % |
| 1.0 m/s | 0 % | 0 % | **0 %** | 0 % | 0 % | 23 % |
| 1.2 m/s | 3 % | 0 % | **0 %** | 0 % | 0 % | 13 % |
| 1.4 m/s | **40 %** | 0 % | **0 %** | 3 % | 0 % | 47 % |
| 1.6 m/s | **60 %** | 10 % | **0 %** | **40 %** | 3 % | **73 %** |

```
SPEED-SWEEP OK: 30 single-pillar courses/speed — crash (reactive/wm/learned/learned-rnn/learned-rnn-edge/learned-rnn-curr) at 0.8 m/s = 0%/10%/0%/0%/0%/7%; at 1.6 m/s = 60%/10%/0%/40%/3%/73% — reaction pays a distance, anticipation pays time
```

The learned stacked-memory policy flies the *entire* speed band without a
single crash — 150 courses, zero. The LSTM variant is the honest control
experiment, four times over. At the stack's own 300k budget it never
converged (3–37 % everywhere). With a *fair* budget — 3× the steps, a
right-sized 64-wide LSTM, a 256-step backprop window — it converged up to
1.4 m/s (0–3 %) and still collapsed at the speed extreme (40 % at 1.6 m/s,
where episodes are shortest and uniform speed sampling starves the edge
twice over). With `--edge-bias` — half the training episodes drawn from the
fast edge, every other knob identical — the extreme closes: **40 % → 3 %**
on the sweep, 45 % → 7 % on the fast rerun. Closed, but not for free: the
re-weighting took its probability mass from the slow cluttered regime, and
the cluttered tail reopened, 2 % → 10 %. You get what you sample — the diet
patches the band you point it at, and the hole moves to where the mass came
from. So the obvious next move was tried too: `--curriculum`, the same
budget through three diets in sequence (natural → edge-drilled → mixed), to
consolidate both bands. It did the opposite — **worse everywhere** (7–73 %
across the sweep, 80 % on the fast rerun), while the uniform and edge runs
reproduced their numbers in the same harness. On-policy learning has no
yesterday: each phase optimized against its own diet and overwrote the
last, and the final mixed phase had neither the steps nor the anchor to
recover both. **Order is not a free knob.** Four runs in, the data axis is
measured out — uniform starves the edge, re-weighting moves the hole,
sequencing loses both bands — and the simple stack needed none of it; it
has yet to lose anywhere.

Read the two scoreboards together, like a robot person would. **Step 4c is the
mechanism**: the reactive trigger fires at a fixed *distance*, so raising the
speed spends its budget — 0 % to 60 % crashes — while the anticipating MPC
triggers at a fixed *time* and holds 0–10 % with 2–3× the clearance at speed.
**Step 4b is the honest limit**: on cluttered courses the side pillars sit at
60–90° bearings, outside the forward camera's FOV, and the memoryless planner
pays a 16 % crash tail there that the privileged-direction baseline does not —
while keeping zero false evasions and a +2 % time cost. The model is no longer
the bottleneck (veer-ranking 1.00); the hand-crafted cost function is — eight
planner configurations were measured to establish that, and step 6 acts on it:
the *learned* policy erases the tail entirely (0 %).

Step 5 prices the sim-to-real trap before hardware pays it: the shipped
(clean-trained) model, measured on conditions it never saw — randomized pillar
shapes/colours, 0–2 control steps of command latency, ±8 % actuation noise,
and Lesson 20's fixed unseen appearance shift on every frame the policies see
— then the Track B fix (`gen --randomize` + `train --robust`), trained across
the variation:

```
ROBUST-WM OK: clean AUC@32=0.96 | randomized(+unseen-shift) AUC@32=0.82 (the gap to buy back), veer-ranking=1.00 (n=8)
  closed loop under randomization — crash reactive 10% / wm 17% / learned 10% / learned-rand 13% / learned-rnn-rand 50% (clearance 0.35 / 0.37 / 0.35 / 0.32 / 0.25 m) — latency + actuation noise + unseen appearance, danger signal still camera-only
  --randomize + --robust retrain: randomized AUC@32=0.92, closed loop crash reactive 10% -> wm 30%, clearance 0.35 -> 0.33 m — train across the variation, not beside it
```

Two honest readings, both part of the lesson. The randomization recipe buys
the *perception* gap back — 0.82 → 0.92 under the unseen shift — and the veer
ranking survives everything. And the robust model, a strictly better
*detector*, flies **worse** through the hand-crafted planner (30 % vs 17 %):
the planner's margins were calibrated against the shipped model's probability
floor, and retraining moved the floor. The refrain, one level up: **a better
score is not a better flight** — the cost function is the bottleneck, so learn
it (step 6), and let Track E measure on hardware what randomization cannot
model.

The learned policies complete the picture. The clean-trained stacked policy
flies the storm at **10 %** — the privileged-direction baseline's level,
reached from vision alone without ever training there — and storm-training it
buys nothing more (13 %, within noise at 30 seeds). The *policy* is already
robust; what remains of the storm lives in *perception*, which is exactly
what `--robust` buys back.

Step 6 does exactly that — Lesson 19's PPO over the world model's outputs,
with one second of stacked memory, and no hand-tuned danger weights anywhere
(measured after 300k steps, ~16 minutes of training; the LSTM, edge-biased,
curriculum and storm-trained variants join the same table):

```
LEARNED-POLICY OK: 60 cluttered courses @ 0.8 m/s — crash reactive 2% / wm-mpc 17% / learned 0% / learned-rnn 2% / learned-rnn-edge 10% / learned-rnn-curr 7% / learned-rand 7% / learned-rnn-rand 37% (clearance 0.41 / 0.40 / 0.32 / 0.28 / 0.29 / 0.27 / 0.32 / 0.28 m)
  single-pillar @ 1.6 m/s — crash reactive 70% / wm-mpc 8% / learned 0% / learned-rnn 45% / learned-rnn-edge 7% / learned-rnn-curr 80% / learned-rand 3% / learned-rnn-rand 45% — the cost function is learned, the world model is the same
```

The learned stacked-memory policy erases the hand planner's cluttered-course
tail outright (**17 % → 0 %**, beating even the privileged-direction baseline's
2 % from vision alone) and stays at **0 %** at speed where reaction crashes
70 % — with the *same* world model underneath. Eight hand-tuned
configurations couldn't do any of this at once; sixteen minutes of learning
did all of it. That is the lesson's closing argument.

## Going further

- **One memory that holds the whole envelope.** The data axis is now measured
  out, four runs deep: uniform sampling starves the edge (40 % at 1.6 m/s),
  `--edge-bias` closes it (3 %) and reopens the cluttered tail (2 % → 10 %),
  and `--curriculum` — three diets in sequence, aimed at consolidating both —
  lost both bands at once (7–73 %): on-policy learning has no yesterday, so
  order is not a free knob. What remains is the *model* axis: a model-side
  GRU over `z_t` (memory inside the world model, not the policy), or a
  constant mixed diet (`edge_p≈0.25` from step one, no phases) as the one
  cheap data-side probe left. The finding, four times measured: **the simple
  stack has yet to lose anywhere** — elegance still hasn't paid its way here.
- **Longer memory, harder worlds.** The stacked second of memory suffices for
  this corridor; denser clutter and moving obstacles will need more — that is
  where the recurrent line (or yaw-aligned flight) earns its keep.
- **Metric-ground the latent (research-grade).** Use **4D-GS offline** to
  produce geometry-consistent occupancy and add a geometry-grounded
  latent-prediction loss, so `ẑ_{t+k}` decodes to a collision-checkable
  distance — V-JEPA's latency with 4D-GS's grounding.
- **Honest gaps.** The counterfactual oracle exists because sim labels are
  privileged anyway; with real-world data you are back to executed-action
  supervision and need far more of it. The danger labels are planar (visual-only
  pillars), which is also why `climb` sits in the model's vocabulary but off the
  planner's menu. And this is a *nano distillation* of the V-JEPA idea — the
  real V-JEPA 2 is Orin-class; here we teach the principle under the GAP8
  budget. Step 5 prices the randomization gap and ships the recipe that buys
  it back; Track E measures what randomization cannot model, on hardware.

---

<a name="中文"></a>
# Lesson 29 — nano 世界模型：隱空間預測做預判式避障

🌐 [English](#lesson-29--nano-world-model-latent-space-prediction-for-proactive-avoidance) · **繁體中文**（以下）

> **一句話：**訓練一個會預測畫面接下來怎麼變的小模型，讓無人機在障礙物逼近之前就提前閃開。 · **建立在：**第 17 課、第 18 課、第 19 課、第 21 課、第 4 課

> 🔭 **本課是課程的 research preview。**它負責教會概念——隱空間預測、collision
> head、預判式避障——並附上量測結果。深入研究（更難的世界、模型側記憶、度量接地、
> 真機）在續作 [**microdrone-world-model**](https://github.com/csinghans/microdrone-world-model) 繼續。

## 為什麼

目前為止每個模型都是**反應式**的：深度網路（Lesson 17）與 RL 策略（Lesson 19）回答的都是
「對**現在**已經很近的障礙我該怎麼辦？」但在高速下，「現在」已經太遲——等柱子塞滿畫面，
一台加速度有上限的 27 克無人機根本轉不過來。

**世界模型**補上缺的那一塊：**預判（anticipation）**。它學會世界將如何變化——以及世界會
**因為你的動作**而如何變化——讓無人機對**即將**發生的事先動作。這正是本課一路指向的前沿，
也是自駕研究用世界模型攻堅的同一個問題。

陷阱是用「預測下一張影像」（擴散／像素生成）來做，那既慢又會幻想出控制器根本用不到的細節。
我們改走 **V-JEPA** 路線——**預測下一個隱空間 embedding，而不是下一格像素**。沒有幻覺影格、
沒有擴散延遲、算力固定。

而招牌動作在前沿再現一次：真正的 V-JEPA 是十億參數、需要 Orin 級 GPU 的大模型，**永遠**塞不進
GAP8。所以你不是下載它，而是在 512KB int8 預算內**訓練你自己的 nano 版**——然後**把迴路閉起來**：
一個 tiny latent MPC 只靠相機飛行、看到 ~667ms 遠，觸發閃避的時間點可量測地早於反應式控制器
（70 條航道平均 +243ms，下方 demo 裡約 500ms）。

## 概念

四個小網路，沿用課程既有的 conv stack，全部可 int8：

- **Encoder** `f_θ`：影像 → 64 維隱向量 `z`。用 Lesson 3 `TinyDronet` 的 conv stack——但把
  全域平均池化換成**方位感知池化**（4 條水平帶 + 一層小投影）。全域池化會把「在哪裡」平均掉：
  它說得出「柱子很近」，說不出「柱子在左邊」——而往左躲還是往右躲，正是「在哪一側」的問題。
  （下方實測：用全域池化時，不管怎麼監督，veer-ranking 都停在隨機；換帶狀池化後到 1.00。）
- **Predictor** `g_φ`：`(z_t, action) → ẑ_{t+k}`，**四個 horizon** `k ∈ {4, 8, 16, 32}` 步
  （48Hz 下約 83 / 167 / 333 / 667 ms）——一個共享 trunk，每個 horizon 一顆小殘差 head
  （`ẑ = z_t + Δ_k`，「什麼都不變」是免費基線）。「預判」是關於**時間**的宣稱：想買回哪怕
  幾百毫秒的反應時間，模型就得看到 ~667ms 遠，而不是固定一跳 167ms。而且每條訓練 rollout 都把整組指令
  乘上一個巡航速度倍率（0.6–1.6 m/s），讓 heads 學會「危險是指令速度的函數」——這正是
  step 4c 的動力來源。
- **Collision heads**：`ẑ_k → P(k 步內進入 0.7m)` **與** `P(k 步內進入 0.35m)`——每個 horizon
  一組**警戒環**與**臨界環**。要兩個環，因為單一個環是「區域測試」不是「風險梯度」：
  在 0.7m 環內，*每個*動作都會「警戒」（繞過柱子本來就會短暫壓進 0.7m——預測是對的），
  只有臨界環還分得出「擦身而過」與「真的要撞」。（單環實測：planner 在邊界凍結成永久懸停。）
- **Danger-now head**：`z_t → P(現在就太靠近)`——**反應式**訊號，刻意保留：它是預判必須在
  「同一顆感測器」條件下打敗的誠實基線。

讓隱空間預測「不用像素 loss 也不塌縮」的關鍵是 **target encoder**：`f_θ` 的 EMA 複本加
stop-gradient（V-JEPA / BYOL 的做法），再加一個 VICReg 式變異數護欄：

```
Σ_k ‖ g_φ(f_θ(x_t), a_t)_k − sg(f_EMA(x_{t+k})) ‖²  +  變異數護欄  +  danger BCEs
```

**資料集是實驗，不是錄影。**每條 rollout 都重置模擬器（全新試驗）、先向前巡航，然後飛一串
**held segments**：每 ~1 秒抽一個高階指令——forward / slow / veer_left / veer_right / climb /
hover——持續到蓋滿最長 horizon。記錄的是**commanded setpoint** 而非量測速度：控制器只能餵指令
給模型，模型就必須以指令為條件。標籤照樣免費（模擬器的招牌動作）而且更進一步：一個
**counterfactual oracle** 對*每一幀 × 每個候選動作 × 每個 horizon* 都給標籤——把指令沿已知柱子
佈局做運動學前推。執行過的 rollout 教「發生了什麼」；planner 需要的是「如果做了會怎樣」。
一條誠實規則：由**相機 60° FOV 之外**的柱子造成的陽性標籤，單幀模型根本答不了，訓練與評分
一律遮罩。

**閉環的代價（全部在本 repo 實測）：**第一版 collision AUC 0.9——但「往左躲 vs 往右躲」的排序
是隨機。AUC 是*距離*問題；規劃是*選擇*問題。把選擇修好，依序花了：intervention segments
（counterfactual 對比）、counterfactual oracle（密集排序監督）、FOV 遮罩（45 個失敗 probe 幀
裡 37 個的威脅在 66–84° 方位——根本看不見）、最後是上面的方位感知池化（根因）。教訓是：
**AUC 高不代表模型會排序動作——要測決策，不是只測偵測。**這個測試以 `veer-ranking` 指標
隨課出貨：在幾何上一側明確較安全的 held-out 幀上，模型是否把較安全的那側排前面？隨機是 0.5；
本模型實測 1.00。

把 *planner* 閉起來用的是同一種紀律，一次量測解一個失效模式：對 horizon 取 max 的危險度在
環邊一票否決造成癱瘓（→ 急迫度加權）；單環在環內沒有梯度（→ 臨界環）；絕對觸發門檻繼承了
heads 依場景而異的機率地板、在每條乾淨航道上誤觸發（→ **相對**邊際觸發）；減速是會爬進環內的
假逃生（→ 只用側移逃生）；長 commit 等於盲飛（→ 短 commit＋用自身里程計做走廊回中先驗）。
後設教訓：**疊在學習頭上的每一條 planner 假設都是待驗證的假說——量測它。**

**為什麼選隱空間（V-JEPA），而非顯式 4D 幾何（4D-GS）？**兩者都是世界模型，但在權衡的兩端。

| | **V-JEPA（本課）** | **4D Gaussian Splatting** |
|---|---|---|
| 表徵 | 隱式*隱空間*流形 | 顯式度量 3D + 時間幾何 |
| 預測 | 未來的*embedding* | 渲染未來的*場景* |
| 強項 | 預判、抽象、**延遲有上限** | 度量精準、擬真 |
| 成本 | 定長 FLOP、**可蒸餾成 int8 < 512KB** | 隨 Gaussian 數擴張 → Orin 級 |
| 幻覺 | 無（不 commit 像素） | 外推時出現幾何 floater |

4D-GS 是你在 **Lesson 14a 建的 2D occupancy grid** 的重量級顯式幾何表親——延伸到 3D 加時間。
它度量精準得很美，卻塞不進 GAP8。V-JEPA 是塞得進的隱空間路線，所以是我們的機上骨幹。

## 動手做

```bash
conda activate nanodrone-ai
python lessons/29_world_model/gen_wm_dataset.py --rollouts 64    # 1. intervention 試驗
python lessons/29_world_model/train_world_model.py --epochs 80   # 2. 訓練 nano V-JEPA
python lessons/29_world_model/proactive_avoid.py                 # 3. 決策時機示意
python lessons/29_world_model/wm_closed_loop.py                  # 4. 純視覺閉環
python lessons/29_world_model/eval_world_model_policy.py         # 4b. 雜訊場景記分板
python lessons/29_world_model/speed_sweep.py                     # 4c. 墜機率 vs 速度
python lessons/29_world_model/eval_robustness.py                 # 5. sim-to-real 陷阱標價
python lessons/29_world_model/learn_policy.py                    # 6. 把策略學出來（~16 分鐘）
```

Step 3（`proactive_avoid.py`）用 privileged 幾何隔離出決策*時機*——一張示意圖。Step 4 拆掉
柺杖：控制迴路裡的危險訊號**只來自相機**（encoder → predictor → collision heads），柱子位置
只用來佈置場景與事後評分。planner 是一個 12Hz 的 tiny **latent MPC**：每幀 encode **一次**，
用小 MLP 把每個候選指令的未來都「想」一遍（昂貴的 encoder 共用，所以在 GAP8 上整段深思幾乎
免費）；當「直行」被預測得比較好的那側側移**明顯更危險**時才觸發逃生——**相對**觸發，因為
絕對門檻會繼承 heads 依場景而異的機率地板——並以臨界環加權的 cost 挑最便宜的側移。反應式
基線用同一顆 encoder——並且**刻意**拿到 privileged 的閃避方向：它只可能輸在時機上，比較因此
把「預判」單獨隔離出來。Step 4c 再把巡航速度往上掃（單柱航道，威脅雙方都看得見）：反應式在
固定*距離*觸發、預判在固定*時間*觸發——速度一拉高，只有一種預算撐得住。

Step 6 對 planner 的裁決採取行動：不再調 cost 函數，改用 PPO（完全是 Lesson 19 的配方——
前進獎勵、墜機懲罰，危險項沒有任何手工塑形）從世界模型自己的輸出學出策略，並把最近一秒的
輸出堆疊成記憶——滑出 60° FOV 的柱子會在觀測裡多留一秒。同一顆 encoder、同一組 heads、
同一套測試工具；只有「做決定的東西」是學出來的。記憶有兩種口味（`--recurrent` 把堆疊換成
sb3-contrib 的 LSTM——全課唯一的可選新依賴），`--randomize` 直接在 step 5 的風暴裡訓練，
`--edge-bias` 把每回合的訓練速度往包絡的高速端加權，`--curriculum` 則把同一份預算依序
跑過三種配方（下方實測——它適得其反）；每個訓練出的變體都會自動加入記分板。

每支腳本都自產所需資料，可用 `--selftest` 獨立執行。

## 驗收 ✅

Step 1（本機實測，完整跑）：

```
WM-DATA OK: 64 rollouts x 120 steps @ 48 Hz, 87 held intervention segments, labels [k=4: n=7076 pos=0.25, k=8: n=6472 pos=0.27, k=16: n=5266 pos=0.30, k=32: n=3020 pos=0.39], saved .../output/wm_dataset.npz
```

Step 2 驗證 predictor 贏過「未來＝現在」、各 horizon 的 AUC（警戒環）、danger-now head、
**以及 veer ranking**——證明模型會「選」而不只會「偵測」的那個數字（rollout 層級切分，零洩漏）：

```
WORLD-MODEL OK: 2442 train seqs, latent MSE@32=1.200 (no-op 2.538), AUC@4/8/16/32=0.96/0.96/0.96/0.96, now-AUC=0.81, veer-ranking=1.00 (n=25), int8 weights=81.3 KB (<512 fits), saved .../output/world_model.pth
```

Step 4 只靠視覺把同一條航道飛兩次，驗證更早觸發、更大淨空、不墜機、抵達終點——控制路徑中
沒有任何 privileged look-ahead：

```
WM-CLOSED-LOOP OK: reactive min-clear=0.38 m (trigger@60), wm min-clear=0.48 m (trigger@36), lead=+24 steps (~500 ms earlier), crashes reactive/wm = 0/0, goal steps = 187/198 — danger signal from camera alone (no privileged look-ahead in control)
```

Step 4b 是雜訊場景記分板——100 條隨機多柱航道（70 條有威脅、30 條乾淨，用來抓 false
positive），加上嵌入式工程師真正會問的機上帳單（weights 不是全部：activation 與 DMA workspace
共用同一塊 512 KB）：

```
WORLD-POLICY OK: seeds=100 (70 in-path / 30 clear)
  crash_rate:        reactive 1% -> wm 16%
  mean_min_clearance: 0.41 m -> 0.38 m
  mean_trigger_lead: +243 ms (n=70 both triggered)
  false_positive:    reactive 0% -> wm 0% of clear runs
  goal_time:         wm +2% vs reactive (n=58 clean)
  decision latency:  0.4 ms measured (this CPU) | ~8 ms est @ GAP8 0.5 GMAC/s (3.9 M MACs, encoder shared across 6 candidates)
ONBOARD-BUDGET OK: weights=81.3 KB + peak_activation=28.0 KB + workspace(dbl-buf)=28.0 KB = 137.3 KB < 512 KB
```

Step 4c 是機制本身的量測——每個巡航速度 30 條單柱航道、跨速度同一組 seeds、
所有可用策略同場（墜機率）：

| 巡航 | reactive | wm（手工 MPC） | **learned（堆疊記憶）** | learned（LSTM） | LSTM＋邊緣過採樣 | LSTM＋課程學習 |
|---|---|---|---|---|---|---|
| 0.8 m/s | 0 % | 10 % | **0 %** | 0 % | 0 % | 7 % |
| 1.0 m/s | 0 % | 0 % | **0 %** | 0 % | 0 % | 23 % |
| 1.2 m/s | 3 % | 0 % | **0 %** | 0 % | 0 % | 13 % |
| 1.4 m/s | **40 %** | 0 % | **0 %** | 3 % | 0 % | 47 % |
| 1.6 m/s | **60 %** | 10 % | **0 %** | **40 %** | 3 % | **73 %** |

```
SPEED-SWEEP OK: 30 single-pillar courses/speed — crash (reactive/wm/learned/learned-rnn/learned-rnn-edge/learned-rnn-curr) at 0.8 m/s = 0%/10%/0%/0%/0%/7%; at 1.6 m/s = 60%/10%/0%/40%/3%/73% — reaction pays a distance, anticipation pays time
```

學出來的堆疊記憶策略把*整條*速度帶飛完、一次都沒撞——150 條航道，零墜機。LSTM 版
是誠實的對照組，而且量了四次。在堆疊版的 300k 預算下它從未收斂（全帶 3–37%）。給了
*公平*預算——三倍步數、合身的 64 寬 LSTM、256 步的 BPTT 窗口——它在 1.4 m/s 以下
收斂（0–3%），卻仍在速度極端崩潰（1.6 m/s 時 40%——那裡回合最短，均勻取樣又對邊緣
雙重稀薄）。加上 `--edge-bias`——一半的訓練回合改抽自高速邊緣、其餘旋鈕一個都不動——
極端關上了：掃描帶 **40% → 3%**、快速重跑 45% → 7%。關上了，但不是免費的：重加權的
質量是從慢速雜訊域挪來的，雜訊尾巴於是重新打開，2% → 10%。你取樣什麼、就得到什麼——
資料配方補上你指著的那條帶，洞就搬去質量被挪走的地方。於是那個「顯然」的下一步也試了：
`--curriculum`，同一份預算依序跑過三種配方（自然 → 鑽邊緣 → 混合收尾），目標是把兩帶
一起鞏固。結果適得其反——**到處都更糟**（掃描帶 7–73%、快速重跑 80%），而同場的均勻版
與邊緣版都重現了自己的數字。on-policy 學習沒有昨天：每一段都對著自己的配方最佳化、
把上一段覆寫掉，最後的混合段既沒有足夠步數、也沒有錨點把兩帶撈回來。**順序不是免費的
旋鈕。**四輪量下來，資料軸已經量盡——均勻餓死邊緣、重加權搬洞、排序兩頭空——而簡單的
堆疊這些全都不需要；它至今沒在任何地方輸過。

把兩張記分板放在一起、用機器人工程師的方式讀。**Step 4c 是機制**：反應式在固定*距離*觸發，
速度一拉高就把預算花光——墜機率 0% 飆到 60%；會預判的 MPC 在固定*時間*觸發，全程壓在
0–10%，高速下淨空還有 2–3 倍。**Step 4b 是誠實的極限**：雜訊航道的側柱位在 60–90° 方位、
在前向相機 FOV 之外，無記憶的 planner 在那裡付出 16% 的墜機尾巴（拿了 privileged 方向的基線
不用付）——但誤閃避保持零、時間成本 +2%。模型已經不是瓶頸（veer-ranking 1.00）；手寫 cost
函數才是——這是量測了八種 planner 配置後確立的結論，而 step 6 對它採取了行動：
*學出來的*策略把尾巴整個抹掉（0%）。

Step 5 在硬體付錢之前先為 sim-to-real 陷阱標價：拿 shipped（clean 訓練）模型，在它沒見過的
條件下量測——隨機柱形／顏色、0–2 步指令延遲、±8% 致動噪音、加上 Lesson 20 的固定「陌生相機」
外觀偏移打在 policy 看到的每一幀上——然後展示 Track B 的解法（`gen --randomize` +
`train --robust`）、讓模型「在變異之中」而非「在變異旁邊」訓練：

```
ROBUST-WM OK: clean AUC@32=0.96 | randomized(+unseen-shift) AUC@32=0.82 (the gap to buy back), veer-ranking=1.00 (n=8)
  closed loop under randomization — crash reactive 10% / wm 17% / learned 10% / learned-rand 13% / learned-rnn-rand 50% (clearance 0.35 / 0.37 / 0.35 / 0.32 / 0.25 m) — latency + actuation noise + unseen appearance, danger signal still camera-only
  --randomize + --robust retrain: randomized AUC@32=0.92, closed loop crash reactive 10% -> wm 30%, clearance 0.35 -> 0.33 m — train across the variation, not beside it
```

兩個誠實讀法，都是課的一部分。域隨機化把**感知** gap 買回來了——陌生偏移下 0.82 → 0.92——
而且 veer ranking 全程撐住。但那個嚴格更好的*偵測器*（robust 模型）過手寫 planner 飛得
**更糟**（30% vs 17%）：planner 的 margin 是對 shipped 模型的機率地板校準的，重訓把地板
移走了。同一句副歌、高一個八度：**分數更好不等於飛得更好**——瓶頸是 cost 函數，把它學出來
（step 6），再讓 Track E 在真機上量測隨機化模擬不了的部分。

學出來的策略把這張圖補完整。clean 訓練的堆疊記憶策略在風暴裡飛出 **10%**——追平拿
privileged 方向的基線，而且純靠視覺、從未在風暴中訓練過——在風暴裡訓練它也沒有買到更多
（13%，30 seeds 下屬噪音範圍）。*策略*本身已經健壯；風暴剩下的部分在*感知*層——
而那正是 `--robust` 買回來的東西。

Step 6 正是這麼做的——Lesson 19 的 PPO 讀世界模型的輸出、帶一秒的堆疊記憶、任何地方都沒有
手調的危險權重（300k 步、約 16 分鐘訓練後實測；LSTM、邊緣過採樣、課程學習與 storm 訓練
變體同表較勁）：

```
LEARNED-POLICY OK: 60 cluttered courses @ 0.8 m/s — crash reactive 2% / wm-mpc 17% / learned 0% / learned-rnn 2% / learned-rnn-edge 10% / learned-rnn-curr 7% / learned-rand 7% / learned-rnn-rand 37% (clearance 0.41 / 0.40 / 0.32 / 0.28 / 0.29 / 0.27 / 0.32 / 0.28 m)
  single-pillar @ 1.6 m/s — crash reactive 70% / wm-mpc 8% / learned 0% / learned-rnn 45% / learned-rnn-edge 7% / learned-rnn-curr 80% / learned-rand 3% / learned-rnn-rand 45% — the cost function is learned, the world model is the same
```

學出來的堆疊記憶策略把手工 planner 的雜訊尾巴**整個抹掉**（**17% → 0%**，純視覺甚至贏過
拿 privileged 方向的基線的 2%），高速下同樣 **0%**（反應式墜 70%）——底下是*同一個*
世界模型。八種手調配置一項都做不到；十六分鐘的學習全做到了。這就是本課的結辯。

## 延伸

- **一個撐得住整個包絡的記憶。**資料軸已經量盡，四輪見底：均勻取樣餓死邊緣（1.6 m/s
  40%）、`--edge-bias` 關掉邊緣（3%）卻重開雜訊尾巴（2% → 10%）、`--curriculum`——
  三段配方依序走、目標是兩帶一起鞏固——反而兩頭空（7–73%）：on-policy 學習沒有昨天，
  順序不是免費的旋鈕。剩下的是*模型*軸：模型側 `z_t` 上的 GRU（把記憶放進世界模型、
  不是策略），或恆定混合配方（`edge_p≈0.25`、從頭到尾不分段）作為資料側最後一探。
  量了四次的發現：**簡單的堆疊至今沒在任何地方輸過**——優雅還沒付清它的路費。
- **更長的記憶、更難的世界。**這條走廊一秒的堆疊記憶就夠；更密的雜訊與會動的障礙物才是
  recurrent 路線（或 yaw 對齊飛行）真正掙飯吃的地方。
- **把隱空間度量接地（研究級）。**用 **4D-GS 離線**產生幾何一致的 occupancy，加一個
  geometry-grounded latent-prediction loss，讓 `ẑ_{t+k}` 能解碼成可做碰撞檢測的距離——
  V-JEPA 的延遲換到 4D-GS 的接地。
- **誠實的落差。**counterfactual oracle 之所以存在，是因為模擬器標籤本來就是 privileged 的；
  換成真實資料就回到只有 executed-action 監督、而且需要多得多的資料。danger 標籤是平面的
  （柱子只有視覺體），這也是為什麼 `climb` 在模型詞彙表裡、卻不在 planner 菜單上。而這是
  V-JEPA 想法的 *nano 蒸餾版*——真正的 V-JEPA 2 是 Orin 級；這裡在 GAP8 預算下教原理。
  Step 5 為隨機化 gap 標了價、也附上買回它的 recipe；隨機化模擬不了的部分，由 Track E
  在真機上量測。
