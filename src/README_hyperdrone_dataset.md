# 用 hyperdrone 生成 world-model 数据集

`gen_wm_dataset_hyperdrone.py` 是 `gen_wm_dataset.py` 的替代实现：**同样的动作词表、held-segment 调度、柱子布局、标签和 npz 格式**（同一个 `--seed` 得到同一批 course），但飞行由 hyperdrone 的 L2F 动力学 + RAPTOR 位置控制完成，图像由 hyperlux 光线追踪渲染，并且可以把场景换成 hyperlux 自带目录里的 **真实家具房间（ProcTHOR）**，或者两种场景混在一份数据集里。`train_world_model.py`、`counterfactual_labels`、`window_valid` 等不需要改动就能读。

为什么要做这一步：

1. **相机可以对齐真实 Tello。** SakanaDreamer 分支 `tello-calibration-drone-ws` 的 ChArUco 标定（960×720，fx≈904）给出水平 56°、垂直 43.5° 的视场，`virtual_maps(64, fov)` 最多只能生成 **42.5°** 的正方形虚拟视图，原训练用的 60° 根本做不出来。hyperdrone 可以直接按 `--fov-deg 40` 渲染训练数据。
2. **批量。** 32 条 rollout 作为一个 batch 同时积分和渲染，64×120 帧的数据集在这台机器上几秒钟生成（第一次运行多出约 2 分钟的一次性 JIT 编译）。
3. **光照和阴影。** 光线追踪的明暗、阴影和 `--randomize` 下的随机灯光、墙色，是和 pybullet 平面渲染不同的外观分布。
4. **真实房间。** hyperlux 以内容哈希固定了 ProcTHOR 200 栋训练房屋和 HSSD 125+43 栋（GLB，托管在 Hugging Face 的 conta 分片上，ProcTHOR 单栋 3–30 MB，首次使用时自动下载）。`--scene rooms` / `--scene mixed` 直接在这些房子里飞。

诚实的边界：这仍然是仿真。柱子没有碰撞物理（和原版一样会穿过去），任何一个 sim 都不是照片级，最终还是要靠 Tello 实测。

## 用法（conda env `dreamer`，在 `drone_ws/` 下）

```bash
# 与 pybullet 等价的 60° 机体固定相机
python src/gen_wm_dataset_hyperdrone.py --rollouts 64 --len 120 --fov-deg 60 --camera body

# 面向 Tello：40° 视场 + 虚拟稳像（Tello 以 0.15 m/s 飞行时相机几乎不倾斜）
python src/gen_wm_dataset_hyperdrone.py --rollouts 64 --len 120 --fov-deg 40 --camera level \
    --out src/output/wm_dataset_hyperdrone_tello40.npz

# 外观/执行随机化版本（柱子形状颜色、灯光、墙色、0–2 步延迟、±8% 执行噪声）
python src/gen_wm_dataset_hyperdrone.py --rollouts 64 --len 120 --fov-deg 40 --camera level --randomize \
    --out src/output/wm_dataset_hyperdrone_tello40_rand.npz

# 混合数据集：32 条柱子 + 32 条房间（4 栋 ProcTHOR 房子各 8 条），Tello 相机
python src/gen_wm_dataset_hyperdrone.py --scene mixed --rollouts 32 --rooms 32 --houses 4 \
    --len 120 --fov-deg 40 --camera level --out src/output/wm_dataset_mixed_tello40.npz

python src/gen_wm_dataset_hyperdrone.py --selftest      # 断言检查（柱子）
python src/wm_rooms.py 1                                # 看一栋房子的占据栅格/起点图

# 训练：--data 选数据集，--out 另存 checkpoint，不要覆盖 Tello 端 SHA 锁定的 world_model.pth
python src/train_world_model.py --epochs 80 --data src/output/wm_dataset_hyperdrone_tello40.npz \
    --out src/output/world_model_hyperdrone_tello40.pth
```

每份数据旁边会写一张 `*_preview.png`（8 条 rollout × 8 个时刻，混合数据会两种场景都抽到）。其他参数：`--rooms N`（房间 rollout 数，默认等于 `--rollouts`）、`--houses H` / `--house-first K` / `--catalog procthor|hssd`（用目录里第 K 起的 H 栋房子）、`--save-depth`（float16 米制深度）、`--drone-model`（`hyperdrone.dynamics.MODELS` 里的机型，默认 `crazyflie`）、`--kinematic`（不跑动力学，相机直接沿参考轨迹走）、`--fidelity`。

## npz 里多出来的键

| 键 | 含义 |
|---|---|
| `fov_deg`, `fov_half_deg` | 相机视场；`fov_half_deg = fov/2 − 2`，`counterfactual_labels` 与 `veer_ranking` 会优先读它来判断"威胁是否在视野内"（没有这个键时仍用 28°） |
| `camera` | `body`（随机体俯仰横滚）或 `level`（只保留偏航，虚拟稳像） |
| `quat` | (R, L, 4) 机体姿态 wxyz，用于诊断倾斜 |
| `scene_type`, `scene_id` | 0 = 柱子 / 1 = 房间；`pillars` 或 ProcTHOR 房屋 id |
| `yaw`, `start_world` | rollout 坐标系（x 朝前）在场景里的朝向和起点；`pos` 始终是 rollout 坐标系 |
| `cf_labels`, `cf_visible` | (R, L, 6, 4, 2) / (R, L, 6) 预先算好的反事实标签与可见性；`gen_wm_dataset.counterfactual_labels` 发现这两个键就直接返回它们，房间 rollout 没有柱子列表也能训练 |
| `source`, `drone_model`, `sim_dt`, `kinematic` | 来源标记 |
| `depth` | 可选，(R, L, 64, 64) float16 |

训练脚本会把 `fov_deg` 和 `source` 写进 checkpoint 的 `meta`，Tello 端可以据此拒绝视场不匹配的权重。

## 和 pybullet 版本的对照（seed 0，64 rollouts × 120 步，80 epoch，相同 rollout 级切分）

| 数据集 | 相机 | latent MSE@32（no-op） | AUC@4/8/16/32 | now-AUC | veer-ranking (n) |
|---|---|---|---|---|---|
| `wm_dataset.npz`（pybullet，对照） | 60° body | 1.603 (2.835) | 0.93 / 0.91 / 0.92 / 0.93 | 0.84 | 1.00 (45, 扩到全部 rollout) |
| `wm_dataset_hyperdrone.npz` | 60° body | 0.946 (3.774) | 0.98 / 0.98 / 0.98 / 0.97 | 0.94 | 1.00 (20) |
| `wm_dataset_hyperdrone_tello40.npz` | 40° level | 0.859 (2.889) | 0.98 / 0.97 / 0.98 / 0.98 | 0.89 | 1.00 (20) |
| `wm_dataset_hyperdrone_tello40_rand.npz` | 40° level，randomize | 0.491 (2.240) | 0.88 / 0.88 / 0.92 / 0.94 | 0.93 | 1.00 (126, 扩到全部 rollout) |
| `wm_dataset_hyperdrone_tello40_body.npz` | 40° body | 0.776 (3.046) | 0.97 / 0.97 / 0.97 / 0.97 | 0.91 | 0.85 (20) |

对应的 checkpoint 在 `src/output/world_model_pybullet_ref.pth`、`world_model_hyperdrone60.pth`、`world_model_hyperdrone_tello40.pth`、`world_model_hyperdrone_tello40_rand.pth`、`world_model_hyperdrone_tello40_body.pth`。各数据集的标签阳性率（k=4/8/16/32）：pybullet 0.25/0.27/0.30/0.39，hyperdrone 干净版 0.16/0.17/0.20/0.30，randomize 版 0.23/0.25/0.28/0.38。指标都是各自 held-out rollout 上的数字，只说明"这份数据能训出和原流程同等水平的模型"，不代表真机表现。

说明：

- RAPTOR 位置控制对移动参考点有约 0.5 s 的滞后，所以生成器把速度意图转换成"指令切换时在平面上重锚定的积分参考 + 0.25 s 渐入的 0.5 s 速度前馈"；高度参考始终是绝对值并带一个小的外环 P/I 补偿（RAPTOR 在 1.6 m/s 时自身会下坠约 0.25 m）。稳态速度与指令比约 1.05，起步比 pybullet 的 PID 慢约 0.25 s，整条 rollout 前进距离少约 0.3 m，这是阳性率略低于 pybullet 的原因。
- `body` 相机下 crazyflie 加速时的俯仰 p95 约 26°，这是机型物理决定的；给 Tello 用的数据建议 `--camera level`。

## 真实房间是怎么打标签的（`wm_rooms.py`）

柱子版的"危险"是到已知柱心的平面距离；房间里没有这样的列表，所以用几何本身：

1. 在飞行高度 1.0 m，把房子占地按 0.1 m 栅格逐格探测：每格 4 个 90° 深度相机组成水平环（垂直只取 ±3.6° 的薄带）+ 1 个朝上的射线。射线命中的表面格记为占据（相当于一张无人机高度的激光雷达地图），头顶没有天花板的格算室外，也记为占据。
2. 对占据图做欧氏距离变换，得到每一点的表面净空和最近障碍格；`dists = 净空 + 0.18`，这样 0.7 m / 0.35 m 两个危险环保持和 0.18 m 柱子一样的**表面**含义（0.52 m / 0.17 m）。
3. 起点：室内、净空 ≥ 0.9 m，随机航向；偶数 rollout 要求正前方 1.3–2.6 m 处有障碍（对应柱子版的 in-path），奇数要求正前方 ≥ 3.5 m 畅通，前 1 m 两侧净空 ≥ 0.5 m。无人机以该航向飞行（复位时直接写入偏航姿态，否则 RAPTOR 会在大角度偏航阶跃时掉高失控），`pos` 记在 rollout 自己的坐标系里。
4. 反事实标签沿每个候选动作的直线路径查距离场，与柱子版 `counterfactual_labels` 的逻辑一一对应；可见性用最近障碍格的方位判断。小测试里预存标签与实飞标签的一致率：柱子 0.95、房间 0.97。

诚实说明：薄带意味着低矮的桌子会被飞过、高处的灯罩不算障碍；无人机穿墙后会看到屋外的黑背景（约 2–3% 的帧），这与原版穿过柱子是同一种"无碰撞物理"语义。`python src/wm_rooms.py <目录序号>` 会输出该房子的占据/距离场和 16 个起点的 PNG，便于检查。

## Hugging Face

`hf_dataset.py` 把若干 npz、预览图、`manifest.json`（形状、阳性率、sha256、生成命令）和数据卡 `README.md` 收进一个目录，再用 `huggingface_hub` 推到一个 dataset repo。训练代码直接读 npz，所以 Hub 上保存的就是训练用的文件，不做会被训练脚本撤销的格式转换。

```bash
python src/hf_dataset.py build src/output/wm_dataset_mixed_tello40.npz src/output/wm_dataset_mixed_60body.npz \
    src/output/wm_dataset_hyperdrone_tello40.npz src/output/wm_dataset_hyperdrone_tello40_rand.npz \
    src/output/wm_dataset_hyperdrone.npz --out hf_dataset --metrics src/output/hf_metrics.md

hf auth login                                   # 或 export HF_TOKEN=...
python src/hf_dataset.py push --folder hf_dataset --repo <你的用户名>/drone-world-model-sim --private
```

拉取并训练：

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("<你的用户名>/drone-world-model-sim", "data/wm_dataset_mixed_tello40.npz", repo_type="dataset")
```

```bash
python src/train_world_model.py --data "$path" --out src/output/world_model_mixed_tello40.pth
```

### 混合数据集的训练检查（80 epoch，rollout 级切分，各自的 held-out）

| 数据集 | 组成 | latent MSE@32（no-op） | AUC@4/8/16/32 | now-AUC | veer-ranking（只在柱子 rollout 上评） |
|---|---|---|---|---|---|
| `wm_dataset_mixed_tello40.npz` | 32 柱子 + 32 房间（4 栋），40° level | 2.119 (2.860) | 0.88 / 0.91 / 0.91 / 0.84 | 0.86 | 0.42 (n=24) |
| `wm_dataset_mixed_60body.npz` | 32 柱子 + 32 房间（4 栋），60° body | 2.037 (2.949) | 0.94 / 0.96 / 0.94 / 0.90 | 0.89 | 0.75 (n=24) |

房间让任务明显变难：同样 64 条 rollout，碰撞 AUC 从柱子版的 0.97–0.98 降到 0.84–0.96，latent 预测相对 no-op 的提升也变小；veer-ranking 只能在 32 条柱子 rollout 上评，40° 版本的 24 个判定帧落在随机水平。这是"小型数据集"的诚实结果：要在真实房间上得到和柱子版一样的指标，需要更多 rollout 和更多房子（生成成本很低，每栋房子约 25 秒）。生成命令、各文件的 sha256 和阳性率都在 `hf_dataset/manifest.json`。

## 对接 SakanaDreamer（Tello 端）

分支 `tello-calibration-drone-ws` 的 `avoidance/model.py` 用 `CHECKPOINT_SHA256` 锁定了旧的 `world_model.pth`，并把输入写死为 60° 虚拟相机；`avoidance/runtime.py` 在 `fov_deg != 60` 时禁止控制。要用 40° 数据训练出的权重真正上机，需要在那边：

1. 把新 checkpoint 的 SHA256 写进 `CHECKPOINT_SHA256`（或改为读取 `meta["source"]` / `meta["fov_deg"]` 做校验）；
2. 把 60° 的硬编码改成读取 `meta["fov_deg"]`，并用 `--fov-deg 40 --camera calibration/tello-charuco-camera-v2.json` 运行；
3. 先按文档走 SHADOW 模式观察预测，再考虑 `--control`。

## 在 pybullet 柱子环境里评估 hyperdrone 数据训出的模型

pybullet 的评估脚本（`eval_world_model_policy.py` 100 条杂乱 course 记分板、`speed_sweep.py` 单柱子速度扫描、`wm_closed_loop.py`）现在接受 `--model` 指定 checkpoint，`speed_sweep.py` 另有 `--no-learned`（只比 reactive 和 wm 两个手写策略；PPO 策略是基于 pybullet 世界模型的潜变量训练的，换了 checkpoint 没有意义）和 `--plot` 输出路径。它们在 `drone_python` 环境里跑（pybullet 物理只有 CPU，上游脚本也把小模型固定在 CPU 上；并行跑多个时请设 `OMP_NUM_THREADS`，否则每个进程会占满所有核）。

pybullet 的画面（白天空、蓝白大格地面、饱和红柱子、无墙）和 hyperdrone 默认场景差别很大，所以增加了 `--arena-style pybullet`：柱子场景改成白背景、蓝白格地面延伸到地平线、红柱子、无墙，用来把"多了数据有没有用"和"画风不同"两个因素分开。`merge_datasets.py` 可以把 pybullet 的 `wm_dataset.npz` 和 hyperdrone 数据按 rollout 拼成一份训练集（为 pybullet 部分补算反事实标签）：

```bash
python src/gen_wm_dataset_hyperdrone.py --scene mixed --rollouts 128 --rooms 128 --houses 16 --len 120 \
    --fov-deg 60 --camera body --arena-style pybullet --out src/output/wm_dataset_mixed_big_60body_pbstyle.npz
python src/merge_datasets.py src/output/wm_dataset.npz src/output/wm_dataset_mixed_big_60body_pbstyle.npz \
    --out src/output/wm_dataset_pb_plus_big_pbstyle.npz
python src/train_world_model.py --epochs 80 --data src/output/wm_dataset_pb_plus_big_pbstyle.npz --out src/output/world_model_pb_plus_big_pbstyle.pth
conda run -n drone_python python src/eval_world_model_policy.py --seeds 100 --model src/output/world_model_pb_plus_big_pbstyle.pth
conda run -n drone_python python src/speed_sweep.py --seeds 30 --no-learned --model src/output/world_model_pb_plus_big_pbstyle.pth --plot src/output/speed_sweep_pb_plus_big_pbstyle.png
```

### 结果（2026-10-27，pybullet 环境，`src/output/pybullet_compare.md`）

训练集：pybullet 原版 64 条；hyperdrone 大混合集 256 条（128 柱子 + 128 房间 / 16 栋）两种画风；以及 pybullet 64 条与两种大混合集的合并（320 条）。每个 checkpoint 用 `eval_world_model_policy.py --seeds 100`（70 条 in-path + 30 条清场）、`speed_sweep.py --seeds 30 --no-learned`、`forest_demo.py --seeds 10 --rows 6 --no-learned` 评估。reactive 基线也用同一 checkpoint 的编码器和 danger-now 头，所以它随 checkpoint 一起变。

| checkpoint | cluttered 100 courses: crash reactive→wm | min clearance | trigger lead | false pos. (clear) | goal time | single-pillar crash wm (reactive) @0.8 / 1.2 / 1.6 m/s | forest 10×6 rows: wm crashes/forest (reactive), crash-free |
|---|---|---|---|---|---|---|---|
| `pybullet_shipped` | 0% → 9% | 0.42 → 0.40 m | +351 ms (n=70) | 0% → 0% | +3% (n=64) | 7% (0%) / 0% (3%) / 7% (57%) | 4.00 (2.30), 0% (0%) |
| `pybullet_ref` | 1% → 10% | 0.41 → 0.37 m | +346 ms (n=70) | 0% → 0% | +2% (n=63) | 7% (0%) / 0% (0%) / 3% (60%) | 2.50 (2.60), 0% (0%) |
| `mixed_big` | 100% → 97% | 0.10 → 0.11 m | +-1208 ms (n=2) | 93% → 100% | – | 83% (100%) / 90% (100%) / 83% (100%) | 3.40 (2.90), 0% (0%) |
| `mixed_big_pbstyle` | 77% → 54% | 0.20 → 0.23 m | +714 ms (n=53) | 63% → 87% | +7% (n=5) | 40% (90%) / 67% (97%) / 13% (100%) | 3.50 (2.30), 0% (10%) |
| `pb_plus_big` | 3% → 31% | 0.41 → 0.33 m | +435 ms (n=70) | 0% → 0% | +3% (n=46) | 20% (0%) / 13% (3%) / 20% (63%) | 3.30 (1.70), 0% (20%) |
| `pb_plus_big_pbstyle` | 0% → 19% | 0.43 → 0.40 m | +418 ms (n=70) | 100% → 100% | +1% (n=57) | 23% (0%) / 10% (0%) / 10% (57%) | 3.70 (2.60), 0% (0%) |

离线指标（各自 held-out rollout）：mixed_big AUC 0.91/0.91/0.90/0.87，mixed_big_pbstyle 0.86/0.83/0.82/0.84，pb_plus_big 0.91/0.89/0.89/0.86，pb_plus_big_pbstyle 0.90/0.89/0.88/0.86；pybullet 对照 0.93/0.91/0.92/0.93。

结论：

- **在 pybullet 柱子环境里，hyperdrone 数据没有带来提升，反而拖后腿。** 只用 hyperdrone 数据训练的模型完全不迁移（撞击率 97%，和 pybullet 的渲染是两个分布）；改成 pybullet 画风后部分迁移（54%，仍远差于 pybullet 自训的 9–10%，而且 87% 的清场 course 误触发）；把 hyperdrone 数据和 pybullet 数据合并训练，闭环撞击率从 10% 升到 19–31%，单柱子 0.8 m/s 的撞击率从 7% 升到 20% 左右。6 行密林对所有手写 MPC 都太难（每片 2.5–4 次穿越），区分不出差异。
- 原因是评估环境本身就是 pybullet 渲染的：在那里，任何别的渲染器的数据都是域外数据，训练集里混入它只会稀释 pybullet 分布，并改变碰撞头的概率标定（MPC 的触发阈值是按 pybullet 模型调的），于是出现更多误触发和更晚/更乱的绕行。离线 AUC 看不出这一点，因为它是在各自的数据上算的。
- 这个实验回答的是"hyperdrone 数据能不能帮 pybullet 里的策略"，答案是不能；它不回答"能不能帮真实 Tello"。对 Tello 来说两个仿真器都是域外，hyperdrone 的价值（可匹配的 40° 相机、真实房间、光照多样性）只能在真机的 shadow 推理里验证。要在仿真里做公平的跨域比较，评估环境也得换成 hyperdrone（在 hyperdrone 上实现 `run_episode` 的闭环接口），或者把两个渲染器的画面对齐到同一外观。
- 如果坚持在 pybullet 里受益，可行的方向是外观对齐和校准：把 hyperdrone 柱子场景的地面纹理、阴影和颜色进一步贴近 pybullet（现在的 pb 风格只做了粗调），训练时打开 `--robust` 外观增强，并对合并数据训出的模型重新标定 MPC 的触发阈值（`wm_closed_loop.py` 里的 `THETA_NOW`、相对触发差值）。

## 反过来：在 hyperdrone 里闭环测试（`hd_closed_loop.py`）

`hd_closed_loop.py` 是 pybullet 记分板在 hyperdrone 里的镜像：同一个 seed 得到同一条柱子 course（`sample_pillars` 复刻了 pybullet 生成器的随机抽样），用同样的 `ReactivePolicy` / `WMPolicy` 对象、同样的 12 Hz 决策、360 步预算、3 m 终点线和 0.22 m 碰撞判定（两个仿真器里柱子都没有接触物理，"碰撞"都是到柱心的平面距离判定；hyperlux 有射线碰撞探针，但为了可比性这里用同一个几何判定），只是无人机换成 hyperdrone 的 crazyflie + RAPTOR，画面来自 hyperlux（`--arena-style pybullet|hyperdrone`）。32 条 course 一个 batch 同时飞，100 条杂乱 course + 30 seed 速度扫描约 1–2 分钟。

```bash
python src/hd_closed_loop.py --model src/output/world_model_pybullet_ref.pth --arena-style pybullet
python src/hd_closed_loop.py --model src/output/world_model_mixed_big.pth --arena-style hyperdrone --seeds 100 --sweep-seeds 30
# 室内：训练没用过的 4 栋 ProcTHOR 房子（目录 16–19）里的杂乱环境生存测试
python src/hd_closed_loop.py --scene rooms --houses 4 --house-first 16 --seeds 100 --model src/output/world_model_mixed_big.pth
python src/hd_video.py --scene rooms --house 17 --seed 1017 --policy wm --model src/output/world_model_mixed_big.pth --tag room17_mixed_big
```

`--scene rooms` 不是柱子 course 的搬运：ProcTHOR 房子里几乎找不到"独立在空间中、两侧都能绕、后面还有 3 m 直路"的障碍（四栋房子各 6000 次采样只有 0–7 个，家具都靠墙），而"前方 1.3–2.6 m 有障碍"的起点基本都是带门洞的墙，绕过去是寻路问题而不是避障问题。所以室内测试定义为**生存测试**：随机起点（净空 ≥ 0.9 m、前方 ≥ 2 m 空、随机航向），策略照常"无威胁就前进、有威胁就绕"，直到第一次接触（距离场表面净空 + 0.18 m < 0.22 m）或 7.5 s；报告存活率、首次接触时间、接触前飞行距离、平均净空。两种策略用相同起点。

> 当前的 [../REPORT.md](../REPORT.md)（英文版 [../REPORT_en.md](../REPORT_en.md)，`make_report_three.py --lang both` 同时生成两份及各自的图）只保留一组干净的对比：纯 pybullet / 纯 hyperdrone 室内 / 两者合并三种训练集训出的模型，在 pybullet 柱子避障和 hyperdrone 室内避障两个任务里的表现（`make_report_three.py` 生成）。更早的探索性结果（画风、随机外观、合并大数据集、六模型泛化网格）归档在 `src/output/report/REPORT_archive_2026-10-27.md`。
>
> 报告用到的辅助脚本：`eval_offline.py`（离线泛化网格：碰撞预判 AUC、绕行排序、潜变量 / 解码帧增益，报告 4.3）、`decode_compare.py`（三个模型在同一批 held-out 窗口上把预测潜变量解码成图像，与真实未来帧并排，报告 4.4；`--lang both` 出中英文两张图）、`decode_latents.py`（单个 checkpoint 的 4 个视界解码 + 每个候选命令的想象帧）、`hd_oracle.py`（室内耐久测试在 MPC 动作菜单下的上限：全知策略 32%，允许悬停 89%）、`draw_pipeline.py`（训练流程图：输入 → 网络各层尺寸与参数量 → 损失 → 输出，报告第 2 节；参数量从模块实际计算）。
>
> 下面两节的表是最初 100 条 course 的结果。之后把每个测试放大到 300 条 course / 60 seed 速度扫描 / 20 片密林 / 300 个室内起点（8 栋房子），并增加了"hyperdrone 画风 + 随机外观"（`hd_closed_loop.py --randomize`）这一条件；放大后的表、95% 置信区间、图和更新的结论统一放在 [../REPORT.md](../REPORT.md)，结论没有变化，另外多了一条：随机外观下 pybullet 训练的模型最鲁棒（撞击率 19–20%），hyperdrone 训练的模型 65–80%——hyperdrone 数据集的固定配色让模型过拟合外观，生成时应该开 `--randomize`。

### 结果（2026-10-27，`src/output/hyperdrone_compare_*.md`，100 条 course）

**hyperdrone 场景，pybullet 画风：**

| checkpoint | cluttered 100 courses: crash reactive→wm | min clearance | trigger lead | false pos. (clear) | goal time | single-pillar crash wm (reactive) @0.8 / 1.2 / 1.6 m/s |
|---|---|---|---|---|---|---|
| `pybullet_shipped@hdpb` | 0% → 9% | 0.43 → 0.52 m | +421 ms (n=70) | 0% → 0% | +7% (n=64) | 17% (0%) / 23% (3%) / 20% (20%) |
| `pybullet_ref@hdpb` | 0% → 11% | 0.44 → 0.52 m | +448 ms (n=70) | 0% → 0% | +6% (n=62) | 3% (0%) / 7% (3%) / 23% (17%) |
| `mixed_big@hdpb` | 99% → 91% | 0.12 → 0.14 m | +123 ms (n=51) | 0% → 97% | 77% (93%) / 77% (97%) / 57% (87%) |
| `mixed_big_pbstyle@hdpb` | 0% → 3% | 0.43 → 0.55 m | +337 ms (n=70) | 0% → 0% | +3% (n=68) | 0% (0%) / 20% (0%) / 23% (23%) |
| `pb_plus_big@hdpb` | 27% → 40% | 0.32 → 0.27 m | +549 ms (n=65) | 97% → 100% | +11% (n=33) | 40% (30%) / 43% (37%) / 33% (53%) |
| `pb_plus_big_pbstyle@hdpb` | 0% → 4% | 0.44 → 0.50 m | +337 ms (n=70) | 0% → 0% | -1% (n=67) | 0% (0%) / 27% (3%) / 17% (13%) |

**hyperdrone 场景，hyperdrone 画风：**

| checkpoint | cluttered 100 courses: crash reactive→wm | min clearance | trigger lead | false pos. (clear) | goal time | single-pillar crash wm (reactive) @0.8 / 1.2 / 1.6 m/s |
|---|---|---|---|---|---|---|
| `pybullet_shipped@hdhd` | 43% → 3% | 0.24 → 0.52 m | +750 ms (n=63) | 0% → 0% | +7% (n=38) | 0% (57%) / 17% (97%) / 0% (90%) |
| `pybullet_ref@hdhd` | 100% → 33% | 0.12 → 0.27 m | +937 ms (n=33) | 0% → 0% | 27% (100%) / 27% (100%) / 20% (90%) |
| `mixed_big@hdhd` | 0% → 4% | 0.42 → 0.44 m | +348 ms (n=70) | 0% → 0% | +11% (n=67) | 0% (0%) / 37% (0%) / 43% (20%) |
| `mixed_big_pbstyle@hdhd` | 100% → 100% | 0.10 → 0.10 m | +42 ms (n=2) | 0% → 3% | 100% (100%) / 100% (100%) / 97% (90%) |
| `pb_plus_big@hdhd` | 4% → 0% | 0.40 → 0.47 m | +359 ms (n=67) | 0% → 0% | +10% (n=67) | 3% (0%) / 7% (0%) / 10% (30%) |
| `pb_plus_big_pbstyle@hdhd` | 99% → 57% | 0.11 → 0.21 m | +353 ms (n=13) | 0% → 0% | +0% (n=1) | 60% (100%) / 77% (100%) / 80% (90%) |

把三个测试环境放在一起（100 条杂乱 course 上 wm 策略的撞击率，越低越好）：

| 训练数据 \ 测试环境 | pybullet | hyperdrone（pybullet 画风） | hyperdrone（hyperdrone 画风） |
|---|---|---|---|
| pybullet 64 条（仓库自带权重） | 9% | 9% | **3%** |
| pybullet 64 条（同脚本重训） | 10% | 11% | 33% |
| hyperdrone 256 条，hyperdrone 画风 | 97% | 91% | **4%** |
| hyperdrone 256 条，pybullet 画风 | 54% | **3%** | 100% |
| pybullet 64 + hyperdrone 画风 256 | 31% | 40% | **0%** |
| pybullet 64 + pybullet 画风 256 | 19% | 4% | 57% |

结论：

- **外观决定一切。** 每个模型只在和训练画风一致的环境里工作；动力学差异（pybullet 的 CF2X PID 对 hyperdrone 的 crazyflie RAPTOR）影响很小——pybullet 权重放进 pybullet 画风的 hyperdrone 场景里，撞击率仍是 9–11%，和它在家里一样。
- **迁移是不对称的。** pybullet 权重进 hyperdrone（pybullet 画风）几乎无损；反过来 pybullet 画风的 hyperdrone 数据训出的权重进 pybullet 只有部分迁移（54%）。pybullet 的渲染带有阴影斑块和更杂的纹理，在"更脏"的图上训出来的模型对更干净的画面鲁棒，反之不成立。
- **hyperdrone 数据在自己的环境里是合格的。** 在 hyperdrone 画风的场景里，hyperdrone 训练的模型撞击率 4%，pybullet 数据 + hyperdrone 数据合并训练到 0%，都优于 pybullet 模型在 pybullet 里的 9–10%。按"同环境闭环"这个和原课程一样的标准，这些数据集能用。
- 仓库自带的 `world_model.pth` 比我用同一脚本重训的 `world_model_pybullet_ref.pth` 鲁棒得多（hyperdrone 画风场景里 3% 对 33%），说明上游训练这份权重时的设置（可能是 `--robust` 增强或更多数据）和脚本默认值不同，拿它当"pybullet 基线"时要注意。
- 对真实 Tello 的含义不变：两个仿真器都是域外，该做的是在真机 shadow 推理里比较，而不是在任一仿真器里。

## 性能注意

- 本机 hyperlux 走的是 WebGPU + llvmpipe（CPU 软件渲染），64×64 仍有约 1000–2000 fps；`LP_NUM_THREADS` 可限制它的线程数。**OptiX 在这台 WSL2 机器上不可用**：装了 CUDA 12.8 工具链和 cuBLAS/cuRAND 开发包后 OptiX 后端能编译通过，但 `optixInit` 报 7805（ENTRY_SYMBOL_NOT_FOUND）——WSL2 驱动只提供一个 14 KB 的 dxcore 壳（`libnvoptix.so.1`）和不透明的 `nvoptix.bin`，没有任何可 dlopen 的库导出 `optixQueryFunctionTable`；NVIDIA 也没有 WSL2 的 Vulkan ICD。GPU 光追需要原生 Windows 或非 WSL 的 Linux。因为环境里现在有 nvcc，hyperlux 的自动选择会挑 OptiX，所以 `dreamer` 环境固化了 `HYPERLUX_BACKEND=WEBGPU`（`conda env config vars`）。CUDA 版动力学（`--device cuda`）在 rl-tools 的 RAPTOR CUDA 模板处编译失败，保持 `--device cpu`（动力学只占几秒）。
- hyperlux 每次构造原生组件都重跑一遍 CMake 陈旧性检查（本机每个组件约 1 分钟）。生成器自动设置 `HYPERLUX_SKIP_BUILD` 走缓存快路径，只有缓存缺失时才真正编译。
- `hyperlux.Mesh`、`hyperlux.make_camera` 这类模块属性每次访问都会重新探测后端（约 50 ms）。生成器只解析一次；自己写场景代码时也要先 `Mesh = hyperlux.Mesh` 再循环。

## 下一步可以做的

- 用 `hyperlux.load_scene` 加载 ProcTHOR/HSSD 的 GLB 作室内背景（conta 缓存里已有 `ProcTHOR-Train-1`），把柱子放进真实房间；需要用 `ClearanceSampler` 选起点并处理墙体距离标签。
- 没有 Tello 的动力学模型；`crazyflie` 是最接近的小机型。若 SakanaDreamer 那边完成了 RC/速度标定，可以把实测的加速度、速度上限反过来约束这里的指令集。
- `--save-depth` 的深度可以作为 world model 的辅助监督。
- `--catalog hssd` 可以换成 HSSD 的房子（单栋 50–90 MB，更大更复杂），目录接口相同。
- 闭环评估脚本（`wm_closed_loop.py`、`speed_sweep.py` 等）仍依赖 pybullet 的 `make_env`；要在房间里做闭环，需要在 hyperdrone 上实现同样的 rollout 接口。
