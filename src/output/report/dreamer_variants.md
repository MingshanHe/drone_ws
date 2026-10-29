**碰撞预判 AUC@32（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, rooms 256, 6k steps | 0.64 | 0.83 | 0.91\* | 0.89 |
| nano-Dreamer, pybullet 64 + rooms 256, 6k steps | 0.99\* | 0.83 | 0.83\* | 0.85 |

**碰撞预判 AUC@8**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, rooms 256, 6k steps | 0.76 | 0.88 | 0.93\* | 0.89 |
| nano-Dreamer, pybullet 64 + rooms 256, 6k steps | 1.00\* | 0.81 | 0.91\* | 0.88 |

**danger-now AUC**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, rooms 256, 6k steps | 0.65 | 0.84 | 0.89\* | 0.89 |
| nano-Dreamer, pybullet 64 + rooms 256, 6k steps | 1.00\* | 0.83 | 0.85\* | 0.90 |

**安全绕行方向排序准确率（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, rooms 256, 6k steps | 0.51 | 0.63 | 0.38\* | 0.24 |
| nano-Dreamer, pybullet 64 + rooms 256, 6k steps | 0.73\* | 0.74 | 0.43\* | 0.30 |

**潜变量预测增益 @32（0 = 不比「未来 = 现在」好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, rooms 256, 6k steps | +0.07 | +0.23 | +0.25\* | +0.26 |
| nano-Dreamer, pybullet 64 + rooms 256, 6k steps | +0.58\* | +0.26 | +0.24\* | +0.25 |

**解码未来帧增益 @32（0 = 不比复制当前帧好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, rooms 256, 6k steps | -4.05 | -0.52 | +0.20\* | -0.07 |
| nano-Dreamer, pybullet 64 + rooms 256, 6k steps | +0.74\* | -0.01 | +0.20\* | +0.15 |

\* = 该数据域参与了这个 checkpoint 的训练（in-sample）；其余单元格是训练时没见过的数据。
