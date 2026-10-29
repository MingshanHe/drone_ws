**碰撞预判 AUC@32（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, pybullet 64 | 1.00\* | 0.73 | 0.64 | 0.75 |
| nano-Dreamer, hyperdrone rooms 256 | 0.59 | 0.76 | 0.84\* | 0.90 |
| nano-Dreamer, pybullet 64 + rooms 256 | 1.00\* | 0.71 | 0.85\* | 0.85 |

**碰撞预判 AUC@8**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, pybullet 64 | 1.00\* | 0.78 | 0.68 | 0.73 |
| nano-Dreamer, hyperdrone rooms 256 | 0.74 | 0.80 | 0.89\* | 0.90 |
| nano-Dreamer, pybullet 64 + rooms 256 | 1.00\* | 0.75 | 0.88\* | 0.86 |

**danger-now AUC**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, pybullet 64 | 0.99\* | 0.71 | 0.62 | 0.70 |
| nano-Dreamer, hyperdrone rooms 256 | 0.63 | 0.75 | 0.86\* | 0.89 |
| nano-Dreamer, pybullet 64 + rooms 256 | 1.00\* | 0.73 | 0.86\* | 0.89 |

**安全绕行方向排序准确率（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, pybullet 64 | 0.92\* | 0.47 | 0.20 | 0.52 |
| nano-Dreamer, hyperdrone rooms 256 | 0.37 | 0.58 | 0.43\* | 0.79 |
| nano-Dreamer, pybullet 64 + rooms 256 | 0.80\* | 0.67 | 0.18\* | 0.67 |

**潜变量预测增益 @32（0 = 不比「未来 = 现在」好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, pybullet 64 | +0.78\* | +0.11 | -0.16 | -0.23 |
| nano-Dreamer, hyperdrone rooms 256 | +0.12 | +0.27 | +0.30\* | +0.27 |
| nano-Dreamer, pybullet 64 + rooms 256 | +0.68\* | +0.30 | +0.29\* | +0.31 |

**解码未来帧增益 @32（0 = 不比复制当前帧好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| nano-Dreamer, pybullet 64 | +0.91\* | -0.74 | -2.64 | -1.44 |
| nano-Dreamer, hyperdrone rooms 256 | -2.55 | -0.24 | +0.16\* | +0.10 |
| nano-Dreamer, pybullet 64 + rooms 256 | +0.85\* | -0.13 | +0.07\* | +0.25 |

\* = 该数据域参与了这个 checkpoint 的训练（in-sample）；其余单元格是训练时没见过的数据。
