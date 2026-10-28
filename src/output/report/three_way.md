**碰撞预判 AUC@32（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| pybullet 64 (retrained) | 0.95\* | 0.85 | 0.51 | 0.46 |
| hyperdrone rooms 256 | 0.44 | 0.57 | 0.76\* | 0.69 |
| pybullet 64 + hyperdrone rooms 256 | 0.96\* | 0.81 | 0.77\* | 0.69 |

**碰撞预判 AUC@8**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| pybullet 64 (retrained) | 0.95\* | 0.82 | 0.48 | 0.43 |
| hyperdrone rooms 256 | 0.45 | 0.58 | 0.71\* | 0.66 |
| pybullet 64 + hyperdrone rooms 256 | 0.96\* | 0.77 | 0.74\* | 0.68 |

**danger-now AUC**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| pybullet 64 (retrained) | 0.87\* | 0.54 | 0.41 | 0.41 |
| hyperdrone rooms 256 | 0.45 | 0.52 | 0.72\* | 0.69 |
| pybullet 64 + hyperdrone rooms 256 | 0.90\* | 0.67 | 0.71\* | 0.74 |

**安全绕行方向排序准确率（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| pybullet 64 (retrained) | 0.93\* | 0.89 | 0.45 | 0.21 |
| hyperdrone rooms 256 | 0.57 | 0.35 | 0.62\* | 0.18 |
| pybullet 64 + hyperdrone rooms 256 | 0.85\* | 0.50 | 0.50\* | 0.48 |

**潜变量预测增益 @32（0 = 不比「未来 = 现在」好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| pybullet 64 (retrained) | +0.80\* | +0.43 | -1.44 | -1.10 |
| hyperdrone rooms 256 | -0.01 | -0.06 | +0.07\* | -0.00 |
| pybullet 64 + hyperdrone rooms 256 | +0.66\* | +0.11 | -0.01\* | +0.01 |

**解码未来帧增益 @32（0 = 不比复制当前帧好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|
| pybullet 64 (retrained) | +0.64\* | +0.33 | -0.58 | -0.54 |
| hyperdrone rooms 256 | +0.28 | +0.27 | +0.12\* | -0.06 |
| pybullet 64 + hyperdrone rooms 256 | +0.57\* | +0.37 | +0.10\* | +0.04 |

\* = 该数据域参与了这个 checkpoint 的训练（in-sample）；其余单元格是训练时没见过的数据。
