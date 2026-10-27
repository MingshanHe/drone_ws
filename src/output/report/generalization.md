**碰撞预判 AUC@32（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, hd look | hyperdrone pillars, pb look | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|---|---|
| pybullet 64 (shipped) | 0.95\* | 0.93 | 0.94 | 0.86 | 0.57 | 0.51 |
| pybullet 64 (retrained) | 0.95\* | 0.90 | 0.94 | 0.85 | 0.51 | 0.46 |
| hyperdrone 256, hd look | 0.48 | 0.98\* | 0.71 | 0.51 | 0.94\* | 0.65 |
| hyperdrone 256, pb look | 0.72 | 0.44 | 0.98\* | 0.53 | 0.94\* | 0.66 |
| pybullet + hyperdrone hd look | 0.96\* | 0.99\* | 0.96 | 0.75 | 0.94\* | 0.68 |
| pybullet + hyperdrone pb look | 0.96\* | 0.73 | 0.99\* | 0.62 | 0.96\* | 0.70 |

**碰撞预判 AUC@8**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, hd look | hyperdrone pillars, pb look | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|---|---|
| pybullet 64 (shipped) | 0.95\* | 0.91 | 0.95 | 0.86 | 0.56 | 0.46 |
| pybullet 64 (retrained) | 0.95\* | 0.85 | 0.95 | 0.82 | 0.48 | 0.43 |
| hyperdrone 256, hd look | 0.37 | 0.99\* | 0.60 | 0.48 | 0.95\* | 0.62 |
| hyperdrone 256, pb look | 0.70 | 0.41 | 0.99\* | 0.46 | 0.91\* | 0.61 |
| pybullet + hyperdrone hd look | 0.96\* | 0.99\* | 0.95 | 0.61 | 0.96\* | 0.64 |
| pybullet + hyperdrone pb look | 0.97\* | 0.70 | 0.99\* | 0.46 | 0.96\* | 0.69 |

**danger-now AUC**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, hd look | hyperdrone pillars, pb look | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|---|---|
| pybullet 64 (shipped) | 0.88\* | 0.73 | 0.89 | 0.63 | 0.43 | 0.40 |
| pybullet 64 (retrained) | 0.87\* | 0.55 | 0.88 | 0.54 | 0.41 | 0.41 |
| hyperdrone 256, hd look | 0.42 | 0.95\* | 0.53 | 0.50 | 0.94\* | 0.69 |
| hyperdrone 256, pb look | 0.57 | 0.39 | 0.94\* | 0.43 | 0.94\* | 0.69 |
| pybullet + hyperdrone hd look | 0.90\* | 0.96\* | 0.83 | 0.55 | 0.96\* | 0.70 |
| pybullet + hyperdrone pb look | 0.91\* | 0.47 | 0.95\* | 0.44 | 0.95\* | 0.70 |

**安全绕行方向排序准确率（0.5 = 随机）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, hd look | hyperdrone pillars, pb look | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|---|---|
| pybullet 64 (shipped) | 0.89\* | 0.96 | 0.95 | 0.97 | 0.40 | 0.21 |
| pybullet 64 (retrained) | 0.93\* | 0.94 | 0.93 | 0.89 | 0.45 | 0.21 |
| hyperdrone 256, hd look | 0.83 | 0.96\* | 0.93 | 0.72 | 0.72\* | 0.27 |
| hyperdrone 256, pb look | 0.87 | 0.76 | 0.93\* | 0.85 | 0.85\* | 0.39 |
| pybullet + hyperdrone hd look | 0.93\* | 0.94\* | 0.92 | 0.86 | 0.89\* | 0.58 |
| pybullet + hyperdrone pb look | 0.84\* | 0.34 | 0.88\* | 0.41 | 0.89\* | 0.91 |

**潜变量预测增益 @32（0 = 不比「未来 = 现在」好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, hd look | hyperdrone pillars, pb look | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|---|---|
| pybullet 64 (shipped) | +0.78\* | -0.06 | +0.55 | +0.46 | -1.49 | -1.32 |
| pybullet 64 (retrained) | +0.80\* | -0.25 | +0.53 | +0.43 | -1.44 | -1.10 |
| hyperdrone 256, hd look | +0.18 | +0.83\* | +0.28 | +0.28 | +0.51\* | -0.09 |
| hyperdrone 256, pb look | +0.07 | -0.06 | +0.86\* | +0.06 | +0.48\* | -0.15 |
| pybullet + hyperdrone hd look | +0.73\* | +0.83\* | +0.68 | +0.41 | +0.65\* | -0.22 |
| pybullet + hyperdrone pb look | +0.78\* | +0.42 | +0.86\* | +0.25 | +0.46\* | -0.24 |

**解码未来帧增益 @32（0 = 不比复制当前帧好）**

| 训练数据 \ 数据域 | pybullet pillars | hyperdrone pillars, hd look | hyperdrone pillars, pb look | hyperdrone pillars, random appearance (unseen) | hyperdrone rooms, houses 0-15 | hyperdrone rooms, houses 16-23 (unseen) |
|---|---|---|---|---|---|---|
| pybullet 64 (shipped) | +0.63\* | +0.49 | +0.44 | +0.31 | -1.27 | -1.05 |
| pybullet 64 (retrained) | +0.64\* | +0.38 | +0.48 | +0.33 | -0.59 | -0.55 |
| hyperdrone 256, hd look | +0.38 | +0.52\* | +0.16 | +0.39 | +0.33\* | -0.07 |
| hyperdrone 256, pb look | +0.32 | +0.33 | +0.63\* | +0.36 | +0.39\* | -0.23 |
| pybullet + hyperdrone hd look | +0.63\* | +0.49\* | +0.47 | +0.39 | +0.41\* | -0.57 |
| pybullet + hyperdrone pb look | +0.64\* | +0.34 | +0.55\* | +0.30 | +0.38\* | -0.04 |

\* = 该数据域参与了这个 checkpoint 的训练（in-sample）；其余单元格是训练时没见过的数据。
