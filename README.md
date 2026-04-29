# 车辆图像检索系统 — 基于 BoF / VLAD / TF-IDF 特征编码

一个完整的车辆图像检索与比对系统，支持 SIFT / ORB 特征提取、BoF / VLAD / TF-IDF 特征编码、KNN 检索，以及完整的 GUI 交互界面和可视化评估。

---

## 项目概述

本项目基于**视觉词袋模型（Bag of Visual Words）**构建车辆图像检索引擎：
- 从监控图像中提取 SIFT / ORB 局部特征描述子
- 通过 K-Means 聚类构建视觉字典（码本）
- 支持 **BoF**、**VLAD**、**BoF+TF-IDF** 三种特征编码策略
- 使用 KNN（K=10）完成相似度检索
- 集成完整的评估体系（mAP、Precision、Recall、PR 曲线）
- 提供 Tkinter GUI 交互界面

---

## 数据集

| 数据集 | 车牌数 | 图像数 | 说明 |
|---|---|---|---|
| `image/`（训练集） | 50 | 946 | 每辆车一个文件夹，文件夹名为车牌号 |
| `test/`（测试集） | 14 | 42 | 用于选择待匹配的查询图像 |

---

## 项目结构

```
LabCB/
├── image/                          # 训练集
│   ├── A03Z78/                     # 车牌号文件夹
│   ├── A0C573/
│   └── ...
├── test/                           # 测试集
│   ├── A0C573/
│   └── ...
├── matcher.py                      # 原始 BFMatcher 特征匹配
├── image_retrieval.py              # BoF / VLAD / TF-IDF 编码 + KNN 检索 + 评估
├── gui.py                          # Tkinter 交互界面（主程序入口）
├── codebook_SIFT_256.pkl           # SIFT 视觉字典
├── codebook_ORB_256.pkl            # ORB 视觉字典
├── idf_SIFT_256.pkl                # SIFT 的 IDF 权重
├── idf_ORB_256.pkl                 # ORB 的 IDF 权重
├── train_encode_*_*.pkl            # 训练集编码缓存
└── test_encode_*_*.pkl             # 测试集编码缓存
```

---

## 功能特性

### 1. 特征提取与匹配
- **SIFT**：尺度不变特征变换，128 维描述子
- **ORB**：快速方向旋转不变特征，32 维描述子
- **原始匹配**：逐张 BFMatcher + Lowe's ratio test

### 2. 特征编码
| 编码方式 | 维度 | 核心思想 |
|---|---|---|
| **BoF** | 256 | 视觉单词频次直方图 |
| **BoF+TF-IDF** | 256 | 归一化频次 × IDF 权重 |
| **VLAD** | 32768 (SIFT) / 8192 (ORB) | 描述子到聚类中心的残差累积 |

### 3. GUI 界面功能
- ✅ 选择待匹配图像
- ✅ 切换特征算法（SIFT / ORB）
- ✅ 切换编码方式（原始匹配 / BoF / VLAD / BoF+TF-IDF）
- ✅ 显示最佳匹配结果（Top-1）
- ✅ 显示 Top-10 检索结果缩略图
- ✅ 显示检索时间、Precision、Recall、mAP
- ✅ **绘制 PR 曲线**（单查询）
- ✅ **绘制平均 PR 曲线**（BoF vs VLAD 对比）
- ✅ **IDF 对比**（PR 曲线 + mAP 柱状图）
- ✅ **编码直方图可视化**（查询图像 + Top-10 近邻）
- ✅ 构建码本（实时生成视觉字典）

### 4. 评估指标
- **mAP@10**：平均精度均值
- **Precision@K**：Top-K 中正确结果的比例
- **Recall@K**：Top-K 中找出的相关结果占全部相关结果的比例
- **AP**：单张查询的平均精度
- **检索时间**：单次查询耗时（ms）

---

## 环境依赖

```bash
pip install opencv-python numpy scikit-learn matplotlib pillow
```

- Python 3.x
- OpenCV 4.x（含 SIFT、ORB）
- NumPy
- scikit-learn（K-Means、KNN）
- Matplotlib（PR 曲线、直方图）
- Pillow（GUI 图像显示）

---

## 运行方式

### 启动 GUI（推荐）

```bash
python gui.py
```

### 运行完整对比实验（命令行）

```bash
python image_retrieval.py
```

### 运行单次实验

```bash
python -c "from image_retrieval import run_pipeline; run_pipeline(method='SIFT', n_words=256, encoding='vlad', k=10)"
```

---

## 实验结果

### 对比实验汇总

| Method | Encoding | **mAP@10** | **Prec@10** | **Recall@10** | **AvgTime(ms)** |
|--------|----------|-----------|------------|--------------|----------------|
| SIFT   | BoF      | 0.8905    | 0.4833     | 0.2380       | 36.92          |
| SIFT   | TF-IDF   | 0.8581    | 0.5119     | 0.2553       | 5.30           |
| SIFT   | VLAD     | **0.9634**| **0.8333** | **0.4222**   | 159.65         |
| ORB    | BoF      | 0.8783    | 0.5738     | 0.2928       | 5.38           |
| ORB    | TF-IDF   | 0.8569    | 0.4595     | 0.2298       | 5.24           |
| ORB    | VLAD     | **0.9370**| **0.7524** | **0.3806**   | 58.58          |

### 关键结论

1. **VLAD 编码全面优于 BoF 和 TF-IDF**：mAP 提升约 7~10%
2. **TF-IDF 在该场景下效果有限**：车辆图像的共享结构特征过多，IDF 惩罚反而削弱了有效信息
3. **ORB 速度极快但精度略低**：ORB+BoF 单次检索仅 5ms，适合大规模快速检索
4. **SIFT+VLAD 精度最高**：mAP 达到 96.3%，但编码维度高（32768 维），检索较慢

---

## 核心算法流程

```
1. 特征提取
   └─ SIFT / ORB → 关键点 + 描述子 (N, 128) 或 (N, 32)

2. 码本构建
   └─ K-Means (MiniBatchKMeans, k=256) → 视觉字典

3. 特征编码
   ├─ BoF: 直方图量化 → L2 归一化
   ├─ TF-IDF: 归一化频次 × log(N/df)
   └─ VLAD: 残差累积 → 幂归一化 → L2 归一化

4. KNN 检索
   └─ NearestNeighbors (k=10, 欧氏距离) → Top-10 结果

5. 评估
   └─ 11 点插值法 → 平均 PR 曲线 + mAP
```

---

## 算法原理详解

### BoF（Bag of Features）

将图像的描述子通过视觉字典量化为频次直方图：

```
h_j = count(word_j in image)
h = h / ||h||₂   (L2 归一化)
```

### TF-IDF

在 BoF 基础上引入 IDF 权重：

```
h_j = (h_j / Σᵢ hᵢ) × log(N / f_j)
```

- `h_j / Σᵢ hᵢ`：归一化频次（TF）
- `log(N / f_j)`：逆文档频率（IDF）

### VLAD（Vector of Locally Aggregated Descriptors）

记录每个视觉单词的残差累积：

```
V_j = Σ_{xᵢ ∈ Ω_j} (xᵢ - c_j)
```

- `xᵢ`：描述子
- `c_j`：第 j 个聚类中心
- `Ω_j`：分配到第 j 类的描述子集合

VLAD 保留了描述子相对于聚类中心的一级统计信息（均值偏差），区分性远强于 BoF 的零级统计（频次）。

---

## 作者

- 课程：计算机视觉 / 视觉计算
- 技术栈：Python + OpenCV + scikit-learn + Matplotlib
