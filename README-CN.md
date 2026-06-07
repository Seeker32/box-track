# Box-Track

**多摄像头纸箱跟踪系统** — 在非重叠视野的多个摄像头之间，持续跟踪外观相似的纸箱目标。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> **[English](README.md)** — Read this project's English documentation.

---

## 概述

Box-Track 解决了**纸箱在摄像头之间移动时保持同一 ID** 的问题，适用于安防监控和物流场景。其核心挑战在于：

- **纸箱外观极为相似**（棕色瓦楞表面，纹理差异极小）
- **摄像头视野不重叠** —— 无法利用时空连续性
- **光照、角度、背景** 在不同摄像头之间差异巨大

核心思路：**YOLO 的 backbone（neck）特征** —— 从检测模型中间特征图中提取的区域特征 —— 可以作为强大的外观描述子，捕捉到印刷文字、封箱胶带、图案等传统 ReID 难以区分的细微差异。而且检测时本身就运行了 YOLO，这些特征**零额外成本**（一鱼两吃）。

### 亮点

- **YOLO backbone 特征匹配** —— 在检测模型的 neck 层注册 forward hook，对每个检测框提取特征向量；无需独立的 ReID 模型
- **BoT-SORT 单摄像头跟踪** —— 基于 Ultralytics + BoT-SORT 的逐摄像头 tracklet 生成
- **匈牙利算法** —— 跨摄像头 tracklet 匹配的最优分配方案
- **时空约束门控** —— 时间窗口约束过滤不合理匹配
- **在线 + 批处理双模式** —— 支持实时 RTSP 流和离线视频文件处理
- **一致颜色可视化** —— 同一全局 ID 在不同摄像头画面中显示相同颜色
- **轨迹持久化** —— 支持 JSON Lines、CSV 和 JSON 摘要导出

---

## 系统架构

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Camera 0    │   │  Camera 1    │   │  Camera N    │
│  (RTSP/文件)  │   │  (RTSP/文件)  │   │  (RTSP/文件)  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  YOLO 检测   │   │  YOLO 检测   │   │  YOLO 检测   │
│  + BoT-SORT  │   │  + BoT-SORT  │   │  + BoT-SORT  │
│  (逐摄像头)  │   │  (逐摄像头)  │   │  (逐摄像头)  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │  Tracklets       │  Tracklets       │  Tracklets
       ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────┐
│                  Backbone 特征提取                       │
│      (YOLO neck 层 → ROI 池化 → 逐框特征向量)          │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│                 跨摄像头匹配引擎                          │
│  · 余弦相似度（backbone 特征）                          │
│  · 时空约束（时间窗口过滤）                             │
│  · 匈牙利算法（全局最优分配）                           │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│                全局 ID 管理器                            │
│  · 新目标注册                                           │
│  · 跨摄像头 ID 传播                                     │
│  · 过期目标清理                                         │
└──────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
box-track/
├── main.py                          # CLI 入口
├── pyproject.toml                   # 项目元数据与依赖
├── configs/
│   ├── pipeline.yaml                # 流水线配置
│   └── botsort.yaml                 # BoT-SORT 跟踪器配置
├── src/
│   ├── detection/
│   │   └── yolo_detector.py         # YOLO 检测器封装
│   ├── tracking/
│   │   └── botsort_tracker.py       # BoT-SORT 单摄像头跟踪
│   ├── features/
│   │   └── backbone_feature.py      # YOLO neck hook + ROI 池化
│   ├── matching/
│   │   ├── similarity.py            # 余弦相似度 & 时空评分
│   │   └── hungarian_matcher.py     # 匈牙利最优分配
│   ├── association/
│   │   └── global_id_manager.py     # 跨摄像头 ID 注册表
│   ├── pipeline/
│   │   ├── cross_camera_pipeline.py # 批处理评估流水线
│   │   └── online_pipeline.py       # 实时流处理流水线
│   ├── io/
│   │   └── __init__.py              # RTSP/文件流读取器（多线程）
│   └── utils/
│       ├── tracklet.py              # BBox & Tracklet 数据结构
│       ├── visualization.py         # 绘制、调色板、VideoWriter
│       └── persistence.py           # JSON Lines / CSV / JSON 导出
├── models/
│   ├── best.pt                      # 微调后的 YOLO 检测模型
│   └── yolo26n.pt                   # YOLO26 nano 预训练权重
├── docs/
│   └── design.md                    # 完整设计文档
├── input/                           # 测试图片（已 gitignore）
└── output/                          # 标注图 & 跟踪导出（已 gitignore）
```

---

## 快速开始

### 安装

```bash
# 需要 Python 3.12+
pip install ultralytics pyyaml scipy opencv-python numpy
```

或使用 uv：

```bash
uv sync
```

### 准备模型

将微调后的 YOLO 模型放在 `models/best.pt`。仓库中也包含 YOLO26n 预训练权重供参考。

### 单图推理

```bash
python main.py --image input/test.jpg
```

---

## 使用模式

### Phase 1：特征验证

测试 YOLO backbone 特征能否区分不同的纸箱。

**可分离性测试** —— 测量单张图片中所有检测框之间的成对余弦相似度：

```bash
python main.py --phase1 --mode separability --image input/IMG_20260605_225534.jpg
```

**合成跨摄像头测试** —— 通过高斯噪声模拟两个摄像头，测试匈牙利匹配能否正确配对每个框：

```bash
python main.py --phase1 --mode synthetic --image input/IMG_20260605_225534.jpg
```

**完整视频评估** —— 处理多摄像头视频文件并计算 Top-1 匹配准确率：

```bash
python main.py --phase1 --mode video --videos cam0.mp4 cam1.mp4
```

### Phase 2：完整跟踪系统

**视频文件模式**（批处理）：

```bash
python main.py --phase2 --mode video --videos cam0.mp4 cam1.mp4
```

**RTSP 实时模式**（直播流）：

```bash
python main.py --phase2 --mode rtsp --streams rtsp://192.168.1.10/stream rtsp://192.168.1.11/stream
```

**附加参数：**

| 参数 | 说明 |
|---|---|
| `--viz` / `--no-viz` | 启用/禁用标注视频输出 |
| `--display` | 显示实时预览窗口（需要 GUI） |
| `--no-persist` | 禁用轨迹日志记录 |
| `--output-dir PATH` | 指定输出目录 |
| `--config PATH` | 自定义流水线 YAML 配置 |
| `--model PATH` | 自定义 YOLO 模型路径 |

---

## 配置说明

### 流水线配置（`configs/pipeline.yaml`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_path` | `models/best.pt` | YOLO 模型权重 |
| `conf` | `0.25` | 检测置信度阈值 |
| `feature.hook_layer` | `-2` | 特征 hook 的 neck 层索引 |
| `feature.normalize` | `true` | L2 归一化特征向量 |
| `feature.aggregation` | `mean` | Tracklet 特征聚合方式：`mean`、`median` 或 `max_conf` |
| `matching.similarity_threshold` | `0.5` | 有效匹配的最低相似度 |
| `matching.min_transit` | `1.0s` | 摄像头间最短转移时间 |
| `matching.max_transit` | `60.0s` | 摄像头间最长转移时间 |
| `global_id.disappearance_timeout` | `60.0s` | 目标消失后保留时长 |
| `online.visualization` | `true` | 输出标注视频 |
| `online.persistence` | `true` | 持久化轨迹到磁盘 |
| `online.match_interval` | `0.5s` | 跨摄像头匹配间隔（在线模式） |

### BoT-SORT 配置（`configs/botsort.yaml`）

标准的 Ultralytics BoT-SORT 参数。`backbone_feature.py` 中的特征提取 hook 取代了内置的 ReID 模块（`with_reid: False`）。

---

## 工作原理

### 1. 单摄像头跟踪

每个摄像头独立运行 YOLO 检测 + BoT-SORT 跟踪。`BOTSORTTracker` 逐帧处理视频，维护一组活跃的 `Tracklet` 对象。

### 2. Backbone 特征提取

在检测模型的 neck 层（检测头之前的最后一个 C2f 模块，索引 `-2`）注册 forward hook。每次前向推理后捕获特征图。对于每个检测框，在特征图上截取对应区域并通过全局平均池化（ROI pooling）生成紧凑的特征向量。

由于 hook 注册在**同一个检测模型**上，无需额外的推理步骤 —— 特征来自检测的那一次前向计算。

### 3. Tracklet 特征聚合

每个 tracklet 在其生命周期内累积逐帧特征。当目标离开摄像头视野时，tracklet 结束，逐帧特征聚合成一个向量：

- **mean**（均值）—— 所有帧特征取平均
- **median**（中位数）—— 跨帧取中位数（对异常值更鲁棒）
- **max_conf**（最高置信度）—— 取检测置信度最高的那一帧的特征

### 4. 跨摄像头匹配

当后序摄像头有 tracklet 完成时，与已有活跃全局目标进行匹配：

1. **时空约束门控** —— 过滤时间间隔不合理或同摄像头的配对
2. **Backbone 特征相似度** —— 聚合后 tracklet 特征向量的余弦相似度
3. **匈牙利算法** —— 寻找最大化总相似度的最优一一分配

### 5. 全局 ID 管理

- 首次在任何摄像头出现 → 分配新全局 ID
- 匹配到已有目标 → 继承该目标的全局 ID
- 超过消失超时时间 → 从活跃列表中移除
- ID 一旦分配不做修订（简单可靠）

---

## 输出文件

Phase 2 启用持久化后，`output/trajectories/` 目录下生成以下文件：

| 格式 | 文件 | 说明 |
|---|---|---|
| JSON Lines | `trajectories_<时间戳>.jsonl` | 每行一个 JSON 对象，记录一个完成的 tracklet |
| CSV | `detections_<时间戳>.csv` | 所有检测的扁平表格，每行一个检测框 |
| JSON 摘要 | `summary_<时间戳>.json` | 按 global_id 分组的 tracklet 摘要 |

启用可视化后，标注视频保存在 `output/videos/` 目录。

---

## 设计思路

### 为什么用 YOLO backbone 特征而不是独立的 ReID 模型？

- **零额外推理成本** —— 特征图在检测时已计算完毕
- **语义特征** —— neck 层编码了高层模式（纹理、印刷、封箱胶带、结构边缘），跨摄像头泛化性好
- **一鱼两吃** —— 简化部署，减少内存占用
- 在此场景下，backbone 特征优于基于行人/车辆数据集训练的通用 ReID 模型

### 为什么用匈牙利算法而不是贪心匹配？

匈牙利算法避免出现早期贪心匹配迫使后续分配错误的问题。当多个纸箱外观得分接近时，全局最优分配尤为关键。

### 为什么"只分配不修正"？

跨摄像头匹配天然具有歧义性，在线修正（回溯修改之前的匹配）会引入不稳定性。一次性分配虽然放弃了理论最优性，但换来了可预测性和简单性。

---

## 许可证

Apache 2.0 —— 详见 [LICENSE](LICENSE)。

---

## 设计文档

详细的设计文档（中文）涵盖架构决策、特征提取策略和实现路线图，请参见 [`docs/design.md`](docs/design.md)。
