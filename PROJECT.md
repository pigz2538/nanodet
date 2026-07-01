# NanoDet-Plus 项目说明

## 项目概述

NanoDet-Plus 是一个面向移动端与边缘设备的轻量级anchor-free目标检测框架，基于 PyTorch 与 PyTorch Lightning 构建。本项目在 NanoDet 的基础上引入了 **Assign Guidance Module (AGM)**、**Dynamic Soft Label Assigner (DSLA)** 以及 **Ghost-PAN** 结构，在几乎不增加推理耗时的前提下显著提升了轻量模型的检测精度。

该仓库采用模块化设计，将模型结构、数据加载、训练流程、评估逻辑与部署工具解耦，便于用户针对自定义数据集调整 backbone、FPN、检测头等组件，并支持训练完成后直接导出为 ONNX 模型。

---

## 整体架构

NanoDet-Plus 属于单阶段无锚框（single-stage anchor-free）检测器，整体网络结构分为三个主要部分：

1. **Backbone（骨干网络）**：用于从输入图像中提取多尺度特征，典型选择为 ShuffleNetV2、MobileNetV2、GhostNet、EfficientNet-Lite、RepVGG 等轻量网络。
2. **FPN / PAN（特征融合模块）**：将 backbone 输出的多尺度特征进行融合，典型实现为 GhostPAN，并支持额外的辅助分支（auxiliary FPN）。
3. **Head（检测头）**：在融合后的特征图上预测分类（classification）与分布回归（distribution regression），典型实现为 NanoDetPlusHead，采用 Generalized Focal Loss 系列损失。

在训练阶段，NanoDet-Plus 额外构建了一个 **auxiliary head**，通过 dual FPN 输出辅助监督信号；在推理阶段仅使用主分支，因此不增加推理开销。

---

## 目录结构详解

### `nanodet/` — 核心 Python 包

#### `nanodet/model/` — 模型定义

- `arch/`
  - `one_stage_detector.py`：所有单阶段检测器的基类，定义了 `forward`、`forward_train` 与 `forward_test` 的基本流程。
  - `nanodet_plus.py`：NanoDet-Plus 的完整架构实现，包含主分支与 auxiliary 分支的交互逻辑。
- `backbone/`
  - 提供多种轻量骨干网络：`shufflenetv2.py`、`mobilenetv2.py`、`ghostnet.py`、`efficientnet_lite.py`、`repvgg.py`、`resnet.py`、`custom_csp.py`。
  - `timm_wrapper.py`：对 `timm` 库中 backbone 的封装，便于引入更多预训练模型。
- `fpn/`
  - `ghost_pan.py`：GhostPAN 实现，NanoDet-Plus 默认使用的特征融合模块。
  - `fpn.py`、`pan.py`、`tan.py`：其他特征金字塔实现。
- `head/`
  - `nanodet_plus_head.py`：NanoDet-Plus 主检测头。
  - `nanodet_head.py`、`simple_conv_head.py`、`gfl_head.py`：其他检测头实现。
  - `assigner/`：标签分配策略，包括 ATSSAssigner、DynamicSoftLabelAssigner 等。
- `loss/`
  - `gfocal_loss.py`：Quality Focal Loss 与 Distribution Focal Loss。
  - `iou_loss.py`：GIoU / DIoU / CIoU 等边界框回归损失。
- `module/`
  - 可复用的基础模块：卷积（`conv.py`）、归一化（`norm.py`）、激活（`activation.py`）、NMS（`nms.py`）、初始化（`init_weights.py`）等。
- `weight_averager/`
  - `ema.py`：指数移动平均（EMA）实现，用于稳定训练末期权重。

#### `nanodet/data/` — 数据管线

- `dataset/`
  - `base.py`：所有数据集的抽象基类。
  - `coco.py`：COCO 格式数据集。
  - `xml_dataset.py`：VOC/XML 格式数据集。
  - `yolo.py`：YOLO 格式数据集。
- `transform/`
  - `pipeline.py`：数据增强流程组合。
  - `warp.py`：几何变换（缩放、旋转、仿射、Mosaic 等）。
  - `color.py`：颜色空间变换（亮度、对比度、饱和度、色相）。
  - `mosaic.py`：Mosaic 数据增强实现。
- `collate.py`：DataLoader 的 batch 组合函数。
- `batch_process.py`：batch 级别的后处理逻辑。

#### `nanodet/trainer/` — 训练任务

- `task.py`：基于 `pytorch_lightning.LightningModule` 的 `TrainingTask`，封装了训练/验证/测试 step、优化器配置、学习率调度、日志记录与模型保存逻辑。

#### `nanodet/evaluator/` — 评估

- `coco_detection.py`：COCO 风格的 mAP 评估，包括 AP@0.5:0.95、AP50、AP75 等指标。

#### `nanodet/optim/` — 优化器构建

- `builder.py`：根据配置文件构建优化器与学习率调度器。

#### `nanodet/util/` — 通用工具

- `config.py` / `yacs.py`：基于 YACS 的配置系统。
- `check_point.py`：权重加载、旧版模型转换。
- `logger.py`：日志与 TensorBoard 封装。
- `visualization.py`：预测结果可视化。
- `box_transform.py`：边界框编码/解码（距离分布 → xyxy）。
- `flops_counter.py`：FLOPs 与参数量统计。
- `env_utils.py`、`misc.py`、`path.py`、`rank_filter.py`、`scatter_gather.py` 等辅助函数。

---

### `config/` — 配置文件

所有训练、验证、导出行为均由 YAML 配置文件驱动。每个配置文件通常包含以下字段：

- `save_dir`：实验输出目录。
- `model`：模型结构定义，包括 `weight_averager`、`arch`（backbone / fpn / head / aux_head）。
- `data`：训练集与验证集路径、输入尺寸、数据增强管线。
- `device`：GPU 编号、batch size、worker 数量、训练精度（32/16）。
- `schedule`：优化器、学习率调度、warmup、总 epoch、val 间隔、resume/load_model 路径。
- `evaluator`：评估器类型与保存指标。
- `log`：日志打印间隔。
- `class_names`：类别名称列表。

仓库内置的配置覆盖了 NanoDet-Plus-m 320/416、NanoDet-Plus-m-1.5x 320/416、YOLO 格式、VOC/XML 格式以及 ConvNeXt backbone 等场景。`legacy_v0.x_configs/` 目录存放旧版 NanoDet 的配置，供历史模型兼容使用。

---

### `tools/` — 训练与导出脚本

- `train.py`：训练入口。加载配置后构建数据集、DataLoader、Evaluator 与 TrainingTask，并调用 `pl.Trainer.fit()` 执行训练。
- `test.py`：验证/测试入口。加载训练好的 checkpoint，在验证集上计算 mAP。
- `export_onnx.py`：将训练好的 PyTorch 模型导出为 ONNX，并可选择使用 `onnx-simplifier` 简化模型。
- `inference.py`：基于 PyTorch 的单张/批量图片推理脚本，用于快速验证训练效果。
- `flops.py`：统计模型 FLOPs 与参数量。
- `export_torchscript.py`：导出 TorchScript 格式模型。
- `convert_old_checkpoint.py`：将旧版 `.pth` checkpoint 转换为新版 lightning 格式。

---

### `tests/` — 单元测试

按照模块划分了测试用例，覆盖 backbone、FPN、head、loss、data transform、evaluator、trainer 与 utils。改模型结构或数据管线后，可通过 pytest 进行回归验证。

---

### `demo*` 与 `docs/` — 示例与文档

- `demo/`：Python 推理示例与 Jupyter Notebook。
- `demo_ncnn/`、`demo_mnn/`、`demo_openvino/`、`demo_libtorch/`：各推理框架的 C++ 部署示例。
- `demo_android_ncnn/`：基于 ncnn 的 Android 完整工程。
- `docs/`：项目文档、配置说明与架构图。

这些目录与训练、验证、ONNX 导出的主链路无直接依赖关系，主要用于模型部署演示与参考。

---

## 核心训练流程

1. **配置解析**：`load_config` 读取 YAML 配置并合并到默认 `CfgNode`。
2. **数据构建**：`build_dataset` 根据配置分别构建训练集与验证集，`naive_collate` 将样本组合为 batch。
3. **评估器构建**：`build_evaluator` 创建 COCO 风格评估器。
4. **训练任务构建**：`TrainingTask` 封装模型、损失、优化器、学习率调度与日志。
5. **权重加载**：若配置中指定 `load_model` 或 `resume`，则加载对应 checkpoint。
6. **Trainer 启动**：使用 `pytorch_lightning.Trainer` 执行 `fit`，支持单卡、多卡与混合精度训练。

---

## ONNX 导出流程

1. 根据配置文件 `model` 字段调用 `build_model` 构建模型。
2. 加载训练好的 checkpoint 权重。
3. 若 backbone 为 RepVGG，执行结构重参数化（re-parameterization）转换。
4. 构造随机输入，调用 `torch.onnx.export` 导出为 ONNX，输入名称为 `data`，输出名称为 `output`。
5. 使用 `onnx-simplifier` 对导出的 ONNX 进行图简化，便于后续部署。

---

## 依赖说明

项目主要依赖：

- `torch >= 1.10, < 2.0`
- `torchvision`
- `pytorch-lightning >= 1.9.0, < 2.0.0`
- `omegaconf >= 2.0.1`
- `onnx`、`onnx-simplifier`
- `opencv-python`、`imagesize`
- `pycocotools`
- `tensorboard`、`torchmetrics`、`tqdm`、`matplotlib`、`tabulate`、`termcolor`、`pyaml`、`Cython`

由于 PyTorch 1.13 与 NumPy 2.x 存在 ABI 不兼容，实际运行时需要将 NumPy 固定在 1.x 版本。

---

## 环境信息

本项目已配置 Conda 环境 `nanodet`：

- Python：3.9
- PyTorch：1.13.1 + CUDA 11.7
- NumPy：1.26.4
- GPU：NVIDIA GeForce RTX 3090

激活环境：

```bash
conda activate nanodet
```

---

## 常用命令示例

### 训练

```bash
python tools/train.py config/nanodet-plus-m_416.yml
```

### 验证

```bash
python tools/test.py --task val --config config/nanodet-plus-m_416.yml --model workspace/nanodet-plus-m_416/model_best/model_best.ckpt
```

### 导出 ONNX

```bash
python tools/export_onnx.py \
  --cfg_path config/nanodet-plus-m_416.yml \
  --model_path workspace/nanodet-plus-m_416/model_best/model_best.ckpt \
  --out_path nanodet-plus-m_416.onnx \
  --input_shape 416,416
```

---

## 开发提示

- 修改 backbone 时，重点关注 `nanodet/model/backbone/` 与对应 `__init__.py` 中的注册逻辑。
- 修改特征融合结构时，重点关注 `nanodet/model/fpn/`。
- 修改检测头时，重点关注 `nanodet/model/head/nanodet_plus_head.py` 与 `simple_conv_head.py`。
- 修改整体训练流程（如增加新的损失项、调整 EMA 逻辑）时，重点关注 `nanodet/trainer/task.py` 与 `nanodet/model/arch/nanodet_plus.py`。
- 新增数据格式时，在 `nanodet/data/dataset/` 中继承 `BaseDataset` 并实现必要接口。
- 配置文件中 `model.arch.head.num_classes` 必须与 `class_names` 的长度一致，否则 `train.py` 会抛出校验错误。
