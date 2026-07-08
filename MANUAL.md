# NanoDet-Plus Barcode/QR Code 训练与部署手册

> 本手册基于 NanoDet-Plus 仓库，针对 **480×640 单通道灰度图** 的条码/二维码检测任务。
> 
> 多服务器环境配置与多卡训练详见 `SERVER_SETUP.md`。

---

## 1. 环境配置

### 1.1 创建 Conda 环境

```bash
conda create -n nanodet python=3.9 -y
conda activate nanodet
```

### 1.2 安装 PyTorch（带 CUDA）

本机 RTX 3090，使用 CUDA 11.7：

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 1.3 安装依赖

```bash
pip install -r requirements.txt
```

注意：`requirements.txt` 会安装 NumPy 2.x，但 PyTorch 1.13 不兼容，需要降级：

```bash
pip install "numpy<2"
```

### 1.4 安装 nanodet 包

```bash
pip install -e .
```

### 1.5 验证环境

```bash
python -c "import nanodet; print(nanodet.__version__); import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 2. 数据准备

数据集放在 `./dataset/`，包含三个来源：

- `Barcode and QR code detection.v3i.coco/`：COCO 格式，416×416
- `qr code.v3i.coco/`：COCO 格式，640×640，只有 QR code
- `archive/`：VOC/XML 格式，约 1600×1200，只有 barcode

### 2.1 运行数据合并脚本

```bash
python tools/prepare_barcode30k.py
```

该脚本会：
1. 解析三个数据集
2. 转换为灰度图
3. 统一类别：`barcode=1`，`qr_code=2`
4. 过滤小于 4×4 像素的 bbox
5. 全局随机打乱，按 **8:1:1** 切分为 train/val/test
6. 计算训练集 mean/std
7. 输出到 `./dataset/barcode30k_final/`

### 2.2 输出结构

```text
dataset/barcode30k_final/
├── train/
│   ├── images/
│   └── train.json
├── val/
│   ├── images/
│   └── val.json
├── test/
│   ├── images/
│   └── test.json
└── norm_stats.json
```

### 2.3 清理临时文件

```bash
rm -rf dataset/unified_pool
```

---

## 3. 配置文件

主配置文件：`config/nanodet-plus-m_480x640_barcode30k.yml`

关键参数：

```yaml
model:
  arch:
    backbone:
      name: MobileNetV2      # 为 NPU 兼容由 ShuffleNetV2 切换
      width_mult: 1.0
      out_stages: [2, 4, 6]
      last_channel: 320        # stage6 输出保持 320，避免默认 1280 过宽
      in_channels: 1          # 单通道灰度输入
      pretrain: False         # 不使用 ImageNet 预训练
    fpn:
      name: GhostPAN
      in_channels: [32, 96, 320]
      out_channels: 96
      upsample_cfg:
        scale_factor: 2
        mode: nearest         # NPU 只支持 nearest，禁用 bilinear
    head:
      num_classes: 2          # barcode + qr_code

data:
  train:
    input_size: [640, 480]    # [w, h]，对应张量 1x1x480x640
    keep_ratio: True          # 等比缩放 + 中心 padding
    grayscale: True
    pipeline:
      normalize: [[114.4859], [66.9615]]   # 训练集统计的 mean/std
  val:
    input_size: [640, 480]
    keep_ratio: True
    grayscale: True

vis:
  num_images: 5               # 每个 epoch 记录 5 张验证图到 TensorBoard
  score_thresh: 0.5

schedule:
  total_epochs: 1000
  val_intervals: 1            # 每个 epoch 验证一次
```

如显存不足，调整：

```yaml
device:
  batchsize_per_gpu: 48       # 或 32
  precision: 32               # 保持 FP32
```

### 3.1 轻量化配置（可选）

仓库还提供了三个不同优化强度的配置，用于在 NPU 上进一步加速：

| 配置 | 文件 | 参数量 | 预估 ONNX | 优化手段 | 适用场景 |
|---|---|---|---|---|---|
| **Baseline** | `config/nanodet-plus-m_480x640_barcode30k.yml` | ~5.2 M | ~9.3 MB | MobileNetV2 1.0x，FPN/Head 96 ch | 精度优先 |
| **Lite** | `config/nanodet-plus-m_480x640_barcode30k_lite.yml` | ~3.4 M | ~7.9 MB | FPN/Head 降到 64 ch | 速度略快，精度损失最小 |
| **Tiny** | `config/nanodet-plus-m_480x640_barcode30k_tiny.yml` | ~1.9 M | ~4.5 MB | MobileNetV2 0.75x，FPN/Head 48 ch，Head 1 conv，3×3 kernel | 速度与精度平衡 |
| **Nano** | `config/nanodet-plus-m_480x640_barcode30k_nano.yml` | ~0.7 M | ~2.0 MB | MobileNetV2 0.5x，FPN/Head 32 ch，去掉 stride 64 | 极速，可能对小目标有影响 |
| **Pico** | `config/nanodet-plus-m_480x640_barcode30k_pico.yml` | ~0.4 M | ~1.3 MB | MobileNetV2 0.4x，FPN/Head 24 ch；调大 batch/aug/lr | 超轻量，尝试维持精度 |
| **Femto** | `config/nanodet-plus-m_480x640_barcode30k_femto.yml` | ~0.2 M | ~0.6 MB | MobileNetV2 0.25x，FPN/Head 16 ch；更强 aug + 大 batch | 极致轻量，精度会有下降 |

切换训练配置只需改命令：

```bash
# Lite
python tools/train.py config/nanodet-plus-m_480x640_barcode30k_lite.yml

# Tiny
python tools/train.py config/nanodet-plus-m_480x640_barcode30k_tiny.yml

# Nano
python tools/train.py config/nanodet-plus-m_480x640_barcode30k_nano.yml
```

---

## 4. 训练

### 4.1 启动训练

```bash
conda activate nanodet
python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

### 4.2 训练输出

```text
workspace/nanodet-plus-m_barcode30k/
├── logs-YYYY-MM-DD-HH-MM-SS/   # TensorBoard 日志
├── model_last.ckpt             # 最后一个 epoch 的模型
└── model_best/
    ├── model_best.ckpt         # 验证 AP 最高的模型
    └── nanodet_model_best.pth  # 仅权重
```

### 4.3 查看 TensorBoard

```bash
conda activate nanodet
tensorboard --logdir workspace/nanodet-plus-m_barcode30k
```

浏览器打开 `http://localhost:6006`。

### 4.4 训练指标说明

| 指标 | 含义 | 目标 |
|------|------|------|
| `Train_loss/total` | 总训练 loss | 稳定下降 |
| `Train_loss/loss_qfl` | 分类 loss | < 0.3 |
| `Train_loss/loss_bbox` | 边框回归 loss | < 0.5 |
| `Train_loss/loss_dfl` | 分布 focal loss | < 0.3 |
| `Val_metrics/mAP` | COCO mAP | > 0.75 |
| `Val_metrics/AP_50` | IoU=0.5 的 AP | > 0.90 |

### 4.5 多卡训练

修改配置文件：

```yaml
device:
  gpu_ids: [0, 1, 2, 3]      # 使用的 GPU
  workers_per_gpu: 8
  batchsize_per_gpu: 48      # 每张卡的 batch size
  precision: 32
```

启动训练：

```bash
./tools/train_multi_gpu.sh
```

或手动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

总 batch size = `batchsize_per_gpu × GPU 数量`。

详细说明参见 `SERVER_SETUP.md`。

---

## 5. 验证与测试

### 5.1 验证集评估

```bash
python tools/test.py --task val \
  --config config/nanodet-plus-m_480x640_barcode30k.yml \
  --model workspace/nanodet-plus-m_barcode30k/model_last.ckpt
```

### 5.2 测试集评估

```bash
python tools/test.py --task test \
  --config config/nanodet-plus-m_480x640_barcode30k.yml \
  --model workspace/nanodet-plus-m_barcode30k/model_last.ckpt
```

### 5.3 预期结果

训练收敛后大致水平：

```text
mAP:    0.815
AP50:   0.978
AP75:   0.934
barcode AP50: 98.2
qr_code AP50: 97.4
```

---

## 6. 单图推理测试

### 6.1 随机抽取 10 张 test 图预测

```bash
python tools/predict_test_samples.py
```

输出到：

```text
workspace/nanodet-plus-m_barcode30k/test_predictions/
```

### 6.2 自定义单图推理

```python
from nanodet.util import Logger, cfg, load_config
from tools.predict_test_samples import Predictor

load_config(cfg, "config/nanodet-plus-m_480x640_barcode30k.yml")
logger = Logger(-1, "./tmp", False)
predictor = Predictor(cfg, "workspace/nanodet-plus-m_barcode30k/model_last.ckpt", logger, device="cuda:0")

meta, results = predictor.inference("path/to/your/image.jpg")
dets = results[0]  # {class_id: [[x1, y1, x2, y2, score], ...]}
```

---

## 7. ONNX 导出

### 7.1 导出命令

```bash
python tools/export_onnx.py \
  --cfg_path config/nanodet-plus-m_480x640_barcode30k.yml \
  --model_path workspace/nanodet-plus-m_barcode30k/model_last.ckpt \
  --out_path workspace/nanodet-plus-m_barcode30k/nanodet_barcode_480x640.onnx \
  --input_shape 480,640
```

### 7.2 导出选项

| 参数 | 说明 |
|------|------|
| `--cfg_path` | 配置文件 |
| `--model_path` | checkpoint 路径 |
| `--out_path` | 输出 ONNX 路径 |
| `--input_shape` | 输入形状 `H,W`，不指定则自动从 config 读取 |

### 7.3 ONNX 模型信息

```text
Input:  data [1, 1, 480, 640]
Outputs:
  cls_dis_stride_8  : [1, 34, 60, 80]
  cls_dis_stride_16 : [1, 34, 30, 40]
  cls_dis_stride_32 : [1, 34, 15, 20]
  cls_dis_stride_64 : [1, 34,  8, 10]
Size:   ~9.3 MB
```

### 7.4 输出格式

每个输出张量的通道维度 `34 = 2 (num_classes) + 32 (4*(reg_max+1))`：

- 前 2 维：分类分数（已做 Sigmoid）
- 后 32 维：bbox 分布回归

4 个输出分别对应 stride 8/16/32/64 的特征图，无需再按 anchor point 数拼接。

### 7.5 NPU 部署后处理

部署时必须实现：

1. **输入预处理**：灰度化 → 等比缩放+中心 padding → 归一化 `mean=[114.4859], std=[66.9615]`
2. **ONNX 推理**：输入 `1×1×480×640`
3. **遍历 4 个 stride 输出**：对每个 `[1, 34, H, W]` 张量：
   - Split：`cls (2)` + `dis (32)`
   - `cls` 已 Sigmoid，直接作为置信度
   - Distribution decode：32 维分布 → 4 个边框偏移
   - BBox decode：特征点坐标 + 偏移 → `x1,y1,x2,y2`
4. **NMS**：按类别分别非极大值抑制

---

## 8. 常见问题

### 8.1 CUDA out of memory

降低 batch size：

```yaml
device:
  batchsize_per_gpu: 32
```

### 8.2 训练 loss 下降但 val AP 很低

通常是 `keep_ratio=True` 时 padding 坐标对齐问题。检查 `nanodet/data/transform/warp.py` 中 padding 是否已合并进 `warp_matrix`。

### 8.3 NumPy 版本不兼容

如出现 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0`：

```bash
pip install "numpy<2"
```

### 8.4 TensorBoard 看不到图像

确认配置中有：

```yaml
vis:
  num_images: 5
  score_thresh: 0.5
```

且 `val_intervals` 不为 0。

---

## 9. 关键文件清单

| 文件 | 作用 |
|------|------|
| `config/nanodet-plus-m_480x640_barcode30k.yml` | 训练/导出配置 |
| `tools/prepare_barcode30k.py` | 数据合并、灰度化、切分 |
| `tools/train.py` | 训练入口 |
| `tools/test.py` | 验证/测试入口 |
| `tools/export_onnx.py` | ONNX 导出 |
| `tools/predict_test_samples.py` | 单图/批量推理可视化 |
| `nanodet/data/transform/warp.py` | 数据增强 + padding 坐标对齐 |
| `nanodet/trainer/task.py` | 训练任务 + TensorBoard 图像可视化 |
| `nanodet/model/backbone/mobilenetv2.py` | 支持 `in_channels=1` 的 MobileNetV2 backbone |
| `nanodet/model/head/nanodet_plus_head.py` | 拆分 cls/reg 输出卷积，ONNX 输出每 stride 一个 4D 张量 |
