# 服务器环境配置与多卡训练手册

本手册针对两台服务器的 CUDA/GPU 差异，分别给出 Conda 环境搭建和多卡训练方法。

| 服务器 | GPU | CUDA 版本 | PyTorch 兼容性 |
|--------|-----|-----------|----------------|
| Server A | 2× RTX 5090 | 13.0 (Driver 580) | **需要 PyTorch 2.6+ / CUDA 12.8+** |
| Server B | 4× RTX 4090 | 12.4 (Driver 550) | 兼容 PyTorch 1.13.1 / CUDA 11.7 |

> **重要**：当前 NanoDet 代码依赖 `torch<2.0` 和 `pytorch-lightning<2.0`，这套组合**不支持 RTX 5090**。5090 服务器必须升级 PyTorch + Lightning，或改用在 4090 服务器上训练。

---

## 目录

1. [Server B：4× RTX 4090 环境搭建](#server-b4--rtx-4090)
2. [Server A：2× RTX 5090 环境搭建](#server-a2--rtx-5090)
3. [多卡训练方法](#多卡训练)
4. [性能调优建议](#性能调优)
5. [常见问题](#常见问题)

---

## Server B：4× RTX 4090

### 1. 创建 Conda 环境

```bash
conda create -n nanodet python=3.9 -y
conda activate nanodet
```

### 2. 安装 PyTorch 1.13.1 + CUDA 11.7

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. 安装项目依赖

```bash
cd /path/to/nanodet
pip install -r requirements.txt
pip install "numpy<2"
```

### 4. 安装 nanodet 包

```bash
pip install -e .
```

### 5. 验证

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

预期输出：

```text
1.13.1+cu117
CUDA: True
GPUs: 4
```

---

## Server A：2× RTX 5090

RTX 5090 是 Blackwell 架构（sm_120），PyTorch 1.13 不支持。有两种方案：

### 方案 1：降级使用（推荐，改动最小）

如果你的项目可以在 4090 服务器上训练，直接把训练好的模型 `.ckpt` 和 ONNX 文件拷到 5090 服务器上**做推理**。推理时 ONNX Runtime / TensorRT 可能有支持 5090 的版本。

### 方案 2：升级训练栈到 PyTorch 2.x + Lightning 2.x

如果必须在 5090 上训练，需要大改代码：

```bash
conda create -n nanodet-py310 python=3.10 -y
conda activate nanodet-py310

# PyTorch 2.6 + CUDA 12.8（5090 需要）
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Lightning 2.x
pip install pytorch-lightning==2.5.0

# 其他依赖
pip install omegaconf onnx onnx-simplifier opencv-python pycocotools tensorboard torchmetrics tqdm matplotlib tabulate termcolor pyaml imagesize Cython
pip install "numpy<2"
```

然后需要修改 NanoDet 代码以适配 Lightning 2.x API：

| Lightning 1.9 | Lightning 2.x |
|---------------|---------------|
| `training_epoch_end` | `on_train_epoch_end` |
| `validation_epoch_end` | `on_validation_epoch_end` |
| `test_epoch_end` | `on_test_epoch_end` |
| `trainer.fit(..., ckpt_path=...)` | 相同 |
| `pl.Trainer(devices=[0,1])` | 相同 |
| `self.log(...)` | 相同 |
| `optimizer_step` 参数列表 | 有变化 |

这是一个独立的迁移任务。如果你确定要在 5090 上训练，我可以帮你做完整迁移。

---

## 多卡训练

### 单卡训练（基准）

```bash
conda activate nanodet
python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

### 多卡训练

修改配置文件：

```yaml
device:
  gpu_ids: [0, 1, 2, 3]   # 使用哪些 GPU
  workers_per_gpu: 8
  batchsize_per_gpu: 48    # 这是**每张卡**的 batch size
  precision: 32
```

总 batch size = `batchsize_per_gpu × len(gpu_ids)`。

例如 4 卡 × 48 = 全局 batch 192。

启动命令和单卡一样：

```bash
python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

PyTorch Lightning 会自动检测到 `len(gpu_ids) > 1`，启用 DDP（DistributedDataParallel）。

### 多卡启动脚本

创建了 `tools/train_multi_gpu.sh`：

```bash
#!/bin/bash
# 4 卡训练示例
export CUDA_VISIBLE_DEVICES=0,1,2,3
python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

使用：

```bash
chmod +x tools/train_multi_gpu.sh
./tools/train_multi_gpu.sh
```

### 指定不同 GPU 数量

只想用 2 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

同时修改配置：

```yaml
device:
  gpu_ids: [0, 1]
```

---

## 性能调优

### 1. 根据显存调整 batch size

| GPU | 显存 | 建议 batchsize_per_gpu |
|-----|------|------------------------|
| RTX 4090 | 24 GB | 48 ~ 64 |
| RTX 5090 | 32 GB | 64 ~ 96（需 PyTorch 2.x） |

### 2. 学习率缩放

多卡训练时，如果增大了全局 batch size，建议按线性缩放学习率：

```yaml
schedule:
  optimizer:
    lr: 0.004    # 单卡 0.001 × 4 卡
```

不过 NanoDet 默认用了 warmup + cosine，通常保持 `lr=0.001` 也能收敛。

### 3. workers_per_gpu

一般设为 `8` 或 `16`。不要设太高，否则 CPU 预处理会成为瓶颈。

### 4. 多卡时关闭可视化（可选）

如果 DDP 训练出现同步问题，可以临时关闭 TensorBoard 图像记录：

```yaml
vis:
  num_images: 0
```

---

## 常见问题

### Q1：多卡训练报错 `NCCL` 相关错误

检查：

```bash
NCCL_P2P_DISABLE=1 python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml
```

如果解决，说明 PCIe 拓扑导致 P2P 通信问题，可以加环境变量：

```bash
export NCCL_P2P_DISABLE=1
```

### Q2：多卡训练时 loss 变成 NaN

可能是学习率太大，或者全局 batch 太大。尝试：

```yaml
schedule:
  optimizer:
    lr: 0.0005
```

### Q3：RTX 5090 上 PyTorch 1.13 报错

典型错误：

```text
CUDA error: no kernel image is available for execution on the device
```

原因：PyTorch 1.13 不支持 Blackwell 架构。必须升级到 PyTorch 2.6+ + CUDA 12.8，并迁移 Lightning 代码。

### Q4：如何只在一台服务器训练，模型拿到另一台推理？

训练完成后：

```bash
# 导出 ONNX
python tools/export_onnx.py \
  --cfg_path config/nanodet-plus-m_480x640_barcode30k.yml \
  --model_path workspace/nanodet-plus-m_barcode30k/model_best/model_best.ckpt \
  --out_path nanodet_barcode_480x640.onnx \
  --input_shape 480,640
```

然后把 ONNX 文件拷到目标服务器。ONNX Runtime / OpenVINO / TensorRT 的推理代码通常不依赖 PyTorch 版本。

---

## 快速命令清单

```bash
# 4090 服务器环境
conda create -n nanodet python=3.9 -y
conda activate nanodet
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install "numpy<2"
pip install -e .

# 4 卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py config/nanodet-plus-m_480x640_barcode30k.yml

# 验证
python tools/test.py --task val --config config/nanodet-plus-m_480x640_barcode30k.yml --model workspace/nanodet-plus-m_barcode30k/model_best/model_best.ckpt

# 导出 ONNX
python tools/export_onnx.py --cfg_path config/nanodet-plus-m_480x640_barcode30k.yml --model_path workspace/nanodet-plus-m_barcode30k/model_best/model_best.ckpt --out_path nanodet_barcode_480x640.onnx --input_shape 480,640
```
