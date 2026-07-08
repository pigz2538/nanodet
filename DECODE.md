# NanoDet-Plus Barcode/QR 检测模型 — 部署解码手册

> 本文档面向 NPU/板端 C++ 部署场景，说明如何将训练好的 `model_best.ckpt` 经过 ONNX 中转后，在板端自有推理引擎上完成：
> 1. 模型输入预处理；
> 2. 4 个 stride 输出的后处理（decode + NMS）；
> 3. 检测框映射回原图、裁剪、二值化；
> 4. 使用 ZXing-C++ 解码二维码/条码；
> 5. 板端 CMake 工程组织与编译。
>
> 配套模型配置：`config/nanodet-plus-m_480x640_barcode30k.yml`

---

## 1. 模型输入

### 1.1 输入张量

| 属性 | 值 | 说明 |
|---|---|---|
| 名称 | `data` | ONNX 输入节点名 |
| 形状 | `[1, 1, 480, 640]` | NCHW，batch=1，单通道灰度 |
| 数据类型 | `float32` | 推理前需从 uint8 转换 |
| 空间尺寸 | 高 480，宽 640 | 与配置中 `input_size: [640, 480]`（[w,h]）对应 |

### 1.2 预处理流程（必须与训练一致）

训练时配置为：

```yaml
data:
  val:
    input_size: [640, 480]   # [w, h]
    keep_ratio: True         # 等比缩放 + 中心 padding
    grayscale: True
    pipeline:
      normalize: [[114.4859], [66.9615]]
```

板端预处理必须严格复现以下步骤：

#### 步骤 1：读取原图并转灰度

- 若原图为彩色（BGR/RGB），按 `Y = 0.299R + 0.587G + 0.114B` 转成单通道灰度。
- 若原图已是灰度，直接使用。

#### 步骤 2：等比缩放 + 中心 Padding

设原图宽高为 `(src_w, src_h)`，目标尺寸 `(dst_w, dst_h) = (640, 480)`。

```cpp
float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
int new_w = (int)(src_w * scale);
int new_h = (int)(src_h * scale);
int pad_x = (dst_w - new_w) / 2;   // 水平左 padding
int pad_y = (dst_h - new_h) / 2;   // 垂直上 padding
```

- 将原图缩放到 `(new_w, new_h)`；
- 放入 `(640, 480)` 画布中央，其余位置填 `0`（黑色）。

> 注意：训练时 `pad_value` 默认为 0，因此 padding 区域应为黑色。某些实现可能用 114，但本仓库未显式设置，按 0 处理。

#### 步骤 3：归一化

对每一个像素值 `p`（范围 0~255）：

```cpp
float norm = (p - 114.4859f) / 66.9615f;
```

将结果按 NCHW 排列写入输入张量：`[1, 1, 480, 640]`。

#### 步骤 4：保存预处理元信息

后续需要将模型输出的 480×640 坐标映射回原图，因此必须保存：

```cpp
struct PreprocInfo {
    float scale;   // 缩放比
    int pad_x;     // 水平左 padding
    int pad_y;     // 垂直上 padding
    int dst_w;     // 640
    int dst_h;     // 480
};
```

---

## 2. 模型输出

### 2.1 输出节点

ONNX 导出后共有 4 个输出节点：

| 输出节点名 | 形状 | 特征图尺寸 | stride |
|---|---|---|---|
| `cls_dis_stride_8`  | `[1, 34, 60, 80]` | 高 60，宽 80  | 8 |
| `cls_dis_stride_16` | `[1, 34, 30, 40]` | 高 30，宽 40  | 16 |
| `cls_dis_stride_32` | `[1, 34, 15, 20]` | 高 15，宽 20  | 32 |
| `cls_dis_stride_64` | `[1, 34,  8, 10]` | 高  8，宽 10  | 64 |

### 2.2 通道含义

每个输出张量的 34 个通道分为两部分：

```
通道 0~1   : 分类分数（已做 Sigmoid）
通道 2~33  : bbox 分布回归（DFL），共 32 个通道
```

DFL 部分再细分为 4 个边界：

```
通道  2~ 9 : 左边界（left）  的 8 个分布值
通道 10~17 : 上边界（top）   的 8 个分布值
通道 18~25 : 右边界（right） 的 8 个分布值
通道 26~33 : 下边界（bottom）的 8 个分布值
```

其中 8 来自 `reg_max + 1 = 7 + 1 = 8`。

### 2.3 数据排布

输出为 **NCHW** 格式。对于节点 `cls_dis_stride_s`，形状 `[1, 34, H, W]`，索引 `(0, c, y, x)` 的内存偏移为：

```cpp
offset = c * H * W + y * W + x
```

其中：

- `c ∈ [0, 33]` 为通道索引；
- `(x, y)` 为特征图像素坐标，`x ∈ [0, W-1]`，`y ∈ [0, H-1]`；
- 该特征点对应原图（480×640）中的锚点坐标为 `(x * stride, y * stride)`。

---

## 3. 后处理流程

后处理目标：从 4 个 stride 输出中得到原图坐标下的检测框 `[x1, y1, x2, y2, score, class]`。

### 3.1 遍历所有 stride

对每个 stride `s ∈ {8, 16, 32, 64}`，遍历其特征图所有 `(x, y)`：

```cpp
for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
        // 读取该点的 34 个通道值
    }
}
```

### 3.2 分类分数解析

ONNX 输出中分类分数已经过 Sigmoid，直接取最大值：

```cpp
score_0 = ptr[0 * H * W + y * W + x];   // barcode
score_1 = ptr[1 * H * W + y * W + x];   // qr_code
score   = max(score_0, score_1);
label   = (score_1 > score_0) ? 1 : 0;
```

若 `score < score_threshold`（如 0.3 或 0.5），则丢弃该点。

### 3.3 DFL 分布解码

对每一边界 `d ∈ {left, top, right, bottom}`，取 8 个原始值，做 softmax，再按索引加权求和：

```cpp
for (int d = 0; d < 4; ++d) {
    const float* dis = ptr + (2 + d * 8) * H * W + y * W + x;
    // softmax
    float maxv = *dis;
    for (int k = 1; k < 8; ++k) maxv = std::max(maxv, dis[k * H * W]);
    float sum = 0;
    for (int k = 0; k < 8; ++k) {
        prob[k] = exp(dis[k * H * W] - maxv);
        sum += prob[k];
    }
    for (int k = 0; k < 8; ++k) prob[k] /= sum;

    // 分布期望
    float dis_pred = 0;
    for (int k = 0; k < 8; ++k) dis_pred += k * prob[k];

    // 乘以 stride，得到像素级距离
    dis_pixel[d] = dis_pred * stride;
}
```

### 3.4 BBox 解码

特征点 `(x, y)` 对应的锚点中心坐标为：

```cpp
cx = x * stride;
cy = y * stride;
```

 bbox 四边：

```cpp
x1 = cx - dis_pixel[0];   // left
y1 = cy - dis_pixel[1];   // top
x2 = cx + dis_pixel[2];   // right
y2 = cy + dis_pixel[3];   // bottom
```

裁剪到输入图范围内：

```cpp
x1 = clamp(x1, 0, 640);
y1 = clamp(y1, 0, 480);
x2 = clamp(x2, 0, 640);
y2 = clamp(y2, 0, 480);
```

### 3.5 NMS

所有 stride 产生的候选框合并后，按类别分别做 NMS。

默认参数（与训练配置一致）：

- 置信度阈值 `score_thr = 0.3`（可根据场景调整到 0.5）
- NMS IoU 阈值 `nms_thr = 0.6`
- 每类最多保留 `max_num = 100`（可选）

NMS 流程：

1. 按 `score` 降序排序；
2. 取当前最高分框加入结果；
3. 计算该框与其余框的 IoU，大于阈值则抑制；
4. 重复步骤 2~3 直到无剩余框。

### 3.6 坐标映射回原图

将 480×640 输入图上的检测框映射回原图 `(src_w, src_h)`：

```cpp
x1_orig = (x1 - pad_x) / scale;
y1_orig = (y1 - pad_y) / scale;
x2_orig = (x2 - pad_x) / scale;
y2_orig = (y2 - pad_y) / scale;
```

再裁剪到原图边界：

```cpp
x1_orig = clamp(x1_orig, 0, src_w);
y1_orig = clamp(y1_orig, 0, src_h);
x2_orig = clamp(x2_orig, 0, src_w);
y2_orig = clamp(y2_orig, 0, src_h);
```

---

## 4. 裁剪与解码

### 4.1 裁剪

对每个检测框，在原图灰度图上裁剪出 ROI：

```cpp
int x1 = (int)std::max(0.0f, box.x1);
int y1 = (int)std::max(0.0f, box.y1);
int x2 = (int)std::min((float)src_w, box.x2);
int y2 = (int)std::min((float)src_h, box.y2);
int crop_w = x2 - x1;
int crop_h = y2 - y1;
```

建议 ROI 外扩一定边距（如 5%），二维码/条码边缘留出白边，解码成功率更高：

```cpp
int margin_x = (int)(crop_w * 0.05f);
int margin_y = (int)(crop_h * 0.05f);
x1 = std::max(0, x1 - margin_x);
y1 = std::max(0, y1 - margin_y);
x2 = std::min(src_w, x2 + margin_x);
y2 = std::min(src_h, y2 + margin_y);
```

### 4.2 二值化

ZXing-C++ 可以处理灰度图，但裁剪后的图若光照不均，建议先做二值化。可用 Otsu 或固定阈值：

```cpp
// 固定阈值（快）
for (auto& p : crop.data) p = (p > 128) ? 255 : 0;

// 或 Otsu（自适应，略慢）
// ... 计算类间方差最大阈值 ...
```

### 4.3 使用 ZXing-C++ 解码

选择 ZXing-C++ 的原因：

- 同时支持 QR Code 和常见 1D 条码（Code 128、EAN-13、UPC-A、Code 39 等）；
- 现代 C++ API，接口简单；
- 可只编译 `Reader` 部分，静态库体积可控（通常 1~3MB）；
- 无运行时依赖。

#### 4.3.1 编译 ZXing-C++ 为静态库

在板端或交叉编译环境中：

```bash
git clone https://github.com/zxing-cpp/zxing-cpp.git
cd zxing-cpp
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BLACKBOX_TESTS=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_READERS=ON \
  -DBUILD_WRITERS=OFF
make -j$(nproc)
make install   # 或手动拷贝 libZXing.a 和头文件
```

#### 4.3.2 C++ 解码代码

```cpp
#include <ZXing/ReadBarcode.h>
#include <ZXing/ImageView.h>

std::string decode_barcode(const uint8_t* gray_data, int w, int h) {
    auto view = ZXing::ImageView(gray_data, w, h, ZXing::ImageFormat::Lum);

    auto opts = ZXing::ReaderOptions()
        .setFormats(
            ZXing::BarcodeFormat::QRCode |
            ZXing::BarcodeFormat::Code128 |
            ZXing::BarcodeFormat::Code39 |
            ZXing::BarcodeFormat::EAN13 |
            ZXing::BarcodeFormat::UPCA
        )
        .setTryHard(true)
        .setTryRotate(true);

    auto result = ZXing::ReadBarcode(view, opts);
    return result.isValid() ? result.text() : "";
}
```

如果希望更激进地提高识别率，可对单张 ROI 尝试多种二值化阈值后多次调用 `ReadBarcode`。

---

## 5. 板端工程结构与 CMakeLists.txt

假设板端已有 NPU 推理引擎，提供类似如下接口：

```cpp
class NpuEngine {
public:
    bool load(const std::string& model_path);
    bool run(const float* input_data, std::vector<float*>& outputs);
};
```

### 5.1 推荐目录结构

```
nanodet_barcode_decode/
├── CMakeLists.txt
├── main.cpp                    // 板端入口
├── nanodet_decode.h            // 后处理声明
├── nanodet_decode.cpp          // 后处理实现
├── image_utils.h               // 图像读写/裁剪/二值化
├── image_utils.cpp
├── stb_image.h                 // 单头文件图像读取
├── stb_image_write.h           // 单头文件图像保存（可选）
├── third_party/
│   └── zxing/
│       ├── include/            // ZXing-C++ 头文件
│       └── lib/
│           └── libZXing.a      // 交叉编译好的静态库
└── models/
    └── nanodet_barcode_480x640.model   // NPU 模型文件
```

### 5.2 CMakeLists.txt 示例

```cmake
cmake_minimum_required(VERSION 3.10)
project(nanodet_barcode_decode)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 源文件
set(SOURCES
    main.cpp
    nanodet_decode.cpp
    image_utils.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

# ZXing-C++
set(ZXING_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zxing")
target_include_directories(${PROJECT_NAME} PRIVATE ${ZXING_ROOT}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${ZXING_ROOT}/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ZXing)

# NPU 推理引擎（根据板端 SDK 调整）
# target_include_directories(${PROJECT_NAME} PRIVATE /path/to/npu/sdk/include)
# target_link_directories(${PROJECT_NAME} PRIVATE /path/to/npu/sdk/lib)
# target_link_libraries(${PROJECT_NAME} PRIVATE npu_infer)

# 基础系统库
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
```

### 5.3 main.cpp 骨架

```cpp
#include "nanodet_decode.h"
#include "image_utils.h"
#include <ZXing/ReadBarcode.h>
#include <ZXing/ImageView.h>
#include <iostream>
#include <vector>

// 板端 NPU 推理引擎封装（由板端 SDK 提供）
// #include "npu_engine.h"

int main(int argc, char** argv) {
    const char* img_path = argv[1];
    const char* model_path = argv[2];

    // 1. 读图并转灰度
    GrayImage src = load_image_gray(img_path);

    // 2. 预处理
    PreprocInfo info;
    std::vector<float> input_tensor(1 * 1 * 480 * 640);
    preprocess(src, input_tensor.data(), 640, 480, info);

    // 3. NPU 推理（伪代码，替换为板端 SDK 接口）
    // NpuEngine engine;
    // engine.load(model_path);
    // std::vector<float*> outputs = engine.run(input_tensor.data());
    // outputs 顺序必须与 ONNX 输出顺序一致：
    // [stride_8, stride_16, stride_32, stride_64]

    // 4. 后处理
    std::vector<BBox> dets = nanodet_decode(
        outputs, 480, 640, 0.3f, 0.6f);

    // 5. 映射回原图坐标
    for (auto& box : dets) {
        box = map_to_original(box, info, src.w, src.h);
    }

    // 6. 裁剪 + 解码
    for (size_t i = 0; i < dets.size(); ++i) {
        GrayImage crop = crop_gray(src, dets[i], 0.05f);
        threshold_otsu(crop);

        std::string text = decode_barcode(crop.data.data(), crop.w, crop.h);
        if (!text.empty()) {
            std::cout << "[" << i << "] class=" << dets[i].label
                      << " score=" << dets[i].score
                      << " text=" << text << std::endl;
        }

        // 可选：保存裁剪图用于调试
        // save_png(("crop_" + std::to_string(i) + ".png").c_str(), crop);
    }

    return 0;
}
```

---

## 6. 关键接口约定

### 6.1 NPU 推理引擎输出约定

板端转换后的模型必须保证输出顺序与 ONNX 一致：

```
output[0] -> cls_dis_stride_8  , shape [1, 34, 60, 80]
output[1] -> cls_dis_stride_16 , shape [1, 34, 30, 40]
output[2] -> cls_dis_stride_32 , shape [1, 34, 15, 20]
output[3] -> cls_dis_stride_64 , shape [1, 34,  8, 10]
```

每个输出必须是连续的 `float32` 内存块，NCHW 排布。

### 6.2 类别定义

| 类别 ID | 名称 |
|---|---|
| 0 | barcode（一维条码） |
| 1 | qr_code（二维码） |

### 6.3 调试建议

1. **对比 Python 输出**：先用 Python 跑通 `tools/predict_test_samples.py`，得到同一张图的检测框；再用 C++ 后处理对比坐标，误差应小于 1 个像素。
2. **可视化中间结果**：将 C++ 后处理得到的 bbox 画在原图上保存，确认 padding/scale 映射正确。
3. **二值化阈值调参**：若 ZXing 解码失败，尝试多种阈值或自适应二值化。

---

## 7. 板端部署注意事项

### 7.1 浮点精度

- NPU 量化模型输出可能与 ONNX 有微小误差，但后处理阈值（0.3）有一定容忍度。
- 若板端模型是 INT8 量化，建议先用浮点 ONNX 在板端 CPU 跑通，再切换到 NPU。

### 7.2 内存布局

- 4 个 stride 输出总大小约为：
  `34 * (60*80 + 30*40 + 15*20 + 8*10) = 34 * 5900 = 200600` floats ≈ **783 KB**。
- 输入张量 `1*1*480*640` ≈ **1.2 MB**。
- 整体内存占用很小，适合嵌入式板端。

### 7.3 多线程

- NPU 推理本身是异步/并行的；
- 后处理（DFL + NMS）可在 CPU 单线程跑，数据量不大；
- 若需处理多图，可将 NPU 推理与 CPU 后处理做成流水线。

### 7.4 ZXing-C++ 体积优化

若静态库体积敏感，编译 ZXing 时只启用需要的码制：

```cmake
-DBUILD_READERS=ON
-DBUILD_WRITERS=OFF
-DBUILD_EXAMPLES=OFF
-DBUILD_BLACKBOX_TESTS=OFF
-DBUILD_UNIT_TESTS=OFF
```

ZXing-C++ 内部会根据 `ReaderOptions::setFormats()` 自动裁剪未使用格式，但编译时关掉 writers 和 tests 仍能显著减少体积。

---

## 8. 后续开发 checklist

板端写代码时按以下顺序验证：

- [ ] 确认 NPU 模型能正常加载并输出 4 个 stride 张量；
- [ ] 确认输出顺序与 ONNX 一致；
- [ ] 确认预处理后的图与 Python 预处理结果一致（像素级比对）；
- [ ] 确认 C++ 后处理得到的 bbox 与 Python `predict_test_samples.py` 一致；
- [ ] 确认 bbox 映射回原图后位置正确；
- [ ] 确认裁剪 + 二值化后的图 ZXing 能识别；
- [ ] 集成 CMake 编译通过；
- [ ] 在真实业务图片上测试 barcode 和 qr_code 两类。

---

## 9. 参考资料

- 训练配置：`config/nanodet-plus-m_480x640_barcode30k.yml`
- ONNX 导出脚本：`tools/export_onnx.py`
- Python 推理参考：`tools/predict_test_samples.py`
- 后处理源码：`nanodet/model/head/nanodet_plus_head.py`、`nanodet/util/box_transform.py`
- ZXing-C++： https://github.com/zxing-cpp/zxing-cpp
