# 测试与复现

以下命令均在项目根目录、独立 CPU 虚拟环境中执行。首次使用先按 README 安装 uv 并同步环境。

模型权重入口见 [模型下载与配置对应关系](model-downloads.md)；下文提供导出参数与参考对照方法。

## 自动化测试

```bash
uv sync --locked --extra service
uv run --locked --no-sync python scripts/test.py all  # 完整 SDK/服务 pytest + smoke
uv run --locked --no-sync python scripts/test.py quick    # 单元回归，排除 real_ort 命名的测试
uv run --locked --no-sync python scripts/test.py smoke    # 合成引擎：warmup、分批、padding和空输入
uv run --locked --no-sync python scripts/test.py ort      # 真实 CPU ORT 执行合成 ONNX 图
uv run --locked --no-sync python scripts/test.py all -- -x  # 向 pytest 传入额外参数，首个失败即停止
```

默认开发依赖提供测试工具；启用 `service` extra 后完整测试包括服务回归。只验证基础 SDK 时使用 `uv sync --locked`，不要求 HTTP 运行依赖；也可追加 `-- --ignore=tests/test_service.py` 明确排除服务。服务安装与接口见 [服务说明](service/README.md)。

脚本使用调用它的 Python 解释器，自动定位仓库并将源码加入子进程 PYTHONPATH，可从任意目录调用绝对路径。任何检查失败返回非零退出码，并停止后续步骤；不会自动安装依赖、下载权重或修改运行时配置。额外 pytest 参数仅作用于 pytest 步骤。

完整测试使用 uv 默认 dev 组（包含 test 组）；quick 需要 pytest 和基础 SDK 依赖，smoke 只需要基础 SDK 依赖。quick/ort 按 `real_ort` 测试名称划分，新添加的实际 ORT 测试应沿用该命名。可继续直接运行 `uv run --locked python -m pytest` 来选择文件或单项测试。

所有默认模型配置与合成 fixture YAML 均省略 `inputs` / `outputs`。真实模型由引擎读取；合成测试由测试助手注入固定契约，或用实际 CPU ORT 的合成图读取。合成测试和 smoke 的固定张量契约位于 `tests/model_contracts.py`；RF-DETR / OM 的处理步骤分别使用 `tests/fixtures/rfdetr_synthetic.yaml` / `tests/fixtures/rfdetr_om_synthetic.yaml`。RT-DETR 符号输出和固定/动态 batch 另有专门回归。

覆盖 OM 自动发现、静态 batch 冲突、原始/解析后元数据日志、RT-DETR 符号输出和 RF-DETR 统一阈值，以及 RF-DETR/YOLO/RT-DETR/YuNet/SFace 前后处理、处理器注入、固定/动态 batch、padding、非法输入、provider失败和模型I/O契约。ort 使用实际 CPU ORT 与合成模型，不代表 GPU、真实权重或识别效果验证。

## YOLO11n 真实模型

已有 `models/yolo11n.onnx` 和 `artifacts/yolo11-test/bus.jpg` 时，可跳过下载和导出。
官方权重来源为 [Ultralytics release](https://github.com/ultralytics/assets/releases/tag/v8.4.0)，
图片来自 [官方 bus.jpg](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)。

### 下载与导出

在独立导出环境中安装工具（不需要加入 SDK 生产依赖）：

```bash
python -m pip install 'ultralytics==8.4.147' 'torch==2.14.0' 'torchvision==0.29.0' 'onnx==1.22.0'
mkdir -p models artifacts/yolo11-test
curl -fL --retry 2 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt -o models/yolo11n.pt
curl -fL --retry 2 https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg -o artifacts/yolo11-test/bus.jpg
python - <<'PY'
from ultralytics import YOLO

YOLO("models/yolo11n.pt").export(
    format="onnx", imgsz=640, batch=1, dynamic=False,
    nms=False, half=False, simplify=False, opset=17, device="cpu",
)
PY
```

以上命令固定导出依赖版本；不同平台需安装其支持的 PyTorch wheel。
导出文件为 `models/yolo11n.onnx`，输入 `images [1,3,640,640]`，输出 `output0 [1,84,8400]`，均为 float32。
配置 `configs/yolo11.yaml` 已包含 COCO 80 类和 CPU 参数。

### SDK 推理

在已安装 SDK 和 CPU ORT 的环境中执行：

```bash
uv run --locked python examples/yolo11.py  # 将 image_path 改为 artifacts/yolo11-test/bus.jpg
```

### 与官方 ONNX 实现对照

使用下文 `scripts/verify_predictions.py`，它固定 640×640 letterbox 和参考参数，并将权重/图片哈希及误差记录到报告。无需手动复制完整比较代码。

## RF-DETR 真实模型

下载[官方 Nano COCO checkpoint](https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth)至 `models/checkpoint_best_regular.pth`，安装 `rfdetr[onnx]==1.11.2`、`onnxruntime` 后运行：

```bash
python scripts/export_rfdetr.py
```

Nano 导出工具与官方 PyTorch 模型比较原始输出。

脚本校验官方 MD5，生成 `models/rfdetr-nano.onnx` 和 `.export.json`，并校验 ONNX 结构及原模型输出数值一致性。`configs/rfdetr.yaml` 对应 384×384、batch=1、COCO 原始类别编号。测试图片使用 `tests/bus.jpg`。
模型契约见 [配置说明](configuration.md#rf-detr)。选择CUDA时修改configs/rfdetr.yaml中的provider：

```bash
uv run --locked python examples/rfdetr.py  # 在示例中修改 yaml_path 与 image_path
```

CPU 可在内存中覆盖运行时配置：

```python
import cv2
from ai_inference import Config, Model

config = Config.from_yaml("configs/rfdetr.yaml")
data = config.model_dump()
data["runtime"]["options"].update(providers="cpu")
model = Model(Config.model_validate(data))
image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("image.jpg")
print(model.predict(image))
```

## CPU 真实权重正确性对照

已提供不下载模型的独立对照脚本。所有参数必须指向已有本地文件，报告记录权重/样本哈希、参考版本、误差与成功/失败状态；任何差异超出容差会返回非零退出码。验证的是给定样本上的实现一致性，不是数据集精度或GPU验收。

```bash
# YOLO：对照 Ultralytics ONNX predictor；在独立参考环境安装 ultralytics==8.4.147 及 SDK/CPU ORT
python scripts/verify_predictions.py \
  --config configs/yolo11.yaml --model models/yolo11n.onnx \
  --image artifacts/yolo11-test/bus.jpg --report artifacts/parity/yolo.json

# YuNet：对照 OpenCV CPU DNN 原始输出和 FaceDetectorYN
uv run --locked python scripts/verify_faces.py --family yunet \
  --config configs/yunet.yaml --model models/face_detection_yunet_2023mar.onnx \
  --image /path/to/face_sample.png --report artifacts/parity/yunet.json

# SFace：对照 FaceRecognizerSF 对齐、原始特征和归一化特征
uv run --locked python scripts/verify_faces.py --family sface \
  --config configs/sface.yaml --model models/face_recognition_sface_2021dec.onnx \
  --image /path/to/face_sample.png --landmarks /path/to/landmarks.json \
  --report artifacts/parity/sface.json
```

SFace landmarks 文件为原图坐标 JSON `[[x,y],...]`，按 YuNet 五点顺序。对照使用相同解码图像，模型文件由命令参数显式指定并记录实际 SHA256，权重来源和使用许可由部署方核实。YuNet 的float框NMS与官方整数框NMS有差异，边界样本对照可能失败，应检查实际保留候选，不能因此扩大容差掩盖差异。

RT-DETR 参考源码固定为 [df2a96c8](https://github.com/lyuwenyu/RT-DETR/tree/df2a96c8c3a098567a1149bf0248da6efe6d495d)，checkpoint 使用官方 README 链接的 `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`。在独立 Python3.10 环境准备 torch2.0.1、torchvision0.15.2、numpy1.26.4、onnx1.14.0、CPU onnxruntime1.20.1、transformers4.34.1、pycocotools、PyYAML、scipy、Pillow、OpenCV与pydantic2：

```bash
python scripts/export_rtdetr.py \
  --reference-source /path/to/pinned/RT-DETR/rtdetr_pytorch \
  --checkpoint /path/to/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth \
  --output models/rtdetr-deployed-v1.onnx
python scripts/verify_rtdetr.py \
  --reference-source /path/to/pinned/RT-DETR/rtdetr_pytorch \
  --checkpoint /path/to/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth \
  --config configs/rtdetr.yaml --model models/rtdetr-deployed-v1.onnx \
  --image artifacts/yolo11-test/bus.jpg --report artifacts/parity/rtdetr.json
```

导出禁止额外 backbone 下载，严格加载完整checkpoint。torch2.0.1不支持将优化SDPA直接导出opset16；导出时使用等价显式attention路径，最终通过与未修改官方 eager 模型的对照检查确认结果。导出记录写入同目录 `.export.json`。验证包括官方 torchvision/Pillow 输入张量、原图 width/height、全部300个候选类别/框/分数、阈值/类别过滤及动态batch。低分候选可能因浮点近似交换TopK排序，按类别和几何做一对一匹配，不忽略候选差异。

SDK 的置信度边界统一为 `>=`；与采用严格 `>` 的参考实现对照时，分数恰好等于阈值的候选可能不同。

CUDA / TensorRT 与 Ascend OM 需要在目标设备上单独验证初始化、warmup、结果一致性及性能。
