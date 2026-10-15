# ai-inference

通用 Python 模型推理 SDK，通过有序 Pipeline 执行前后处理，Engine 执行模型，Batcher 负责分批、补齐和结果合并。发行包名为 `ai-inference`，Python 导入名为 `ai_inference`。

支持 ONNX Runtime 和静态 Ascend OM，提供 RF-DETR、YOLOv8 / YOLO11、RT-DETR、YuNet 和 SFace 的处理步骤。检测结果可包含关键点和掩码；当前没有内置分割模型。HTTP 接口由可选的 [service](docs/service/README.md) 模块提供，与 SDK 使用同一发行包及版本；通过 `service` extra 安装 HTTP 依赖。

## 安装

Python ≥3.10。当前版本为 **0.2.2**，变更见 [CHANGELOG](CHANGELOG.md)。以下命令用于 `0.2.2` 发行版；发布前可先[从源码安装](#开发与验证)。

选择一个推理后端，在独立虚拟环境中安装：

```bash
# CPU，包括 macOS
python -m pip install 'ai-inference[onnx-cpu]==0.2.2'

# NVIDIA CUDA ≥12.8 / cuDNN 9，Linux x86_64 或 Windows x64
python -m pip install 'ai-inference[onnx-gpu-cu12]==0.2.2'

# NVIDIA CUDA ≥13.0 / cuDNN 9，Python ≥3.11
# Linux x86_64 / ARM64 或 Windows x64
python -m pip install 'ai-inference[onnx-gpu-cu13]==0.2.2'
```

**Linux ARM64（aarch64）的 NVIDIA GPU 环境需要 CUDA ≥13.0、cuDNN 9、Python ≥3.11 和 glibc ≥2.34。** Python 3.10 或 CUDA 12 的 ARM64 环境需要设备厂商提供的兼容运行时或自行构建。Jetson 还需核对 JetPack、驱动和 GPU 型号；ARM64 wheel 本身不保证设备兼容。

GPU extra 不会检测本机 CUDA，也不安装 NVIDIA 驱动或系统 CUDA/cuDNN。不支持的 Python/平台组合可能跳过 GPU 依赖；安装成功后仍须验证真实 CUDA 推理。详细要求见 [CUDA 部署说明](docs/configuration.md#cuda-部署与失败)。旧 `onnx-gpu` 保留为兼容入口，新部署建议选择明确的 CUDA 系列。请只安装一个后端，不要混装 CPU 和 GPU 运行时。

需要 HTTP 服务时，组合 `service` 与所选后端：

```bash
python -m pip install 'ai-inference[service,onnx-cpu]==0.2.2'
# GPU 服务也可组合 service,onnx-gpu-cu12 或 service,onnx-gpu-cu13
```

基础安装不引入 FastAPI/Uvicorn，`service` extra 安装 HTTP 依赖。启动方法见 [HTTP 服务](docs/service/README.md)。

Ascend 需要自行准备 CANN、驱动和设备环境，并安装 SDK 与独立的厂商依赖：

```bash
python -m pip install 'ai-inference==0.2.2'
# 从对应版本源码取得 requirements/ascend.txt 后执行
python -m pip install -r requirements/ascend.txt
```

[Ascend 依赖文件](requirements/ascend.txt) 支持 Linux x86_64 / aarch64、CPython 3.10–3.12；其他环境需自行准备兼容运行时。`ascend` 保留为空的兼容 extra。模型文件不随包分发，示例 YAML 见 [configs](configs)，权重准备见 [模型下载](docs/model-downloads.md)。

## 检测调用

准备本地模型文件和图片，修改 YAML 中的模型路径与类别配置：

```python
import cv2
from ai_inference import Model

model = Model("configs/yolo11.yaml")
image = cv2.imread("image.jpg")  # HWC uint8；示例配置使用 BGR
if image is None:
    raise FileNotFoundError("image.jpg")

result = model.predict(image)[0]
print(result.boxes, result.scores, result.class_ids)
results = model.predict([image, image])  # 与输入顺序一致
```

检测结果为 `list[Prediction]`，空输入返回空列表。框为 float32 `[K,4]` 原图像素 xyxy；分数为 `[K]`，类别 ID 为 int64，`original_shape` 为 `(H,W)`。可选关键点为 `[K,P,2]` 原图 xy，掩码为 `[K,H,W]`。

也可分步执行，`RawOutput` 携带当前请求的预处理元数据：

```python
prepared = model.preprocess([image, image])
raw = model.infer(prepared)
results = model.postprocess(raw)
```

## 人脸关键点与特征

YuNet 提供人脸框和五点关键点，SFace 根据五点对齐后提取特征：

```python
from ai_inference import FaceInput, Model

detector = Model("configs/yunet.yaml")
encoder = Model("configs/sface.yaml")
faces = detector.predict(image)[0]
if len(faces.boxes):
    feature = encoder.predict(FaceInput(image, faces.keypoints[0]))[0]
    print(feature.embedding)  # float32 [D]，默认 L2 归一化
```

已经对齐的 112×112 裁剪须在 `face_align` 步骤中显式配置 `aligned_input: true`。人脸质量评估、人员关联和特征库检索由应用实现。模型输入契约见 [配置说明](docs/configuration.md#结果人脸输入与扩展模型契约)。

## 配置与失败行为

| 示例配置 | 模型契约 |
| --- | --- |
| [yolo11.yaml](configs/yolo11.yaml) | YOLO11n COCO，固定 batch=1 |
| [rfdetr.yaml](configs/rfdetr.yaml) | RF-DETR Nano COCO，384×384，固定 batch=1 |
| [rtdetr.yaml](configs/rtdetr.yaml) | 原始 RT-DETR deploy 导出，动态 batch 上限 2 |
| [yunet.yaml](configs/yunet.yaml) | YuNet 2023mar 五点检测，640×640 |
| [sface.yaml](configs/sface.yaml) | SFace 2021dec 特征提取 |
| [rfdetr_om.yaml](configs/rfdetr_om.yaml) | 官方 RF-DETR Nano 转换的静态 OM / AIPP |

ONNX 示例默认使用 CPU 和 BGR 图像。CPU / CUDA / TensorRT 共用模型契约，通过 `runtime.options.providers` 选择 `cpu`、`cuda`、`trt` 或有序列表 `[trt, cuda, cpu]`。单项严格使用指定后端；列表允许按序初始化降级和算子回退，执行失败直接抛错。

相对模型路径以 YAML 所在目录解析。ONNX / OM 自动读取真实输入输出的名称、dtype、shape，YAML 中的 `inputs` / `outputs` 可作为可选约束；预后处理步骤仍须显式配置并匹配模型契约。静态模型不能通过修改 YAML 变成动态模型。默认执行 3 次引擎 warmup，可用 `runtime.warmup` / `warmup_runs` 调整。

公开张量和内置前后处理使用 CPU NumPy。权重不随仓库分发，不自动下载；来源、哈希和许可由部署方核实。详细规则见 [配置与模型契约](docs/configuration.md)。

## 标注与批量示例

`Plotter` 只消费图像和既有结果，可独立绘制框、关键点和掩码：

```python
from ai_inference import Plotter

Plotter().save("results/annotation.png", image, result, names={0: "person"})
```

类别名称只影响标注，未提供名称时显示原始 ID。默认不添加顶部标题或统计；可通过 `title`、`note`、`show_summary=True` 启用。

准备五个模型后，可对同一图片执行推理并保存 PNG / JSON：

```bash
uv run --locked python scripts/predict_all.py --image tests/bus.jpg --results results
```

SFace 使用 YuNet 检测的五点；没有有效人脸时输出 `not_run` 状态。最小调用见 [examples](examples/README.md)，完整场景工具见 [scripts](scripts/README.md)。

## 开发与验证

从仓库根目录使用 [uv](https://docs.astral.sh/uv/) 安装源码和开发依赖。默认开发组包含 CPU ONNX Runtime；GPU / Ascend 环境使用 `--no-default-groups` 并只选择对应后端 extra，具体命令见 [运行环境配置](docs/configuration.md#cuda-部署与失败)。

```bash
uv sync --locked --extra service
uv lock --check
uv run --locked --no-sync ruff check src tests scripts examples
uv run --locked --no-sync ruff format --check src tests scripts examples
uv run --locked --no-sync python scripts/test.py all
uv build
```

以上命令同时验证 SDK 和服务；只验证基础 SDK 时可不启用 `service` extra，服务测试会在缺少 HTTP 依赖时跳过。服务说明见 [HTTP 服务](docs/service/README.md)。新增运行依赖用 `uv add`，开发工具用 `uv add --dev`，测试依赖用 `uv add --group test`，并提交 `pyproject.toml` 和 `uv.lock`。

## 文档

- [CHANGELOG](CHANGELOG.md)：版本变化与升级迁移。
- [模型下载](docs/model-downloads.md)：各配置对应的官方权重、ONNX 文件与导出方式。
- [源码目录与职责](docs/architecture.md)：组件职责与执行边界。
- [可组合处理管道](docs/pipeline.md)：步骤、Context、几何信息与设备策略。
- [配置与模型契约](docs/configuration.md)：张量、batch 和运行时选项。
- [处理器与扩展](docs/extension.md)：自定义步骤、处理器、Engine 和 Predictor。
- [日志与性能测量](docs/logging-and-benchmark.md)：日志上下文与 benchmark。
- [测试与复现](docs/testing.md)：自动化测试、真实模型导出与参考对照。
- [发布工作流](docs/releases/actions.md)：构建、发布与失败重试。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方运行时和模型权重各自的许可须单独遵守。
