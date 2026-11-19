# ai-inference

通用 Python 模型推理 SDK，通过有序 Pipeline 执行前后处理，Engine 执行模型，Batcher 负责分批、补齐和结果合并。发行包名为 `ai-inference`，Python 导入名为 `ai_inference`。

支持 ONNX Runtime 和静态 Ascend OM，提供 RF-DETR、YOLOv8 / YOLO11、RT-DETR、YuNet 和 SFace 的处理步骤。检测结果可包含关键点和掩码；当前没有内置分割模型。HTTP 接口由可选的 [service](docs/service/README.md) 模块提供，与 SDK 使用同一发行包及版本；通过 `service` extra 安装 HTTP 依赖。

## 安装

Python ≥3.10，已验证 Python 3.10 / 3.11 / 3.12。当前源码版本为 **0.2.3**，变更见 [CHANGELOG](CHANGELOG.md)。推荐从源码使用 uv 安装，Quick Start 选择 CPU；GPU 或 Ascend 部署应按目标环境选择后端。

| 安装方式 | 适用场景 | 说明 |
| --- | --- | --- |
| [源码 + uv（推荐）](docs/installation.md#11-使用-uv推荐) | 示例体验、开发、可复现部署 | 使用仓库锁文件，显式选择 Python 与后端 |
| [源码 + pip](docs/installation.md#12-使用-pip-本地安装) | 已有 Python 环境 | 支持普通安装和可编辑安装 |
| [Release wheel](docs/installation.md#21-从-release-安装) | 安装已发布 SDK | 下载指定版本 wheel，选择后端 extra |
| [自行构建 wheel](docs/installation.md#22-自行构建) | 内部分发、定制版本 | 从源码构建后安装 |
| [离线安装](docs/installation.md#23-离线安装) | 无网络的目标环境 | 提前准备 SDK 和全部依赖 wheel |

后端选择、HTTP 服务、Ascend 厂商包和安装检查见 [完整安装指南](docs/installation.md)；CUDA、cuDNN、CANN 与平台约束见 [配置说明](docs/configuration.md#环境与后端选择)。

## Quick Start

以下命令在 macOS / Linux shell 中执行，使用 CPU 运行 YuNet 人脸检测，无需自行导出模型。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再获取源码并安装运行依赖：

```bash
git clone https://github.com/Jasper-T/ai-inference.git
cd ai-inference
uv sync --locked --python 3.12 --no-default-groups --extra onnx-cpu
```

下载与示例配置对应的 YuNet ONNX 权重，再运行仓库自带图片示例：

```bash
mkdir -p models
curl -fL --retry 3 \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o models/face_detection_yunet_2023mar.onnx
uv run --locked --no-sync python examples/yunet.py
```

结果保存为 `results/example_yunet.png`；图片中没有符合阈值的人脸时，检测结果可以为空。模型来源、SHA256 和许可见 [模型下载](docs/model-downloads.md)。模型权重不随 SDK 分发；wheel 用户需另行取得对应版本的配置和示例，具体见安装指南。

需要 HTTP 服务时，安装 `service` 与一个后端，例如 `uv sync --locked --python 3.12 --no-default-groups --extra service --extra onnx-cpu`，再按 [HTTP 服务](docs/service/README.md) 启动。

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
uv run --locked --no-sync python scripts/predict_all.py --image tests/bus.jpg --results results
```

SFace 使用 YuNet 检测的五点；没有有效人脸时输出 `not_run` 状态。最小调用见 [examples](examples/README.md)，完整场景工具见 [scripts](scripts/README.md)。

## 开发与验证

从仓库根目录使用 [uv](https://docs.astral.sh/uv/) 安装源码和开发依赖。默认 `dev` 组包含测试和发布检查工具，不安装 CPU/GPU 推理后端。CPU、CUDA、Ascend 均显式选择后端；开发环境可保留默认组，纯运行环境添加 `--no-default-groups`。安装方式见 [完整安装指南](docs/installation.md)，后端同步规则见 [环境与后端选择](docs/configuration.md#环境与后端选择)。

```bash
uv sync --locked --extra service --extra onnx-cpu
uv lock --check
uv run --locked --no-sync ruff check src tests scripts examples
uv run --locked --no-sync ruff format --check src tests scripts examples
uv run --locked --no-sync python scripts/test.py all
uv build --build-constraint .github/build-constraints.txt
```

以上命令同时验证 SDK 和服务；不启用 `service` extra 时服务测试会跳过。无 ORT 的 Ascend 开发环境运行 `scripts/test.py quick` 与 `smoke`；完整 `all`/`ort` 需要所选 ONNX Runtime 后端。服务说明见 [HTTP 服务](docs/service/README.md)。新增运行依赖用 `uv add`，开发工具用 `uv add --dev`，测试依赖用 `uv add --group test`，并提交 `pyproject.toml` 和 `uv.lock`。修改 Ascend 厂商来源后先运行 `uv lock`，再运行 `python scripts/ascend_requirements.py`，从锁文件哈希同步生成 pip 安装文件。

## 文档

- [安装指南](docs/installation.md)：源码、Release wheel、自行构建、离线安装与后端选择。
- [CHANGELOG](CHANGELOG.md)：版本变化与升级迁移。
- [模型下载](docs/model-downloads.md)：各配置对应的官方权重、ONNX 文件与导出方式。
- [源码目录与职责](docs/architecture.md)：组件职责与执行边界。
- [可组合处理管道](docs/pipeline.md)：步骤、Context、几何信息与设备策略。
- [配置与模型契约](docs/configuration.md)：张量、batch 和运行时选项。
- [处理器与扩展](docs/extension.md)：自定义步骤、处理器、Engine 和 Predictor。
- [日志与性能测量](docs/logging-and-benchmark.md)：日志上下文与 benchmark。
- [测试与复现](docs/testing.md)：自动化测试、真实模型导出与参考对照。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方运行时和模型权重各自的许可须单独遵守。
