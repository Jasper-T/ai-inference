# ai-inference

通用 Python 模型推理 SDK，通过有序 Pipeline 执行前后处理，Engine 执行模型，Batcher 负责分批、补齐和结果合并。发行包名为 `ai-inference`，Python 导入名为 `ai_inference`。

支持 ONNX Runtime 和静态 Ascend OM，提供 RF-DETR、YOLOv8 / YOLO11、RT-DETR、YuNet 和 SFace 的处理步骤。检测结果可包含关键点和掩码；当前没有内置分割模型。HTTP 接口由可选的 [service](docs/service/README.md) 模块提供，与 SDK 使用同一发行包及版本；通过 `service` extra 安装 HTTP 依赖。

## 安装

Python >= 3.10。项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖、环境、测试和构建。先安装 uv，再从仓库根目录运行：

```bash
# 开发环境：SDK、CPU ONNX Runtime、pytest、ONNX 和 Ruff
uv sync --locked

# CPU 运行环境，包括 macOS
uv sync --locked --no-default-groups --extra onnx-cpu

# NVIDIA Linux / Windows 运行环境
uv sync --locked --no-default-groups --extra onnx-gpu

# Ascend Linux 运行环境，CANN、驱动和设备环境须单独准备
uv sync --locked --python 3.12 --no-default-groups --extra ascend
```

需要 HTTP 服务时可组合安装：

```bash
pip install 'ai-inference[service,onnx-cpu]'
# 在本仓库安装 CPU 服务运行环境
uv sync --locked --no-default-groups --extra service --extra onnx-cpu
```

上述安装方式需要包含 `service` extra 的新版本；此前的 `0.2.0` 制品不包含该 extra。基础安装不引入 FastAPI/Uvicorn，服务代码包含在统一 wheel 中。

三个后端 extras 互斥。默认 dev/test 组包含 CPU ONNX Runtime，GPU / Ascend 环境须关闭默认组。切换后端建议使用独立环境。Ascend extra 按 Linux x86_64 / aarch64、CPython 3.10–3.12 选择官方 ACLRuntime / ais_bench 0.0.2 wheel，其他环境需自行准备兼容运行时。

当前源码版本为 **0.2.0**，变更和迁移说明见 [CHANGELOG](CHANGELOG.md)。当前快照包含尚未发布的服务等更新，请先按上文从源码安装。后续正式版本将通过 [GitHub Releases](https://github.com/Jasper-T/ai-inference/releases) 提供 wheel、源码包及 `SHA256SUMS`；取得经过验收的 wheel 后可按以下形式安装：

```bash
uv venv --python 3.12
uv pip install './ai_inference-0.2.0-py3-none-any.whl[onnx-cpu]'
```

GPU 和 Ascend 分别使用 `[onnx-gpu]`、`[ascend]`。GPU 运行时必须与 NVIDIA 驱动、CUDA、cuDNN 兼容，见 [CUDA 部署说明](docs/configuration.md#cuda-部署与失败)。

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
