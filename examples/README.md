# SDK 与 HTTP 服务调用示例

模型准备见 [官方模型下载与导出说明](../docs/model-downloads.md)，包含五个默认模型的链接和本地文件名。

示例直接读取本仓库 `models/` 中的本地权重，统一使用 `tests/bus.jpg`；不下载模型或图片。先运行 `uv sync --locked` 安装 SDK 与 CPU ONNX Runtime。以下命令从仓库根目录执行；代码按示例文件位置定位仓库，从其他目录使用脚本绝对路径运行也可：

```bash
uv run --locked python examples/yolo11.py
uv run --locked python examples/rfdetr.py
uv run --locked python examples/rtdetr.py
uv run --locked python examples/yunet.py
uv run --locked python examples/sface.py
```

每个文件直接展示 `import → ROOT → yaml_path → Model → image_path → predict → Plotter.save`，没有额外封装。YAML中的模型路径相对于配置文件解析，均指向本仓库 `models/`。六份默认配置不手写 I/O 张量契约，由 ONNX / OM 引擎发现；步骤名称仍须匹配真实张量。RT-DETR 示例使用动态 batch（每批上限 2），其他 ONNX 示例为固定 batch=1。图片为 BGR，转换和归一化由对应 YAML/处理器负责。结果固定写入本仓库 `results/example_<模型>.png`。

| 示例 | YAML | 本地模型 |
| --- | --- | --- |
| yolo11.py | configs/yolo11.yaml | models/yolo11n.onnx |
| rfdetr.py | configs/rfdetr.yaml | models/rfdetr-nano.onnx |
| rtdetr.py | configs/rtdetr.yaml | models/rtdetr-deployed-v1.onnx |
| yunet.py | configs/yunet.yaml | models/face_detection_yunet_2023mar.onnx |
| sface.py | configs/sface.yaml | models/face_recognition_sface_2021dec.onnx |

SFace不是检测模型，额外调用本地YuNet获取五点，同样使用 `tests/bus.jpg`。检测到人脸时提取第0张（最高分）人脸的特征；没有有效人脸时保存not_run说明。context显示人脸检测几何，特征不是身份标签。

示例不下载权重。SDK配置只管理推理所需的模型路径、处理器和输入输出契约；模型来源、版本和许可如有管理需求，由上层应用负责。

名称只作为 Plotter 的可选 names 输入，不写入推理 YAML；YOLO 示例使用 `{0: "person", 5: "bus"}`，省略或未映射的 ID 直接显示数字标签。

标注默认不添加顶部标题或统计，类别颜色与Ultralytics的20色调色板一致，同类保持相同颜色。

## HTTP 服务调用

[service.py](service.py) 调用已经启动的服务，客户端不加载模型。示例使用 `httpx`，仓库默认开发依赖已包含它；独立客户端环境可执行 `pip install httpx`，不需要安装 SDK 或推理后端。

先在一个终端启动 YOLO CPU 服务（需本地权重）：

```bash
uv sync --locked --extra service
AI_INFERENCE_MODEL_CONFIG=configs/yolo11.yaml \
  uv run --locked --no-sync uvicorn ai_inference.service.app:app --host 127.0.0.1 --port 8000 --workers 1
```

另一个终端从仓库根目录调用，默认上传 `tests/bus.jpg` 并打印 JSON 响应：

```bash
uv run --locked --no-sync python examples/service.py
uv run --locked --no-sync python examples/service.py --source base64
uv run --locked --no-sync python examples/service.py --source bytes --image /path/to/image.jpg
uv run --locked --no-sync python examples/service.py --source url --image-url https://example.com/image.jpg
```

URL 示例中的地址需替换为实际可下载的公网图片地址。服务地址通过 `--base-url http://host:8000` 指定。调用按顺序执行，不自动重试；失败显示 HTTP 状态及服务错误响应。客户端读取超时为 60 秒，超时或断连不代表服务推理已取消。

调用 SFace 时，先把服务启动配置换成 `configs/sface.yaml`。准备与图片对应的五点 JSON 文件，例如 `/path/to/keypoints.json`，其格式为 `[[x,y],[x,y],[x,y],[x,y],[x,y]]`；点的顺序是右眼、左眼、鼻尖、右嘴角、左嘴角，使用原图像素坐标。然后执行：

```bash
uv run --locked --no-sync python examples/service.py --source base64 \
  --image /path/to/face.jpg --keypoints /path/to/keypoints.json
```

`--keypoints` 对四种输入均有效：Base64/URL 写入逐图 JSON，file 写入 multipart metadata，bytes 写入 metadata 查询参数。普通 YOLO 请求不传该参数。完整接口与限制见 [服务说明](../docs/service/README.md)。
