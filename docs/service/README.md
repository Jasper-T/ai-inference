# 单模型 HTTP 服务

每个进程启动一个 SDK `Model`，多模型路由由上层服务负责。服务不修改 SDK 的处理流程，不自动下载权重、不接收客户端模型路径。所有四种输入共用推理、响应和错误契约。

维护资料：[统一变更记录](../../CHANGELOG.md)、[代码审查与验证记录](review.md)、[统一版本发布规划](release-plan.md)。发布规划列出了迁移状态和剩余验收条件，已确认的边界问题见审查记录。

## 启动

在仓库根目录执行，模型权重和 YAML 应提前准备：

各模型的官方链接、文件名及导出要求见 [模型下载](../model-downloads.md)。

```bash
uv sync --locked --no-default-groups --extra service --extra onnx-cpu
AI_INFERENCE_MODEL_CONFIG=configs/yolo11.yaml \
  uv run --locked --no-sync uvicorn ai_inference.service.app:app --host 127.0.0.1 --port 8000 --workers 1
```

服务代码位于 `src/ai_inference/service/`，随 `ai-inference` 统一发行。`service` extra 安装 FastAPI、Uvicorn 和 multipart 依赖，可与任一后端 extra 组合。环境变量中的配置路径相对进程工作目录，YAML 内的权重路径仍相对 YAML 目录。

发行版安装使用 `pip install 'ai-inference[service,onnx-cpu]'`；此前的 `0.2.0` 制品尚无 `service` extra，须使用包含本次迁移的新版本。安装后可从其他目录运行 `uvicorn ai_inference.service.app:app`，部署时建议使用绝对配置路径。纯 SDK 使用者无需安装 `service` extra，基础导入不加载 HTTP 框架。

启动阶段加载模型，按模型 YAML 的 `runtime.warmup` 设置预热；加载失败则启动失败。不要使用多个 worker 扩容同一设备：每个 worker 都会独立加载模型。单进程只接纳一个推理请求，包含输入解析、下载、解码、推理和响应发送；忙时立即返回 503，不排队。健康检查不等待推理。推理在工作线程执行，SDK 当前没有统一的显式 `close()` 接口，停止时释放服务持有的模型引用，设备资源最终由进程退出回收。

本版不执行模型热更新、跨请求动态合批或推理中断；上层按服务实例分发。URL 下载和慢请求可占用当前执行名额，上层应设置请求读取超时、并发连接数及访问控制。请求断开不等于底层设备推理已取消。CUDA/OM 的实际设备行为仍需部署环境验收。

## 接口

交互文档：`/docs`；机器可读契约：`/openapi.json`。

| 接口 | 输入 |
| --- | --- |
| `POST /v1/predict/file` | multipart，重复 `files` 字段可上传多张图片 |
| `POST /v1/predict/base64` | JSON `{"images":[{"data":"…"}]}`，支持纯 Base64 或 `data:image/png;base64,…` |
| `POST /v1/predict/url` | JSON `{"images":[{"url":"https://example.com/image.jpg"}]}` |
| `POST /v1/predict/bytes` | 单张编码图片，Content-Type 为 `application/octet-stream` 或对应的 `image/jpeg`、`image/png`、`image/webp`、`image/bmp` |
| `GET /v1/model` | 当前模型名称、后端、I/O、物理 batch 策略及服务输入限制，不暴露权重路径 |
| `GET /health/live` | 进程可响应 |
| `GET /health/ready` | 模型已加载；忙时仍就绪 |

```bash
curl -F 'files=@tests/bus.jpg' http://127.0.0.1:8000/v1/predict/file
curl -H 'Content-Type: image/jpeg' --data-binary @tests/bus.jpg \
  http://127.0.0.1:8000/v1/predict/bytes
curl -H 'Content-Type: application/json' \
  -d '{"images":[{"url":"https://example.com/image.jpg"}]}' \
  http://127.0.0.1:8000/v1/predict/url
```

Python 调用示例见 [examples/service.py](../../examples/service.py)，支持四种输入及 SFace 关键点。启动服务后运行 `uv run --locked --no-sync python examples/service.py`；更多参数见 [示例说明](../../examples/README.md#http-服务调用)。

图片限定为单帧 JPEG、PNG、WEBP、BMP；按图片内容解码，转换成 HWC uint8 三通道。透明度丢弃，不自动应用 EXIF 旋转；关键点与输出坐标对应编码图片的原始方向。默认 BGR，`AI_INFERENCE_INPUT_COLOR=RGB` 可切换，但必须匹配模型 `preprocess.source_color`（未声明时按 BGR），不一致会拒绝启动。bytes 不支持裸像素缓冲区。

所有输入校验通过后才调用模型；任一输入错误则整批拒绝，无部分结果。响应 `results` 与输入顺序一致，每个结果为：

```json
{"results":[{"type":"detection","original_shape":[640,480],"boxes":[],"scores":[],"class_ids":[]}]}
```

检测可包含 `keypoints` 和布尔 `masks`；特征结果为 `type: embedding`、`original_shape` 和 `embedding`。掩码以原图大小的非零布尔数组输出。模型的物理 batch 与一次 HTTP 请求的图片数量是两回事，SDK 继续负责拆分和补齐。

## 人脸关键点

需要原图五点的 SFace 配置要求每张图片提供 `keypoints`，顺序沿用 SDK：右眼、左眼、鼻尖、右嘴角、左嘴角。Base64/URL 在每个图片对象中附加该字段：

```json
{"images":[{"data":"…","keypoints":[[10,10],[20,10],[15,15],[10,20],[20,20]]}]}
```

`configs/sface.yaml` 的 `face_align.aligned_input` 为 `false`，因此 `keypoints` 必填。每张图片各自携带五组 `[x,y]`，使用编码图片原始方向的像素坐标，不是归一化坐标；可使用 YuNet 对同一原图输出的 `keypoints`。Base64 请求使用 `data`，URL 请求使用 `url`：

```json
{
  "images": [
    {
      "url": "https://example.com/face.jpg",
      "keypoints": [[38,52],[74,52],[56,72],[42,92],[71,92]]
    }
  ]
}
```

上面的数值仅展示格式，实际坐标必须来自所提交的人脸图片。一个图片对象对应一次人脸特征提取；同一原图有多张人脸时，为每个人脸分别提交图片与对应五点。响应为每个人脸的 `type: embedding` 和 128 维 `embedding`，服务不接收姓名或人员 ID。

文件接口可传 multipart `metadata` 字段，其内容为与文件等长的 JSON 数组，例如 `[{"keypoints":[…]}]`。bytes 接口用 URL 编码后的 `metadata` 查询参数传单个 JSON 对象。已对齐图片须在模型 YAML 的 `face_align.aligned_input` 显式设为 `true`。服务不会自动检测人脸或关联人员。

| SFace 输入方式 | 图片字段 | 关键点位置 |
| --- | --- | --- |
| Base64 JSON | `images[i].data` | 请求体 `images[i].keypoints` |
| URL JSON | `images[i].url` | 请求体 `images[i].keypoints` |
| multipart 文件 | 重复 `files` 字段 | 请求体 `metadata` 中与文件同索引的对象 |
| bytes | 原始编码图片请求体 | 查询参数 `metadata` 中的对象 |

bytes 的请求体保留为图片字节，因此关键点通过查询参数传入。若希望图片和关键点都在请求体内，使用 Base64、URL 或 multipart 接口。

缺少五点、点数错误或非有限坐标返回 422。当前退化五点仍可能返回 500，已列为[发布前修复项](review.md#p2客户端关键点错误返回模型故障-500)。已对齐输入不传关键点时须是 112×112 裁剪；传入关键点则仍执行对齐。

## 输入限制与 URL

环境变量前缀均为 `AI_INFERENCE_`：

| 变量后缀 | 默认值 | 含义 |
| --- | --- | --- |
| `MODEL_CONFIG` | 必填 | 单模型 YAML 路径 |
| `INPUT_COLOR` | BGR | 解码后的图像颜色 |
| `MAX_REQUEST_BYTES` | 25165824 | 整个 HTTP 请求体上限，包含 Base64/multipart 包装 |
| `MAX_IMAGE_BYTES` | 8388608 | 单张编码图片字节上限 |
| `MAX_IMAGE_PIXELS` | 16000000 | 单张图片像素上限，在展开像素前校验 |
| `MAX_BATCH_PIXELS` | 32000000 | 一次请求总像素上限 |
| `MAX_BATCH_SIZE` | 8 | 一次请求最大图片数 |
| `URL_TIMEOUT_SECONDS` | 10 | 网络操作超时及下载循环时间预算 |

URL 仅允许标准端口的公网 HTTP/HTTPS。拒绝凭据、内网/回环/链路本地地址、IPv4 映射或隧道 IPv6 地址，以及混合公网和私网 DNS 结果。连接固定到已检查 IP，HTTPS 保留原域名进行证书校验；不使用环境代理、不跟随重定向、不解压 HTTP Content-Encoding。远端须返回 200，实际下载字节也受限制。DNS 查询耗时由操作系统解析器控制，网络超时不构成整个 HTTP 请求的硬截止时间；推理不设置伪取消超时。

错误统一返回 `{"error":{"message":"…","index":null}}`；逐图输入错误的 `index` 为从 0 开始的输入索引。422 表示输入无效，413 表示大小/像素超限，415 表示 bytes 媒体类型错误，503 表示模型未就绪或忙，504 表示下载超时，500 表示模型执行失败。模型异常详情仅记录在服务日志中。调用方是否重试由上层决定。

## 验证

```bash
uv sync --locked --extra service
uv run --locked --no-sync python -m pytest tests/test_service.py -q
```

服务回归使用可控模型替身及模拟下载，覆盖四种输入一致性、颜色、原子批量、请求/图片限制、URL 防护和单请求准入；不代表真实权重精度或目标设备并发验收。

## 运行与维护

启动前确认服务与 SDK 安装版本、模型 YAML、权重文件和实际后端依赖。记录这组版本与配置，便于重现和回滚。配置路径相对启动目录，建议部署时使用绝对路径；模型文件路径按 YAML 所在目录解析。

启动后先检查 `/health/ready`，再用固定样本验证结果；`/health/live` 只说明进程可响应。忙时 readiness 仍成功，不表示当前实例有空闲名额。启动加载或预热失败，应检查服务日志、配置与设备环境。

单个进程从接收图片到响应发送都占用唯一名额，慢上传、下载和慢客户端会延长占用。网关应设置客户端读取超时、请求体限制、连接数限制和访问控制；仅设置上游响应超时不能约束客户端慢上传。只开放所需的预测接口；若不需要 URL 输入，可在网关限制该路由，当前服务没有关闭 URL 输入的环境开关。

维护时关注请求量、503/504/500 比例、端到端时延、进程内存和设备占用。这些指标需由网关或部署监控采集，当前服务没有内置 metrics 接口或请求级结构化计时。413/415/422 通常要求调用方修正输入；503 可由上层按容量重试，避免立即无限重试；500 应结合服务日志定位。

停止前先从上层路由摘除实例，等待已接纳请求完成，再停止进程。断开 HTTP 请求不会保证设备推理停止，不能以此作为同一设备重入依据。SDK 资源最终随进程退出回收。升级与回滚按[发布规划](release-plan.md#回滚与后续版本)保存并恢复完整版本、依赖和配置组合。
