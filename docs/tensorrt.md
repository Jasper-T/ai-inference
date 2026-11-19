# 原生 TensorRT 后端

原生后端使用 NVIDIA TensorRT Python API 加载序列化 engine；`runtime.backend: tensorrt` 与 ONNX Runtime 的 `runtime.options.providers: trt` 是两条独立路径。SDK 和 HTTP 均使用现有 Model 入口，前后处理仍由 Pipeline 配置。

## 安装与适用范围

仓库锁定 TensorRT **10.16.0.72 / CUDA 12**、`cuda-bindings==12.9.4`。可选依赖覆盖 CPython 3.10–3.12、Linux x86_64（glibc ≥2.28）和 Windows AMD64。该版本没有对应的 Linux ARM wheel；macOS、其他架构和 Python 版本不会安装此 extra 中的厂商包。基础 SDK 的 Python 要求不变。

```bash
# 从源码安装运行环境；HTTP 服务额外添加 --extra service
uv sync --locked --python 3.12 --no-default-groups --extra tensorrt-cu12

# pip 从 NVIDIA 索引获取实际 wheel
python -m pip install --extra-index-url https://pypi.nvidia.com '.[tensorrt-cu12]'
```

安装包含此 extra 的 wheel 时，将 pip 命令最后一个参数替换为 `'路径/ai_inference-版本-py3-none-any.whl[tensorrt-cu12]'`。核对该版本的 [CHANGELOG](../CHANGELOG.md)。需要复现锁定环境时，先按哈希安装导出的依赖，再安装 SDK：

```bash
uv export --locked --no-default-groups --extra tensorrt-cu12 --no-emit-project --emit-index-url --output-file /tmp/tensorrt-requirements.txt
python -m pip install --require-hashes -r /tmp/tensorrt-requirements.txt
python -m pip install --no-deps .
python -m pip check
```

基础安装不会加载 TensorRT/CUDA。后端 extras 按项目约定互斥；`service` 可与任一后端组合。uv 为两个 TensorRT 包指定显式 NVIDIA 索引；pip 使用上述额外索引。`tensorrt-cu12` 前端包提供标准 `import tensorrt` 入口；TensorRT 的 PyPI libs 源码包是下载器，实际二进制 wheel 来自 [NVIDIA 索引](https://pypi.nvidia.com/tensorrt-cu12-libs/)。包版本、wheel URL 和哈希记录在 `uv.lock`。裸 `tensorrt` 包的 CUDA 默认版本会随官方版本变化，因此本 extra 显式选择 cu12，详见 [官方安装说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)。

部署机器仍需兼容的 NVIDIA GPU、驱动和 CUDA 环境。TensorRT engine 通常受构建版本、平台、GPU 及插件约束；SDK 不重建产物或回退到 ORT。仅加载可信产物。构建方应在目标环境验证兼容性，参见 [NVIDIA engine 兼容性说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)。

## 配置和批策略

```yaml
model:
  name: detector
  path: ../models/detector.engine
runtime:
  backend: tensorrt
  options:
    device_id: 0
  warmup: true
# inputs、outputs 和 batch 可省略。
# preprocess / postprocess 复用对应模型的有序 Pipeline 配置。
```

`runtime.options` 仅接收非负整数 `device_id`。名称、dtype、声明 shape 从 engine 发现，显式 `inputs` / `outputs` 用于核对或约束动态维。无需在 YAML 声明 profiles、I/O 格式、位置，也不需要转换清单。

TensorRT 的默认批策略与旧后端区分：

| 配置 | 行为 |
| --- | --- |
| 省略 `runtime.batch` 或显式 `mode: auto` | 加载后解析；静态产物采用真实固定 batch，动态产物采用首个 profile 的共同范围和首个输入的 opt batch，必要时限制在共同范围内 |
| 显式 `mode: fixed` | 始终使用 `size`，尾批按 `padding` 补齐；尺寸必须落在当前所有输入共同支持的 profile 内 |
| 显式 `mode: dynamic` | `size` 是物理批次上限；依据当前全部输入的非 batch shape 选择范围，尾批可补齐到合法下限 |

显式填写 `batch` 但省略其中字段时，仍采用 BatchConfig 的字段默认值（fixed、4、repeat_last）；需要自动推导时应省略整项或写 `mode: auto`。auto 不适用于 ORT、OM 或自定义后端。

例如 profile batch 范围为 `[2,8]`、动态上限为 4，请求 5 个样本时执行 `[4,2]`，第二批含 1 个补齐样本，最后返回 5 个结果。补齐使用 `repeat_last` 或 `zeros`，要求样本相互独立。范围间隙不会被当成合法尺寸；每个物理批次的所有输入必须匹配同一个 profile。无法共同匹配时明确报错，不分别为不同输入选择 profile。

ONNX / OM 保持现有固定 batch=4 默认值及显式策略校验。Batcher 仅消费通用能力数据，拆批、补齐、裁剪和合并均位于 `inference/`，TensorRT 后端只执行收到的一个物理批次。

## 张量和 warmup

公开边界是 CPU NumPy 命名张量，输入可非连续，内部通过 pinned host staging 和设备缓冲区传输；返回数组不共享后端复用缓冲区。支持现有契约中的 float32、float16、float64、int64、int32、uint8、bool，具体还受 TensorRT 产物支持能力约束。

后端读取每个 profile 的格式、每个 I/O 的位置。首版支持 HOST / DEVICE 和连续 LINEAR 格式，其他格式、dtype、无首维 batch 的张量以及 shape-inference I/O 会明确拒绝。shape tensor 的值不属于当前按样本拆批的通用契约。

设置输入 shape 后，通过执行上下文查询输出 shape；仍依赖执行结果的尺寸使用 `IOutputAllocator` 分配，在执行完成后读取通知的实际 shape。输出仍必须符合首维 batch 契约；各物理批次的实际非 batch 维必须可拼接，不提供 ragged tensor 支持。

默认 warmup 沿用 BaseEngine 的零输入和次数。固定模式使用固定 batch；TensorRT 动态模式采用当前输入 shape 可执行的最小 batch，避免 profile 下限大于 1 时失败。不自动猜测非 batch 动态维；这种模型应提供具体输入约束，或设置 `warmup: false` 后调用 `model.warmup(tensors)`。

## 资源生命周期

```python
from ai_inference import Model

with Model("configs/detector.local.yaml") as model:
    results = model.predict(images)

# 长生命周期服务可在停止接收请求并等待请求结束后调用：
# model.close()
```

`Model.close()` 可重复调用，关闭后 predict、分阶段操作和 warmup 明确报错。Model 关闭时拒绝新操作，并等待已经开始的公开操作完成；Model 不额外串行化并行请求。所有内置后端继承 BaseEngine 的关闭管理，直接 Engine 调用也会受到保护；TensorRT 额外保留物理执行锁。

TensorRT 后端持有 runtime、engine、执行上下文、CUDA primary context 引用、stream 和缓冲区。每次执行在当前工作线程 push/pop CUDA context，恢复调用线程原有 context；只释放自身 retain 的引用，不 reset 设备。初始化失败会回收已经创建的资源，执行失败也会先等待已提交工作结束，再允许回收或复用缓冲区。

HTTP 服务在退出时停止对外暴露 runtime，等待已准入请求结束，再调用关闭接口；启动后的服务配置检查失败也会关闭模型。自建 HTTP/gRPC 服务决定创建和关闭时机，清理细节由后端负责。ORT 在关闭时释放自身的会话引用，不调用私有销毁接口；外部持有的会话引用仍需由外部释放。OM 调用实例级 `free_resource()`，不 finalize 全局 ACL runtime。旧自定义 Engine 尚无 close 时，Model 保留放弃引用的兼容路径。

## ONNX 转换

仓库提供独立的 [转换工具](../scripts/convert_tensorrt.py)，SDK 运行时不依赖该脚本。在目标 NVIDIA GPU 环境运行：

```bash
python scripts/convert_tensorrt.py models/model.onnx models/model.engine \
  --device-id 0 --workspace-mib 1024 --precision fp16 \
  --profile 'images=2x3x640x640:4x3x640x640:8x3x640x640'
```

同一个 profile 的多个输入以分号分隔；重复 `--profile` 创建多个 profiles。每个 profile 必须覆盖所有动态输入，静态输入可省略；固定维度不能被覆盖。静态 ONNX 无需 profile。默认精度为 fp32 且禁用 TF32；fp16 允许构建器使用 FP16，不保证每层都使用 FP16。工具不做图像预处理、模型导出、INT8 校准或插件编译；构建失败不会覆盖已有目标文件。

## 验证边界

无 GPU 测试使用模拟 TensorRT 和 CUDA Driver API，覆盖真实后端代码的 metadata、范围选择、动态最小 batch、输出分配、pinned/device 缓冲区增长与独立输出、context 线程切换、执行失败及关闭等待；还覆盖转换参数和 Builder/Parser 错误、SDK/HTTP 生命周期、依赖导出和平台约束。模拟结果不证明 CUDA 驱动、TensorRT ABI、内核或数值正确性。

实机测试的环境准备和运行命令见 [原生 TensorRT 验证](testing.md#原生-tensorrt-验证)。

GPU 验收还应单独覆盖：

1. 真实业务模型转换、反序列化，与 ORT/参考实现按 FP32/FP16 容差比较数值和解码结果。
2. 多 profile 切换、全部输入共同匹配、最小/最大尺寸和 profile 范围间隙。
3. 代表性数据依赖输出模型的 allocator 回调、实际输出 shape 和跨批可拼接性。
4. HOST/DEVICE 传输、缓冲区扩容、多设备选择和 HTTP 工作线程执行。
5. 初始化/执行故障、并发关闭、反复创建销毁的显存占用，以及与同进程其他 CUDA 使用方的 context 共存。
