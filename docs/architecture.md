# 模块边界与演进约定

包顶层是SDK入口，阶段实现归入对应子包。推荐应用使用 `from ai_inference import Model, Config, Plotter` 等公开导出；内部实现按下表的组件职责组织。

```text
ai_inference/
├── __init__.py          公开导出
├── config.py            YAML配置与配置约束
├── model.py             初始化与逐请求编排
├── contracts/           阶段间的数据与接口契约
├── engines/             物理推理与引擎生命周期
├── inference/           逻辑推理与批次准备/恢复
├── processing/          前后处理管道
│   ├── steps/           预处理、解码、结果处理的步骤注册
│   └── decoders/        各模型的输出解释算法
├── logging.py           日志模板与应用提供的上下文
├── service/             可选 HTTP 接口、输入边界与单模型请求准入
└── visualization/       已有结果的绘制和保存
```

## 组件职责

| 组件 | 职责 |
| --- | --- |
| `config.py` / `model.py` | 配置解析与校验，组件初始化，公开请求阶段编排和日志 |
| `contracts/` | 输入输出数据类型、处理器协议、通用注册与张量边界校验 |
| `engines/` | 物理 batch 执行、模型 I/O 读取、provider 初始化、共享张量契约解析和 warmup |
| `inference/` | Batcher 分批/补齐/裁剪/合并；DefaultPredictor 编排 Engine，不实现图像算法 |
| `processing/pipeline.py` / `context.py` | 步骤编译、连接校验与请求级 Context / Geometry |
| `processing/steps/` | 通用图像步骤、解码器注册及结果过滤/L2 归一化 |
| `processing/decoders/` | YOLO、RF-DETR、RT-DETR、YuNet、SFace 的输出算法；SFace 同时提供对齐模板 |
| `processing/common.py` / `geometry.py` | 图像校验、letterbox、框格式转换、恢复、裁剪和 NMS |
| `logging.py` / `visualization/` | 应用提供的日志上下文，以及消费既有结果的独立 Plotter |
| `service/` | 四种 HTTP 图片输入、校验/解码、模型执行与响应；通过 `service` extra 安装 HTTP 依赖 |
| `scripts/`（仓库目录） | 通过显式命令执行模型导出、离线转换、性能测量等开发和部署准备工作 |

`__init__.py` 定义包边界和公开导出；`processing/__init__.py` 还加载内置步骤注册。`py.typed` 标记类型信息，不参与执行。

服务与 SDK 使用统一发行包和版本，但根入口不导入 HTTP 模块；纯 SDK 安装无需 FastAPI/Uvicorn。服务启动入口为 `ai_inference.service.app:app`，详见 [服务说明](service/README.md)。

## 执行路径与边界

初始化由Model编排：配置校验 → 处理流程编译/Engine加载与契约读取 → 创建Batcher和DefaultPredictor → Engine.warmup → ready。显式提供完整契约和批策略时可先编译处理流程。需要发现 I/O 或解析自动批策略时，先加载产物，再用解析后的契约编译 Pipeline。随后输出原始/解析后元数据和校验后的 batch 摘要；这些日志不代表 warmup 已成功。

逐请求由Model调用：Preprocessor → 检查逻辑输入 → DefaultPredictor循环执行Batcher.split / Engine.infer / Batcher.trim → Batcher.merge → 检查逻辑输出 → Postprocessor。Engine只处理一个物理batch；Batcher不依赖具体后端，不保存请求累计状态。模型原始输出shape在后处理前检查，Model 和 Pipeline 检查最终结果的类型与数量；自定义处理器负责具体字段的算法契约。

公共张量边界仍为CPU NumPy，batch在第一维，补齐要求样本独立；RT-DETR 允许符号候选维度，但批次合并仍要求实际非 batch 维可以拼接，不提供 ragged tensor 支持。

业务告警、匹配和服务调度由应用管理。

性能测量位于仓库scripts/benchmark.py，通过公开接口包装Model。SDK 不保存请求统计或计时状态，参见 [日志与独立性能测量](logging-and-benchmark.md)。

## 职责归属与依赖方向

以下约定用于判断新增能力的归属。模块之间通过配置、数据契约和执行接口协作。

| 模块 | 接收与交付 | 状态归属与依赖 |
| --- | --- | --- |
| `config.py` | 将 YAML 或调用方提供的配置转换为 `Config`。显式张量字段用于核对或细化模型契约。 | 持有配置结构和通用约束。后端专用选项由对应后端解析和校验。 |
| `contracts/` | 定义 `PreparedInput`、`RawOutput`、结果类型和处理器协议，提供命名张量的通用校验。 | 可引用配置类型。模型运行库、图像算法和传输协议的规则由各自模块持有。 |
| `engines/` | 加载模型产物，发现 I/O 契约，执行一个物理 batch，按名称返回原始 NumPy 张量。 | 后端实例持有会话和执行资源。共享检查复用 `contracts/`，厂商库及设备细节封装在具体后端，按需加载。 |
| `inference/` | 将 `PreparedInput` 拆为物理批次，按策略补齐，调用 `Engine` 后裁剪并合并为 `RawOutput`。 | 批次和结果属于当前请求。通过 `Engine` 接口执行，`Batcher` 保持对具体后端及其厂商对象的独立。 |
| `processing/` | 将应用输入转换为 `PreparedInput`，将 `RawOutput` 转换为预测结果。颜色、归一化、几何变换和解码属于这层。 | 流水线持有已编译的步骤，请求数据和几何信息由 `Context` 管理。处理步骤使用张量契约和配置，可以独立运行。 |
| `model.py` / `__init__.py` | `Model` 负责组件初始化及请求阶段的编排。包入口负责公开导出。 | `Model` 持有已初始化的组件。阶段实现各自管理内部状态，HTTP 由独立入口加载。 |
| `service/` | 将 HTTP 请求转换为 SDK 输入，再将预测结果转换为 HTTP 响应。负责准入、请求限制及服务生命周期。 | 服务进程持有 `Model`，通过 SDK 接口执行推理。模型算法留在 SDK 的处理层。 |
| `logging.py` / `visualization/` | 日志记录执行事件，绘图消费已有预测结果。 | 各层可调用统一的日志接口。绘图依赖结果契约，独立于模型会话和 HTTP 请求。 |
| `scripts/` | 读取显式输入，输出模型产物或测量结果。 | 构建参数由产物生成方管理。SDK 运行时可消费产物，源码包不应依赖脚本入口。 |

### 示例：新增原生 TensorRT 后端

ONNX → TensorRT engine 的转换工具放在 `scripts/`。推理部署交付序列化 engine，`engines/` 负责加载。后端把实际 I/O 的名称、dtype 和声明 shape 映射到 `TensorMetadata`。profile、I/O 格式和内存位置由后端读取和使用，无需在 YAML 重复声明。设置实际输入后，后端从执行上下文读取已经解析的输出 shape。依赖输入值的输出尺寸可能要到执行时才确定。

模型的静态尺寸和动态范围由产物决定。SDK 的拆批和补齐策略由 `inference/` 负责，后端负责检查收到的物理批次能否执行。设备选择、显存与 stream 属于执行资源，颜色转换、归一化和输出解码仍由 `processing/` 负责。当前的扩展入口见[处理步骤与扩展](extension.md)。

### 原生 TensorRT 的能力与生命周期

`contracts/execution.py` 定义不含厂商对象的输入尺寸范围及 profile 数据。TensorRT 后端从产物生成 `batch_capabilities`；Model / DefaultPredictor 将能力传给 Batcher。Batcher 按全部实际输入的非 batch shape 寻找共同 profile 的合法批范围，负责拆批和尾批补齐，Engine 再检查整个物理批次共同匹配并执行。不同 profile 的范围间隙不合并。

原生 TensorRT 省略 batch 时使用待解析的 `mode=auto`：静态产物解析为真实固定 batch，动态产物以首个 profile 的 opt batch 和共同范围确定上限。显式策略保留并校验。旧后端默认规则保持不变。`tests/test_tensorrt.py`、`tests/test_batcher_predictor.py` 验证动态最小 batch 大于 1、尾批裁剪和共同 profile 匹配。

`Engine` 协议定义 `close()`；`BaseEngine` 统一管理关闭准入、等待在途 infer / 整次 warmup、重复关闭和关闭后的调用检查，各后端只实现 `_close()`。通用生命周期不会串行化并行推理；TensorRT 的执行锁仍由后端持有。内置后端初始化使用 `_initialization()` 在失败时调用同一关闭入口，清理异常不覆盖原始初始化错误。ORT 释放自身会话引用，OM 调用实例级 `free_resource()` 而不 finalize 全局 ACL runtime，TensorRT 回收自身 CUDA/TRT 资源。

Model 提供幂等 `close()` 和上下文管理，等待自己的在途阶段操作后委托 Engine 关闭。HTTP 启动检查失败及服务退出均接入同一入口。旧自定义 Engine 未实现 close 时仍保持运行时兼容，Model 放弃自身及 Predictor 对 Engine 的引用；新后端应实现 Engine 协议或继承 BaseEngine。`tests/test_engine_lifecycle.py` 验证共同关闭规则、并行执行、整次 warmup 与失败回收，Model / HTTP 回归分别位于 `tests/test_model_lifecycle.py` 和 `tests/test_service.py`。具体设备限制见 [原生 TensorRT](tensorrt.md)。

## 边界如何演进

新增能力先按最小职责定位。处理整次请求的批策略归 `inference/`，执行单个物理批次的设备细节归 `engines/`，会改变张量或结果语义的处理逻辑归 `processing/`。能通过现有接口完成的实现，可以直接扩展对应模块。

如果改动影响跨模块的数据契约、公开配置的含义、依赖方向、资源所有权或现有默认行为，在同一个 PR 更新本页。说明哪个模块接手决定、调用方如何适配，以及哪条测试验证新约定。引入新的自动批策略时，引擎负责报告可执行范围，推理层负责选批和补齐，公开配置定义省略批策略时的行为。共享接口的改动应验证现有后端及新行为。
