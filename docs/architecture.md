# 源码目录与职责

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

`__init__.py` 定义包边界和公开导出；`processing/__init__.py` 还加载内置步骤注册。`py.typed` 标记类型信息，不参与执行。

服务与 SDK 使用统一发行包和版本，但根入口不导入 HTTP 模块；纯 SDK 安装无需 FastAPI/Uvicorn。服务启动入口为 `ai_inference.service.app:app`，详见 [服务说明](service/README.md)。

## 执行路径与边界

初始化由Model编排：配置校验 → 处理流程编译/Engine加载与契约读取 → 创建Batcher和DefaultPredictor → Engine.warmup → ready。完整契约可先编译处理流程；ONNX / OM 默认先加载会话，读取真实 I/O 并通过 BaseEngine 解析约束，再编译 Pipeline。随后输出原始/解析后元数据和校验后的 batch 摘要；这些日志不代表 warmup 已成功。

逐请求由Model调用：Preprocessor → 检查逻辑输入 → DefaultPredictor循环执行Batcher.split / Engine.infer / Batcher.trim → Batcher.merge → 检查逻辑输出 → Postprocessor。Engine只处理一个物理batch；Batcher不依赖具体后端，不保存请求累计状态。模型原始输出shape在后处理前检查，Model 和 Pipeline 检查最终结果的类型与数量；自定义处理器负责具体字段的算法契约。

公共张量边界仍为CPU NumPy，batch在第一维，补齐要求样本独立；RT-DETR 允许符号候选维度，但批次合并仍要求实际非 batch 维可以拼接，不提供 ragged tensor 支持。

业务告警、匹配和服务调度由应用管理。

性能测量位于仓库scripts/benchmark.py，通过公开接口包装Model。SDK 不保存请求统计或计时状态，参见 [日志与独立性能测量](logging-and-benchmark.md)。
