# 日志与独立性能测量

## SDK日志

日志实现位于 `src/ai_inference/logging.py`，使用Loguru。推理和业务事件使用模板：

```text
[model.name]-[stage]-[step] 消息摘要
```

INFO记录加载、校验后采用的 batch 策略、模型原始与解析后的输入输出张量契约、预热、主要处理阶段与batch计划；DEBUG展开具体步骤和物理batch编号；WARNING记录失败步骤和错误类型，保留原异常向上传递。SDK不记录图片、张量内容或异常原文。

```text
[rfdetr_nano_coco]-[init]-[engine.metadata] Input 'input' | float32 | shape=[1, 3, 384, 384] | resolved_shape=[1, 3, 384, 384]
[rfdetr_nano_coco]-[init]-[batching] Validated | mode=fixed | size=1 | size_role=fixed | padding=repeat_last
[rfdetr_nano_coco]-[infer]-[batching] images=1 | mode=fixed | batches=1 | size=1 | padding=0
[rfdetr_nano_coco]-[infer]-[engine.predict] batch=1/1 | Completed | valid=1 | physical=1 | padding=0
```

初始化摘要在模型契约与 Pipeline 校验通过后输出；不需要发起推理请求。ONNX / OM 共用元数据日志，每个 input/output 打印真实名称、dtype、模型原始 `shape` 和实际采用的 `resolved_shape`。静态模型 batch 冲突先报错，不输出通过校验的摘要。固定策略会将模型的符号 batch 约束为 size；动态策略的未知维度保持 `None`，实际尺寸在推理时校验。

```text
[rtdetr]-[init]-[engine.metadata] Input 'images' | float32 | shape=['N', 3, 640, 640] | resolved_shape=[None, 3, 640, 640]
[rtdetr]-[init]-[batching] Validated | mode=dynamic | size=2 | size_role=max | padding=none
```

`size_role=fixed` 表示固定物理 batch，`size_role=max` 表示每批上限；动态模式不补齐，因此有效策略打印 `padding=none`。结构化字段为 `shape`、`resolved_shape`、`batch_mode`、`batch_size`、`batch_size_role`、`batch_padding`。初始化校验失败时不输出 `Validated` 摘要；warmup 成功与模型 ready 由后续日志单独确认。

SDK不生成request/run/instance编号，不保存计时状态，不做请求汇总或日志终态去重。PreparedInput/RawOutput仅携带阶段数据及原输入元数据。每次实际阶段执行独立输出日志，包括重复后处理；请求统计和告警归应用管理。

应用配置sink，SDK不添加或移除sink：

```python
import sys
from loguru import logger
from ai_inference import LOG_FORMAT, Model

logger.remove()
logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
model = Model("configs/yolo11.yaml", logger=logger.bind(application="inference"))
results = model.predict(image)
```

`LOG_FORMAT`是 `{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}`。使用 `serialize=True` 可输出结构化日志，字段保留于 `record["extra"]`。如需请求或场景关联，应用可自行绑定：

```python
from ai_inference import log_context, log_event

with log_context(model=model.config.model.name, request_id="frame-001", scene="camera-1"):
    results = model.predict(image)
    log_event("business", "alarm.evaluate", "No alarm", alarm_count=0)
```

这些自定义字段仅透传到日志，不进入张量对象，也不改变Model接口。Plotter 独立消费已有结果，使用组件日志模板：

```text
[plotter]-[save] Saved | path=results/annotation.png
[plotter]-[save] Failed | error=OSError
```

Plotter 共用应用配置的 logger、sink 和 `log_context`，透传 `request_id`、`scene` 等关联字段。模型名仅在应用提供时作为结构化上下文保留，不参与消息前缀；独立使用时不填充模型占位符。结构化字段保留 `stage=visualize`、`step=plotter.save`，增加 `component=plotter`；保存成功使用 INFO，失败使用 WARNING 并继续抛出原异常。

## benchmark工具

性能测量位于 `scripts/benchmark.py`，包装已有Model的公开接口，不继承Model，不依赖私有方法。核心SDK没有阶段计时。

```bash
uv run --locked python scripts/benchmark.py --config configs/yolo11.yaml --image tests/bus.jpg --runs 20 --warmup 3 --samples 1
```

默认使用本仓库YOLO11配置及bus.jpg；默认路径不依赖当前工作目录。`--samples`表示每次调用输入的图片数量，复用同一张图片；固定/动态batch策略仍由模型配置决定。

工具关闭Model自动预热，由自身执行指定次数的有效图片预热，排除初始化和预热后再重复测量。失败时保留异常，不输出成功报告。测量预处理、完整逻辑推理、后处理和端到端耗时，单位ms，输出mean/P50/P95/min/max及真实样本吞吐率。推理阶段包含分批、补齐、引擎执行、输出裁剪和合并，不能当成纯设备内核时间；阶段和总耗时包含工具的同步回调开销。

CLI使用WARNING级别日志减少输出干扰，结果在终端以表格显示，默认不保存文件。添加 `--save results/benchmark_yolo11.json` 可额外保存完整精度的JSON报告；相对保存路径按当前工作目录解析。导入 `benchmark(model, images, runs=..., warmup=..., synchronize=...)` 时不会改变日志配置。

当前内置引擎通过同步调用返回CPU NumPy结果。自定义异步后端需要提供 `synchronize` 回调，工具会在测量开始前及各阶段结束后等待完成。未提供同步回调时只能测量接口返回时间。报告不代表生产请求分布、数据集精度或其他设备性能。
