# 处理步骤与扩展

默认架构为Pipeline、StepRegistry、Context与模型I/O契约，完整配置与步骤列表见 [处理管道](pipeline.md)。
Model负责请求编排与阶段日志；DefaultPredictor编排独立Batcher与Engine；Batcher准备和恢复批次，Engine执行物理batch。
内置模型算法通过注册步骤执行，逐请求数据统一由Context与Geometry传递。

## 分步调用与独立管道

```python
from ai_inference import Config, Model, Pipeline

config = Config.from_yaml("configs/yolo11.yaml")
model = Model(config)
prepared = model.preprocess(images)
raw = model.infer(prepared)
results = model.postprocess(raw)

# 以下 Pipeline 构造不会另建推理 session。
resolved = model.config  # 使用加载后完整的 I/O 契约
pre = Pipeline("preprocess", resolved.preprocess, resolved)
post = Pipeline("postprocess", resolved.postprocess, resolved)
prepared = pre(images)
```

PreparedInput是未补齐的CPU NumPy张量及逐样本metadata；RawOutput去除补齐样本并携带其PreparedInput。
`model.postprocess(raw)` 使用 RawOutput 携带的请求元数据。
SDK 不保存请求统计；直接 Pipeline 调用仅执行处理步骤和阶段日志。

## 注册处理步骤

在构造Model前注册步骤工厂。工厂收到完整Config及YAML中的步骤参数，返回Step：

```python
import numpy as np
from pydantic import Field
from ai_inference import Step, StepRegistry
from ai_inference.config import Section

class ClipOptions(Section):
    lower: float = Field(default=0, allow_inf_nan=False)
    upper: float = Field(default=1, allow_inf_nan=False)

@StepRegistry.register("clip_pixels")
def clip_pixels(config, **options):
    params = ClipOptions.model_validate(options)
    if params.lower > params.upper:
        raise ValueError("lower must not exceed upper")

    def plan(state):
        if state.dtype not in ("float32", "float16"):
            raise ValueError("clip_pixels requires float data")

    def run(context):
        for sample in context.data:
            sample.data = np.clip(sample.data, params.lower, params.upper)

    return Step("images", "images", run, plan)
```

放在normalize之后：`- clip_pixels: {lower: 0, upper: 1}`。

`input_kind/output_kind`描述数据边界，内置种类为images、prepared、raw、results；
可定义专用中间kind并注册下一步消费它。改变kind的步骤必须在run中显式更新context.kind。
plan只在编译时运行，校验并更新State中的layout、color、dtype、尺寸及几何能力。
运行时几何步骤必须同步更新SampleContext.geometry，不能只变像素而遗失坐标关系。
框架校验已声明的kind/device与最终I/O，第三方工厂负责参数schema和自己的算法语义。
重复注册和未知步骤报错，YAML不执行任意脚本或动态导入。

当前设备能力只支持 cpu；扩展设备需要提供对应执行实现并在 Step.devices 中声明。
请求数据只能放在Context内，不能写入共享工厂缓存或Pipeline.state。

## 整体处理器注入

`Preprocessor/Postprocessor` 是 callable Protocol。前处理返回 PreparedInput，后处理必须返回 list[Result]，每个真实输入恰好对应一个 Prediction 或 FeaturePrediction；Model 统一检查类型与数量。

```python
model = Model(config, preprocessor=my_preprocessor, postprocessor=my_postprocessor)
```

一侧替换需保持对侧metadata兼容；成对替换可定义自己的metadata。
配置会先编译两侧步骤契约，再替换指定执行入口。内置步骤使用Geometry；自定义步骤通过Context传递逐请求数据。

## Engine 与 Predictor 扩展

用EngineRegistry.register(name)注册接收Config的工厂，返回提供config、infer(tensors)、warmup(tensors=None)、close()的Engine。推荐继承BaseEngine，只实现_infer并按需覆盖预热输入。
物理batch输入输出保持名称、dtype和batch维；模型算法不放进Engine。
内置ORT/OM/TensorRT继承BaseEngine，子类实现_infer，公共infer统一检查张量契约，warmup经过相同入口。
`BaseEngine.create_warmup_inputs()`默认按输入契约创建零张量，可由子类覆盖以提供合法尺寸、索引或其他模型特定值。`BaseEngine.warmup(tensors=None)`接收可选的合法物理batch输入并调用公共infer入口；推荐保留公共 warmup 方法以使用统一生命周期保护；覆盖整个方法时，扩展方须自行保证关闭准入和在途调用保护。第三方引擎需要实现 Engine 契约。
`read_tensor_metadata()`返回后端无关的张量信息（名称、dtype、shape），内置后端从加载的会话或 engine 读取；基类默认返回配置契约，自定义后端默认须在 Config 中提供完整契约。支持自动发现的扩展引擎应先创建 session，再覆盖 read_tensor_metadata 并调用 BaseEngine._load_tensor_contract；解析后的 config 保存在引擎实例。`log_tensor_metadata(logger)`使用Loguru打印这些信息。
引擎的 `_load_session()` 负责会话初始化，BaseEngine 的 `_load_tensor_contract()` 负责解析和校验张量契约；图片处理和解码在Pipeline中，跨物理batch的准备与恢复在Batcher中。

公开Predictor Protocol接收PreparedInput、返回RawOutput。统一默认实现为DefaultPredictor(engine,config,batcher=...)，单张/多张、固定/动态均经过相同路径。
Batcher(config)接收解析后的张量契约，只依赖CPU NumPy张量和batch策略，不按backend名称分支：split(prepared)返回BatchChunk迭代器，记录张量、起始位置、真实/物理数量和批次编号；trim(outputs,chunk)校验并复制有效输出，避免引擎复用缓冲区污染此前结果；merge(parts,prepared)按输入顺序合并并校验真实结果数量。Batcher不调用Engine，不保存请求累计状态。
Model.batcher与Model.predictor.batcher为同一实例，所有chunk与输出集合都属于当前请求。共享运行时校验函数位于contracts.validation模块，公共Engine与逻辑阶段使用同一契约规则。
自定义Engine默认收到3次零张量warmup，runtime.warmup:false可关闭。

模型来源、版本、许可、业务关联和人员检索仍由上层负责；SDK只管理推理契约。

## Engine 生命周期

调用方通过 `Model.close()` 或 `with Model(...)` 决定释放时机；HTTP 服务已在退出流程接入。直接使用 Engine 的调用方应在不再使用它时调用 `engine.close()`。

`BaseEngine.close()` 阻止新 infer / warmup，等待已经准入的操作完成，然后仅调用一次后端 `_close()`。完整 warmup（包括其内部多次 infer）作为一个在途操作；关闭期间不会中断已开始的 warmup。公共生命周期保护允许并行 infer，不替后端决定具体会话是否需要执行锁。不得从当前 Engine 的 infer / warmup 内同步关闭同一个 Engine，否则明确报错以避免自等待。

资源型后端在 `super().__init__(config)` 后先初始化 `_close()` 需要读取的字段，再将加载、元数据发现及契约校验放入 `with self._initialization():`。初始化抛错会通过相同的 close 入口清理，保留原始异常。子类实现 `_close()`，不覆盖公共 close；无资源后端可继承默认空钩子。清理钩子应尝试回收全部自有资源；若仍抛错，首次关闭报告该错误，引擎保持不可使用状态，重复关闭不重试钩子。

ORT 通过清空会话引用释放自身所有权，不使用私有 ORT 接口；若调用方另持会话引用，其生存期由调用方负责。OM 使用锁定 ais_bench 版本的实例级 `free_resource()`，不调用会影响其他模型的全局 finalize。TensorRT 将 CUDA/TRT 清理实现放入 `_close()`，不重置整个设备。

新自定义 Engine 应提供 close 或继承 BaseEngine。旧的非 BaseEngine 实现没有 close 时，Model 仍允许运行并在关闭时放弃引用；这条兼容路径不提供厂商级显式释放保证。自定义后端若覆盖公共 infer / warmup / close，其生命周期行为由该实现负责。
