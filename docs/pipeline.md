# 可组合处理管道

默认模型通过 `Pipeline + StepRegistry + Context + 模型 I/O 契约` 执行。
YAML 声明步骤顺序与参数；Model 编排前处理、Predictor、后处理及阶段日志。
处理算法由显式注册步骤决定。

## 配置

下面是完整的 YOLO 前处理步骤；相应输入必须是 float32 `[N,3,640,640]`：

```yaml
preprocess:
  source_color: BGR
  device: cpu
  steps:
    - resize:
        width: 640
        height: 640
        keep_ratio: true
        scale_up: true
    - padding:
        width: 640
        height: 640
        value: 114
        alignment: center
    - color:
        source: BGR
        target: RGB
    - normalize:
        divisor: 255
        dtype: float32
    - layout:
        source: HWC
        target: CHW
    - pack:
        input_name: images
```

每项只允许一个操作，参数字典不能为空值（无参数操作写 `{}`）。
`width` 是有效字段，`witdth` 等拼写错误报错。步骤按列表顺序执行，不自动调整顺序。

`source_color` 声明调用方传入的颜色顺序，不会转换数据；`color` 才执行转换。
`normalize` 可用 `scale: 0.00392156862745098`，但 scale 与 divisor 不能同时配置。
divisor=255 使用除法归一化；float32 中乘倒数和除法不必逐位一致。
`pack` 显式命名张量、堆叠真实样本，输出 PreparedInput；物理 batch padding 仍由 Predictor 负责。

后处理也是有序步骤，例如 SFace：

```yaml
postprocess:
  steps:
    - sface_features:
        output_name: fc1
        dimensions: 128
    - l2_normalize: {}
```

## 内置步骤

| 名称 | 作用与主要参数 |
| --- | --- |
| resize | width/height，keep_ratio，scale_up，backend: opencv/pillow；等比缩放输出实际尺寸，后接 padding 才形成固定目标画布 |
| padding | width/height，value，alignment: center/top_left；只补边，不裁剪 |
| letterbox | 已验证的组合操作；width/height，value，scale_up，rounding: floor/ultralytics；必须是首个几何操作 |
| color | source/target: BGR/RGB；要求当前 HWC 与声明 source 一致 |
| normalize | HWC 转浮点；dtype: float32/float16，scale 或 divisor，mean/std；std 必须为正 |
| cast | dtype: float32/float16/uint8；显式转换，不归一化、不转换颜色 |
| layout | source/target: HWC/CHW；校验当前布局后转置 |
| face_align | FaceInput 五点相似变换到112×112；aligned_input:true 允许已对齐裁剪；必须在其他几何操作之前 |
| pack | input_name；可选 sizes_name，生成原图 width/height 的额外输入；校验模型输入名称、dtype、layout和shape |
| yolo_decode | 已验证的完整YOLO输出解码，包括最高类别、阈值、NMS、原图坐标恢复与结果构造 |
| rfdetr_decode | RF-DETR logits/框解码、统一 threshold 与坐标恢复，支持浮点NCHW和AIPP uint8 NHWC契约 |
| rtdetr_decode | 原始 RT-DETR deploy 输出校验、过滤及结果构造；允许符号候选维度，运行时检查形状；不重复坐标缩放或 NMS |
| yunet_decode | 12路stride解码、float框NMS、五点恢复 |
| sface_features | 特征输出校验与结果构造；归一化由独立l2_normalize步骤完成 |
| l2_normalize | 对 FeaturePrediction 单独执行L2归一化；拒绝非有限或零向量 |
| filter_detections | 对 Prediction 进一步筛选，threshold/class_ids；框、分数、类别、关键点和掩码同步选择 |

normalize 无 scale 时，执行 `(pixels - mean * divisor) / (std * divisor)`，divisor默认1；
有 scale 时执行 `(pixels * scale - mean) / std`。参数按输出 dtype 表示，输出非有限值报错。

模型解码步骤是有明确契约的复合步骤。
后接 filter_detections 只能进一步筛选，不能恢复解码步骤已经丢弃的候选；改变NMS和初始阈值要修改解码步骤参数。
可通过注册新步骤进一步细分解码流程，Step 的 kind 可以描述自定义中间数据。

## Context 与几何信息

每次调用创建独立 Context，包含 data、kind、device、attributes。
前处理 data 为 SampleContext 列表：当前像素数据、颜色、布局、设备、可选五点及 Geometry。
Geometry 包含 original_shape、实际scale_x/scale_y、左上padding、理想统一gain和restorable。

resize/padding 更新几何信息，支持连续缩放/补边时组合scale和偏移。
YOLO使用理想统一gain；RF-DETR/YuNet使用实际比例，保持各自既有坐标契约。
face_align 属于非轴向变换，标记restorable=false；检测解码不能用简单scale/padding逆变换恢复它。
SFace 只使用原图尺寸，不执行检测框恢复。RT-DETR deploy 已用原图尺寸换算框，前处理只允许无 padding 的直接 resize；不能改为 letterbox。

pack 将 Geometry 作为每张真实样本的metadata传入 PreparedInput；RawOutput携带同一请求。
Context 不保存在共享 Model 上，内置步骤不保存请求数据；重复/并发请求不会覆盖彼此的几何信息。
不可并发修改输入数组、注册表或步骤参数；第三方步骤与Engine的并发安全仍由其实现负责。

## 校验与设备

创建 Model 时，引擎读取张量契约后编译两侧 Pipeline，检查操作注册、参数 schema、步骤 kind/布局/颜色顺序、输入尺寸与 dtype、模型解码输出契约和前后处理连接。若程序已提供完整契约，则可在加载会话前编译。
执行期间再次校验步骤输出kind/device、最终PreparedInput和RawOutput张量，以及每个真实样本的Result数量。
第三方步骤需要在plan中校验自身的算法条件，框架不会猜测任意自定义函数的数学语义。

设备默认cpu。可在Pipeline顶层配置device，或在每个步骤参数中覆盖：

```yaml
- resize:
    width: 640
    height: 640
    device: cpu
```

当前仅实现CPU步骤；cuda/trt或其他未实现设备明确报错，不回退CPU。
CUDA/TRT推理仍配置 `runtime.options.providers`，与处理设备独立。
处理设备直接配置在 preprocess/postprocess 或步骤参数中。

张量边界仍为CPU NumPy，无设备张量零拷贝、异步调度或自动跨设备复制。
SDK不保留计时状态；性能测量由独立benchmark工具完成，见 [日志与性能测量](logging-and-benchmark.md)。

## 配置与扩展契约

六份默认 YAML 统一采用有序 steps，model 仅需 name/path；ONNX / OM 自动发现输入输出契约。构造独立 Pipeline 时应传入 model.config 或其他已解析的完整契约。
前后处理必须使用有序 steps；每个步骤的参数由对应工厂严格校验。
内置模型解码位于processing/decoders，仅消费模型输出与Geometry；processing/steps/decode.py负责将这些算法注册为后处理步骤。
SFace特征构造与L2归一化独立配置。

完整处理器注入 `Model(config, preprocessor=..., postprocessor=...)` 可用于自定义入口；
配置中的Pipeline先编译验证，再替换指定执行入口。只替换一侧时需保持另一侧Geometry契约；
成对替换可自定义metadata。公共predict及分步调用、PreparedInput/RawOutput、Prediction/FeaturePrediction保持不变。
