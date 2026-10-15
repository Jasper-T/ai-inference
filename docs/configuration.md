# 配置与模型契约

所有命令在项目根目录执行。配置拒绝未知字段；名称、dtype、shape 必须与实际模型一致。

## 配置字段

| 字段 | 说明 |
| --- | --- |
| `model` | 模型名称、路径 |
| `runtime` | 公共字段：`backend`、`batch`、`options`、`warmup`、`warmup_runs` |
| `runtime.options` | 对应 Engine 的专有参数，由该 Engine 严格校验 |
| `inputs` / `outputs` | ONNX / OM 均可省略，默认读取模型；显式提供时作为预期约束。自定义后端默认仍须提供完整契约 |
| `preprocess` / `postprocess` | 有序steps与设备；平铺配置不再支持 |

YAML 相对路径以 YAML 所在目录为基准；直接构造 `Config` 时，相对模型路径以进程工作目录为基准。
ONNX / OM 加载后自动读取输入输出的名称、dtype、shape，用于预处理检查、解码器检查、warmup 和推理校验。`Model.config` 和 `Model.engine.config` 保存解析后的完整契约；原始配置对象不被修改。`preprocess` / `postprocess` 的处理步骤仍须显式配置，不根据 shape 猜测颜色、归一化或输出含义。

可以完全省略 `inputs` / `outputs`，也可以只提供其中一侧；提供某一侧时，名称列表须覆盖该侧所有模型张量，`dtype` 和 `shape` 可分别省略，例如：

```yaml
inputs:
  - name: input
    shape: [4, 3, 560, 560]  # 可选的预期约束；dtype 自动读取
# outputs 自动读取
```

显式配置的名称、dtype、rank 和固定维度不一致时，加载失败。固定模式下，模型的符号 batch 由 `runtime.batch.size` 约束；其余符号维度转换为 `null`；显式配置的具体尺寸可约束模型的符号维度，并在运行时继续校验。无法确定的非 batch 尺寸仍可能需要显式约束，以满足内置解码器或 warmup 的要求。批处理策略仍由 `runtime.batch` 配置，不自动推断。

加载 ONNX / OM 后，引擎通过类内 `read_tensor_metadata()` 读取实际张量信息；`Model` 调用引擎的日志方法，通过 Loguru 在 INFO 级别记录每个输入输出的实际名称、dtype、原始 shape（包含符号维度）和解析后采用的 resolved_shape。模型契约与 Pipeline 校验通过后，还会打印采用的 batch mode、size、size_role 和有效 padding 策略。默认输出到终端，也可指定应用的 Loguru logger：

```python
from loguru import logger
from ai_inference import Model

model = Model("configs/rfdetr.yaml", logger=logger.bind(application="inference"))
```

示例消息：`[rfdetr_nano_coco]-[init]-[engine.metadata] Input 'input' | float32 | shape=[1, 3, 384, 384] | resolved_shape=[1, 3, 384, 384]`。SDK 不修改 Loguru sink 配置。统一格式、INFO/DEBUG 层级与应用提供的关联字段见 [日志与独立性能测量](logging-and-benchmark.md)。SDK 不自动生成请求或实例编号。

也可独立检查真实模型契约：

```python
import onnxruntime as ort

session = ort.InferenceSession("models/yolo11n.onnx", providers=["CPUExecutionProvider"])
for tensor in session.get_inputs() + session.get_outputs():
    print(tensor.name, tensor.type, tensor.shape)
```

## 执行设备

`runtime.options.providers`接受简写名称或有序列表：

```yaml
runtime:
  options:
    providers: cpu              # 也可写cuda或trt
```

```yaml
runtime:
  options:
    providers: [trt, cuda, cpu]  # 按顺序选择，允许显式降级
```

cpu/cuda/trt分别对应ONNX Runtime的CPU、CUDA和TensorRT Execution Provider。单项或单元素列表均严格使用该provider，不可用或初始化失败直接报错。多项列表会跳过不可用provider；初始化失败时按顺序尝试剩余provider。没有列出的后端不会用于推理，未列cpu时禁止CPU graph fallback。

同一会话内ORT按列表优先级分配算子，例如 `[cuda, cpu]` 可让CUDA不支持的节点在CPU执行；它不是每次请求先在CUDA完整执行再重跑CPU。执行错误直接抛出，不自动重试。YAML/模型I/O契约错误也不会因为换provider而被忽略。

`model.engine.requested_providers`记录配置顺序，`selected_providers`记录成功选用的顺序，`providers`为ORT实际注册列表。ORT可能注册CPU但禁止它承担图节点，因此注册列表不证明算子的实际执行位置。

需要高级选项时仍可使用对象形式：

```yaml
runtime:
  options:
    providers:
      - name: CUDAExecutionProvider
        options: {device_id: "1"}
      - name: CPUExecutionProvider
```

简写cuda/trt默认device_id=0，cpu没有设备参数。CPU/CUDA/TRT仍需安装相应运行时依赖；列表配置只是允许降级，不保证每个provider在机器上都可用。

OM 设备选项保持在同一个 `runtime.options` 层级；`providers` 和线程数属于 ORT，OM 使用 `device_id`：

```yaml
runtime:
  backend: om
  warmup: true
  warmup_runs: 3
  options:
    device_id: 0
  batch:
    mode: fixed
    size: 4
    padding: repeat_last
```

## CUDA 部署与失败

GPU 环境使用 `uv sync --locked --no-default-groups --extra onnx-gpu`，避免与默认测试组的 CPU ORT 混装。实际 ORT 版本见 `uv.lock`，CUDA、cuDNN 与 NVIDIA 驱动必须按该版本的 [官方兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 准备，不能从 `onnx-gpu` extra 的宽版本范围推断兼容性。

OpenCV 前后处理仍在 CPU 执行。目标设备应单独验证 provider 初始化、真实权重 warmup、显存、延迟和结果一致性；provider 降级及执行失败行为见上文“执行设备”。

## 批处理

- `mode: fixed`：`size` 等于模型固定 batch。5 张输入、size=4 时执行两个批次，返回 5 项。
- `padding: repeat_last`：不足一批时复制最后一张；`zeros` 填充归一化张量域的零。
- `mode: dynamic`：`size` 为每批上限，不补齐；模型实际支持动态 batch。自动读取时符号维度转为 `null`；显式填写 I/O shape 时第一维必须为 `null`。

模型能力与 SDK 组批策略分开校验：

| 模型第一维 | runtime.batch | 结果 |
| --- | --- | --- |
| 固定 B | fixed，size=B | 允许 |
| 固定 B | fixed，size 不等于 B | 初始化报错 |
| 固定 B | dynamic | 初始化报错 |
| 符号 batch | fixed | 允许，按配置 size 组批和补齐 |
| 符号 batch | dynamic | 允许，size 为每批上限，不补齐 |

输入和输出都按实际模型元数据检查。冲突错误包含张量名称、模型 batch、配置 mode/size 和修改建议；SDK 不自动改变 batch 策略。运行时继续检查所有张量的实际 batch 一致性。OM 当前只允许静态模型与 fixed 策略。

修改 YAML 不能改变模型的固定 batch。OM 当前仅支持静态固定 batch；动态模式仅适用于支持它的后端。补齐要求各样本推理独立；各批输出的非 batch 维必须可拼接。
图片会一次性预处理并保存在内存，大量输入建议由调用方分段提交。

## RF-DETR

解码步骤 `rfdetr_decode`。示例配置采用：

| 项目 | 契约 |
| --- | --- |
| 输入 | `input`，float32 `[1,3,384,384]` |
| 框输出 | `dets`，float32 `[1,300,4]`，归一化 `cxcywh` |
| 分类输出 | `labels`，float32 `[1,300,91]`，类别 logits |
| 类别 | COCO 80 类，保留 91 个原始编号槽位；阈值统一为 0.4 |

默认直接 resize、BGR→RGB、ImageNet 归一化；后处理 sigmoid、每个 query 取最高分类、按统一的 `threshold`（默认 0.4）过滤，不做 NMS。
业务层可根据返回的类别和分数进一步筛选；SDK 的 `threshold` 应不高于业务所需的最低阈值，已过滤的结果无法恢复。
resize步骤设置keep_ratio:true并后接padding可改为等比补边；`integer_boxes` 控制坐标截断，`clip_boxes` 控制越界裁剪。
Nano 示例关闭整数截断、开启裁剪，输出框 dtype 为 float32。

## YOLOv8 / YOLO11

解码步骤yolo_decode支持YOLOv8/YOLO11 detect模型的nms=False导出。

| 项目 | 契约 |
| --- | --- |
| 输入 | float32 / float16 `[N,3,H,W]`，固定 H/W |
| 输出 | `layout: NCQ` 对应 `[N,4+C,Q]`；`NQC` 对应 `[N,Q,4+C]` |
| 坐标 | 输入图像像素域 `cxcywh` |
| 分类 | C 个类别概率，无 objectness，不再 sigmoid |

`num_classes`必须匹配输出类别通道数。类别名称只由Plotter的可选names参数提供。640×640 标准导出通常 Q=8400，COCO 的 C=80。
前处理为居中 letterbox、默认填充 114、BGR→RGB、除以 255。
后处理取最高分类、置信度过滤、按类别 NMS、坐标还原并默认裁剪；结果按分数降序排列。

常用参数：`confidence_threshold`、`iou_threshold`、`max_detections`。
`agnostic_nms: true` 开启跨类别抑制；`max_nms` 限制进入 NMS 的候选数。
不支持内置 NMS 输出、YOLOv5 objectness 格式、分割、姿态和旋转框模型。

YOLO的yolo_decode参数output_name 可省略，默认使用 `outputs[0].name`；显式填写时必须与其一致。
RF-DETR 保留 `boxes_name` / `logits_name`，用于区分两个输出的语义。
两种模型均保留置信度 **大于或等于** 阈值的候选；NMS 仍在 IoU **大于** 阈值时抑制。

`intra_op_num_threads` 控制 ORT CPU 算子内部线程数，默认 1。
执行模式为 `ORT_SEQUENTIAL`；不支持配置 `execution_mode` 或 `inter_op_num_threads`。

ORT 专有参数统一放在 `runtime.options`，由 ONNXRuntimeOptions 严格校验。
`ONNXRuntimeOptions` 位于 `engines/onnxruntime.py`，在创建 Engine 时校验，未知选项报错。
公共 `Config` 不解释 options 内容；自定义 Engine 应定义自己的严格配置类。每个部署使用一份 YAML。

## 结果、人脸输入与扩展模型契约

检测模型仍返回 `list[Prediction]`。可选 `keypoints` 为 float32 `[K,P,2]` 原图 xy 坐标，`masks` 为 `[K,H,W]` 原图大小的实例掩码；没有对应结果时为 None。空人脸检测的关键点形状为 `[0,5,2]`。掩码由自定义后处理器填充，具体 dtype、类别与阈值由该处理器定义，当前没有内置分割模型。

SFace 返回 `list[FeaturePrediction]`，每项包含 float32 `[D]` 的 `embedding` 和输入原图的 `original_shape`。默认 L2 归一化，零向量与非有限向量报错。`Model` 与 Postprocessor 公共契约使用 `Result = Prediction | FeaturePrediction`，保持同一 PreparedInput/RawOutput、批处理与 warmup 路径。

`FaceInput(image, keypoints)` 用于原图五点对齐，关键点必须是 YuNet 的右眼、左眼、鼻尖、右嘴角、左嘴角顺序。face_align步骤的aligned_input:true 可显式允许已经对齐的 112×112 uint8 裁剪；不能将普通人脸框裁剪当成对齐图。无样本调用仍返回空列表，不执行引擎。

### RT-DETR、YuNet 与 SFace

| 模型 | 输入 | 输出 / 处理 |
| --- | --- | --- |
| `rtdetr` | RGB float32 NCHW，0..1，默认Pillow双线性抗锯齿resize；`orig_target_sizes` `[N,2]` 为原图 width,height，int64 或 float32 必须匹配导出 | 原始 RT-DETR **组合 deploy 后处理导出**的 labels int64 `[N,K]`、boxes float32 `[N,K,4]`、scores float32 `[N,K]`；boxes 已是原图 xyxy。仅阈值/类别过滤与可选裁边，不再次 sigmoid、缩放或 NMS |
| `yunet_2023` | BGR float32 原始像素 NCHW；固定输入边长为32倍数，本 SDK 拉伸 resize 后恢复原图坐标 | `cls/obj/bbox/kps` × stride 8/16/32 共12路 float32 `[N,H/s*W/s,C]`，C 为1/1/4/10；解码关键点与检测框，分数 sqrt(clamp(cls)*clamp(obj))，float框 NMS |
| `sface_2021` | 五点相似变换到112×112，RGB float32 原始像素，无 /255、均值或标准差外部变换 | float32 `[N,D]`，示例 D=128；可选 L2 归一化 |

示例 ONNX 使用动态 batch，YAML 配置 `mode: dynamic`、`size: 2`，每批最多两张，不补齐。候选数和框维度若为符号，初始化时校验已知维度，推理后检查 labels/scores 的候选数一致且 boxes 为 `[N,K,4]`；不将候选数写死为 300。

RT-DETR 的 bare `pred_logits/pred_boxes` 导出以及 RF-DETR `dets/labels` 是其他契约，不会自动猜测或兼容。COCO 示例使用连续80类，person ID=0；自定义标签或导出要重新核对。

源码依据固定为 [OpenCV 4.10.0 YuNet 实现](https://github.com/opencv/opencv/blob/4.10.0/modules/objdetect/src/face_detect.cpp) 与 [OpenCV 4.10.0 对齐/特征实现](https://github.com/opencv/opencv/blob/4.10.0/modules/objdetect/src/face_recognize.cpp)。模型版本限定 `face_detection_yunet_2023mar.onnx` 和 `face_recognition_sface_2021dec.onnx` 的浮点契约；不支持旧 YuNet 或 int8 导出。YuNet 官方可直接处理32倍数补边；SDK 为固定批处理采用拉伸 resize，其小脸表现与官方原尺寸补边可能不同。SDK 使用浮点框 NMS，官方实现先取整，边界例子的抑制结果可能不同。

RT-DETR 参考 [官方 export_onnx.py](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/tools/export_onnx.py) 和 [deploy 后处理](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py)。部署时记录实际导出源码 commit、checkpoint、类别映射和预处理版本，示例 `rtdetr-deployed-v1` 是本 SDK 的契约标识，不代表已固定上游 checkpoint。

### 示例配置与模型来源

使用 `configs/rtdetr.yaml`、`configs/yunet.yaml`、`configs/sface.yaml`。这些配置是部署模板，使用前须核对路径、输入输出名称和形状。示例模型契约：YuNet 2023mar固定640×640，SFace为data `[1,3,112,112]` → fc1 `[1,128]`；其它导出仍需重新核验。YuNet / SFace 示例权重为固定 batch=1，不能把 YAML 改为 batch=4 或 dynamic。RT-DETR 示例要求实际 ONNX 支持动态 batch；若使用静态导出，应按实际 batch 修改策略。引擎在初始化时拒绝静态维度不匹配。

model仅包含name/path；preprocess/postprocess必须声明有序steps。SDK不管理模型版本、来源、SHA256、许可或采用状态，不提供engine.model_record，也不在加载时做文件哈希校验。文件来源与采用记录由上层应用按自身需求管理。YAML只选择已注册处理器和严格参数，不执行任意脚本。输入输出名称、dtype、shape与CUDA provider检查仍保留，避免执行错误契约。手动验证工具可在独立报告中计算哈希以复现实验，不作为SDK配置或加载条件。

YuNet 目录声明 [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE)，SFace 目录声明 [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE)，具体 SFace 权重训练来源仍有 [公开待澄清事项](https://github.com/opencv/opencv_zoo/issues/313)。部署前须单独核实具体权重的来源和使用许可。

RT-DETR deploy 输出已经换算为原图坐标，前处理必须使用无 padding 的直接 resize；不要替换为 letterbox 或添加 padding。Model 管道连接与解码阶段都会检查该限制。

## Warmup

`runtime.warmup` 默认 true，`runtime.warmup_runs` 默认 3，次数必须为正整数。
Model 在引擎加载、处理流程和Batcher配置完成后触发 `Engine.warmup()`，不执行图像前后处理。默认输入由引擎的 `create_warmup_inputs()` 按契约名称、dtype、shape创建零张量。
固定 batch 使用配置大小，动态 batch 使用 1；非 batch 输入维度未确定时需补齐尺寸或关闭 warmup。
warmup 异常直接向上传递；模型若不接受全零输入，应关闭 warmup，由调用方使用有效样本预热。
直接创建 Engine 不自动 warmup。默认预热会增加 Model 构造耗时。
预热实现位于 `BaseEngine.warmup()`；Model 按配置触发，引擎也可独立调用 `engine.warmup()`。内置 ONNX/OM 共用这套逻辑，预热输入输出经过公共 `infer()` 校验。
特殊引擎可覆盖 `create_warmup_inputs()` 或整个 `warmup()`。调用方也可先关闭自动预热，再通过 `model.warmup(tensors)` 或 `engine.warmup(tensors)` 提供合法输入；动态尺寸可由这些张量确定。输入须满足物理batch约束，不会自动拆批/补齐。例如固定batch=4时，可以使用Batcher准备的第一批：

```python
model = Model("configs/rfdetr.yaml")  # 配置 runtime.warmup: false
prepared = model.preprocess(image)
chunk = next(model.batcher.split(prepared))
model.warmup(chunk.tensors)
```

提供的张量必须包含全部输入并匹配dtype和已知尺寸；不采用统一随机输入，尺寸、索引、掩码等值约束由模型/引擎或调用方负责。

## 阶段间张量校验

`preprocess → infer → postprocess` 的边界统一检查张量名称、dtype、rank 和 shape，替换自定义处理器或 Predictor 后仍执行检查。失败立即抛错，不继续执行下一阶段，并通过请求日志标记失败阶段。

| 边界 | batch 第一维校验 | 其他维度 |
| --- | --- | --- |
| 预处理返回、逻辑推理入口 | 等于真实样本数量 N，可以小于或大于物理 batch | 匹配模型 inputs 的已知维度 |
| 引擎执行前 | 固定模式必须等于 size；动态模式为 1 到 size | 匹配模型 inputs 的已知维度 |
| 引擎执行后 | 等于本次物理 batch 大小 | 匹配模型 outputs 的已知维度 |
| 逻辑推理返回、后处理入口 | 去除补齐后等于真实样本数量 N | 匹配模型 outputs 的已知维度 |

shape 中的 `null` 表示该维度没有具体尺寸约束，不强制等于固定数值。第一维由上述逻辑/物理 batch 规则单独检查；不同张量第一维均须一致。空请求允许逻辑 batch=0，不进入引擎执行。

模型加载错误由各后端运行时报告；Engine 在初始化时校验 session I/O 元数据。
SDK同时保留Pipeline步骤、模型解码契约和执行边界的张量校验。

## Ascend OM 与 RF-DETR AIPP

`runtime.backend: om` 使用 `ais_bench.infer.interface.InferSession`，`runtime.options.device_id` 默认 0。
在目标 Ascend 环境安装兼容的 CANN、ACLRuntime 和 ais_bench；SDK 基础依赖不安装这些厂商组件。
当前仅支持静态固定 batch，调用 `InferSession(device_id, model_path).infer(feeds)`。
与 ONNX Runtime 一样，OM 按加载会话 → 读取元数据 → 解析张量契约 → 推理的结构执行；两者共用 BaseEngine 的契约解析、张量校验和 warmup。
通过 `get_inputs()` / `get_outputs()` 读取模型真实名称、dtype 和 shape；`inputs` / `outputs` 可省略，或作为可选预期约束。显式 YAML 的顺序不影响执行，输入列表和输出绑定始终使用会话的真实顺序。
`pack.input_name` 和解码步骤的输出名称必须与模型真实名称一致。不根据 shape 猜测输出语义；动态维度和 SDK 不支持的 dtype 在初始化时拒绝。
接口依据 [Ascend ais_bench API 指南](https://gitee.com/ascend/tools/blob/ec952b56d45be60becf2f5f2717db8cf09569502/ais-bench_workload/tool/ais_bench/API_GUIDE.md)。
加载和执行异常向上传递，不复制模型或调整全局线程变量。session 生命周期跟随 Engine，当前未提供显式资源释放接口。

示例见 [rfdetr_om.yaml](../configs/rfdetr_om.yaml)，对应官方 RF-DETR Nano ONNX 转换后的 `models/rfdetr-nano.om`，与 ONNX 示例一样省略 `inputs` / `outputs`。默认采用 384×384、静态 batch=1、91 个 COCO 类别槽位，后处理保留浮点框并裁剪至原图边界，与 ONNX 示例一致。转换后的张量名称、输入格式和 batch 必须按实际 OM 核对；自动读取张量契约不会自动修改处理步骤。
rfdetr_om.yaml的resize→pack步骤形成BGR uint8 NHWC输入，不是启用AIPP的开关：颜色转换及归一化必须已包含在模型静态AIPP中。
不支持动态 AIPP 参数设置。普通浮点NCHW OM配置相应颜色/归一化/layout步骤，不能仅依据文件后缀猜测输入。

Model 仍默认执行 3 次零张量 warmup。准备模型并核实配置后运行：

```python
import cv2
from ai_inference import Model, Plotter

model = Model("configs/rfdetr_om.yaml")
image = cv2.imread("image.jpg")
result = model.predict(image)[0]
Plotter().save("results/om.png", image, result)
```

自动化测试使用模拟 InferSession 验证 SDK 行为；实际 OM 的兼容性和结果须在目标 Ascend 设备验收。

## 独立的处理设备策略

Pipeline顶层device默认cpu，每个步骤可通过device覆盖。当前仅实现CPU步骤，CUDA/TRT明确拒绝。
推理设备仍使用runtime.options.providers；预后处理与推理的设备选择独立。
处理设备直接配置在 preprocess/postprocess 或步骤参数中。
完整语义与可用步骤见 [处理管道](pipeline.md)。
