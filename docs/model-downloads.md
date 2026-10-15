# 模型下载与配置对应关系

以下来源按 2026-10-15 查询的官方仓库及本项目配置整理。SDK 和 HTTP 服务使用相同的本地模型配置；不会自动下载权重。文件放入仓库根目录 `models/`，配置中的模型路径相对 YAML 文件解析。

## 默认模型

| 模型 / 配置 | 官方下载 | 放入 `models/` 的运行文件 | 准备方式 |
| --- | --- | --- | --- |
| YOLO11n / [yolo11.yaml](../configs/yolo11.yaml) | [PT checkpoint](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt)、[ONNX](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx) | `yolo11n.onnx` | 本项目已验证的流程是下载 PT，按 [YOLO 导出说明](testing.md#下载与导出) 导出；官方现成 ONNX 是可选来源，未在本项目验收 |
| RF-DETR Nano COCO / [rfdetr.yaml](../configs/rfdetr.yaml) | [checkpoint_best_regular.pth](https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth) | `rfdetr-nano.onnx` | 下载为 `models/checkpoint_best_regular.pth`，按 [RF-DETR 导出说明](testing.md#rf-detr-真实模型) 执行 `scripts/export_rfdetr.py` |
| RT-DETR R18 / [rtdetr.yaml](../configs/rtdetr.yaml) | [rtdetr_r18vd_dec3_6x_coco_from_paddle.pth](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth) | `rtdetr-deployed-v1.onnx` | 使用固定参考源码和 [RT-DETR 导出说明](testing.md#cpu-真实权重正确性对照)；运行 `scripts/export_rtdetr.py` |
| YuNet 2023mar / [yunet.yaml](../configs/yunet.yaml) | [ONNX 文件](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | `face_detection_yunet_2023mar.onnx` | 下载即可，无需自行导出 |
| SFace 2021dec / [sface.yaml](../configs/sface.yaml) | [ONNX 文件](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) | `face_recognition_sface_2021dec.onnx` | 下载即可；输入图片需提供五点关键点，或显式配置为已对齐裁剪 |

`.pt` / `.pth` 是导出源权重，不能直接填入本项目 ONNX Runtime 配置，也不能靠改后缀转换成 ONNX。RF-DETR 与 RT-DETR 的表格链接提供 checkpoint，不是可直接运行的 ONNX。

导出在单独环境中安装工具依赖，具体版本和参数见 [测试与复现](testing.md)。不需要把 PyTorch、Ultralytics 或 RF-DETR 导出依赖加入 SDK/服务运行环境。YOLO 需匹配当前 640×640、非内置 NMS 的原始检测输出；RT-DETR 需匹配 deploy 输入输出，不能直接替换为 Ultralytics 的 RT-DETR 导出模型。

## 官方来源与文件校验

- YOLO11：[Ultralytics v8.4.0 发布文件](https://github.com/ultralytics/assets/releases/tag/v8.4.0)。官方提供 PT 和 ONNX；使用预导出 ONNX 时另行校验其输入输出与当前配置及样本结果。
- RF-DETR：[Roboflow 模型权重注册表](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/assets/model_weights.py)。Nano checkpoint 的官方 MD5 为 `fb6504cce7fbdc783f7a46991f07639f`，本仓库导出脚本会检查它。
- RT-DETR：[固定参考版本的 PyTorch Model Zoo](https://github.com/lyuwenyu/RT-DETR/blob/df2a96c8c3a098567a1149bf0248da6efe6d495d/rtdetr_pytorch/README.md)。使用表中的 R18 checkpoint，与本仓库导出脚本默认配置对应。
- YuNet：[OpenCV Zoo 模型目录](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。下载指定的 `2023mar` FP32 文件，与当前配置对应；不要直接替换为新版或 int8 变体。
- SFace：[OpenCV Zoo 模型目录](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)。下载不带 `_int8` / `_int8bq` 后缀的文件。

OpenCV Zoo 的 ONNX 文件使用 Git LFS，普通 `raw.githubusercontent.com` 链接可能返回文本指针；上表使用 `media.githubusercontent.com` 获取实际权重。[YuNet LFS 记录](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)和 [SFace LFS 记录](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)提供下列 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

这些链接在对应官方页面或权重注册记录中核对过；本次没有重新下载所有二进制文件，也没有把链接可见性等同于模型兼容性验收。

## RF-DETR Ascend OM

[rfdetr_om.yaml](../configs/rfdetr_om.yaml) 对应官方 RF-DETR Nano COCO ONNX 转换后的 `models/rfdetr-nano.om`，与 `rfdetr.yaml` 使用相同模型来源，不是另一个业务专用模型。先按上面的 RF-DETR 链接下载 checkpoint 并导出 `rfdetr-nano.onnx`，再在目标 Ascend/CANN 环境转换成 OM；本仓库未提供可直接下载的通用 OM 制品。

默认示例按 Nano 的 384×384、静态 batch=1 和 91 个 COCO 类别槽位配置。当前 OM YAML 使用 BGR uint8 NHWC 输入，转换时必须嵌入静态 AIPP，包含与 ONNX 前处理相同的 BGR→RGB 和归一化。若仅将 ONNX 转换为普通 float32 NCHW OM、没有嵌入 AIPP，应沿用 ONNX 配置的颜色、归一化和 layout 步骤，并切换后端与文件路径。实际张量名、batch、AIPP 和数值一致性须在目标设备核对。

下载后的 SDK 示例见 [examples](../examples/README.md)，HTTP 服务启动与 SFace 输入见 [服务说明](service/README.md)。
