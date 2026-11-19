# 手动调用与验证工具

本目录面向 SDK 使用者和开发者：脚本封装完整场景，参数明确、输出可检查，便于手动执行和修改。通常从仓库根目录运行，并使用装好相应依赖的Python环境；不自动下载模型。参考源码或模型路径由参数明确传入。verify_predictions.py 当前只对照 YOLO，参数为 --config / --model / --image / --report。RF-DETR Nano 原始输出对照由 export_rfdetr.py 完成。

| 工具 | 用途与调用 |
| --- | --- |
| smoke.py | 无权重的共享管道检查：`uv run --locked --no-sync python scripts/smoke.py` |
| predict_all.py | 同一图像调用五个模型并独立标注：`uv run --locked --no-sync python scripts/predict_all.py --image tests/bus.jpg --results results` |
| verify_predictions.py | YOLO真实权重参考对照，输出JSON误差报告 |
| verify_faces.py | YuNet/SFace与OpenCV参考API对照 |
| verify_rtdetr.py | RT-DETR ONNX与官方PyTorch模型对照 |
| export_rfdetr.py | 官方 Nano COCO checkpoint 导出并验证 ONNX 原始输出：`python scripts/export_rfdetr.py` |
| export_rtdetr.py | 本地官方checkpoint导出，输出ONNX和导出记录 |
| [convert_tensorrt.py](convert_tensorrt.py) | 在目标 NVIDIA GPU 环境将 ONNX 转换为 TensorRT engine |
| clean_python_cache.py | 清理仓库 __pycache__ / pyc / pyo：`uv run --locked --no-sync python -B scripts/clean_python_cache.py` |
| test.py | 维护测试的便捷入口：`uv run --locked --no-sync python scripts/test.py all`，另支持quick、smoke、ort |
| ascend_requirements.py | 从 Ascend extra 的 wheel 来源和锁定哈希生成 pip 依赖清单；`--check` 检查文件是否一致 |
| check_dist.py | 检查 wheel/sdist 元数据、全新 wheel 安装和解压后的源码包完整测试；`uv run --locked --no-sync python scripts/check_dist.py dist`；只检查，不发布 |

调用示例、依赖、容差及验证边界见 [测试说明](../docs/testing.md)。smoke本身不需要pytest或ONNX权重，只需基础SDK依赖；test.py的all/quick/ort需要相应测试依赖。

优先通过公开Model/Config/Plotter接口模拟实际使用。参考一致性工具会明确访问原始输出、模型session或官方参考实现来定位差异，这些属于验证细节，不是应用必须依赖的内部接口。模拟结果、CPU参考一致性、数据集效果和GPU验收分别记录。

`examples/`只保留最小逐模型调用；`tests/`负责pytest/CI中的边界、异常和回归检查。test.py 是自动化测试的便捷入口。

processing_reference.py仅供参考对照脚本读取标准步骤参数，实际请求仍通过Model执行配置管道；自定义管道需要自己的参考契约。

## 独立性能测量

`uv run --locked --no-sync python scripts/benchmark.py --config configs/yolo11.yaml --runs 20 --warmup 3`

默认使用本地YOLO11和tests/bus.jpg。工具包装Model公开接口，不继承Model，输出阶段和总耗时的mean/P50/P95/min/max及真实样本吞吐；初始化和预热不计入。详见[日志与性能测量](../docs/logging-and-benchmark.md)。CLI接受普通图片模型；SFace等需要FaceInput的模型通过Python接口传入已准备的输入。

benchmark默认只显示终端表格。添加 `--save results/benchmark_yolo11.json` 可选择保存JSON报告，终端结果仍显示。

## uv 与缓存清理

CPU 开发环境先执行 `uv sync --locked --extra onnx-cpu`。`uv sync --locked` 仅安装 SDK 与开发工具；不安装 ORT 时使用 quick/smoke 验证。导出工具在各自参考依赖环境执行，不将 PyTorch/Ultralytics 加入 SDK 生产依赖。

需要清理缓存时运行 `uv run --locked --no-sync python -B scripts/clean_python_cache.py`；添加 `--dry-run` 仅检查缓存。工具清理整个仓库（包括本地环境）的 Python 字节码，跳过 `.git` 与符号链接。

仅验证 SDK 时使用 `uv run --locked --no-sync python scripts/test.py all -- --ignore=tests/test_service.py`；包含服务的完整验证先执行 `uv sync --locked --extra service --extra onnx-cpu`，服务测试见 [服务说明](../docs/service/README.md#验证)。

## 发行包验收

执行 `uv build --build-constraint .github/build-constraints.txt` 后运行 `uv run --locked --no-sync python scripts/check_dist.py dist`。检查脚本在临时环境中安装基础 wheel 和 HTTP/CPU wheel，再解压 sdist，仅用归档内的文件安装锁定的测试依赖并执行 `scripts/test.py all`。检查会下载依赖，结束后自动清理临时环境。

源码包内的 `test_release_requires_acceptance_of_selected_tag` 读取工作流配置，检查发布任务必须依赖目标 tag 的验收；它不启动 GitHub Actions。`check_dist.py` 可独立在本地运行，现有 Package acceptance 和 GitHub Release 工作流也执行同一检查，任何失败都会阻止后续发布。

正式打标签前，将 `CHANGELOG.md` 中本次发布的变更移入对应版本章节，并同步 `pyproject.toml`、`ai_inference.__version__` 和 `uv.lock`。在 Python 3.12 环境运行 `python scripts/release.py validate --tag vX.Y.Z --source .` 检查版本和发布说明，将 `vX.Y.Z` 替换为目标标签。原生 TensorRT 的设备验收见 [测试说明](../docs/testing.md#原生-tensorrt-验证)。

## ONNX → TensorRT engine

`python scripts/convert_tensorrt.py models/model.onnx models/model.engine --profile 'images=2x3x640x640:4x3x640x640:8x3x640x640'`

工具直接使用 OnnxParser 和 Builder，需在目标 NVIDIA GPU 与 TensorRT 10 环境执行。`--precision fp16` 允许 FP16，默认 fp32；`--workspace-mib` 控制构建 workspace。多个输入用分号分隔，重复 `--profile` 添加 profiles。静态模型无需 profile。转换选项由产物生成者管理，SDK 只消费 engine，不导入转换脚本；详见 [原生 TensorRT](../docs/tensorrt.md)。
