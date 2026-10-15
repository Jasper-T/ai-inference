# 手动调用与验证工具

本目录面向 SDK 使用者和开发者：脚本封装完整场景，参数明确、输出可检查，便于手动执行和修改。通常从仓库根目录运行，并使用装好相应依赖的Python环境；不自动下载模型。参考源码或模型路径由参数明确传入。verify_predictions.py 当前只对照 YOLO，参数为 --config / --model / --image / --report。RF-DETR Nano 原始输出对照由 export_rfdetr.py 完成。

| 工具 | 用途与调用 |
| --- | --- |
| smoke.py | 无权重的共享管道检查：`uv run --locked python scripts/smoke.py` |
| predict_all.py | 同一图像调用五个模型并独立标注：`uv run --locked python scripts/predict_all.py --image tests/bus.jpg --results results` |
| verify_predictions.py | YOLO真实权重参考对照，输出JSON误差报告 |
| verify_faces.py | YuNet/SFace与OpenCV参考API对照 |
| verify_rtdetr.py | RT-DETR ONNX与官方PyTorch模型对照 |
| export_rfdetr.py | 官方 Nano COCO checkpoint 导出并验证 ONNX 原始输出：`python scripts/export_rfdetr.py` |
| export_rtdetr.py | 本地官方checkpoint导出，输出ONNX和导出记录 |
| clean_python_cache.py | 清理仓库 __pycache__ / pyc / pyo：`uv run --locked python -B scripts/clean_python_cache.py` |
| test.py | 维护测试的便捷入口：`uv run --locked python scripts/test.py all`，另支持quick、smoke、ort |

调用示例、依赖、容差及验证边界见 [测试说明](../docs/testing.md)。smoke本身不需要pytest或ONNX权重，只需基础SDK依赖；test.py的all/quick/ort需要相应测试依赖。

优先通过公开Model/Config/Plotter接口模拟实际使用。参考一致性工具会明确访问原始输出、模型session或官方参考实现来定位差异，这些属于验证细节，不是应用必须依赖的内部接口。模拟结果、CPU参考一致性、数据集效果和GPU验收分别记录。

`examples/`只保留最小逐模型调用；`tests/`负责pytest/CI中的边界、异常和回归检查。test.py 是自动化测试的便捷入口。

processing_reference.py仅供参考对照脚本读取标准步骤参数，实际请求仍通过Model执行配置管道；自定义管道需要自己的参考契约。

## 独立性能测量

`uv run --locked python scripts/benchmark.py --config configs/yolo11.yaml --runs 20 --warmup 3`

默认使用本地YOLO11和tests/bus.jpg。工具包装Model公开接口，不继承Model，输出阶段和总耗时的mean/P50/P95/min/max及真实样本吞吐；初始化和预热不计入。详见[日志与性能测量](../docs/logging-and-benchmark.md)。CLI接受普通图片模型；SFace等需要FaceInput的模型通过Python接口传入已准备的输入。

benchmark默认只显示终端表格。添加 `--save results/benchmark_yolo11.json` 可选择保存JSON报告，终端结果仍显示。

## uv 与缓存清理

开发环境先执行 `uv sync --locked`。导出工具在各自参考依赖环境执行，不将 PyTorch/Ultralytics 加入 SDK 生产依赖。

需要清理缓存时运行 `uv run --locked python -B scripts/clean_python_cache.py`；添加 `--dry-run` 仅检查缓存。工具清理整个仓库（包括本地环境）的 Python 字节码，跳过 `.git` 与符号链接。

仅验证 SDK 时使用 `uv run --locked python scripts/test.py all -- --ignore=tests/test_service.py`；包含服务的完整验证先执行 `uv sync --locked --extra service`，服务测试见 [服务说明](../docs/service/README.md#验证)。

## Release 工具

release.py（Python 3.12）供 Actions 校验版本、提取 tag 的 CHANGELOG、检查构建包并发布 GitHub Release；默认分支工具可处理没有该脚本的旧 tag。tag/validate/prepare 不发布，publish 会修改 Release。流程与失败保护见[发布工作流说明](../docs/releases/actions.md)。
