# SDK 与 HTTP 服务统一发布规划

SDK 与 HTTP 服务使用同一个 `ai-inference` 发行包、版本、`vMAJOR.MINOR.PATCH` tag 和根目录 [CHANGELOG](../../CHANGELOG.md)。服务位于 `src/ai_inference/service/`，`service` extra 只控制 HTTP 依赖安装，代码随统一 wheel 发布。基础 SDK 导入不加载 HTTP 依赖。

当前源码版本仍为 `0.2.0`，本次迁移记录在 `Unreleased`，尚未发布。建议下一功能版本为 `0.3.0`，正式准备发布时再同时修改 `pyproject.toml` 与公开 `__version__`，填写 CHANGELOG 对应版本及实际日期；不覆盖已有 `0.2.0` 制品。版本号和发布日期目前是规划。

## 安装与迁移

新版本提供 `pip install 'ai-inference[service,onnx-cpu]'`。GPU 或 Ascend 使用 `service` 与相应后端 extra 组合；三个后端 extras 仍互斥。HTTP 依赖可与后端独立选择。

本地开发和 CI 使用：

```bash
uv sync --locked --extra service
uv lock --check
uv run --locked --no-sync ruff check src tests scripts examples
uv run --locked --no-sync ruff format --check src tests scripts examples
uv run --locked --no-sync python scripts/test.py all
```

从仓库源码安装后，启动入口改为：

```bash
AI_INFERENCE_MODEL_CONFIG=/absolute/path/to/configs/yolo11.yaml \
  uv run --locked --no-sync uvicorn ai_inference.service.app:app --workers 1
```

原独立服务包配置及顶层 `service.app:app` 入口已移除。迁移时在新环境安装统一包，替换部署启动入口，并保留原模型 YAML 与权重；旧环境若装过 `ai-inference-service`，应在退役该环境时卸载旧发行包。HTTP `/v1`、图片坐标、响应格式和环境变量保持现有契约。`FastAPI(version="1.0")` 是当前 HTTP 文档版本，不作为制品版本。

## 发布条件

| 阶段 | 验收要求 |
| --- | --- |
| 服务边界 | 修复 [审查记录](review.md) 的下载预算与关键点错误分类问题，补齐对应回归 |
| 自动化验证 | 服务源码参与 lint/format；包含 extra 的新 tag 安装 HTTP 依赖，运行完整 SDK/服务回归及 smoke；缺依赖时不能把跳过当成功 |
| 安装包 | wheel 包含 `ai_inference/service`；sdist 包含服务源码、文档和测试；排除 build、缓存与权重 |
| 可选依赖 | 在干净环境仅安装基础依赖时 SDK 可导入；组合安装 `service,onnx-cpu` 后可在仓库外启动服务；从 sdist 重建并验证 wheel |
| 部署 | 实际 Uvicorn/网关验收四种输入、错误、超限、慢上传、忙请求、断连与停止行为；真实 CPU 模型样本验收 |
| 发布材料 | 版本/tag/源码提交一致；根 CHANGELOG、构建工具约束、uv.lock、SHA256、迁移和回滚说明齐全 |

迁移、安装 extra、文档和发布检查已调整。边界问题仍待修复，生产网关及设备验收不由源码测试替代。CUDA/OM 只有目标设备验收后才能列为已验证能力。

## 构建与发布入口

使用现有 [.github 工作流说明](../releases/actions.md)，不新增服务 tag 或独立 Release。构建前确保输出目录不含旧制品；本地可选择新的临时输出目录检查候选包：

```bash
uv build --out-dir /tmp/ai-inference-candidate-dist --build-constraint .github/build-constraints.txt
```

正式工作流在新的 checkout 中向 `dist/` 构建，由 `scripts/release.py` 检查 wheel/sdist 并生成 `SHA256SUMS`。服务模块与测试已允许进入统一源码包，构建副本、权重和字节码仍被拒绝。工作流检查被选 tag 是否声明 `service` extra：新 tag 验证完整 SDK/服务，旧 tag 按其自身依赖和源码验证，支持历史 SDK 补发。

从候选 wheel 在仓库外安装，确认 `ai_inference.service` 来自安装环境。用固定样本检查 readiness、model 信息和四种接口。发布制品仍为统一 wheel、sdist 和 `SHA256SUMS`；先草稿、上传校验，再公开，不能覆盖同版本已发布制品。本次没有修改版本、创建 tag 或执行发布。

## 灰度与回滚

上线前保留上一版 wheel、依赖记录、模型与 YAML，以及对应启动入口。候选实例 readiness 成功并通过固定样本后切流。设备无法容纳两份模型时，在维护窗口停止旧进程再启动候选实例。

出现结果契约不一致、500/503/504 比例、时延或设备资源异常时，恢复上一组制品、依赖、配置和入口，再检查 readiness 与样本。HTTP 断连不保证设备推理停止，停止或回滚前须先摘除实例并等待已接纳请求完成。
