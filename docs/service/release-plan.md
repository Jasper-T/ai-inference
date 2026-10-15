# v0.2.1 统一发布与验收流程

SDK 与 HTTP 服务使用同一个 `ai-inference` 发行包、版本、`vMAJOR.MINOR.PATCH` tag 和根目录 [CHANGELOG](../../CHANGELOG.md)。服务位于 `src/ai_inference/service/`，`service` extra 只控制 HTTP 依赖安装，代码随统一 wheel 发布。基础 SDK 导入不加载 HTTP 依赖。

本轮发布版本为 `0.2.1`，包含 Ascend 厂商 wheel 独立安装、MIT 许可证、发行包验收和 GitHub Actions Trusted Publishing 配置。`pyproject.toml`、公开 `__version__` 与新标签 `v0.2.1` 必须一致；保留已有 `v0.2.0` 标签和历史记录，不移动或重新发布旧标签。

根 CHANGELOG 的 `0.2.1` 章节记录本轮变更，日期为 2026-10-15；`0.2.0` 保留 SDK 与 HTTP 服务统一发行及服务边界修复的历史。按下列顺序完成本地验收、提交和打标签，并以 GitHub Actions 的完成状态及发布附件为准。

## 执行顺序与完成标准

| 顺序 | 工作 | 完成标准 |
| --- | --- | --- |
| 1 | 修复下载时间预算 | 连接、TLS、响应头与响应体读取受剩余预算约束；慢分块头和 `Connection: close` 场景有回归；超时返回 504 后正常恢复准入，不留下继续下载的后台任务 |
| 2 | 修复模型相关输入校验 | 不支持关键点的模型拒绝关键点；SFace 缺失或退化五点、错误的已对齐裁剪返回带图片索引的 422；整批拒绝且模型故障仍保留 500 |
| 3 | 整理本轮发布文档 | 把已完成的 `Unreleased` 内容归入 `0.2.1` 章节，保留 `0.2.0` 历史；同步版本、安装示例、许可证与 Ascend 安装迁移说明 |
| 4 | 完成回归和安装验收 | 全套 SDK/服务测试、smoke、Ruff、锁文件与发布工具检查通过；基础安装不依赖 HTTP；wheel/sdist 安装后在仓库外验证组合 extras、启动及接口 |
| 5 | 提交到 main | 修复、测试和文档全部提交并推送；工作区干净，本地 main 与远程一致；不需要再次重写主线历史 |
| 6 | 创建并推送新标签 | 确认远程 `v0.2.1` 不存在，创建指向上述验收提交的 annotated tag，再推送标签触发自动发布 |
| 7 | 检查发布结果 | GitHub Actions 验证和发布 job 成功；Release 正文包含 SDK 与服务完整变更；wheel、sdist 和 SHA256SUMS 齐全，下载后按哈希核对 |

阶段 1–4 通过后才执行阶段 5–7。服务边界修复已包含在 `0.2.0`；本轮继续执行对应回归，并验证新增发行包检查，不把历史测试结果当作新版本验收结果。CPU 验收与 GPU/Ascend 实机验收分别表述，不把 OM 模拟测试列为实际设备验证。

发布范围保持现有 SDK、可选 HTTP 服务、四种输入、SFace 关键点、示例和模型准备文档。不在本轮新增多模型路由、热更新、动态合批或推理取消。

标签推送在验收后发布 GitHub Release 并上传 PyPI；手动运行 Release 工作流时也可选择 `publish_to=testpypi` 或 `publish_to=pypi`，通过 Trusted Publishing 上传验收后的同一组制品。账号和 environment 配置见[发布说明](../releases/actions.md#通过-actions-上传-pypi)。`pip install ai-inference` 从包索引安装，不能用来证明 GitHub 上的新制品已发布到该索引。

## 标签与工作流操作

已有 `v0.2.0` 不含本轮 PyPI 修复，保留其历史，不将其作为本轮上传入口。不要在阶段 1–5 完成前使用 `Run workflow` 输入 `v0.2.1`。

以下命令仅在修复、验收和 main 推送完成后执行：

```bash
git ls-remote --tags origin refs/tags/v0.2.1
# 上一步确认没有该标签，再创建并推送。
git tag -a v0.2.1 -m "Release v0.2.1"
git push origin v0.2.1
```

标签推送会触发构建并正式发布，不是单纯保存版本标记。发布失败先查看日志：若只是依赖下载等临时失败且源码未变，可以重跑同一任务；若必须修改源码，应使用后续版本号重新验收并创建新标签，不能把已经验收或已发布的标签静默移到其他提交。

## 安装与迁移

本轮 `0.2.1` 提供 `service` 与后端组合的 extras。取得新 wheel 后可执行 `pip install './ai_inference-0.2.1-py3-none-any.whl[service,onnx-cpu]'`。GPU 使用 `service,onnx-gpu`。`ascend` 保留为空的兼容 extra，厂商运行时须从同版本源码包的 `requirements/ascend.txt` 单独安装，CANN 与驱动仍须另行准备；三个后端 extras 在 uv 中仍互斥。HTTP 依赖可与后端独立选择。

本地开发和 CI 使用：

```bash
uv sync --locked --extra service
uv lock --check
uv run --locked --no-sync ruff check src tests scripts examples
uv run --locked --no-sync ruff format --check src tests scripts examples
uv run --locked --no-sync python scripts/test.py all
uv build --build-constraint .github/build-constraints.txt
uv run --locked --no-sync twine check --strict dist/*.whl dist/*.tar.gz
uv run --locked --no-sync python scripts/check_dist.py dist
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

迁移、安装 extra、文档、两项边界修复和发布检查已调整。生产网关及设备验收不由源码测试替代。CUDA/OM 只有目标设备验收后才能列为已验证能力。

## 构建与发布入口

使用现有 [.github 工作流说明](../releases/actions.md)，不新增服务 tag 或独立 Release。构建前确保输出目录不含旧制品；本地可选择新的临时输出目录检查候选包：

```bash
uv build --out-dir /tmp/ai-inference-candidate-dist --build-constraint .github/build-constraints.txt
```

正式工作流在新的 checkout 中向 `dist/` 构建，由 `scripts/release.py` 检查 wheel/sdist 并生成 `SHA256SUMS`。服务模块与测试已允许进入统一源码包，构建副本、权重和字节码仍被拒绝。工作流检查被选 tag 是否声明 `service` extra：新 tag 验证完整 SDK/服务，旧 tag 按其自身依赖和源码验证，支持历史 SDK 补发。

从候选 wheel 在仓库外安装，确认 `ai_inference.service` 来自安装环境。用固定样本检查 readiness、model 信息和四种接口。发布制品仍为统一 wheel、sdist 和 `SHA256SUMS`；先草稿、上传校验，再公开，不能静默覆盖同版本已发布制品。本轮新建 `v0.2.1`，保留 `v0.2.0`；后续变更继续按新版本号演进。PyPI 上传复用同一次运行的制品，须先完成 Trusted Publishing 配置。

## 灰度与回滚

上线前保留上一版 wheel、依赖记录、模型与 YAML，以及对应启动入口。候选实例 readiness 成功并通过固定样本后切流。设备无法容纳两份模型时，在维护窗口停止旧进程再启动候选实例。

出现结果契约不一致、500/503/504 比例、时延或设备资源异常时，恢复上一组制品、依赖、配置和入口，再检查 readiness 与样本。HTTP 断连不保证设备推理停止，停止或回滚前须先摘除实例并等待已接纳请求完成。
