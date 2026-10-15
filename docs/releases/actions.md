# GitHub Actions 发布

Release 正文只从选定 tag 的 CHANGELOG.md 提取：包含对应版本二级标题，截止下一个二级标题；不生成 commit 列表，不在本文件重复发布说明。版本缺失、重复或正文为空时失败。tag 必须是 vMAJOR.MINOR.PATCH，并与 pyproject.toml、公开 __version__ 一致。

## 执行入口

工作流为 `.github/workflows/release.yml`。推送包含此工作流的 `v*` tag 时自动执行；验证成功后会正式发布 GitHub Release 并上传 PyPI，推送 tag 前应完成 Trusted Publishing 配置并确认发布意图。也可在 GitHub 仓库 Actions → GitHub Release → Run workflow，选择默认分支 main，输入已存在的 tag（如 v0.2.1），启动补发。已有 tag 不会因后来添加工作流而自动执行。

## 通过 Actions 上传 PyPI

`.github/workflows/release.yml` 使用 PyPI Trusted Publishing（OIDC），无需在 GitHub Secrets
保存 PyPI API Token。推送新版本 tag 会先发布 GitHub Release，再上传 PyPI；手动运行时，`publish_to` 可选
`github`（默认）、`testpypi` 或 `pypi`。选择包索引也会先发布 GitHub Release。

首次使用需要账号持有人完成一次配置：

1. 在 GitHub 仓库 Settings → Environments 建立 `pypi`；试发布时另建 `testpypi`。
2. 在 [PyPI Publishing](https://pypi.org/manage/account/publishing/) 添加 pending publisher，填写：

   | 字段 | 值 |
   | --- | --- |
   | PyPI project name | `ai-inference` |
   | Owner | `Jasper-T` |
   | Repository name | `ai-inference` |
   | Workflow name | `release.yml`（仅文件名） |
   | Environment name | `pypi` |

   包已存在时，在该项目 Publishing 设置中添加 publisher。首次成功上传会创建项目；
   pending publisher 不预留包名。TestPyPI 是独立站点，须在
   [TestPyPI Publishing](https://test.pypi.org/manage/account/publishing/) 单独配置，environment 填 `testpypi`。
3. 提交并推送工作流与发布修复到默认分支，更新版本和 CHANGELOG 后创建新的版本 tag。
   已有 `v0.2.0` 不含这些修复，不应移动；本次发布使用 `v0.2.1`。
4. 在 Actions → GitHub Release → Run workflow 中选择默认分支，输入新 tag，先选择
   `publish_to=testpypi`，确认试发布和安装，再以相同 tag 选择 `publish_to=pypi`。

构建 job 没有 OIDC 上传权限；它从选定 tag 测试、构建和验收，再传递制品。
`publish-index` job 仅下载同一次运行验收并用于 GitHub Release 的 bundle，核对 SHA256，
仅将 wheel 和源码包交给固定提交号的官方 PyPA Action，不重新构建、不运行源码安装。
只有该 job 获得 `id-token: write`，并使用与目标索引一致的 environment。

TestPyPI 下载验收可在仓库外的新环境执行，`VERSION` 替换为实际版本：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --no-deps --index-url https://test.pypi.org/simple/ 'ai-inference==VERSION'
uv pip install --python .venv/bin/python 'ai-inference[service,onnx-cpu]==VERSION'
.venv/bin/python -I -c 'import ai_inference; from ai_inference.service.app import app; print(ai_inference.__version__)'
```

第一步仅从 TestPyPI 安装 SDK，第二步从默认索引补齐依赖；这里不升级已安装的 SDK。
避免将 TestPyPI 与 PyPI 同时设为候选索引，以免下载到错误来源的同名包。

上传目标已有同名版本时默认报错，不设置 `skip-existing`。网络故障导致部分上传时，先核对
索引现有文件和本次制品；工作流不会自动覆盖或跳过。正式发布身份校验和端到端上传只能在
配置好账号后于 GitHub Actions 验证。

参考：[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)、
[首次发布创建项目](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)。

## 源码与工具来源

验证 job 从默认分支获取发布工具，再单独检出指定的 `refs/tags/<tag>` 到 source/。所有 SDK/服务测试、依赖锁、构建源码和 CHANGELOG 均来自 tag。发布工具 `tooling/scripts/release.py`、测试及构建约束来自默认分支；记录这份工具的提交号，发布 job 必须检出同一提交，不能中途换成最新 main。这也支持没有 release.py 或新工作流的旧 tag。

验证使用 Ubuntu 24.04、Python 3.12 和 uv 0.12.5：包含 `service` extra 的新 tag 使用 uv sync --locked --extra service，旧 tag 使用 uv sync --locked；随后执行 uv lock --check、完整 Ruff lint/format、scripts/test.py all、发布工具回归及 uv build。新 tag 额外验证服务模块和 HTTP 测试依赖可导入，确保服务回归实际执行。固定隔离构建的 setuptools/wheel 版本，并以源码提交时间设置 SOURCE_DATE_EPOCH；新工作目录避免上传本机旧 dist/。Actions 不下载模型权重，合成 ONNX 回归不代表 GPU、Ascend 实机或真实权重验收。

wheel、源码包经过内容检查，允许 `ai_inference/service`、服务文档和源码包中的服务测试，禁止构建目录副本、权重及 Python 字节码；再生成 SHA256SUMS。跨 job bundle 还含 tag/源码提交元数据及提取的 notes.md，仅 wheel、源码包、SHA256SUMS 上传 Release。

## 失败与重试

验证 job 仅有 contents:read；发布 job 才获得 contents:write。外部 Actions 固定到提交号，输入通过环境变量/参数传递，不直接插入 shell。按 tag 排队且不取消正在发布的运行。发布前及最后转为公开前核对远端 tag 仍指向已验证源码。

新 Release 先建立草稿，上传并下载核验附件后才正式公开。失败留下草稿供检查；下次可更新同 tag 草稿的同名附件，发现未知附件则失败，不自行删除。未绑定该 tag 的其他草稿不会被触碰。

同版本已正式发布时，只有附件名称、全部文件字节与正文完全一致才按成功处理，不做修改；任何差异都失败，不能静默覆盖。源码归档或构建环境发生变化可能产生不同字节，即使版本号相同也会拒绝覆盖，应检查原因并使用新版本。不同 tag 工作流无法防止人为改动；发布期间不要手动移动 tag 或修改 Release。

## 本地复现

### PyPI 发布前验收

标签推送在验收后发布 GitHub Release 和 PyPI；手动运行可选择目标索引。`.github/workflows/package.yml` 在推送和 PR 时使用
Python 3.10 / 3.12 验证源码、构建和安装；包含 `scripts/check_dist.py` 的新 tag 也会在
GitHub Release 发布前执行以下验收，旧 tag 保持原有检查。

在全新检出、没有旧构建产物的目录中运行：

```bash
uv sync --locked --extra service
uv run --locked --no-sync python scripts/test.py all -- -rs
uv build --build-constraint .github/build-constraints.txt
uv run --locked --no-sync twine check --strict dist/*.whl dist/*.tar.gz
uv run --locked --no-sync python scripts/check_dist.py dist
```

Twine 检查发行元数据和 README 渲染。`check_dist.py` 额外检查 wheel 和源码包的 MIT
许可声明、LICENSE 文件、版本和依赖一致性、源码包内的 Ascend 安装文件，并拒绝任何
直接 URL 依赖（包括 extras）。随后创建两个临时环境，分别安装基础 wheel 和
`[service,onnx-cpu]`，检查依赖、公开导出、版本及服务路由，导入位置必须在新环境内。
临时环境从索引解析依赖，需要网络，不复用开发环境的包。此检查不代表 GPU 或 Ascend 实机验收。

Ascend 的厂商 wheel 位于 `requirements/ascend.txt`，不进入 `Requires-Dist`；空的
`ascend` extra 仅保留名称兼容。发布后使用者需要另外安装厂商依赖、CANN 和驱动。

验收通过后仍须确认 PyPI 包名和版本可用、维护账号权限，再配置 API token 或
Trusted Publishing。若先做 TestPyPI 试发布，须显式选择 TestPyPI，并在独立环境验证
上传制品；本地验收和 Package acceptance 工作流不会上传；Release 工作流在新版本标签推送时上传 PyPI，手动运行时按所选目标发布。已存在的版本不能覆盖，
发布新内容应更新项目版本、公开 `__version__` 和 CHANGELOG，并创建匹配的新 tag。

### GitHub Release 制品复现

从选定 tag 创建全新的临时检出（不能混入未跟踪文件）。在该目录运行与工作流相同的检查，然后：

```bash
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
uv build --build-constraint /path/to/tooling/.github/build-constraints.txt
python3.12 /path/to/tooling/scripts/release.py prepare \
  --source . --tag v0.2.1 --output /path/to/new-bundle
```

prepare 仅校验和准备文件，不发布。publish 子命令会使用 runner 的 gh / GH_TOKEN 创建或更新 Release，只应在明确准备发布时执行。
