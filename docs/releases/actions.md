# GitHub Actions 发布

Release 正文只从选定 tag 的 CHANGELOG.md 提取：包含对应版本二级标题，截止下一个二级标题；不生成 commit 列表，不在本文件重复发布说明。版本缺失、重复或正文为空时失败。tag 必须是 vMAJOR.MINOR.PATCH，并与 pyproject.toml、公开 __version__ 一致。

## 执行入口

工作流为 `.github/workflows/release.yml`。推送包含此工作流的 `v*` tag 时自动执行；验证成功后会正式发布 GitHub Release，推送 tag 前应确认发布意图。也可在 GitHub 仓库 Actions → GitHub Release → Run workflow，选择默认分支 main，输入已存在的 tag（如 v0.2.0），启动补发。已有 tag 不会因后来添加工作流而自动执行。

## 源码与工具来源

验证 job 从默认分支获取发布工具，再单独检出指定的 `refs/tags/<tag>` 到 source/。所有 SDK/服务测试、依赖锁、构建源码和 CHANGELOG 均来自 tag。发布工具 `tooling/scripts/release.py`、测试及构建约束来自默认分支；记录这份工具的提交号，发布 job 必须检出同一提交，不能中途换成最新 main。这也支持没有 release.py 或新工作流的旧 tag。

验证使用 Ubuntu 24.04、Python 3.12 和 uv 0.12.5：包含 `service` extra 的新 tag 使用 uv sync --locked --extra service，旧 tag 使用 uv sync --locked；随后执行 uv lock --check、完整 Ruff lint/format、scripts/test.py all、发布工具回归及 uv build。新 tag 额外验证服务模块和 HTTP 测试依赖可导入，确保服务回归实际执行。固定隔离构建的 setuptools/wheel 版本，并以源码提交时间设置 SOURCE_DATE_EPOCH；新工作目录避免上传本机旧 dist/。Actions 不下载模型权重，合成 ONNX 回归不代表 GPU、Ascend 实机或真实权重验收。

wheel、源码包经过内容检查，允许 `ai_inference/service`、服务文档和源码包中的服务测试，禁止构建目录副本、权重及 Python 字节码；再生成 SHA256SUMS。跨 job bundle 还含 tag/源码提交元数据及提取的 notes.md，仅 wheel、源码包、SHA256SUMS 上传 Release。

## 失败与重试

验证 job 仅有 contents:read；发布 job 才获得 contents:write。外部 Actions 固定到提交号，输入通过环境变量/参数传递，不直接插入 shell。按 tag 排队且不取消正在发布的运行。发布前及最后转为公开前核对远端 tag 仍指向已验证源码。

新 Release 先建立草稿，上传并下载核验附件后才正式公开。失败留下草稿供检查；下次可更新同 tag 草稿的同名附件，发现未知附件则失败，不自行删除。未绑定该 tag 的其他草稿不会被触碰。

同版本已正式发布时，只有附件名称、全部文件字节与正文完全一致才按成功处理，不做修改；任何差异都失败，不能静默覆盖。源码归档或构建环境发生变化可能产生不同字节，即使版本号相同也会拒绝覆盖，应检查原因并使用新版本。不同 tag 工作流无法防止人为改动；发布期间不要手动移动 tag 或修改 Release。

## 本地复现

从选定 tag 创建全新的临时检出（不能混入未跟踪文件）。在该目录运行与工作流相同的检查，然后：

```bash
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
uv build --build-constraint /path/to/tooling/.github/build-constraints.txt
python3.12 /path/to/tooling/scripts/release.py prepare \
  --source . --tag v0.2.0 --output /path/to/new-bundle
```

prepare 仅校验和准备文件，不发布。publish 子命令会使用 runner 的 gh / GH_TOKEN 创建或更新 Release，只应在明确准备发布时执行。
