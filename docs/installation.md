# 安装指南

首次体验推荐使用 [README Quick Start](../README.md#quick-start)。本文介绍源码和 wheel 两类安装方式；所有源码命令均在仓库根目录执行。

## 安装前选择

Python 下限为 **3.10**，已验证 3.10 / 3.11 / 3.12。推荐 Python 3.12；CUDA 11 环境选择 3.10–3.12，CUDA 13 至少使用 3.11。依赖按 Python 版本自动选择适用范围，无需额外的 Python 分组，最低组合见 [依赖兼容说明](configuration.md#最低依赖组合)。

安装时选择一个推理后端，需要 HTTP 服务时再加 `service`：

| extra | 用途与主要要求 |
| --- | --- |
| `onnx-cpu` | CPU 推理，包括 macOS |
| `onnx-gpu-cu11` | NVIDIA CUDA 11.x / cuDNN 8；Python 3.10–3.12；Linux x86_64 或 Windows x64 |
| `onnx-gpu-cu12` | NVIDIA CUDA ≥12.8 / cuDNN 9；Linux x86_64 或 Windows x64 |
| `onnx-gpu-cu13` | NVIDIA CUDA ≥13.0 / cuDNN 9；Python ≥3.11；Linux x86_64 / aarch64 或 Windows x64 |
| `tensorrt-cu12` | [原生 TensorRT 10.16 / CUDA 12](tensorrt.md#安装与适用范围)；CPython 3.10–3.12；Linux x86_64（glibc ≥2.28）或 Windows x64；当前源码新增 |
| `ascend` | 静态 OM 推理；Linux x86_64 / aarch64、CPython 3.10–3.12；需匹配的 CANN、驱动和设备 |
| `service` | 可与一个后端组合，提供 FastAPI、Uvicorn、multipart 和 httpx 依赖 |

每个环境选择一个后端 extra。uv 会检查互斥，使用 pip 时由调用方选择。GPU 后端按目标环境显式选择 CUDA 系列。切换后端时建议新建环境，避免 CPU/GPU ORT 包覆盖同名模块。

驱动、CUDA、cuDNN、CANN 和模型文件需另行准备。CUDA 11 extra 使用 CUDA 11.8 构建的 ORT，CUDA 11.2 仍需目标机器实测；Linux ARM64 CUDA 13 wheel 要求 glibc ≥2.34，Jetson 还需核对 JetPack 和设备兼容性。完整约束及失败行为见 [CUDA 部署说明](configuration.md#cuda-部署与失败) 和 [Ascend OM](configuration.md#ascend-om-与-rf-detr-aipp)。

ORT wheel 的系统要求由发行文件的平台标签决定，安装工具会检查这些标签。当前锁文件在 macOS ARM64 / Python 3.11–3.12 选择 ORT 1.31.0，其 `macosx_14_0_arm64` wheel 要求 macOS ≥14；CUDA 13 的 Linux ARM64 wheel 使用 `manylinux_2_34_aarch64` 标签，要求 glibc ≥2.34。文件见 [CPU ORT 1.31.0](https://pypi.org/project/onnxruntime/1.31.0/#files) 和 [GPU ORT 1.31.0](https://pypi.org/project/onnxruntime-gpu/1.31.0/#files)。pip 按版本范围解析时可能选择兼容旧系统的较早 ORT；`uv sync --locked` 不会自动降级锁定版本。旧系统部署需单独选择、固定并验证兼容的完整依赖组合。

## 1. 源码安装

获取源码，或解压所需版本的源码归档后进入项目目录：

```bash
git clone https://github.com/Jasper-T/ai-inference.git
cd ai-inference
```

默认分支用于获取当前源码；固定版本部署应选择所需 tag 或提交，并保留对应的 `uv.lock`。

### 1.1 使用 uv（推荐）

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。以下命令分别对应不同环境，选择其中一个执行：

```bash
# CPU 运行环境
uv sync --locked --python 3.12 --no-default-groups --extra onnx-cpu

# CPU + HTTP 服务
uv sync --locked --python 3.12 --no-default-groups --extra service --extra onnx-cpu

# CUDA 11 / cuDNN 8
uv sync --locked --python 3.10 --no-default-groups --extra onnx-gpu-cu11

# ONNX Runtime GPU，CUDA 12 / cuDNN 9
uv sync --locked --python 3.12 --no-default-groups --extra onnx-gpu-cu12

# CUDA 13 / cuDNN 9
uv sync --locked --python 3.12 --no-default-groups --extra onnx-gpu-cu13

# 原生 TensorRT，CUDA 12
uv sync --locked --python 3.12 --no-default-groups --extra tensorrt-cu12

# Ascend；自动安装锁定来源的 aclruntime 和 ais-bench
uv sync --locked --python 3.12 --no-default-groups --extra ascend
```

`--python` 选择解释器，`--extra` 选择后端或服务。`--no-default-groups` 排除默认开发工具；开发环境去掉此参数即可。uv 在 `.venv` 中安装项目，源码修改会立即生效。

重复同步时保留相同的后端 extra；省略后端的 `uv sync` 会移除之前选择的可选依赖。运行时复用环境：

```bash
uv run --locked --no-sync python examples/yunet.py
```

示例运行前须按 [模型下载](model-downloads.md) 准备权重。开发与测试命令见 [开发与验证](../README.md#开发与验证)。

### 1.2 使用 pip 本地安装

使用 Python 3.10–3.12 创建独立环境，再安装本地源码。以下为 macOS / Linux shell 命令：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[onnx-cpu]'
```

Windows PowerShell 使用 `py -3.12 -m venv .venv` 和 `.venv\Scripts\Activate.ps1`。如果已有虚拟环境，可直接使用该环境的 Python。

ONNX Runtime GPU 使用对应的 extra，例如 `'.[onnx-gpu-cu12]'`。服务使用 `'.[service,onnx-cpu]'` 等组合。原生 TensorRT 还需配置 [NVIDIA 索引](tensorrt.md#安装与适用范围)。pip 根据依赖范围解析版本，生产部署需要另行固定完整依赖集合。

Ascend 的 pip 安装须同时使用厂商 wheel 清单：

```bash
python -m pip install '.[ascend]' -r requirements/ascend.txt
# HTTP 服务：将上面的 extra 改为 .[service,ascend]
```

`ascend` 严格依赖 `aclruntime==0.0.2` 和 `ais-bench==0.0.2`。uv 自动读取仓库中的厂商来源；pip 不读取 uv 来源配置，需显式传入 [requirements/ascend.txt](../requirements/ascend.txt)，按 Python 与架构选择带 SHA256 的厂商 wheel。

### 1.3 可编辑安装

使用 pip 开发源码时，在已激活的虚拟环境中执行：

```bash
python -m pip install -e '.[service,onnx-cpu]'
```

`-e` 使源码修改立即生效；普通本地安装在修改源码后需重新安装。pip 的可编辑安装不会安装 uv 的 `dev` / `test` 组，完整开发环境推荐使用 uv。

## 2. wheel 安装

适用于直接使用 SDK 或在内部环境分发。SDK wheel 为 `py3-none-any`，但后端及其他依赖仍受 Python、操作系统、架构和设备环境限制。

### 2.1 从 Release 安装

在 [GitHub Releases](https://github.com/Jasper-T/ai-inference/releases) 中选择所需版本，下载 Assets 中的 `ai_inference-<版本>-py3-none-any.whl`。仅在该版本已发布对应资产时使用此方式；需要尚未发布的功能时，从源码安装或自行构建。

以下以 `0.2.5` 文件名为例，在独立虚拟环境中执行，将路径和版本替换为实际下载的文件：

```bash
# CPU
python -m pip install './ai_inference-0.2.5-py3-none-any.whl[onnx-cpu]'

# CPU + HTTP 服务；与上面的命令二选一
python -m pip install './ai_inference-0.2.5-py3-none-any.whl[service,onnx-cpu]'
```

如 Release 提供 `SHA256SUMS`，安装前计算 wheel 的 SHA256，并与清单中该文件的记录比较。macOS 可用 `shasum -a 256 <wheel路径>`，Linux 可用 `sha256sum <wheel路径>`。

ONNX Runtime GPU 安装替换为相应的 `onnx-gpu-cu11` / `onnx-gpu-cu12` / `onnx-gpu-cu13` extra，并确认下载的版本提供该 extra。原生 TensorRT 的 wheel 安装命令见 [TensorRT 安装指南](tensorrt.md#安装与适用范围)，上例的 `v0.2.5` 标签尚未包含该 extra。Ascend 安装还需从相同版本源码取得厂商清单：

```bash
python -m pip install './ai_inference-0.2.5-py3-none-any.whl[ascend]' -r requirements/ascend.txt
```

wheel 不包含仓库的 `configs/`、`examples/`、测试图片或模型权重。需要使用示例时，另行取得相同版本的源码；业务部署可自行提供 YAML 与模型，配置格式见 [配置与模型契约](configuration.md)。

如果所需版本已发布到 PyPI，也可按包名安装，例如 `python -m pip install 'ai-inference[onnx-cpu]==0.2.5'`；实际可用版本以 [PyPI 项目页](https://pypi.org/project/ai-inference/) 为准。

### 2.2 自行构建

在包含 `pyproject.toml` 的源码根目录执行，使用仓库固定的构建工具版本：

```bash
uv build --build-constraint .github/build-constraints.txt
```

构建约束文件随 Git 仓库和源码发行包一起提供，解压源码发行包后也可执行同一命令。

构建结果位于 `dist/`：`.whl` 为安装包，`.tar.gz` 为源码发行包。当前源码版本 `0.2.5` 的 wheel 安装示例：

```bash
python -m pip install './dist/ai_inference-0.2.5-py3-none-any.whl[onnx-cpu]'
```

构建 SDK 无需先安装 GPU 或 Ascend 运行时；目标环境安装时再选后端。wheel 不内嵌后端依赖，也不锁定所有间接依赖。验收自建 wheel 应在仓库外的全新虚拟环境执行，避免源码的可编辑安装掩盖打包问题。发布与产物检查见 [脚本说明](../scripts/README.md)。

### 2.3 离线安装

在与目标环境具有相同操作系统、架构和 Python 版本的联网机器上，准备 SDK wheel，并下载完整依赖集合。以下以 CPU 为例：

```bash
python -m pip download --only-binary=:all: --dest wheelhouse \
  './ai_inference-0.2.5-py3-none-any.whl[onnx-cpu]'
```

将整个 `wheelhouse/` 复制到离线机器，在独立虚拟环境中安装：

```bash
python -m pip install --no-index --find-links=wheelhouse \
  'ai-inference[onnx-cpu]==0.2.5'
```

HTTP 服务在下载和安装两步都选择 `[service,onnx-cpu]`。ONNX Runtime GPU 替换为对应后端。`--only-binary=:all:` 要求每个依赖都提供可用 wheel。

Ascend 的联网下载还需添加 `-r requirements/ascend.txt`，并选择 `[ascend]`，把两个厂商 wheel 及全部依赖一并收入 `wheelhouse/`。离线安装使用 `--no-index --find-links=wheelhouse 'ai-inference[ascend]==0.2.5'`，不再传入包含远程 URL 的厂商清单。CANN、驱动、CUDA、cuDNN 和模型文件分别准备。

原生 TensorRT 的前端包以源码包分发。可在匹配目标平台的联网机器上用 `pip wheel --wheel-dir wheelhouse --extra-index-url https://pypi.nvidia.com '.[tensorrt-cu12]'` 预构建 SDK 与前端 wheel，并收集全部依赖。离线环境使用 `pip install --no-index --find-links=wheelhouse 'ai-inference[tensorrt-cu12]'`，随后执行设备验收。

## 3. 安装检查与首次运行

在已激活的 pip 虚拟环境中检查包版本与导入：

```bash
python -m pip check
python -c "from importlib.metadata import version; from ai_inference import Model; print(version('ai-inference'))"
```

uv 环境使用 `uv pip check`，导入检查使用 `uv run --locked --no-sync python -c "from importlib.metadata import version; from ai_inference import Model; print(version('ai-inference'))"`。

导入通过后，还需运行一次目标模型。源码用户可按 [Quick Start](../README.md#quick-start) 执行 CPU YuNet 示例；pip 源码安装使用 `python examples/yunet.py`。wheel 用户需先准备配置、模型和输入图片，再调用 [检测示例](../README.md#检测调用)。

GPU / Ascend extra 在不支持的 Python 或平台上可能跳过运行时依赖。ORT CUDA 验收使用 `runtime.options.providers: cuda` 执行真实推理；原生 TensorRT 使用 `runtime.backend: tensorrt` 和 [GPU 实测说明](testing.md#原生-tensorrt-验证)。Ascend 在目标 CANN 环境使用 [OM 配置](../configs/rfdetr_om.yaml) 验证。HTTP 服务启动及健康检查见 [服务说明](service/README.md)。

## 原生 TensorRT

原生后端使用独立的 `tensorrt-cu12` extra，实际 wheel 来自 NVIDIA 官方索引。uv 源已配置在项目中；pip 需添加 `--extra-index-url https://pypi.nvidia.com`。版本、支持平台、锁定依赖导出及安装示例见 [原生 TensorRT](tensorrt.md#安装与适用范围)。这与 ORT 的 `trt` provider 安装路径不同。
