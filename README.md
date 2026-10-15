# ai-inference 维护文档

此分支只维护发布操作、审查记录和发布计划。SDK 源码、面向使用者的文档、工作流、脚本、测试及 CHANGELOG 在 [main](https://github.com/Jasper-T/ai-inference/tree/main)。该公开仓库的所有分支均公开，此分支不提供访问隔离。

## 文档

- [发布操作](docs/releases/actions.md)：Trusted Publishing、验收门禁、失败重试和旧版本处理。
- [0.2.2 发布计划](docs/service/release-plan.md)：统一 SDK / HTTP 服务发布与安装验收。
- [服务审查记录](docs/service/review.md)：下载预算、人脸关键点校验与历史验收记录。

## 维护方式

在独立目录检出此分支进行文档更新，不将此分支合并回 main，也不在此维护另一份 SDK 代码。文档中的源码链接指向 main；修改发布流程时，同时核对 main 的工作流和脚本。

```bash
git clone --single-branch --branch maintainers git@github.com:Jasper-T/ai-inference.git ai-inference-maintainers
```

0.2.2 候选源码审查提交为 `365301e`。本地 449 项测试、严格发行包检查和独立安装验收通过；[Linux x86_64 / ARM64 和 Windows 的 10 项原生平台验收](https://github.com/Jasper-T/ai-inference/actions/runs/38688508254)全部通过。此记录不代表真实 GPU / Ascend 设备推理验收，也不表示已发布 0.2.2。
