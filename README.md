# KuaiSAR 精排模型 Scaling：RankMixer + Sparse MoE

面向搜广推算法实习的可复现实验项目。目标是在**相同数据、特征和训练协议**下，回答一个具体问题：扩大 RankMixer 的专家总容量时，能否在可接受的激活计算和推理成本内改善点击预测？项目分别评估推荐流（rec）和搜索结果（search），不把二者混成一个未经验证的“统一模型”。

> **结果状态**：本仓库不预填实验增益。模型代码、数据处理、结果文件和图表只有在实际运行并核验后才算完成；简历中的 AUC、LogLoss、延迟和提升幅度均须来自同一实验协议的真实记录。

## 为什么做这个项目

[RankMixer](https://arxiv.org/abs/2507.15551) 用多头 token mixing 和每个 token 独立的 FFN 建模异质特征，并已研究 Sparse MoE、ReLU 动态路由和稠密训练/稀疏推理。[TokenMixer-Large](https://arxiv.org/abs/2602.06563) 进一步研究了真正的稀疏训练与推理、共享专家等设计。因此，“把 RankMixer 与 SMoE 组合”**不是本项目提出的新算法**。本项目的贡献定位是：在公开数据上实现可复现的精排对照，量化**模型容量、激活计算、训练成本与预测质量**之间的关系，并解释专家路由何时有收益、何时失效。

可检验的假设是：在相近的激活 FFN 计算量下，增加专家数可能改善 AUC/LogLoss，但收益可能因数据规模、专家训练不足或稀疏派发开销而消失。负面结果也需要如实记录。

## 数据与任务

使用 [KuaiSAR 官方数据](https://kuaisar.github.io/)；全量版本下载文件为 `KuaiSAR_v2.zip`，小版本为 `KuaiSAR.zip`。官方统计的全量数据包含约 **1461 万**条推荐交互、**506 万**条搜索交互，跨度约 19 天。这些是**数据集统计**，不是本项目实验结果。数据仅覆盖在该时段使用过搜索与推荐两类服务的用户，结论不能直接推广到整个平台。

| 任务 | 输入记录 | 预测目标 | 主指标 | 补充指标 |
| --- | --- | --- | --- | --- |
| 推荐点击精排 | `rec_inter.csv` | `click` | AUC、LogLoss | 暂无其他已实现指标；UAUC 可作为后续扩展 |
| 搜索点击精排 | `src_inter.csv` | `click_cnt > 0` | AUC、LogLoss | 按搜索会话计算 NDCG@10，并报告有效会话数 |

搜索有 `search_session_id`，可以对**同一会话内已展示的候选**计算排序指标；推荐表没有可靠的请求级候选分组，不从相同时间戳臆造请求并报告 NDCG。UAUC 目前尚未实现；若未来增加，须只统计同时有正负样本的用户并报告覆盖率。`click_cnt` 的二值化规则须在读取数据时检查取值并固定，不能按测试结果改变。字段定义见 [官方数据说明](https://kuaisar.github.io/detailed_statistics.html)。

默认只使用打分时可获得的特征：用户和候选 ID、静态侧信息、搜索关键词与来源、时间上下文，以及推荐侧曝光前已知的视频时长 `duration_ms`。推荐记录中的 `playing_time`、`like`、`forward`、`follow`、`search` 等反馈不能作为预测当前 `click` 的输入；搜索记录中的 `click_cnt` 只能作标签。当前代码按时间戳分位点切分，训练集拟合类别词表；具体口径和限制见 [实验方案](docs/EXPERIMENTS.md)。

## 模型与公平比较

同一套编码与特征输入下比较：

1. **MLP**：常规 embedding + 多层感知机，作为低复杂度基线。
2. **DCNv2**：显式特征交叉基线。
3. **Dense RankMixer**：特征分组形成 token，经多头 token mixing 与 per-token FFN 处理。
4. **RankMixer + top-k Sparse MoE**：将 per-token FFN 扩展为每 token 多专家，从中选取 `top-k` 个参与计算。

比较至少包含两条视角：**等激活计算**看增加总容量是否有效，**等总参数**看稀疏结构是否胜过同规模稠密结构。总参数、交互层参数、embedding 参数和每样本激活参数分开报告。`experts=1` 对应单专家 RankMixer；`experts>1` 且 `top-k<experts` 对应稀疏专家实验。即使理论 FLOPs 下降，PyTorch 的派发和小矩阵计算也可能令实测延迟上升；只有确认未执行未选中的专家、并在固定硬件与 batch 下测量，才能写“推理加速”。

## 运行入口

先准备 **Python 3.10+** 和 PyTorch，在项目根目录激活相应虚拟环境（下文的 `python` 指向该环境）并安装。将官方数据解压到本地，把含 `rec_inter.csv`、`src_inter.csv` 的目录传给 `--data-dir`；程序也会在指定目录下递归查找任务文件。数据文件不随仓库分发。以下命令使用项目的主入口；完整矩阵与记录格式见 [实验方案](docs/EXPERIMENTS.md)。

```bash
python -m pip install -e .

# 无真实数据的流程检查；输出只证明代码可运行，不构成模型效果结论
python -m rmscaling.train --dataset synthetic --task rec --model rankmixer --experts 2 --top-k 1 --skip-test --device cpu --output-dir results/smoke

# 全量推荐任务的单专家 RankMixer：先用验证集比较
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model rankmixer --experts 1 --top-k 1 --max-rows 0 --skip-test --device auto --output-dir results/rec_rankmixer_e1_val

# 全量搜索任务的 top-1 Sparse MoE：先用验证集比较
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task search --model rankmixer --experts 4 --top-k 1 --max-rows 0 --skip-test --device auto --output-dir results/search_rankmixer_e4k1_val
```

`--max-rows` 默认值为 `200000`，它会在时间切分前取 CSV 的前 N 行，仅用于快速检查；`--max-rows 0` 才表示全量。`--skip-test` 只输出验证集结果，用于调参和模型选择；确定配置后去掉该选项，单独运行一次最终测试。小样本运行应标为开发实验，不与全量结果混在一张结论表里。训练后原始配置、指标和延迟写入对应目录的 `metrics.json`。

## 计划产出与简历边界

实验完成后，仓库应有：固定切分与特征清单、配置和随机种子、各模型结果、专家负载统计、质量与成本曲线、失败配置和复现实验说明。建议重点展示 AUC/LogLoss 与实测延迟的 Pareto 图，以及专家数、宽度或深度扩展后的收益曲线。少数几个模型点只能称为 **scaling 实验或曲线**，不能据此宣称发现通用 scaling law。

**简历占位模板，仅在对应实验完成后填写：**

- 基于 KuaiSAR 构建推荐/搜索点击精排实验，完成时间切分、训练集特征编码和 MLP、DCNv2、RankMixer 基线；在推荐/搜索任务分别达到 AUC **[实际值] / [实际值]**、LogLoss **[实际值] / [实际值]**。
- 将 RankMixer per-token FFN 扩展为 top-k 稀疏专家，在**相同激活计算口径**下对比稠密模型；AUC 变化 **[实际值]**，交互层总参数 **[实际值]**、激活参数 **[实际值]**，分析专家使用率与训练/推理成本。
- 如实报告指定设备、精度和 batch 下的推理延迟 **[实际值]**，并通过专家数/宽度/深度消融解释质量与成本的取舍。

未经运行不要写“复现论文效果”“达到工业级性能”“线上 A/B 收益”或论文作者报告的参数量、MFU 与业务指标。

## 参考资料

- [KuaiSAR 官方站点与数据说明](https://kuaisar.github.io/)、[数据发布页](https://zenodo.org/records/8181109)、[KuaiSAR 论文](https://arxiv.org/abs/2306.07705)
- [RankMixer 论文](https://arxiv.org/abs/2507.15551)
- [TokenMixer-Large 论文](https://arxiv.org/abs/2602.06563)
- [DCNv2 论文](https://arxiv.org/abs/2008.13535)

KuaiSAR 官方标注为 **CC BY-NC-SA 4.0**；请遵守其授权条件，不将原始数据提交到仓库。
