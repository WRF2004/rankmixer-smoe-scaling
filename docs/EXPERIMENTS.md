# 实验方案与记录规范

本页是运行前约定的实验协议，不是已完成的实验报告。所有结论只从实际生成的结果和配置推导；未运行或缺少对照的表格单元保持 `待测`。

## 1. 研究问题与优先级

**主问题**：对于公开点击精排任务，RankMixer 的 per-token FFN 扩展成 top-k Sparse MoE 后，在相同激活计算量下，增加总专家容量是否稳定改善预测质量？

**次问题**：专家数增加后，训练吞吐、实际推理延迟、显存占用和专家负载如何变化？推荐与搜索任务的结论是否一致？

先完成 **MLP → DCNv2 → Dense RankMixer → RankMixer + Sparse MoE** 四类对照，再扩展宽度、深度和专家数。RankMixer 的 token mixing、per-token FFN 和 Sparse MoE 均有论文先例；本项目重点是公开数据、固定预算、公平比较与诊断。参考 [RankMixer](https://arxiv.org/abs/2507.15551) 和 [TokenMixer-Large](https://arxiv.org/abs/2602.06563)。

## 2. 数据版本与任务定义

主实验使用 [KuaiSAR 全量版](https://zenodo.org/records/8181109) `KuaiSAR_v2.zip`。官方数据表明，全量有 25,877 名用户、14,605,716 条推荐记录、5,059,169 条搜索记录，覆盖 2023-05-22 至 2023-06-10。小版本 `KuaiSAR.zip` 可用于开发，但与全量分开记结果。这里的规模是[数据集发布者统计](https://kuaisar.github.io/)，不是运行统计。

| 任务 | 文件 | 标签 | 基础指标 | 任务特有指标 |
| --- | --- | --- | --- | --- |
| rec | `rec_inter.csv` | `click` | AUC、LogLoss | UAUC 暂未实现；可作为后续扩展 |
| search | `src_inter.csv` | `click_cnt > 0` | AUC、LogLoss | NDCG@10：按 `(user_id, search_session_id)` 的已展示候选分组，当前输出有效会话数 |

搜索表把 `click_cnt` 描述为是否点击展示项；当前代码固定以 `click_cnt > 0` 二值化，正式运行前应检查原始值域。当前 NDCG@10 口径只统计**至少 2 个曝光且至少 1 个正例**的会话；全正例会话也计入，缺少负例时该会话的 NDCG 通常为 1。`metrics.json` 记录 `ndcg_groups`，若需覆盖率，还须另统计测试集全部会话及被排除的无点击/单曝光会话，不能把有效会话均值称为全会话结果。推荐记录虽可能共享分发时间戳，官方未提供可靠请求 ID，故不从时间戳拼接伪请求计算推荐 NDCG。

KuaiSAR 只纳入研究时段内同时使用过推荐和搜索的用户；不要把在此样本上的提升写作全平台收益，也不要从离线指标推断线上 A/B 效果。[字段定义](https://kuaisar.github.io/detailed_statistics.html)应作为数据解析的依据。

## 3. 切分与泄漏防控

当前实现对每个任务**分别**按事件时间排序，默认约 80%/10%/10% 划分训练、验证、测试；边界由 `--val-ratio 0.1 --test-ratio 0.1` 对该任务的时间戳分位点确定，同一时间戳的记录不拆到不同集合。推荐使用 `timestamp`，搜索使用 `search_session_timestamp`；同一搜索会话必须全部落在同一集合。两个任务因记录分布不同，实际时间边界可能不同，因此当前实验是分别建模与评估，不把搜索训练段与推荐测试段混用构造跨任务历史。每次运行把 `metrics.json` 中的 `train_before_ms`、`test_from_ms`、各段行数和数据版本一起保存；不要看过测试结果后改比例。

`--max-rows` **先取交互 CSV 前 N 行，再在这 N 行内按时间戳分位点切分**。默认 N=200000，仅供开发检查；它不是对完整时间范围的随机抽样，也不是只限制训练集。正式实验必须传 `--max-rows 0`，再检查各段时间范围、正例率、用户覆盖与搜索会话数量。不同 `--max-rows` 的结果不能直接比较或写成全量结论。

默认白名单仅使用打分时可知的信息：

- **rec**：用户 ID、候选 item ID、用户/物品静态侧信息、事件发生时间的小时和星期，以及视频在曝光前已知的总时长 `duration_ms`；`click` 只作标签。`playing_time`、`forward`、`like`、`follow`、`search`、`search_item_related` 是当前或后续反馈，不进入当前点击预测。当前实现从曝光记录读取 `duration_ms`，依据其“视频总时长”含义将其视为打分前可知；若后续发现该字段并非曝光前可得，应移除并重跑全部模型。
- **search**：用户 ID、匿名 query 的 `keyword`、`search_source`、候选 item ID、打分前可知的 `item_type` 和物品侧信息、搜索会话时间；`click_cnt` 只作标签，`search_session_id` 只作分组键而非特征。当前实现把整条匿名 `keyword` 序列作为一个类别值编码，并未实现分词级文本模型或 query–caption 语义匹配。
- `user_features.csv` 中的 `search_active_level`、`rec_active_level` 可能来自特定历史窗口；未证实其值在每条曝光前已可用时默认排除。物品 `upload_time` 应早于曝光时间；未来上传、缺失或不可信值按固定规则处理。

当前类别词表及频次阈值只用训练段拟合，再原样应用到验证/测试；未见过的 ID 走 OOV。当前未使用历史反馈或基于标签的统计特征。若以后加入数值归一化、历史行为或目标编码，也只能用训练段拟合，并按用户和严格小于当前事件的时间构造；同一毫秒内的曝光不能互相读取反馈。查询的匿名词序列可在当前请求使用；任何标签或由完整数据算出的点击率、观看统计都不得进入特征。

数据检查至少记录：文件校验值或来源版本、时间范围、标签取值、重复记录、缺失率、会话是否跨段、训练集外新 ID 比例。发现问题先修数据协议，再运行所有模型。

## 4. 模型矩阵与预算对齐

第一轮固定同一输入字段、embedding 维度、token 构造、时间切分比例与优化器协议。当前代码的 token 组由任务字段表确定，不能通过命令行直接改变。基础矩阵：

| 编号 | 模型 | 设置 | 目的 | 状态 |
| --- | --- | --- | --- | --- |
| B1 | MLP | 同一输入 embedding，固定层数/宽度 | 非交叉网络基线 | 待测 |
| B2 | DCNv2 | 同一输入 embedding，固定交叉层数 | 显式交叉基线 | 待测 |
| B3 | RankMixer | `experts=1, top-k=1` | 稠密 per-token FFN 基线 | 待测 |
| M1 | RankMixer + SMoE | `experts=2, top-k=1` | 容量扩展起点 | 待测 |
| M2 | RankMixer + SMoE | `experts=4, top-k=1` | 主稀疏配置 | 待测 |
| M3 | RankMixer + SMoE | `experts=8, top-k=1` | 检查专家利用率与收益递减 | 待测 |

默认先用单一随机种子检查流程，再对**关键结论**至少重复 3 个种子。所有模型在同一训练样本和相同的调参预算下选择验证集模型；训练命令按验证集 LogLoss 选择轮次。如果每个模型的最佳学习率不同，允许在相同候选网格内独立选择并记录。固定最多 epoch 和 batch，另记录早停后的实际训练轮次与总时长，避免把“更多训练”误认作架构收益。探索阶段 `train --skip-test` 只评估验证集，`sweep` 默认跳过测试并只汇总验证指标；锁定配置后，单独运行不带 `--skip-test` 的 `train` 报告最终测试指标。不能根据测试结果继续改配置。

扩展实验只改变一项轴：

| 轴 | 示例配置 | 要回答的问题 |
| --- | --- | --- |
| 宽度 `D` | 64 / 128 / 256 | 稠密加宽与增加专家哪种更划算？ |
| 深度 `L` | 1 / 2 / 3 | 更深模型是否带来稳定收益？ |
| token 数 `T` | 后续扩展：改变特征分组代码 | 特征分组粒度与开销如何变化？当前无直接 CLI 参数 |
| 专家数 `E` | 1 / 2 / 4 / 8，固定 `top-k=1` | 只扩大总容量、近似保持激活专家数时效果如何？ |
| 激活数 `K` | 1 / 2（`E≥2`） | 更多激活计算是否值得？ |

这是实验候选而非已运行配置。数据有限时，优先完成 `E=1/2/4/8` 与一组**等激活计算的加宽稠密对照**。`--shared-expert` 可开启每 token 常驻专家；它增加每样本激活计算，比较时须配一个相近激活预算的稠密/稀疏对照。`--balance-weight` 控制辅助负载损失，可做有/无消融。缩减或改变 FFN 隐层宽度时，明确记录实际总参数、激活 FFN 参数；当前代码尚未测量 FLOPs，不能仅凭 `E×K` 推算完整模型成本。embedding 往往占大量参数，须与交互层参数分列。少量配置点只支撑本数据上的容量曲线，不能拟合或宣称普适 scaling law。

### 成本口径

- **质量**：当前评估全局 AUC、LogLoss，搜索另加会话 NDCG@10。UAUC 属于后续扩展，若实现，须仅纳入同时有正负样本的用户并报告覆盖率。当前未计算置信区间；关键差值未来可用用户聚类 bootstrap 或多种子分布给出不确定性，不将小幅波动直接写成“显著提升”。
- **参数**：`metrics.json` 已记录总参数、embedding 参数、非 embedding 参数，以及 RankMixer 的交互层总参数和估计的每样本激活交互参数；这些参数计数是成本代理，不能直接当成 FLOPs。
- **训练**：当前按 epoch 记录训练耗时与 rows/s，记录总耗时和最佳轮次；未记录峰值显存。正式比较需同时写明样本数、实际 epoch、设备与精度。
- **推理**：当前对同一验证集 batch 做模型前向计时，默认预热 5 次、计时 30 次，输出 median/p95；设备计算会同步。该数值不包括文件读取和特征预处理，也不是完整线上服务延迟。p95 仅 30 次会比较不稳，可通过 `--latency-batches` 增加重复次数；只比较相同设备、batch 和精度下的值。
- **路由**：`metrics.json` 已记录验证集（以及最终运行时的测试集）各层、各 token、各专家的调用次数与使用率。负载变异系数、零调用专家比例和路由熵可据此继续分析；当前并未直接输出这些汇总值。

稀疏计算必须**跳过未选专家的 FFN**。如果实现先计算全部专家再将未选输出乘零，只能称“稀疏门控”，不能宣称 FLOPs 或延迟下降。`top-k=1` 的激活 FFN FLOPs 与同宽单专家模型接近，但路由和派发仍会增加开销；最终结论依据实测。

## 5. 运行命令

先激活 Python 3.10+ 虚拟环境（下文的 `python` 指向该环境），在项目根目录执行 `python -m pip install -e .`。把官方压缩包解压到 `data/raw/`，确认传给 `--data-dir` 的目录下能找到数据文件。以下为当前主命令；每次训练保存配置、切分边界、指标、参数计数和延迟到 `metrics.json`。`--max-rows` 默认 `200000`，`0` 才表示全量。

```bash
# 1. 合成数据 smoke test；只检查前向、反向和日志流程
python -m rmscaling.train --dataset synthetic --task rec --model rankmixer --experts 2 --top-k 1 --skip-test --device cpu --output-dir results/smoke

# 2. 推荐任务强基线与主稀疏配置：只看验证集
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model mlp --max-rows 0 --skip-test --device auto --output-dir results/rec_mlp_val
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model dcnv2 --max-rows 0 --skip-test --device auto --output-dir results/rec_dcnv2_val
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model rankmixer --experts 1 --top-k 1 --max-rows 0 --skip-test --device auto --output-dir results/rec_rm_e1_val
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model rankmixer --experts 4 --top-k 1 --max-rows 0 --skip-test --device auto --output-dir results/rec_rm_e4k1_val

# 3. 搜索任务：相同矩阵另跑一遍，保留会话分组评估
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task search --model rankmixer --experts 1 --top-k 1 --max-rows 0 --skip-test --device auto --output-dir results/search_rm_e1_val
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task search --model rankmixer --experts 4 --top-k 1 --max-rows 0 --skip-test --device auto --output-dir results/search_rm_e4k1_val

# 4. 可选：共享专家与负载均衡损失的消融；共享专家增加激活成本
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model rankmixer --experts 4 --top-k 1 --shared-expert --balance-weight 0.01 --max-rows 0 --skip-test --device auto --output-dir results/rec_rm_e4k1_shared_val

# 5. 扫描宽度和专家数；输出 summary.csv 与各次运行的 metrics.json
python -m rmscaling.sweep --data-dir data/raw/KuaiSAR_v2 --task rec --max-rows 0 --output-dir results/sweep_rec
python -m rmscaling.sweep --data-dir data/raw/KuaiSAR_v2 --task search --max-rows 0 --output-dir results/sweep_search

# 6. 只根据成功运行的真实验证集结果生成图；专家热图读取单次稀疏模型结果
python -m rmscaling.report --summary results/sweep_rec/summary.csv --routing-metrics results/rec_rm_e4k1_val/metrics.json --output-dir results/figures_rec

# 7. 锁定配置后，去掉 --skip-test，单独报告最终测试集；下方 E4/K1 仅为命令示例
python -m rmscaling.train --dataset kuaisar --data-dir data/raw/KuaiSAR_v2 --task rec --model rankmixer --experts 4 --top-k 1 --max-rows 0 --device auto --output-dir results/rec_final_example
```

若因算力先运行默认 `200000` 行，应在结果表里写“开发抽样”，不要与全量 benchmark 或论文结果并列比较。测试时固定 `--device`，不要把 CPU、MPS 与 CUDA 延迟放在同一成本曲线上。

## 6. 待填写结果与图表

下表为**最终测试报告**占位；验证阶段用 `summary.csv` 比较和选型。表中的参数、AUC、延迟等全部从运行文件提取，禁止手工补造。

| 任务 | 模型 | 非 embedding 参数 | 激活交互参数 | AUC | LogLoss | 前向 median/p95（设备、batch） | 种子数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rec | MLP | 待测 | 不适用 | 待测 | 待测 | 待测 | 待测 |
| rec | DCNv2 | 待测 | 不适用 | 待测 | 待测 | 待测 | 待测 |
| rec | Dense RankMixer | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| rec | RankMixer + SMoE E4/K1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| search | Dense RankMixer | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| search | RankMixer + SMoE E4/K1 | 待测 | 待测 | 待测 | 待测 | 待测 |

当前 `rmscaling.report` 可从 sweep 的 `summary.csv` 生成三幅基于**验证集 AUC** 的散点图，并可选读取一个稀疏模型的 `metrics.json` 生成第一层专家使用率热图。根据验证图锁定最终配置后，单独报告测试集结果：

1. **`val_auc_vs_latency.svg`**：横轴为实测单 batch 前向 median 延迟，纵轴为验证集 AUC。不能把它称作端到端服务延迟。
2. **`val_auc_vs_parameters.svg`**：横轴为非 embedding 总参数（对数尺度），纵轴为验证集 AUC。
3. **`val_auc_vs_active_parameters.svg`**：横轴为每样本预计激活的非 embedding 参数（对数尺度），用于与总容量图配合判断稀疏化收益。这是参数计数代理，不等于实际 FLOPs。
4. **`expert_utilization.svg`**：第一层各 token 的专家使用率热图。要分析全部层，继续查看原始 `expert_counts`。

分析每个结论时写明对照模型、固定了什么预算、实际差值及其不确定性。若稀疏模型 AUC 更好但延迟更慢，应直说“质量提升但服务成本未改善”。

## 7. 消融与面试问题

当前可直接做的消融：比较 `E=1/2/4/8`、`top-k=1/2`、`--shared-expert` 有/无、`--balance-weight` 设为 `0` 或正值。移除 token mixing、把 per-token FFN 改为共享 FFN 需要新增模型开关，属于后续扩展。每次只改一个因素，并记录总参数和激活计算是否也变化。

面试时应能解释：

- RankMixer 的 token mixing 如何做跨特征交互？为什么 per-token FFN 与普通共享 FFN 不同？
- `E` 增大而 `K` 固定时，总参数、激活参数、FLOPs 与延迟分别如何变化？
- 为什么稀疏专家可能理论 FLOPs 更低、实测却更慢？如何验证计算确实跳过了未选专家？
- 专家训练不足或路由塌缩怎样在调用分布、梯度和质量曲线上体现？
- 为什么搜索可以按 session 报 NDCG，而推荐表不能直接报请求级 NDCG？
- 为什么时间切分、训练集拟合词表和严格过去的历史特征对离线结果可信度重要？

## 8. 简历使用规则

完成代码但未跑出结果时，只能写“实现了什么、建立了什么评测协议”。跑出结果后，注明**数据版本、任务、对照、预算、设备与真实指标**。没有线上实验就不写线上收益；不将论文中的十亿参数、工业 MFU 或业务增量移植到个人项目。项目价值来自一套能复现、能解释、能承认失败的比较，而非编造一个好看的提升百分比。
