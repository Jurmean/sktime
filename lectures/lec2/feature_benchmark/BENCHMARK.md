# Lecture 2 首版运行结果

已完成六个仓库自带数据集、38 个接口（核心 8、扩展 30）的完整运行。共 228 个数据集—方法组合，每个组合完整预热一次、正式测量三次，共 912 次运行。所有组合三次测量的特征、数组及状态在 rtol=1e-7、atol=1e-9 下保持一致。

实际基准代码提交：`d7e309017fe6e5e86ade2107613582104b018333`；基线：`3b70129e7aa2ef1a9d4ffdeef8bcebe37018e4cb`。运行时工作区干净；Python 3.11.15，macOS-26.5.1-arm64-arm-64bit，10 个逻辑 CPU，使用 CPU 单进程并限制数值线程为 1。依赖版本和完整参数保存在 baseline 下的 JSON，实际 TSFresh 计算器参数还保存在结果包内各 metadata.json。

基准后仅修正了显式预处理流水线中缺失依赖的诊断分类并完善对照表；本次两个预设均使用空预处理流水线，算法和参数未变。最终版本另行提取了六份合并核心表，并与基准输出逐值比较通过。

## 核心特征表

| 数据集 | 样本数 | 通道 | 每样本长度 | 标量列数 | 有限值比例 | 分方法中位耗时之和（秒） |
|---|---:|---:|---:|---:|---:|---:|
| GunPoint | 200 | 1 | 150 | 835 | 88.30% | 6.656 |
| ArrowHead | 211 | 1 | 251 | 835 | 99.83% | 7.581 |
| ItalyPowerDemand | 1096 | 1 | 24 | 835 | 55.48% | 31.577 |
| BasicMotions | 80 | 6 | 100 | 5010 | 76.38% | 15.337 |
| Airline | 1 | 1 | 144 | 840 | 86.90% | 0.059 |
| DailyDelhiClimate | 1 | 4 | 114 | 3360 | 79.73% | 0.175 |

表中耗时是各方法完整提取的中位数之和，不是一次联合提取的测量值，也不是下游预测效果。单个方法的三次耗时、加载/写盘耗时和每轮输出计数见 [核心明细](baseline/core-summary.csv) 与 [完整运行记录](baseline/core-benchmark.json)。
TSFresh efficient 在整数时间索引上输出 777 列；在 Airline 和 DailyDelhiClimate 的真实日期索引上额外输出 5 个 timewise 趋势特征，共 782 列。核心还包括其余 58 个逐通道标量，因此表的维度随通道数和真实时间索引变化。
四个分类数据集保留全部 TRAIN/TEST；Airline 和 DailyDelhiClimate 按原始连续序列整体处理。DailyDelhiClimate 使用该仓库加载器提供的 114 条日观测。所有标签仅作元数据，未参与特征提取。

## 适用性与数值诊断

核心组合：36 个成功、12 个部分成功，无执行失败或缺失依赖。Catch22 和 TSFresh 的 NaN/Inf 均原样保留；ItalyPowerDemand 序列仅长 24，许多长滞后/大窗口特征无有效值，这解释了其较低的有限值比例。
扩展组合：130 个成功、19 个部分成功、31 个不适用。部分成功包含既有输出也有失败的组合，不能理解为全部样本拟合成功。第一次正式测量共有 652 个样本—通道调用失败，三次正式测量的失败集合相同，均为曲线拟合达到 maxfev=10000。

| 数据集 | 方法 | 成功 | 部分成功 | 失败 |
|---|---|---:|---:|---:|
| GunPoint | fit_sigmoid | 119 | 20 | 61 |
| ArrowHead | fit_sigmoid | 143 | 8 | 60 |
| ItalyPowerDemand | fit_sigmoid | 664 | 65 | 367 |
| BasicMotions | fit_exponential | 478 | 0 | 2 |
| BasicMotions | fit_sigmoid | 272 | 46 | 162 |

- BasicMotions：PCA 80/80 成功；ICA 78 成功、2 个不收敛；DFM 29 成功、51 个不收敛。不收敛但有数值结果的情况标记 partial，因子仍保存。DailyDelhiClimate 的 PCA/ICA/DFM 均成功。
- 31 个不适用组合：其他五个数据集没有明确周期，跳过三种季节分解（15）；四个单变量数据集跳过三种通道降维（12）；四个无日期索引的分类数据集跳过日期重采样（4）。
- Airline 的 STL 成功；经典加法/乘法分解保留边界 NaN；小波全部适用组合成功。Airline 从月度重采样到 2D 会产生大量空区间，作为独立预处理示例保留 NaN，不将其用于默认核心提取。
- 拟合误差和重构误差仅为当前样本的数值诊断，不能据此推断预测能力、特征价值或跨样本因子含义。
- 扩展命令按设计返回退出码 1，提示存在上述拟合失败；整个批次完整执行，没有用缺失文件伪装完成。

全部方法的状态、耗时、维度和原因见 [扩展明细](baseline/extended-summary.csv) 与 [扩展运行记录](baseline/extended-benchmark.json)。各次运行原始状态计数和失败原因均保存。

## 验证与交付文件

31 项针对性测试通过，新增代码 Ruff 检查及 Git 空白检查通过。测试涵盖正弦单边谱/Parseval、线性/多项式参数、分解重构、逐通道与样本对应、不等长输入、常量/短序列/缺失值、非法参数、依赖缺失、CLI/Python 一致性、时间索引和重复运行。详见 [验证记录](baseline/validation.json)。
全量合并表与基准表逐值一致；已实际读取并校验 235591 个 NPZ 数组的形状和类型（allow_pickle=False）。TSFresh comprehensive 额外在 Airline 实际提取得到 788 列，状态为 partial，原始输出收录于可选示例目录。

完整数据包 `lec2-feature-results.tar.gz` 单独保存在本地交付目录，不进入 Git。包含：

- `lec2-features/`：六个数据集合并核心 CSV、数组与元数据，可直接使用。
- `lec2-core/`、`lec2-extended/`：每个组合第一次正式测量的完整 CSV/NPZ/JSON/诊断，以及全部 912 次运行的记录。
- `lec2-comprehensive-example/`：可选完整 TSFresh 配置的实际输出。
- `config/` 与 `FILES.sha256`：锁定配置、依赖、使用说明及逐文件校验值。

如需重跑，请使用 [中文说明](README.md) 中的命令。大型文件未上传到 GitHub；解压后的目录结构可与记录中的相对路径直接对应。

结果包 SHA256：`81eb1ae5f0bab0a0c3f80735fc3c2843363e8a574e7e8125ee2ea1ab545c9627`。大小：101.8 MiB。
