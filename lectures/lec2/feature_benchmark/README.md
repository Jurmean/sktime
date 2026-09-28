# Lecture 2 一键特征提取与运行基准

本工具将课件中的已有方法统一为 Python 和命令行入口。输出特征表、频谱/分解/降维数组、数值诊断及可复现的运行记录。它不训练分类器，不产生准确率排行榜；拟合与重构误差仅描述当前样本。

## 安装与一键运行

在仓库根目录，使用 Python 3.11 的独立环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r lectures/lec2/feature_benchmark/requirements-lock.txt
python -m pip install -e . --no-deps

# 列出全部方法、默认参数与六个离线数据集
python -m sktime.benchmarking.feature_extraction list

# 一键提取全部六个数据集的核心特征
python -m sktime.benchmarking.feature_extraction extract \
  --config lectures/lec2/feature_benchmark/core.json --out outputs/lec2-features

# 正式运行：每个数据集 × 方法，完整预热一次、测量三次
python -m sktime.benchmarking.feature_extraction benchmark \
  --config lectures/lec2/feature_benchmark/core.json --out outputs/lec2-core
python -m sktime.benchmarking.feature_extraction benchmark \
  --config lectures/lec2/feature_benchmark/extended.json --out outputs/lec2-extended
```

输出目录必须是新目录，避免覆盖历史实验。数据全部使用仓库自带文件；安装完成后，提取和基准无需联网。`--datasets Airline BasicMotions` 可缩小范围；`--methods summary spectrum` 可显式选择方法（不要同时传包含其他方法参数的配置文件）。命令行显式参数覆盖配置文件；未设置项使用 `methods.json` 中的默认值。

`lec2-core` 包含统计量、ACF/PACF、ADF、排列熵、单边频谱、Catch22 和 TSFresh efficient。`lec2-extended` **只包含扩展方法**：拟合、经典/STL/小波分解、PCA/ICA/DFM 与独立预处理操作，不重复核心方法。

TSFresh 的 `minimal`、`efficient`、`comprehensive` 可通过 `parameters.tsfresh.feature_set` 选择。完整预设计算量更大；课件表格中已移除的计算器或需要额外输入的计算器不会被伪造出来，详见 [完整对照表](ALGORITHMS.md)。

## Python 与输入约定

```python
from sktime.datasets import load_basic_motions
from sktime.benchmarking.feature_extraction import extract_features, run_benchmark

X, labels = load_basic_motions(split="TRAIN")
result = extract_features(X, methods=["summary", "spectrum", "catch22"])
print(result.features.shape)
print(result.diagnostics.groupby(["method", "status"]).size())
result.save("outputs/my-features")

# labels 未传给提取器，也不参与任何特征筛选。
# 返回内存中的汇总字典，并同时导出所有实际运行文件。
report = run_benchmark(datasets=["Airline"], methods=["summary", "spectrum"],
                       output_dir="outputs/my-benchmark")
```

接受 pandas Series、数值 DataFrame，以及 sktime 的 nested DataFrame、MultiIndex Panel、numpy3D `(样本, 通道, 时间)`、DataFrame 列表。不等长样本可用 nested/列表，不做隐式补齐；同一样本内的各通道必须有共同时间索引。索引必须唯一且递增，样本/通道标识不得重复。`sample_ids` 可覆盖样本标识。

每个原始样本默认整体处理。`window_length=48, stride=24`（CLI 为 `--window-length 48 --stride 24`）产生完整窗口；窗口不跨样本或数据划分，尾部不足一个窗口时丢弃，原样本 ID 与窗口位置保留。样本短于窗口时显式报错。

预处理仅按用户指定顺序运行，例如：

```python
result = extract_features(X, methods=["summary", "spectrum"], preprocessing=[
    {"method": "interpolate"},
    {"method": "zscore"},
])
```

扩展预设中的各预处理方法是**互相独立地作用于原始输入**，不自动串成一个流水线。L2 对零范数返回原零序列；Z-score/Min-max 由当前样本拟合。线性插值只填内部空缺，ffill/bfill 保留其方向无法解决的边界空缺。差分不补零；Pooling 丢弃不完整尾窗；Padding 只补长，不截断。前向/中心平滑使用样本内部的未来观测，工具定位为离线特征分析，不是在线因果预测。

## 输出与可追溯性

每次提取输出：

- `features.csv`：每个样本一行，列为 `通道::方法::完整特征名`；通道名称使用 URL 编码，避免分隔符冲突。
- `artifacts.npz`：原始数值、时间索引、FFT 实虚部、功率谱、分解分量、拟合曲线、降维因子等；使用 `np.load(..., allow_pickle=False)`。
- `metadata.json`：数组 ID 与样本/通道/方法的对应关系、原始索引类型/频率、数据划分、标签元数据、实际配置和 TSFresh 函数参数字典。索引值保存为字符串数组，类型/频率另存，可恢复日期或 PeriodIndex。
- `diagnostics.csv`：每次方法调用的状态、有效值比例、参数、警告、失败原因和耗时。
- `report.md` 与 `write_timing.json`：可读摘要与独立写盘耗时。

基准另外生成 `summary.csv`、`benchmark.json`、`command.json` 和汇总报告；每个方法保存第一次正式测量的完整提取结果，全部四轮的耗时/状态均保留。基准逐个组合落盘，中断前已完成的结果可检查；当前版本不自动续跑已有目录。

`ok` 表示数值完整；`partial` 表示保留了 NaN/Inf 或拟合警告；`skipped` 表示输入不适用；`blocked` 表示依赖缺失；`failed` 表示执行错误。CLI 遇到 blocked/failed 或重复不一致返回 1，非法配置/目录返回 2。科学上合法的 partial/skipped 不等于程序失败。CSV/NPZ 保留非有限值，严格 JSON 中非有限元数据使用 null。

## 参数和科学含义

- 原始数据默认不归一化、不插补。多变量核心特征逐通道独立计算，保留原通道名。
- FFT 要求等间隔采样；保留 DC，正频率能量加倍但排除 DC 和偶数长度的 Nyquist。功率和等于原信号均方值。未知采样率标注 cycles/sample_interval；显式 `sampling_rate` 以每秒样本数解释并输出 Hz。平均频率为功率加权谱质心。
- ACF/PACF 默认输出滞后 1–10，不可计算的短序列滞后为 NaN。ADF 输出检验统计量、p 值及 0.05 阈值拒绝单位根的指示，不把检验结果当成平稳性证明。
- 排列熵为 TSFresh 的自然对数、未归一化定义，dimension=5、tau=1； ties 沿用 TSFresh argsort。Catch22 导出全部 22 个完整名称，不用有歧义的短别名；常量或长度小于 10 的序列保留 22 个 NaN。
- TSFresh 关闭插补和标签筛选；保留真实日期索引（PeriodIndex 转为时间戳供其调用），不凭空为无时间戳序列补日期。部分特征依赖长度、重复值或日期索引，出现 NaN 是诊断结果。
- 拟合时间轴统一为 `(position+1)/N`；线性/幂函数/指数参数为 a,b，Sigmoid 为 L,k,t0，多项式为常数项开始的升幂系数。使用仓库 CurveFitForecaster/PolynomialTrendForecaster，明确记录不收敛、不可辨识参数和样本内 MAE/RMSE。
- 季节周期仅对具备数据集身份的 Airline 默认设为 12；其他数据显式传 `sp`，不猜周期。直接传 `load_airline()` 得到的裸 Series 时也需显式设置。经典乘法分解要求正值，经典分解边界 NaN 原样保存。
- 小波默认 db1、2 层、symmetric 边界；可选 db4。层数超过当前长度允许范围时跳过，不默默调整。
- PCA/ICA/DFM 均在**当前样本内部**逐通道标准化并拟合，默认 2 个因子，需要至少 3 个通道；不借用其他样本、测试标签或合成真值。因子编号仅在本样本内有意义。DFM 的平滑因子是离线输出，其重构使用平滑状态和观测矩阵。

## 基准协议和验证

固定 seed=42，单进程，`threadpoolctl` 限制数值库线程为 1，TSFresh `n_jobs=0` 不启用进程池。每个数据集—方法完整预热一次，再运行三次；报告中位数及原始耗时。提取计时包括输入规范化、校验值和内存结果构造，不包括加载、写盘、环境采集与重复结果比较。预热可能包含编译/导入成本，保留原始耗时，不纳入中位数。有效值比例分别统计标量特征和数值数组，时间字符串不计入分母。

记录提交、方法源文件 SHA256、全部依赖版本、Python、平台、CPU 架构、原始数据内容/索引/标签元数据的 SHA256。数值重复一致性以 rtol=1e-7、atol=1e-9 检查，包括特征表、数组与状态；不同硬件耗时不可直接当作算法排名。

```bash
python -m pytest -o addopts='' sktime/benchmarking/feature_extraction/tests -q
python lectures/lec2/feature_benchmark/generate_inventory.py
```

数据范围为 GunPoint、ArrowHead、ItalyPowerDemand、BasicMotions、Airline、DailyDelhiClimate。TRAIN/TEST 原样保留；连续序列标为 unsplit。DailyDelhiClimate 的异常原始观测不会被静默剔除。全量特征包独立保存，Git 仅提交配置、程序、测试、对照表和运行摘要。
