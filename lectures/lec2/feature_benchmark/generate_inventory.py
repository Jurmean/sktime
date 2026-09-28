"""Regenerate the audited lecture inventory against installed feature libraries.

Run from repository root: python lectures/lec2/feature_benchmark/generate_inventory.py
PDF pages are physical (1-based); printed slide numbers jump at PDF page 16.
The feature names below were visually transcribed from PDF image tables.
"""

import csv
import json
from pathlib import Path

from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
)

from sktime.benchmarking.feature_extraction._methods import DEFAULTS, list_methods

OUT = Path(__file__).resolve().parent
BASE = "https://github.com/NetManAIOps/sktime/blob/3b70129e7aa2ef1a9d4ffdeef8bcebe37018e4cb/"
rows = []


def add(
    name,
    page,
    method="",
    source="",
    output="",
    status="supported",
    note="",
    category="method",
):
    rows.append(
        dict(
            name=name,
            pdf_page=page,
            slide_page=page if page <= 15 else page + 1,
            category=category,
            method=method,
            parameters=json.dumps(DEFAULTS.get(method, {}), ensure_ascii=False),
            source=source,
            output=output,
            status=status,
            note=note,
        )
    )


# Algorithm identities, not every illustrative plot or repeated mention.
entries = [
    (
        "Min-max scaling",
        6,
        "minmax",
        "sktime/transformations/series/adapt.py",
        "series",
    ),
    (
        "Z-normalization",
        7,
        "zscore",
        "sktime/transformations/series/adapt.py",
        "series",
    ),
    (
        "Lp normalization (implemented L2 case)",
        8,
        "l2",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Backward moving average",
        9,
        "ma_backward",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Forward moving average",
        9,
        "ma_forward",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Centered moving average",
        9,
        "ma_centered",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Interval resampling / aggregation",
        12,
        "resample",
        "sktime/transformations/series/adapt.py",
        "series",
    ),
    (
        "Linear interpolation / alignment",
        12,
        "interpolate",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Contiguous slicing",
        13,
        "window_length",
        "sktime/transformations/panel/segment.py",
        "samples",
    ),
    ("Patching", 14, "patch", "sktime/transformations/panel/segment.py", "array"),
    (
        "Missing points: linear interpolation",
        15,
        "interpolate",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Missing points: forward fill",
        15,
        "ffill",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Missing points: backward fill",
        15,
        "bfill",
        "sktime/transformations/series/impute.py",
        "series",
    ),
    (
        "Missing points: moving average fill",
        15,
        "mean_fill",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    ("Padding", 16, "padding", "sktime/transformations/panel/padder.py", "series"),
    (
        "Instance normalization",
        17,
        "zscore",
        "lectures/lec2/Preprocessing_Techniques.ipynb",
        "series",
    ),
    (
        "Average pooling",
        18,
        "pool_mean",
        "sktime/transformations/series/binning.py",
        "series",
    ),
    (
        "Max pooling",
        18,
        "pool_max",
        "sktime/transformations/series/binning.py",
        "series",
    ),
    ("ACF", 23, "acf", "sktime/transformations/series/acf.py", "scalar vector"),
    ("PACF", 26, "pacf", "sktime/transformations/series/acf.py", "scalar vector"),
    (
        "Augmented Dickey-Fuller test",
        29,
        "adf",
        "sktime/param_est/stationarity/_statsmodels.py",
        "statistic/pvalue/rejection",
    ),
    (
        "First-order differencing",
        30,
        "difference1",
        "sktime/transformations/series/difference.py",
        "series",
    ),
    (
        "Second-order differencing",
        31,
        "difference2",
        "sktime/transformations/series/difference.py",
        "series",
    ),
    (
        "Permutation entropy",
        33,
        "permutation_entropy",
        "sktime/transformations/panel/tsfresh.py",
        "scalar",
    ),
    (
        "FFT / Fourier transform",
        38,
        "spectrum",
        "lectures/lec2/fft.ipynb",
        "complex coefficients (real/imag)",
    ),
    (
        "Inverse FFT",
        43,
        "spectrum",
        "lectures/lec2/fft.ipynb",
        "reconstruction from stored coefficients",
    ),
    (
        "Power spectrum",
        45,
        "spectrum",
        "lectures/lec2/frequency_feature.ipynb",
        "frequency/power arrays",
    ),
    (
        "Mean frequency / power-weighted spectral centroid",
        46,
        "spectrum",
        "lectures/lec2/frequency_feature.ipynb",
        "scalar",
    ),
    (
        "Peak frequency",
        47,
        "spectrum",
        "lectures/lec2/frequency_feature.ipynb",
        "scalar",
    ),
    (
        "Spectral roll-off",
        48,
        "spectrum",
        "lectures/lec2/frequency_feature.ipynb",
        "scalar",
    ),
    (
        "Spectral bandwidth",
        49,
        "spectrum",
        "lectures/lec2/frequency_feature.ipynb",
        "scalar",
    ),
    (
        "Linear fitting",
        59,
        "fit_linear",
        "sktime/forecasting/trend/_curve_fit_forecaster.py",
        "parameters/fitted/residual",
    ),
    (
        "Power fitting",
        63,
        "fit_power",
        "sktime/forecasting/trend/_curve_fit_forecaster.py",
        "parameters/fitted/residual",
    ),
    (
        "Exponential fitting",
        65,
        "fit_exponential",
        "sktime/forecasting/trend/_curve_fit_forecaster.py",
        "parameters/fitted/residual",
    ),
    (
        "Polynomial fitting",
        67,
        "fit_polynomial",
        "sktime/forecasting/trend/_polynomial_trend_forecaster.py",
        "parameters/fitted/residual",
    ),
    (
        "Sigmoid fitting",
        69,
        "fit_sigmoid",
        "sktime/forecasting/trend/_curve_fit_forecaster.py",
        "parameters/fitted/residual",
    ),
    (
        "Classical additive decomposition",
        79,
        "classical_additive",
        "lectures/lec2/decomposition.ipynb",
        "trend/seasonal/residual",
    ),
    (
        "Classical multiplicative decomposition",
        81,
        "classical_multiplicative",
        "sktime/transformations/series/detrend/_deseasonalize.py",
        "trend/seasonal/residual",
    ),
    (
        "Moving-average trend",
        84,
        "ma_centered",
        "sktime/transformations/series/moving_average.py",
        "series",
    ),
    (
        "Least-squares trend",
        86,
        "fit_linear",
        "sktime/forecasting/trend/_curve_fit_forecaster.py",
        "parameters/series",
    ),
    (
        "STL / LOESS decomposition",
        91,
        "stl",
        "sktime/transformations/series/detrend/_deseasonalize.py",
        "trend/seasonal/residual",
    ),
    (
        "Wavelet transform / decomposition",
        93,
        "wavelet",
        "lectures/lec2/decomposition.ipynb",
        "coefficients/reconstruction",
    ),
    (
        "Channel-independent extraction",
        101,
        "all univariate methods",
        "sktime/transformations/panel/catch22.py",
        "one feature block per channel",
    ),
    (
        "PCA",
        102,
        "pca",
        "lectures/lec2/mts_pca_ica.ipynb",
        "local factors/loadings/reconstruction",
    ),
    (
        "ICA / FastICA",
        113,
        "ica",
        "lectures/lec2/mts_pca_ica.ipynb",
        "local factors/loadings/reconstruction",
    ),
    (
        "Dynamic Factor Model",
        117,
        "dfm",
        "sktime/forecasting/dynamic_factor.py",
        "local factors/loadings/reconstruction",
    ),
    (
        "TSFresh",
        122,
        "tsfresh",
        "sktime/transformations/panel/tsfresh.py",
        "feature table",
    ),
    (
        "Catch22",
        128,
        "catch22",
        "sktime/transformations/panel/catch22.py",
        "22 named features per channel",
    ),
]
for e in entries:
    add(*e)
for name, page, note in [
    (
        "Outlier-aware smoothing",
        11,
        "Principle only; the lecture does not specify a detector. No detector is silently chosen.",
    ),
    (
        "Spline interpolation",
        12,
        "Mentioned as an alternative; v1 exposes linear interpolation.",
    ),
    (
        "Model-based imputation",
        15,
        "Generic family, no particular estimator specified.",
    ),
    (
        "Logarithmic fitting",
        58,
        "Mentioned in overview; no concrete lecture implementation.",
    ),
    (
        "Hilbert transform",
        44,
        "Introductory mention; no matching lecture adapter verified.",
    ),
    (
        "Hilbert-Huang / EMD",
        44,
        "Introductory mention; no matching lecture adapter verified.",
    ),
    (
        "General basis-function least-squares decomposition",
        96,
        "RBFTransformer generates bases, not fitted decomposition coefficients; not substituted.",
    ),
]:
    add(name, page, status="gap/deferred", note=note)
add(
    "Missing points: deletion",
    15,
    "drop_missing",
    "sktime/transformations/series/dropna.py",
    "series",
)
for name, page, note in [
    (
        "Trend and seasonality",
        22,
        "Properties represented by fitted/decomposition outputs; not independent algorithms.",
    ),
    (
        "Stationarity",
        27,
        "Concept; ADF fails to reject a unit root does not prove nonstationarity.",
    ),
    (
        "Entropy",
        32,
        "Concept; permutation entropy is the implemented standalone measure.",
    ),
    (
        "Least squares",
        61,
        "Optimization principle underlying fitting, not a separate feature set.",
    ),
    (
        "Newton's method",
        65,
        "Teaching optimizer discussion; adapters preserve existing scipy/statsmodels solvers.",
    ),
    (
        "MAE, MAPE, MSE, RMSE",
        55,
        "Evaluation metrics. V1 records in-sample MAE/RMSE; MAPE is not ranked on near-zero targets.",
    ),
    (
        "Correlation, distance, GNN-based relationships",
        119,
        "Lecture 3 preview, outside Lecture 2 execution scope.",
    ),
    (
        "Causal relationships",
        120,
        "Lecture 9 preview, outside Lecture 2 execution scope.",
    ),
    ("hctsa", 127, "Background package; no complete compatible repository adapter."),
]:
    add(
        name,
        page,
        status="reference only",
        note=note,
        category="concept/metric/preview",
    )

# Four image tables, physical PDF pages 123-126, printed 124-127.
tables = {
    123: "abs_energy absolute_maximum absolute_sum_of_changes agg_autocorrelation agg_linear_trend approximate_entropy ar_coefficient augmented_dickey_fuller autocorrelation benford_correlation binned_entropy c3 change_quantiles cid_ce count_above count_above_mean count_below count_below_mean",
    124: "cwt_coefficients energy_ratio_by_chunks fft_aggregated fft_coefficient first_location_of_maximum first_location_of_minimum fourier_entropy friedrich_coefficients has_duplicate has_duplicate_max has_duplicate_min index_mass_quantile kurtosis large_standard_deviation last_location_of_maximum last_location_of_minimum lempel_ziv_complexity",
    125: "length linear_trend linear_trend_timewise longest_strike_above_mean longest_strike_below_mean matrix_profile max_langevin_fixed_point maximum mean mean_abs_change mean_change mean_n_absolute_max mean_second_derivative_central median minimum number_crossing_m number_cwt_peaks number_peaks",
    126: "partial_autocorrelation percentage_of_reoccurring_datapoints_to_all_datapoints percentage_of_reoccurring_values_to_all_values permutation_entropy quantile query_similarity_count range_count ratio_beyond_r_sigma ratio_value_number_to_time_series_length root_mean_square sample_entropy set_property skewness spkt_welch_density standard_deviation sum_of_reoccurring_data_points sum_of_reoccurring_values sum_values symmetry_looking time_reversal_asymmetry_statistic value_count variance variance_larger_than_standard_deviation variation_coefficient",
}
efficient, comprehensive = EfficientFCParameters(), ComprehensiveFCParameters()
for page, names in tables.items():
    for name in names.split():
        if name == "set_property":
            add(
                name,
                page,
                category="utility",
                status="excluded utility",
                note="Decorator helper, not a feature calculator.",
            )
            continue
        status = (
            "efficient"
            if name in efficient
            else "comprehensive only"
            if name in comprehensive
            else "requires custom settings / not in preset"
            if hasattr(feature_calculators, name)
            else "unavailable in installed tsfresh"
        )
        add(
            name,
            page,
            "tsfresh",
            "sktime/transformations/panel/tsfresh.py",
            "scalar(s)",
            status,
            "Image-table transcription; exact expanded parameters are saved per extraction.",
            "tsfresh calculator",
        )
        rows[-1]["parameters"] = json.dumps(comprehensive.get(name), ensure_ascii=False)

catch = [
    ("DN_HistogramMode_5", "mode_5"),
    ("DN_HistogramMode_10", "mode_10"),
    ("DN_OutlierInclude_p_001_mdrmd", "outlier_timing_pos"),
    ("DN_OutlierInclude_n_001_mdrmd", "outlier_timing_neg"),
    ("CO_f1ecac", "acf_timescale (slide: first1e_acf_tau)"),
    ("CO_FirstMin_ac", "acf_first_min (slide: firstMin_acf)"),
    ("SP_Summaries_welch_rect_area_5_1", "low_freq_power"),
    ("SP_Summaries_welch_rect_centroid", "centroid_freq"),
    ("FC_LocalSimple_mean3_stderr", "forecast_error"),
    ("FC_LocalSimple_mean1_tauresrat", "whiten_timescale"),
    ("MD_hrv_classic_pnn40", "high_fluctuation"),
    ("SB_BinaryStats_mean_longstretch1", "stretch_high"),
    ("SB_BinaryStats_diff_longstretch0", "stretch_decreasing"),
    ("SB_MotifThree_quantile_hh", "entropy_pairs"),
    ("CO_HistogramAMI_even_2_5", "ami2"),
    ("CO_trev_1_num", "trev"),
    ("IN_AutoMutualInfoStats_40_gaussian_fmmi", "ami_timescale"),
    ("SB_TransitionMatrix_3ac_sumdiagcov", "transition_variance"),
    ("PD_PeriodicityWang_th0_01", "periodicity"),
    ("CO_Embed2_Dist_tau_d_expfit_meandiff", "embedding_dist"),
    ("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", "rs_range"),
    ("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "dfa"),
]
for name, alias in catch:
    add(
        name,
        130,
        "catch22",
        "sktime/transformations/panel/catch22.py",
        "scalar",
        "supported",
        f"Slide alias: {alias}; exported using full canonical name.",
        "catch22 feature",
    )

with (OUT / "algorithm_inventory.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(OUT / "methods.json").write_text(
    json.dumps(list_methods(), ensure_ascii=False, indent=2) + "\n"
)
lines = [
    "# 课件算法与代码对照",
    "",
    "来源：lec2-2026.pdf，共 132 页。PDF 页码为物理页码；第 16 页起页脚编号比 PDF 页码大 1。",
    "图片特征表已逐项核对。概念、预告、评价指标、装饰器不计为可执行特征。",
    "状态针对本次锁定依赖；TSFresh 的 efficient 与 comprehensive 参数组合不同。",
    "",
    "| 方法/特征 | PDF / 页脚 | 类别 | 接口 ID | 支持状态 | 来源/说明 |",
    "|---|---|---|---|---|---|",
]
for r in rows:
    source = f"[代码]({BASE + r['source']})" if r["source"] else ""
    note = r["note"].replace("|", "/")
    lines.append(
        f"| {r['name']} | {r['pdf_page']} / {r['slide_page']} | {r['category']} | {r['method']} | {r['status']} | {source} {note} |"
    )
lines += [
    "",
    "完整参数见 algorithm_inventory.csv、methods.json 以及实际运行的 metadata.json。",
    "",
    "## 必要的语义修正",
    "",
    "- 频谱统计采用正确的单边能量权重，保留 DC；平均频率与功率加权谱质心为同一量。",
    "- 原课程双边谱平均会正负抵消；50 Hz 单频信号已作为回归测试。",
    "- Catch22 使用完整特征名，避免仓库中低频功率/质心的短别名对调影响解释。",
    "- 平滑边界只使用已观测点，避免示例卷积的隐式零填充；平均填补维持向前递归定义。",
    "- 拟合使用 t=(position+1)/N，避免时间原点溢出；幂函数采用原值域最小二乘，不复用示例 log(y) 拟合的不同目标。",
    "- PCA/ICA 为通道降维；不使用将整条序列展平的 PCATransformer。",
    "- DFM 使用现有状态空间模型及平滑因子，不读取合成真值；后端访问集中在一个适配器内。",
    "- 经典分解保留边界 NaN；ADF 的 p 值只表达单位根检验，不把未拒绝原假设当作证明。",
    "- 简单均值/标准差/分位数使用 SummaryTransformer；原始标准差 ddof=1。",
    "- 预处理、切片、Pooling 只执行显式配置；不暗中补齐缺失/不等长样本。",
]
(OUT / "ALGORITHMS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(
    f"{len(rows)} catalog rows; {sum(r['category'] == 'tsfresh calculator' for r in rows)} TSFresh calculators; {len(catch)} Catch22 features"
)
