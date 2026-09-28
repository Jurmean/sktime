"""Lecture methods with explicit semantics, lazy optional imports and raw outputs."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd


class NotApplicable(ValueError):
    """The method's mathematical/input requirements are not satisfied."""


@dataclass
class Outcome:
    """Scalars and native arrays from one method invocation."""

    features: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)


CORE = (
    "summary",
    "acf",
    "pacf",
    "adf",
    "permutation_entropy",
    "spectrum",
    "catch22",
    "tsfresh",
)
FIT = ("fit_linear", "fit_power", "fit_exponential", "fit_polynomial", "fit_sigmoid")
DECOMPOSE = ("classical_additive", "classical_multiplicative", "stl", "wavelet")
MULTIVARIATE = ("pca", "ica", "dfm")
PREPROCESS = (
    "minmax",
    "zscore",
    "l2",
    "ma_backward",
    "ma_forward",
    "ma_centered",
    "interpolate",
    "ffill",
    "bfill",
    "mean_fill",
    "drop_missing",
    "difference1",
    "difference2",
    "patch",
    "padding",
    "pool_mean",
    "pool_max",
    "resample",
)
EXTENDED = FIT + DECOMPOSE + MULTIVARIATE + PREPROCESS
PRESETS = {"lec2-core": CORE, "lec2-extended": EXTENDED}

DEFAULTS = {
    "summary": {},
    "acf": {"n_lags": 10},
    "pacf": {"n_lags": 10, "method": "ywadjusted"},
    "adf": {"maxlag": None, "regression": "c", "autolag": "AIC"},
    "permutation_entropy": {"dimension": 5, "tau": 1},
    "spectrum": {"sampling_rate": None, "rolloff": 0.85},
    "catch22": {},
    "tsfresh": {"feature_set": "efficient"},
    "fit_linear": {},
    "fit_polynomial": {"degree": 5},
    "fit_power": {"maxfev": 10000},
    "fit_exponential": {"maxfev": 10000},
    "fit_sigmoid": {"maxfev": 10000},
    "classical_additive": {"sp": None},
    "classical_multiplicative": {"sp": None},
    "stl": {"sp": None, "robust": False},
    "wavelet": {"wavelet": "db1", "level": 2},
    "pca": {"n_components": 2},
    "ica": {"n_components": 2, "max_iter": 2000, "tol": 0.0001},
    "dfm": {"k_factors": 2, "factor_order": 1, "maxiter": 100},
    "minmax": {},
    "zscore": {},
    "l2": {},
    "ma_backward": {"window": 5},
    "ma_forward": {"window": 5},
    "ma_centered": {"window": 5},
    "interpolate": {},
    "ffill": {},
    "bfill": {},
    "mean_fill": {"window": 2},
    "drop_missing": {},
    "difference1": {},
    "difference2": {},
    "patch": {"length": 16, "stride": 16},
    "padding": {"length": 256, "fill_value": 0.0},
    "pool_mean": {"window": 4},
    "pool_max": {"window": 4},
    "resample": {"rule": "2D"},
}


def list_methods():
    """Return method IDs, default parameters, scope and optional dependencies."""
    deps = {
        "acf": ["statsmodels"],
        "pacf": ["statsmodels"],
        "adf": ["statsmodels"],
        "permutation_entropy": ["tsfresh"],
        "catch22": ["numba"],
        "tsfresh": ["tsfresh"],
        "wavelet": ["PyWavelets"],
        "dfm": ["statsmodels"],
    }
    return [
        {
            "id": m,
            "preset": "lec2-core" if m in CORE else "lec2-extended",
            "scope": "sample" if m in MULTIVARIATE else "channel",
            "parameters": dict(DEFAULTS[m]),
            "dependencies": deps.get(m, ["statsmodels"] if m in DECOMPOSE else []),
        }
        for m in CORE + EXTENDED
    ]


def resolve(method, params=None):
    """Validate method configuration before running any samples."""
    if method not in DEFAULTS:
        raise ValueError(f"Unknown method {method!r}")
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise ValueError("Method parameters must be a dictionary")
    unknown = set(params) - set(DEFAULTS[method])
    if unknown:
        raise ValueError(f"{method}: unknown parameters {sorted(unknown)}")
    p = dict(DEFAULTS[method], **params)
    for key in (
        "n_lags",
        "dimension",
        "tau",
        "degree",
        "maxfev",
        "sp",
        "level",
        "n_components",
        "max_iter",
        "k_factors",
        "factor_order",
        "maxiter",
        "window",
        "length",
        "stride",
    ):
        if key in p and p[key] is not None:
            value = p[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{method}.{key} must be a positive integer")
    if (
        "maxlag" in p
        and p["maxlag"] is not None
        and (not isinstance(p["maxlag"], int) or p["maxlag"] < 0)
    ):
        raise ValueError("maxlag must be a nonnegative integer or None")
    if method == "spectrum":
        if not 0 < p["rolloff"] <= 1 or (
            p["sampling_rate"] is not None
            and (not np.isfinite(p["sampling_rate"]) or p["sampling_rate"] <= 0)
        ):
            raise ValueError(
                "rolloff must be in (0, 1]; sampling_rate must be positive and finite"
            )
    if method == "tsfresh" and p["feature_set"] not in (
        "minimal",
        "efficient",
        "comprehensive",
    ):
        raise ValueError(
            "tsfresh.feature_set must be minimal, efficient or comprehensive"
        )
    if method == "adf" and (
        p["regression"] not in ("c", "ct", "ctt", "n")
        or p["autolag"] not in ("AIC", "BIC", "t-stat", None)
    ):
        raise ValueError("Invalid ADF regression or autolag")
    if method == "ica" and (not np.isfinite(p["tol"]) or p["tol"] <= 0):
        raise ValueError("ICA tol must be positive and finite")
    return p


def _identity_impute(frame):
    # sktime's wrapper treats None as its default imputer, so use an explicit no-op.
    return frame


@lru_cache(maxsize=3)
def tsfresh_parameters(feature_set):
    """Resolve and record the actual version-dependent calculator dictionary."""
    from tsfresh.feature_extraction.settings import (
        ComprehensiveFCParameters,
        EfficientFCParameters,
        MinimalFCParameters,
    )

    return dict(
        {
            "minimal": MinimalFCParameters,
            "efficient": EfficientFCParameters,
            "comprehensive": ComprehensiveFCParameters,
        }[feature_set]()
    )


def _linear(t, a, b):
    return a * t + b


def _power(t, a, b):
    return a * t**b


def _exponential(t, a, b):
    return a * np.exp(b * t)


def _sigmoid(t, L, k, t0):
    from scipy.special import expit

    return L * expit(k * (t - t0))


def _errors(y, fitted):
    residual = np.asarray(y) - np.asarray(fitted)
    return {
        "fit_mae": float(np.mean(np.abs(residual))),
        "fit_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def preprocess(method, frame, p):
    """One preprocessing step; no automatic chain and no cross-sample fitting."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    f = frame.copy()
    if method in ("minmax", "zscore"):
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor

        scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
        values = TabularToSeriesAdaptor(scaler).fit_transform(f)
        return pd.DataFrame(np.asarray(values), index=f.index, columns=f.columns)
    if method == "l2":
        norm = np.sqrt((f**2).sum(axis=0))
        return f / norm.where(norm != 0, 1)
    if method.startswith("ma_"):
        w = p["window"]
        if len(f) < w:
            raise NotApplicable("series shorter than moving-average window")
        # Real observations only at edges; unlike the notebook, do not zero pad.
        if method == "ma_forward":
            return f.iloc[::-1].rolling(w, min_periods=1).mean().iloc[::-1]
        return f.rolling(w, min_periods=1, center=method == "ma_centered").mean()
    if method in ("interpolate", "ffill", "bfill"):
        # Keep unresolved edges explicit; sktime Imputer fills edges automatically.
        if method == "interpolate":
            return f.interpolate(method="linear", limit_area="inside")
        # Leading/trailing gaps are left missing by these directional operations.
        return f.ffill() if method == "ffill" else f.bfill()
    if method == "drop_missing":
        return f.dropna()
    if method == "mean_fill":
        for col in f:
            for i in range(len(f)):
                if pd.isna(f.iloc[i][col]):
                    previous = f[col].iloc[:i].dropna().iloc[-p["window"] :]
                    if len(previous):
                        f.iat[i, f.columns.get_loc(col)] = previous.mean()
        return f
    if method in ("difference1", "difference2"):
        from sktime.transformations.series.difference import Differencer

        order = 1 if method == "difference1" else 2
        if len(f) <= order:
            raise NotApplicable("series shorter than difference order")
        return Differencer(lags=[1] * order, na_handling="drop_na").fit_transform(f)
    if method == "padding":
        if len(f) > p["length"]:
            raise NotApplicable(
                "padding length shorter than sample; no implicit truncation"
            )
        from sktime.transformations.panel.padder import PaddingTransformer

        nested = pd.DataFrame({c: [f[c]] for c in f})
        padded = PaddingTransformer(
            pad_length=p["length"], fill_value=p["fill_value"]
        ).fit_transform(nested)
        return pd.DataFrame({c: padded[c].iloc[0].to_numpy() for c in f})
    if method in ("pool_mean", "pool_max"):
        w = p["window"]
        if len(f) < w:
            raise NotApplicable("series shorter than pooling window")
        n = len(f) // w * w
        grouped = f.iloc[:n].groupby(np.arange(n) // w)
        out = grouped.mean() if method == "pool_mean" else grouped.max()
        out.index = f.index[w - 1 : n : w]
        return out
    if method == "resample":
        if isinstance(f.index, pd.PeriodIndex):
            f.index = f.index.to_timestamp()
        if not isinstance(f.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            raise NotApplicable("resampling requires a dated index and explicit rule")
        return f.resample(p["rule"]).mean()
    raise ValueError(f"{method} cannot be used as a shape-preserving pipeline step")


def run_method(method, frame, p, seed=42):
    """Compute a univariate method or a per-sample multivariate decomposition."""
    x = frame.iloc[:, 0].to_numpy(dtype=float)
    n = len(x)
    if method in PREPROCESS:
        if method == "patch":
            length, stride = p["length"], p["stride"]
            if n < length:
                raise NotApplicable("series shorter than patch length")
            starts = np.arange(0, n - length + 1, stride)
            return Outcome(
                {"patch_count": len(starts)},
                {
                    "patches": np.stack([x[i : i + length] for i in starts]),
                    "starts": starts,
                },
                {"tail_policy": "drop incomplete patch"},
            )
        transformed = preprocess(method, frame, p)
        return Outcome(
            {"output_length": len(transformed)},
            {
                "values": transformed.to_numpy(),
                "index": np.asarray(list(map(str, transformed.index))),
            },
            {
                "index_type": type(transformed.index).__name__,
                "tail_policy": "drop incomplete bins"
                if method.startswith("pool_")
                else None,
            },
        )
    if not np.isfinite(frame.to_numpy()).all():
        raise NotApplicable(
            "nonfinite input: choose explicit preprocessing; values were not "
            "silently imputed"
        )
    if method == "summary":
        from sktime.transformations.series.summarize import SummaryTransformer

        out = SummaryTransformer(
            summary_function=("mean", "std", "min", "max", "median"),
            quantiles=(0.25, 0.75),
        ).fit_transform(frame.iloc[:, 0])
        return Outcome(
            {str(k): float(v) for k, v in out.iloc[0].items()}, details={"std_ddof": 1}
        )
    if method in ("acf", "pacf"):
        from sktime.transformations.series.acf import (
            AutoCorrelationTransformer,
            PartialAutoCorrelationTransformer,
        )

        count = p["n_lags"]
        if np.std(x) == 0 or n < 3:
            raise NotApplicable(
                "correlation undefined for constant or very short series"
            )
        valid = min(count, n - 1 if method == "acf" else n // 2 - 1)
        trafo = (
            AutoCorrelationTransformer(n_lags=valid, fft=False)
            if method == "acf"
            else PartialAutoCorrelationTransformer(n_lags=valid, method=p["method"])
        )
        vals = np.asarray(trafo.fit_transform(pd.Series(x))).ravel()
        features = {
            f"lag_{i}": float(vals[i]) if i <= valid else np.nan
            for i in range(1, count + 1)
        }
        return Outcome(
            features,
            details={
                "effective_n_lags": valid,
                "unavailable_lags": list(range(valid + 1, count + 1)),
            },
        )
    if method == "adf":
        from sktime.param_est.stationarity import StationarityADF

        if n < 8 or np.std(x) == 0:
            raise NotApplicable(
                "ADF requires at least 8 observations and nonconstant input in this"
                " preset"
            )
        fitted = StationarityADF(**p).fit(pd.Series(x))
        return Outcome(
            {
                "statistic": fitted.test_statistic_,
                "pvalue": fitted.pvalue_,
                "reject_unit_root_0.05": float(fitted.stationary_),
            }
        )
    if method == "permutation_entropy":
        from tsfresh.feature_extraction.feature_calculators import permutation_entropy

        if n < (p["dimension"] - 1) * p["tau"] + 1:
            raise NotApplicable("series shorter than permutation embedding")
        return Outcome(
            {"entropy": permutation_entropy(x, **p)},
            details={
                "log_base": "e",
                "normalized": False,
                "tie_policy": "tsfresh argsort",
            },
        )
    if method == "spectrum":
        rate = p["sampling_rate"] or 1.0
        fft = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, 1 / rate)
        power = np.abs(fft) ** 2 / n**2
        # Double positive-frequency energy except DC and the even-N Nyquist bin.
        power[1 : -1 if n % 2 == 0 else None] *= 2
        total = power.sum()
        mf = np.sum(freq * power) / total if total > 0 else np.nan
        feat = {
            "mean_frequency": mf,
            "peak_frequency": freq[np.argmax(power)] if total > 0 else np.nan,
            "rolloff_frequency": freq[
                min(
                    np.searchsorted(np.cumsum(power), p["rolloff"] * total),
                    len(freq) - 1,
                )
            ]
            if total > 0
            else np.nan,
            "spectral_bandwidth": np.sqrt(np.sum((freq - mf) ** 2 * power) / total)
            if total > 0
            else np.nan,
            "mean_square_power": total,
        }
        return Outcome(
            feat,
            {
                "frequency": freq,
                "power": power,
                "fft_real": fft.real,
                "fft_imag": fft.imag,
            },
            {
                "frequency_unit": "Hz"
                if p["sampling_rate"] is not None
                else "cycles/sample_interval",
                "DC": "retained",
                "mean_frequency_alias": "power-weighted spectral centroid",
                "input_length": n,
            },
        )
    if method == "catch22":
        from sktime.transformations.panel.catch22 import CATCH22_FEATURE_NAMES, Catch22

        if n < 10 or np.std(x) == 0:
            return Outcome(
                dict.fromkeys(CATCH22_FEATURE_NAMES, np.nan),
                details={
                    "reason": (
                        "Catch22 undefined/unstable for constant "
                        "or fewer than 10 observations"
                    )
                },
            )
        out = Catch22(replace_nans=False, col_names="str_feat").fit_transform(
            pd.Series(x)
        )
        return Outcome({str(k): float(v) for k, v in out.iloc[0].items()})
    if method == "tsfresh":
        from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

        class DatedTSFreshFeatureExtractor(TSFreshFeatureExtractor):
            """Keep timestamps on values after sktime's long-format conversion."""

            def _transform(self, X, y=None):
                if pd.api.types.is_datetime64_any_dtype(X.iloc[:, 1]):
                    X = X.copy()
                    X.index = pd.DatetimeIndex(X.iloc[:, 1])
                    X.index.name = None
                return super()._transform(X, y)

        # Each sample is one id. Use the repository adapter, no labels/selection.
        cfg = tsfresh_parameters(p["feature_set"])
        signal = frame.iloc[:, 0].rename("signal")
        if isinstance(signal.index, pd.PeriodIndex):
            signal = signal.copy()
            signal.index = signal.index.to_timestamp()
        out = DatedTSFreshFeatureExtractor(
            default_fc_parameters=cfg,
            n_jobs=0,
            disable_progressbar=True,
            show_warnings=True,
            impute_function=_identity_impute,
        ).fit_transform(signal)
        return Outcome(
            {str(k).removeprefix("signal__"): float(v) for k, v in out.iloc[0].items()}
        )
    if method in FIT:
        # A normalized positive time axis avoids epoch-sized powers and t=0 poles.
        t = np.arange(1, n + 1, dtype=float) / n
        if method == "fit_polynomial":
            from sktime.forecasting.trend import PolynomialTrendForecaster

            if n <= p["degree"]:
                raise NotApplicable(
                    "polynomial degree must be smaller than sample length"
                )
            # The native forecaster uses integer time; rescale its fitted coefficients.
            fitted = PolynomialTrendForecaster(degree=p["degree"]).fit(
                pd.Series(x, index=pd.RangeIndex(1, n + 1))
            )
            coef = fitted.regressor_.steps[-1][1].coef_ * n ** np.arange(
                p["degree"] + 1
            )
            pred = np.polynomial.polynomial.polyval(t, coef)
        else:
            from sktime.forecasting.trend import CurveFitForecaster

            if n < 4:
                raise NotApplicable("curve fitting requires at least 4 observations")
            fn = {
                "fit_linear": _linear,
                "fit_power": _power,
                "fit_exponential": _exponential,
                "fit_sigmoid": _sigmoid,
            }[method]
            initial = (
                [float(np.ptp(x)), float(x.mean())]
                if method == "fit_linear"
                else [float(x.mean()), 1.0]
            )
            if method == "fit_sigmoid":
                initial = [float(np.max(x)), 1.0, 0.5]

            # Native CurveFitForecaster converts indices to integers: close over the
            # normalization, keeping its original scipy objective and parameter API.
            def normalized_fn(index, *params):
                return fn(np.asarray(index) / n, *params)

            fitted = CurveFitForecaster(
                function=normalized_fn,
                curve_fit_params={"p0": initial, "maxfev": p.get("maxfev", 10000)},
            ).fit(pd.Series(x, index=pd.RangeIndex(1, n + 1)))
            coef = fitted.params_[0]
            pred = fn(t, *coef)
        features = {f"coefficient_{i}": float(c) for i, c in enumerate(coef)}
        features.update(_errors(x, pred))
        return Outcome(
            features,
            {"fitted": pred, "residual": x - pred},
            {
                "time_axis": "t=(position+1)/sample_length",
                "coefficient_order": "ascending polynomial powers"
                if method == "fit_polynomial"
                else {
                    "fit_linear": ["a", "b"],
                    "fit_power": ["a", "b"],
                    "fit_exponential": ["a", "b"],
                    "fit_sigmoid": ["L", "k", "t0"],
                }[method],
                "fit_scope": "sample",
                "error_scope": "in-sample, not forecast",
            },
        )
    if method in DECOMPOSE:
        if method == "wavelet":
            import pywt

            maximum = pywt.dwt_max_level(n, pywt.Wavelet(p["wavelet"]).dec_len)
            if p["level"] > maximum:
                raise NotApplicable(
                    f"requested wavelet level exceeds valid maximum {maximum}"
                )
            coefs = pywt.wavedec(x, p["wavelet"], level=p["level"], mode="symmetric")
            arrays = {f"coefficient_{i}": c for i, c in enumerate(coefs)}
            recon = pywt.waverec(coefs, p["wavelet"], mode="symmetric")[:n]
            arrays["reconstruction"] = recon
            return Outcome(
                {"reconstruction_rmse": np.sqrt(np.mean((x - recon) ** 2))},
                arrays,
                {
                    "order": [f"A{p['level']}"]
                    + [f"D{j}" for j in range(p["level"], 0, -1)],
                    "boundary_mode": "symmetric",
                },
            )
        sp = p["sp"]
        if sp is None:
            raise NotApplicable("seasonal period unknown: supply sp explicitly")
        if sp < 2 or n < 2 * sp:
            raise NotApplicable("decomposition requires sp>=2 and at least two cycles")
        if method == "stl":
            from sktime.transformations.series.detrend import STLTransformer

            comp = STLTransformer(
                sp=sp, robust=p["robust"], return_components=True
            ).fit_transform(pd.Series(x))
            trend, seasonal, resid = (
                comp[c].to_numpy() for c in ("trend", "seasonal", "resid")
            )
        else:
            from statsmodels.tsa.seasonal import seasonal_decompose

            if method == "classical_multiplicative" and np.any(x <= 0):
                raise NotApplicable(
                    "multiplicative decomposition requires positive values"
                )
            comp = seasonal_decompose(
                x, model=method.removeprefix("classical_"), period=sp
            )
            trend, seasonal, resid = comp.trend, comp.seasonal, comp.resid
        reconstruction = (
            trend * seasonal * resid
            if method == "classical_multiplicative"
            else trend + seasonal + resid
        )
        valid = np.isfinite(reconstruction)
        return Outcome(
            {
                "reconstruction_rmse": np.sqrt(
                    np.mean((x[valid] - reconstruction[valid]) ** 2)
                )
                if valid.any()
                else np.nan
            },
            {
                "trend": trend,
                "seasonal": seasonal,
                "residual": resid,
                "reconstruction": reconstruction,
            },
            {"boundary_nans_preserved": True},
        )
    if method in MULTIVARIATE:
        from sklearn.preprocessing import StandardScaler

        d = frame.shape[1]
        r = p.get("n_components", p.get("k_factors"))
        if d < 2 or r >= d or n <= r + 2:
            raise NotApplicable(
                "requires multiple channels, fewer components than channels, and "
                "enough time points"
            )
        if (frame.std(ddof=0) == 0).any():
            raise NotApplicable("constant channel makes this reduction ill-conditioned")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(frame)
        arrays = {"channel_mean": scaler.mean_, "channel_scale": scaler.scale_}
        details = {
            "fit_scope": "sample",
            "standardization": "within-sample per-channel",
            "factor_identity": "local to this sample; no ground-truth alignment",
        }
        if method in ("pca", "ica"):
            from sklearn.decomposition import PCA, FastICA

            model = (
                PCA(n_components=r, svd_solver="full")
                if method == "pca"
                else FastICA(
                    n_components=r,
                    max_iter=p["max_iter"],
                    tol=p["tol"],
                    random_state=seed,
                    whiten="unit-variance",
                )
            )
            factors = model.fit_transform(scaled)
            reconstruction = scaler.inverse_transform(model.inverse_transform(factors))
            arrays["loadings"] = model.components_ if method == "pca" else model.mixing_
            if method == "pca":
                arrays["explained_variance_ratio"] = model.explained_variance_ratio_
            else:
                details["n_iter"] = int(model.n_iter_)
                details["converged"] = bool(model.n_iter_ < p["max_iter"])
        else:
            from sktime.forecasting.dynamic_factor import DynamicFactor

            model = DynamicFactor(
                k_factors=r,
                factor_order=p["factor_order"],
                maxiter=p["maxiter"],
                full_output=True,
            ).fit(pd.DataFrame(scaled))
            # Isolate the version-sensitive backend access here, covered by tests.
            backend = model._fitted_forecaster
            factors = np.asarray(backend.factors.smoothed).T
            design = np.asarray(backend.model.ssm["design"])
            if design.ndim == 3:
                design = design[:, :, 0]
            common = (design @ backend.smoothed_state).T
            reconstruction = scaler.inverse_transform(common)
            arrays["loadings"] = design
            arrays["fitted_parameters"] = np.asarray(backend.params)
            details["converged"] = bool(backend.mle_retvals.get("converged", False))
            details["backend_parameter_names"] = list(backend.param_names)
        arrays.update(factors=factors, reconstruction=reconstruction)
        return Outcome(
            {
                "reconstruction_rmse": np.sqrt(
                    np.mean((frame.to_numpy() - reconstruction) ** 2)
                ),
                "components": r,
            },
            arrays,
            details,
        )
    raise ValueError(f"Unimplemented method {method}")
