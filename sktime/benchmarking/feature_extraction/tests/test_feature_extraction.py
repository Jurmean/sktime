"""Test known signals, boundaries, serialization and benchmark semantics."""

import json

import numpy as np
import pandas as pd
import pytest

from sktime.benchmarking.feature_extraction import extract_features, run_benchmark
from sktime.benchmarking.feature_extraction.__main__ import main
from sktime.benchmarking.feature_extraction._data import Sample, load_dataset


def feature(result, suffix):
    """Return one named scalar without coupling to a channel name."""
    return result.features.filter(like=suffix).iloc[0, 0]


def artifact(result, method, name, sample_id=None):
    """Find the native array via the public metadata mapping."""
    for key, meta in result.metadata["artifact_index"].items():
        if (
            meta["method"] == method
            and meta["name"] == name
            and (sample_id is None or meta["sample_id"] == sample_id)
        ):
            return result.artifacts[key]
    raise AssertionError(f"No artifact {method}/{name}")


@pytest.mark.parametrize("n", [999, 1000])
def test_single_sided_frequency_and_parseval(n):
    # A frequency exactly on a DFT bin; catches +/- frequency cancellation and
    # missing odd-N/Nyquist energy weighting.
    fs = float(n)
    x = np.sin(2 * np.pi * 50 * np.arange(n) / fs)
    r = extract_features(
        pd.Series(x),
        methods=["spectrum"],
        parameters={"spectrum": {"sampling_rate": fs}},
    )
    assert feature(r, "mean_frequency") == pytest.approx(50)
    assert feature(r, "peak_frequency") == pytest.approx(50)
    assert feature(r, "mean_square_power") == pytest.approx(np.mean(x**2))
    recovered = np.fft.irfft(
        artifact(r, "spectrum", "fft_real") + 1j * artifact(r, "spectrum", "fft_imag"),
        n=n,
    )
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_zero_power_preserves_nan():
    r = extract_features(pd.Series(np.zeros(32)), methods=["spectrum"])
    assert np.isnan(feature(r, "mean_frequency"))
    assert r.diagnostics.status.tolist() == ["partial"]


@pytest.mark.parametrize(
    "method,expected", [("fit_linear", [3, 2]), ("fit_polynomial", [2, 3, 4])]
)
def test_known_curve_parameters(method, expected):
    t = np.arange(1, 101) / 100
    x = 3 * t + 2 if method == "fit_linear" else 2 + 3 * t + 4 * t**2
    p = {method: {"degree": 2}} if method == "fit_polynomial" else None
    r = extract_features(pd.Series(x), methods=[method], parameters=p)
    assert r.diagnostics.status.tolist() == ["ok"]
    np.testing.assert_allclose(
        r.features.filter(like="coefficient").iloc[0], expected, atol=1e-9
    )
    assert feature(r, "fit_rmse") < 1e-10


def test_seasonal_decomposition_and_wavelet():
    pytest.importorskip("statsmodels")
    pytest.importorskip("pywt")
    x = 20 + np.arange(120) * 0.1 + np.sin(2 * np.pi * np.arange(120) / 12)
    r = extract_features(
        pd.Series(x),
        methods=["stl", "classical_additive", "wavelet"],
        parameters={"stl": {"sp": 12}, "classical_additive": {"sp": 12}},
    )
    for m in ("stl", "classical_additive", "wavelet"):
        reconstructed = artifact(r, m, "reconstruction")
        valid = np.isfinite(reconstructed)
        np.testing.assert_allclose(reconstructed[valid], x[valid], atol=1e-10)
    assert np.isnan(artifact(r, "classical_additive", "trend")[:6]).all()


def test_channel_identity_unequal_lengths_and_splits():
    panel = pd.DataFrame(
        {
            "a": [pd.Series([1.0, 2.0, 3.0]), pd.Series([10.0, 11.0, 12.0, 13.0])],
            "b": [pd.Series([4.0, 5.0, 6.0]), pd.Series([20.0, 21.0, 22.0, 23.0])],
        },
        index=["TEST/b", "TRAIN/a"],
    )
    r = extract_features(panel, methods=["summary"])
    assert r.features.index.tolist() == ["TEST/b", "TRAIN/a"]
    assert {c.split("::")[0] for c in r.features} == {"a", "b"}
    np.testing.assert_allclose(r.features["a::summary::mean"], [2, 11.5])
    assert artifact(r, "input", "values", "TRAIN/a").shape == (4, 2)


def test_sktime_panel_representations():
    x = np.arange(2 * 2 * 8, dtype=float).reshape(2, 2, 8)
    a = extract_features(x, methods=["summary"])
    frames = [pd.DataFrame(v.T, columns=["dim_0", "dim_1"]) for v in x]
    b = extract_features(frames, methods=["summary"])
    pd.testing.assert_frame_equal(a.features, b.features)
    mi = pd.concat(frames, keys=[0, 1])
    c = extract_features(mi, methods=["summary"], sample_ids=["0", "1"])
    pd.testing.assert_frame_equal(a.features, c.features)


def test_window_boundaries_and_pipeline_order():
    panel = [pd.Series(np.arange(12.0)), pd.Series(np.arange(12.0) + 100)]
    r = extract_features(
        panel,
        methods=["summary"],
        window_length=4,
        stride=4,
        preprocessing=[{"method": "difference1"}, {"method": "zscore"}],
    )
    assert len(r.features) == 6
    assert r.metadata["samples"][3]["parent_id"] == "1"
    for name in r.features.index:
        np.testing.assert_allclose(artifact(r, "preprocessed", "values", name), 0)


def test_missing_values_require_explicit_choice():
    x = pd.Series([0.0, 1.0, np.nan, 3.0, 4.0])
    raw = extract_features(x, methods=["spectrum"])
    assert raw.diagnostics.status.iloc[0] == "skipped"
    filled = extract_features(
        x, methods=["spectrum"], preprocessing=[{"method": "interpolate"}]
    )
    assert filled.diagnostics.status.iloc[0] == "ok"
    assert np.isnan(x.iloc[2])
    leading = extract_features(
        pd.Series([np.nan, 1.0, 2.0]),
        methods=["summary"],
        preprocessing=[{"method": "interpolate"}],
    )
    assert leading.diagnostics.status.iloc[0] == "skipped"


def test_short_constant_and_invalid_conditions():
    pytest.importorskip("statsmodels")
    r = extract_features(
        pd.Series([1.0, 1.0, 1.0]), methods=["acf", "adf", "catch22", "stl", "pca"]
    )
    assert set(r.diagnostics.status) <= {"skipped", "partial"}
    assert r.features.filter(like="catch22").shape[1] == 22
    r = extract_features(pd.Series(np.arange(7.0)), methods=["pacf"])
    assert r.features.shape[1] == 10
    assert r.features.iloc[0, 2:].isna().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"methods": ["unknown"]},
        {"methods": ["acf"], "parameters": {"acf": {"n_lags": -1}}},
        {"window_length": 0},
        {"stride": 2},
        {"seed": -1},
        {"methods": ["spectrum"], "parameters": {"spectrum": {"sampling_rate": 0}}},
        {"methods": ["summary"], "parameters": {"typo": {}}},
        {"preprocessing": [{"method": "patch"}]},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        extract_features(pd.Series(np.arange(20.0)), **kwargs)


def test_nonuniform_fft_rejected():
    r = extract_features(
        pd.Series([1.0, 2.0, 3.0], index=[0, 1, 3]), methods=["spectrum"]
    )
    assert r.diagnostics.status.iloc[0] == "skipped"


def test_pca_is_channel_reduction_and_sample_local():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(80, 2))
    x = pd.DataFrame(latent @ rng.normal(size=(2, 4)))
    r = extract_features(x, methods=["pca"])
    assert artifact(r, "pca", "factors").shape == (80, 2)
    np.testing.assert_allclose(artifact(r, "pca", "reconstruction"), x, atol=1e-10)
    panel = extract_features([x, x * 3 + 100], methods=["pca"])
    np.testing.assert_allclose(
        artifact(r, "pca", "factors"), artifact(panel, "pca", "factors", "0")
    )


def test_dfm_backend_factors_are_accessible():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(5)
    latent = np.cumsum(rng.normal(size=60))
    x = pd.DataFrame(
        latent[:, None] * np.array([[1.0, 2.0, -1.0]]) + rng.normal(size=(60, 3)) * 0.1
    )
    r = extract_features(
        x, methods=["dfm"], parameters={"dfm": {"k_factors": 1, "maxiter": 5}}
    )
    assert artifact(r, "dfm", "factors").shape == (60, 1)
    assert r.diagnostics.status.iloc[0] in {"ok", "partial"}


def test_tsfresh_retains_nonfinite_output():
    pytest.importorskip("tsfresh")
    r = extract_features(pd.Series(np.zeros(24)), methods=["tsfresh"])
    assert r.features.shape[1] > 100
    assert r.features.isna().any().any()
    assert r.diagnostics.status.iloc[0] == "partial"
    assert "autocorrelation" in r.metadata["tsfresh_calculators"]


def test_tsfresh_preserves_dated_index():
    pytest.importorskip("tsfresh")
    x = pd.Series(np.arange(24.0), index=pd.date_range("2020-01-01", periods=24))
    r = extract_features(x, methods=["tsfresh"])
    timewise = r.features.filter(like="linear_trend_timewise")
    assert timewise.shape[1] > 0
    assert np.isfinite(timewise.to_numpy()).all()


def test_preprocessing_native_results():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    r = extract_features(
        x,
        methods=[
            "difference1",
            "difference2",
            "patch",
            "pool_mean",
            "pool_max",
            "padding",
        ],
        parameters={"patch": {"length": 4, "stride": 2}, "padding": {"length": 10}},
    )
    np.testing.assert_array_equal(
        artifact(r, "difference1", "values").ravel(), np.ones(7)
    )
    np.testing.assert_array_equal(
        artifact(r, "difference2", "values").ravel(), np.zeros(6)
    )
    np.testing.assert_array_equal(
        artifact(r, "pool_mean", "values").ravel(), [2.5, 6.5]
    )
    np.testing.assert_array_equal(artifact(r, "pool_max", "values").ravel(), [4.0, 8.0])
    assert artifact(r, "patch", "patches").shape == (3, 4)
    np.testing.assert_array_equal(
        artifact(r, "padding", "values").ravel(), np.r_[x, 0, 0]
    )


def test_ica_repeated_seed_is_reproducible():
    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(size=(64, 4)))
    a = extract_features(x, methods=["ica"])
    b = extract_features(x, methods=["ica"])
    np.testing.assert_allclose(
        artifact(a, "ica", "factors"), artifact(b, "ica", "factors")
    )


def test_missing_dependency_does_not_block_other_methods(monkeypatch):
    from sktime.benchmarking.feature_extraction import _engine

    original = _engine.run_method

    def blocked(method, *args):
        if method == "acf":
            raise ModuleNotFoundError("test missing statsmodels")
        return original(method, *args)

    monkeypatch.setattr(_engine, "run_method", blocked)
    r = extract_features(pd.Series(np.arange(20.0)), methods=["acf", "summary"])
    assert r.diagnostics.status.tolist() == ["blocked", "ok"]


def test_export_roundtrip(tmp_path):
    idx = pd.period_range("2020-01", periods=32, freq="M")
    r = extract_features(
        pd.Series(np.zeros(32), index=idx),
        methods=["spectrum"],
        output_dir=tmp_path / "out",
    )
    metadata = json.loads((tmp_path / "out/metadata.json").read_text())
    assert metadata["samples"][0]["index_type"] == "PeriodIndex"
    arrays = np.load(tmp_path / "out/artifacts.npz", allow_pickle=False)
    for k in arrays:
        np.testing.assert_array_equal(arrays[k], r.artifacts[k])
    with pytest.raises(FileExistsError):
        r.save(tmp_path / "out")


def test_cli_matches_python(tmp_path, capsys):
    assert (
        main(
            [
                "extract",
                "--datasets",
                "Airline",
                "--methods",
                "summary",
                "spectrum",
                "--out",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    actual = pd.read_csv(tmp_path / "cli/Airline/features.csv", index_col=0)
    expected = extract_features(
        load_dataset("Airline"), methods=["summary", "spectrum"]
    )
    pd.testing.assert_frame_equal(actual, expected.features, check_exact=False)


def test_benchmark_protocol_and_label_independence(tmp_path):
    r = run_benchmark(
        datasets=["Airline"],
        methods=["summary", "spectrum"],
        output_dir=tmp_path / "bench",
    )
    assert len(r["repetitions"]) == 8
    assert all(s["repeat_consistent"] for s in r["summary"])
    assert all(len(s["repeat_seconds"]) == 3 for s in r["summary"])
    f = pd.DataFrame({"x": np.arange(12.0)})
    a = extract_features(
        [Sample("id", f, {"label": "A", "split": "TRAIN"})], methods=["summary"]
    )
    b = extract_features(
        [Sample("id", f, {"label": "B", "split": "TEST"})], methods=["summary"]
    )
    pd.testing.assert_frame_equal(a.features, b.features)
    assert a.metadata["input_sha256"] != b.metadata["input_sha256"]
