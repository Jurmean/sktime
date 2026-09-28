"""Extraction, provenance, portable artifacts and serial benchmark protocol."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as packages
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sktime.benchmarking.feature_extraction._data import (
    DATASETS,
    fingerprint,
    load_dataset,
    normalize,
    window_samples,
)
from sktime.benchmarking.feature_extraction._methods import (
    DECOMPOSE,
    MULTIVARIATE,
    PREPROCESS,
    PRESETS,
    NotApplicable,
    preprocess,
    resolve,
    run_method,
    tsfresh_parameters,
)


def json_safe(value):
    """Strict JSON metadata; raw nonfinite numbers remain untouched in CSV/NPZ."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def provenance():
    """Capture dependency versions, code hashes, commit and machine characteristics."""
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        try:
            return subprocess.check_output(  # noqa: S603 -- fixed read-only git commands
                [shutil.which("git") or "/usr/bin/git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for name in (
        "sktime",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-base",
        "statsmodels",
        "tsfresh",
        "numba",
        "PyWavelets",
        "threadpoolctl",
    ):
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = None
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_worktree_status": git("status", "--porcelain"),
        "source_sha256": sources,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "versions": versions,
    }


@dataclass
class ExtractionResult:
    """Feature table, native arrays, metadata and per-invocation diagnostics.

    ``features`` has one row per sample and channel::method::feature columns.
    ``artifacts`` contains only pickle-free arrays; see metadata.artifact_index.
    ``save`` refuses to overwrite an existing directory.
    """

    features: pd.DataFrame
    artifacts: dict
    metadata: dict
    diagnostics: pd.DataFrame

    def save(self, output_dir):
        """Write CSV, compressed NPZ, strict JSON and a human-readable report."""
        start = time.perf_counter()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)
        self.features.to_csv(out / "features.csv", na_rep="NaN")
        self.diagnostics.to_csv(out / "diagnostics.csv", index=False, na_rep="NaN")
        np.savez_compressed(out / "artifacts.npz", **self.artifacts)
        write_json(out / "metadata.json", self.metadata)
        counts = (
            self.diagnostics.status.value_counts().to_dict()
            if len(self.diagnostics)
            else {}
        )
        report = [
            "# Feature extraction report",
            "",
            f"Samples: {len(self.features)}; scalar columns: {self.features.shape[1]}",
            f"Statuses: {counts}",
            "",
            (
                "Numerical reconstruction/fit errors are in-sample diagnostics, not "
                "predictive scores."
            ),
            (
                "NaN and Inf are preserved in features.csv and artifacts.npz. JSON uses"
                " null for nonfinite metadata."
            ),
            (
                "See diagnostics.csv for every invocation, actual parameters, warnings "
                "and failures."
            ),
            (
                "See metadata.json for original sample IDs, splits, labels (unused), "
                "time indices and array mappings."
            ),
        ]
        (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        write_seconds = time.perf_counter() - start
        # This timing covers payload/report serialization; the tiny receipt is excluded.
        write_json(
            out / "write_timing.json",
            {"write_seconds": write_seconds, "excludes": "write_timing.json"},
        )
        return write_seconds


def _configuration(preset, methods, parameters, preprocessing):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose {list(PRESETS)}")
    methods = list(PRESETS[preset] if methods is None else methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a dictionary keyed by method")
    if set(parameters) - set(methods):
        raise ValueError("parameters contains methods not selected for this run")
    params = {m: resolve(m, parameters.get(m)) for m in methods}
    chain = []
    for step in preprocessing or []:
        if not isinstance(step, dict) or set(step) - {"method", "parameters"}:
            raise ValueError(
                "Each preprocessing step requires method and optional parameters"
            )
        m = step.get("method")
        if m not in PREPROCESS or m == "patch":
            raise ValueError(
                "Pipeline preprocessing must be a series-to-series method; use "
                "window_length for patches"
            )
        chain.append({"method": m, "parameters": resolve(m, step.get("parameters"))})
    return methods, params, chain


def _quality(features, arrays):
    scalar = np.asarray(list(features.values()), dtype=float)
    numeric = [
        np.asarray(a).ravel()
        for a in arrays.values()
        if np.issubdtype(np.asarray(a).dtype, np.number)
    ]
    values = np.concatenate(numeric) if numeric else np.array([])
    return {
        "n_features": len(scalar),
        "finite_features": int(np.isfinite(scalar).sum()),
        "feature_valid_fraction": float(np.isfinite(scalar).mean())
        if len(scalar)
        else np.nan,
        "array_values": len(values),
        "finite_array_values": int(np.isfinite(values).sum()),
        "array_valid_fraction": float(np.isfinite(values).mean())
        if len(values)
        else np.nan,
    }


def extract_features(
    X,
    *,
    preset="lec2-core",
    methods=None,
    parameters=None,
    preprocessing=None,
    sample_ids=None,
    window_length=None,
    stride=None,
    seed=42,
    output_dir=None,
    _capture_provenance=True,
):
    """Extract label-free features from Series or Panel data.

    Parameters are method-ID dictionaries (see ``list_methods``). Every fitted
    method operates within one observed sample, including PCA/ICA/DFM. Inputs
    are never concatenated across samples/splits. Optional ``window_length``
    and ``stride`` create complete windows per sample, dropping trailing tails.
    No scaling, imputation, feature selection or label use is automatic.

    Invalid configuration raises ValueError. Data/dependency/numerical failures
    are returned in diagnostics; healthy methods/samples continue. NaN/Inf are
    retained. Set output_dir to export all four result components.
    """
    methods, params, chain = _configuration(preset, methods, parameters, preprocessing)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    originals = normalize(X, sample_ids)
    original_checksum = fingerprint(originals)
    samples = window_samples(originals, window_length, stride)
    checksum = fingerprint(samples)
    arrays, mapping, records, rows, sample_meta = {}, {}, [], [], []

    def store_array(sample, method, channel, name, value):
        key = f"a{len(arrays):07d}"
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not permitted in portable artifacts")
        arrays[key] = value
        mapping[key] = {
            "sample_id": sample.sample_id,
            "method": method,
            "channel": channel,
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    with threadpool_limits(limits=1):
        for sample in samples:
            frame = sample.frame.copy()
            sample_meta.append(
                dict(
                    sample.metadata,
                    sample_id=sample.sample_id,
                    channels=list(map(str, frame.columns)),
                    original_length=len(frame),
                    index_type=type(frame.index).__name__,
                    index_dtype=str(frame.index.dtype),
                    index_names=list(frame.index.names),
                    index_frequency=getattr(frame.index, "freqstr", None),
                )
            )
            store_array(sample, "input", None, "values", frame.to_numpy())
            store_array(
                sample,
                "input",
                None,
                "time_index",
                np.asarray(list(map(str, frame.index))),
            )
            row = {}
            pipeline_error = None
            for step in chain:
                try:
                    frame = preprocess(step["method"], frame, step["parameters"])
                    if frame.empty:
                        raise NotApplicable("preprocessing produced an empty sequence")
                except Exception as exc:
                    pipeline_error = (
                        f"preprocessing {step['method']}: {type(exc).__name__}: {exc}"
                    )
                    break
            if chain and pipeline_error is None:
                store_array(sample, "preprocessed", None, "values", frame.to_numpy())
                store_array(
                    sample,
                    "preprocessed",
                    None,
                    "time_index",
                    np.asarray(list(map(str, frame.index))),
                )
            for method in methods:
                targets = (
                    [(None, frame)]
                    if method in MULTIVARIATE
                    else [(str(c), frame[[c]]) for c in frame]
                )
                for channel, part in targets:
                    actual = dict(params[method])
                    if (
                        method in DECOMPOSE
                        and "sp" in actual
                        and actual["sp"] is None
                        and sample.metadata.get("dataset") == "Airline"
                    ):
                        actual["sp"] = 12
                    record = {
                        "sample_id": sample.sample_id,
                        "channel": channel,
                        "method": method,
                        "status": "ok",
                        "reason": "",
                        "parameters": json.dumps(json_safe(actual), sort_keys=True),
                        "n_features": 0,
                        "finite_features": 0,
                        "array_values": 0,
                        "finite_array_values": 0,
                    }
                    started = time.perf_counter()
                    captured = []
                    try:
                        if pipeline_error:
                            raise NotApplicable(pipeline_error)
                        if method == "spectrum" and len(part) > 2:
                            index = part.index
                            numeric_index = (
                                index.asi8
                                if isinstance(
                                    index,
                                    (
                                        pd.DatetimeIndex,
                                        pd.TimedeltaIndex,
                                        pd.PeriodIndex,
                                    ),
                                )
                                else np.asarray(index)
                            )
                            if np.issubdtype(
                                numeric_index.dtype, np.number
                            ) and not np.allclose(
                                np.diff(numeric_index), np.diff(numeric_index)[0]
                            ):
                                raise NotApplicable(
                                    "FFT requires evenly sampled observations; "
                                    "resample explicitly"
                                )
                        with warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter("always")
                            result = run_method(method, part, actual, seed)
                        record.update(_quality(result.features, result.arrays))
                        record["details"] = json.dumps(
                            json_safe(result.details),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        nonfinite = (
                            record["finite_features"] < record["n_features"]
                            or record["finite_array_values"] < record["array_values"]
                        )
                        convergence = result.details.get("converged") is False or any(
                            "converg" in str(w.message).lower()
                            or "covariance of the parameters" in str(w.message).lower()
                            for w in captured
                        )
                        if nonfinite or convergence:
                            record["status"] = "partial"
                            record["reason"] = "; ".join(
                                s
                                for s in (
                                    "nonfinite output preserved" if nonfinite else "",
                                    "solver did not converge or parameters "
                                    "unidentifiable"
                                    if convergence
                                    else "",
                                    result.details.get("reason", ""),
                                )
                                if s
                            )
                        for name, value in result.features.items():
                            prefix = (
                                quote(channel, safe="")
                                if channel is not None
                                else "@sample"
                            )
                            column = f"{prefix}::{method}::{name}"
                            row[column] = float(value)
                        for name, value in result.arrays.items():
                            store_array(sample, method, channel, name, value)
                    except NotApplicable as exc:
                        record.update(status="skipped", reason=str(exc))
                    except (ImportError, ModuleNotFoundError) as exc:
                        record.update(
                            status="blocked", reason=f"{type(exc).__name__}: {exc}"
                        )
                    except Exception as exc:
                        record.update(
                            status="failed", reason=f"{type(exc).__name__}: {exc}"
                        )
                    record["seconds"] = time.perf_counter() - started
                    record["warnings"] = json.dumps(
                        sorted(
                            {f"{w.category.__name__}: {w.message}" for w in captured}
                        ),
                        ensure_ascii=False,
                    )
                    records.append(record)
            rows.append(row)
    metadata = {
        "schema_version": 1,
        "config": {
            "preset": preset,
            "methods": methods,
            "parameters": params,
            "preprocessing": chain,
            "window_length": window_length,
            "stride": stride,
            "seed": seed,
            "threads": 1,
        },
        "input_sha256": original_checksum,
        "windowed_input_sha256": checksum,
        "samples": sample_meta,
        "artifact_index": mapping,
        "fit_scope": "each observed sample independently",
        "labels_used": False,
    }
    if "tsfresh" in methods:
        try:
            metadata["tsfresh_calculators"] = tsfresh_parameters(
                params["tsfresh"]["feature_set"]
            )
        except ImportError:
            metadata["tsfresh_calculators"] = None
    if _capture_provenance:
        metadata["runtime"] = provenance()
    features = pd.DataFrame(
        rows, index=pd.Index([s.sample_id for s in samples], name="sample_id")
    )
    result = ExtractionResult(features, arrays, metadata, pd.DataFrame(records))
    if output_dir is not None:
        result.save(output_dir)
    return result


def run_benchmark(
    *,
    datasets=None,
    preset="lec2-core",
    methods=None,
    parameters=None,
    preprocessing=None,
    seed=42,
    repeats=3,
    warmups=1,
    output_dir=None,
    window_length=None,
    stride=None,
    progress=None,
):
    """Run every dataset-method combination serially with independent repetitions.

    Returns a JSON-compatible dictionary with summary rows, raw repetition
    records and provenance. If output_dir is supplied, exports the first
    measured result of every combination and writes incremental receipts.
    Timing covers extraction/normalization/artifact construction, excludes
    loading, disk I/O, machine-provenance collection and repeat comparison.
    """
    methods, params, chain = _configuration(preset, methods, parameters, preprocessing)
    names = list(DATASETS) if datasets is None else list(datasets)
    if not names or len(set(names)) != len(names) or set(names) - set(DATASETS):
        raise ValueError(
            f"datasets must be a unique nonempty subset of {list(DATASETS)}"
        )
    if (
        not isinstance(repeats, int)
        or repeats < 1
        or not isinstance(warmups, int)
        or warmups < 0
    ):
        raise ValueError("repeats>=1 and warmups>=0 must be integers")
    # Validate remaining public arguments even if all datasets subsequently fail.
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    out = Path(output_dir) if output_dir is not None else None
    if out:
        out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "runtime": provenance(),
        "protocol": {
            "datasets": names,
            "preset": preset,
            "methods": methods,
            "parameters": params,
            "preprocessing": chain,
            "seed": seed,
            "warmups": warmups,
            "repeats": repeats,
            "threads": 1,
            "window_length": window_length,
            "stride": stride,
            "timing_scope": (
                "whole extraction, excluding load/write/provenance/comparison"
            ),
            "comparison_tolerance": {"rtol": 1e-7, "atol": 1e-9},
        },
        "datasets": {},
        "summary": [],
        "repetitions": [],
    }
    for name in names:
        load_start = time.perf_counter()
        try:
            samples = load_dataset(name)
            # Check window parameters outside the per-method error boundary.
            window_samples(samples, window_length, stride)
        except Exception as exc:
            for method in methods:
                manifest["summary"].append(
                    {
                        "dataset": name,
                        "method": method,
                        "status": "failed",
                        "reason": (
                            f"dataset load/validation: {type(exc).__name__}: {exc}"
                        ),
                    }
                )
            continue
        manifest["datasets"][name] = {
            "load_seconds": time.perf_counter() - load_start,
            "input_sha256": fingerprint(samples),
            "samples": len(samples),
            "channels": sorted({s.frame.shape[1] for s in samples}),
            "lengths": sorted({len(s.frame) for s in samples}),
        }
        for method in methods:
            if progress:
                progress(f"{name}: {method}")
            timings, statuses, reference, first_diag, same, write_seconds = (
                [],
                [],
                None,
                None,
                True,
                0.0,
            )
            for iteration in range(-warmups, repeats):
                start = time.perf_counter()
                result = extract_features(
                    samples,
                    preset=preset,
                    methods=[method],
                    parameters={method: params[method]},
                    preprocessing=chain,
                    seed=seed,
                    window_length=window_length,
                    stride=stride,
                    _capture_provenance=False,
                )
                elapsed = time.perf_counter() - start
                counts = result.diagnostics.status.value_counts().to_dict()
                receipt = {
                    "dataset": name,
                    "method": method,
                    "phase": "warmup" if iteration < 0 else "measured",
                    "iteration": iteration,
                    "seconds": elapsed,
                    "status_counts": counts,
                }
                manifest["repetitions"].append(receipt)
                if iteration < 0:
                    continue
                timings.append(elapsed)
                statuses.append(counts)
                if reference is None:
                    reference = result.features.copy()
                    reference_arrays = {
                        k: v.copy() for k, v in result.artifacts.items()
                    }
                    first_diag = result.diagnostics.copy()
                    if out:
                        result.metadata["runtime"] = manifest["runtime"]
                        write_seconds = result.save(out / name / method)
                else:
                    same = (
                        same
                        and reference.index.equals(result.features.index)
                        and reference.columns.equals(result.features.columns)
                    )
                    if same:
                        same = bool(
                            np.allclose(
                                reference.to_numpy(),
                                result.features.to_numpy(),
                                rtol=1e-7,
                                atol=1e-9,
                                equal_nan=True,
                            )
                        )
                    same = same and reference_arrays.keys() == result.artifacts.keys()
                    if same:
                        for key, a in reference_arrays.items():
                            b = result.artifacts[key]
                            if a.shape != b.shape or (
                                not np.allclose(
                                    a, b, rtol=1e-7, atol=1e-9, equal_nan=True
                                )
                                if np.issubdtype(a.dtype, np.number)
                                else not np.array_equal(a, b)
                            ):
                                same = False
                                break
                    same = same and first_diag[
                        ["sample_id", "channel", "status", "reason"]
                    ].equals(
                        result.diagnostics[["sample_id", "channel", "status", "reason"]]
                    )
            counts = statuses[0]
            success = counts.get("ok", 0) + counts.get("partial", 0)
            status = (
                "ok"
                if counts.get("ok", 0) == len(first_diag)
                else "partial"
                if success
                else "blocked"
                if counts.get("blocked", 0)
                else "failed"
                if counts.get("failed", 0)
                else "skipped"
            )
            row = {
                "dataset": name,
                "method": method,
                "status": status,
                "median_seconds": float(np.median(timings)),
                "repeat_seconds": timings,
                "repeat_consistent": same,
                "samples": len(reference),
                "scalar_columns": reference.shape[1],
                "status_counts": counts,
                "feature_valid_fraction": float(
                    first_diag.finite_features.sum() / first_diag.n_features.sum()
                )
                if first_diag.n_features.sum()
                else None,
                "array_valid_fraction": float(
                    first_diag.finite_array_values.sum() / first_diag.array_values.sum()
                )
                if first_diag.array_values.sum()
                else None,
                "write_seconds": write_seconds,
                "reasons": first_diag.loc[first_diag.reason != "", "reason"]
                .value_counts()
                .to_dict(),
            }
            manifest["summary"].append(row)
            if out:
                write_json(out / "benchmark.json", manifest)
                pd.DataFrame(manifest["summary"]).to_csv(
                    out / "summary.csv", index=False
                )
    if out:
        write_json(out / "benchmark.json", manifest)
        pd.DataFrame(manifest["summary"]).to_csv(out / "summary.csv", index=False)
        lines = [
            "# Lecture 2 extraction benchmark",
            "",
            (
                "Efficiency and numerical diagnostics only; this is not a downstream "
                "accuracy ranking."
            ),
            "",
            (
                "| Dataset | Method | Status | Median seconds | Scalar columns | "
                "Reproducible |"
            ),
            "|---|---|---|---:|---:|---|",
        ]
        for row in manifest["summary"]:
            lines.append(
                f"| {row['dataset']} | {row['method']} | {row['status']} | "
                f"{row.get('median_seconds', float('nan')):.6f} | "
                f"{row.get('scalar_columns', 0)} | "
                f"{row.get('repeat_consistent', False)} |"
            )
        lines.extend(
            [
                "",
                (
                    "See benchmark.json for load/write times, all warmup/repetition "
                    "timings, failure reasons, parameters and environment."
                ),
                (
                    "Each dataset/method directory contains the first measured "
                    "extraction, including nonfinite outputs."
                ),
            ]
        )
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_safe(manifest)
