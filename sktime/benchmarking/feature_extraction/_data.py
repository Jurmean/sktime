"""Lossless sample boundaries and offline dataset loaders."""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Sample:
    """One observed series; labels are metadata only."""

    sample_id: str
    frame: pd.DataFrame
    metadata: dict = field(default_factory=dict)


DATASETS = {
    "GunPoint": "load_gunpoint",
    "ArrowHead": "load_arrow_head",
    "ItalyPowerDemand": "load_italy_power_demand",
    "BasicMotions": "load_basic_motions",
    "Airline": "load_airline",
    "DailyDelhiClimate": "load_daily_delhi_climate",
}


def normalize(X, sample_ids=None):
    """Accept Series, numeric/nested DataFrame, MultiIndex Panel, numpy3D, df-list."""
    if isinstance(X, list) and X and all(isinstance(x, Sample) for x in X):
        samples = [Sample(s.sample_id, s.frame.copy(), dict(s.metadata)) for s in X]
    elif isinstance(X, pd.Series):
        samples = [
            Sample("0", X.to_frame(name=X.name if X.name is not None else "value"))
        ]
    elif isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
        levels = list(range(X.index.nlevels - 1))
        samples = []
        for key, frame in X.groupby(
            level=levels[0] if len(levels) == 1 else levels, sort=False, dropna=False
        ):
            name = json.dumps(key, ensure_ascii=False, default=str)
            samples.append(Sample(name, frame.droplevel(levels), {"original_id": key}))
    elif isinstance(X, pd.DataFrame) and X.size and isinstance(X.iloc[0, 0], pd.Series):
        samples = []
        for key, row in X.iterrows():
            if not all(isinstance(v, pd.Series) for v in row):
                raise ValueError("Nested Panel cells must all be pandas Series")
            if not all(v.index.equals(row.iloc[0].index) for v in row):
                raise ValueError("Channels within a sample must share an index")
            samples.append(
                Sample(
                    str(key), pd.concat(list(row), axis=1).set_axis(X.columns, axis=1)
                )
            )
    elif isinstance(X, pd.DataFrame):
        samples = [Sample("0", X.copy())]
    elif isinstance(X, np.ndarray) and X.ndim == 3:
        samples = [
            Sample(
                str(i),
                pd.DataFrame(v.T, columns=[f"dim_{j}" for j in range(v.shape[0])]),
            )
            for i, v in enumerate(X)
        ]
    elif (
        isinstance(X, list)
        and X
        and all(isinstance(x, (pd.DataFrame, pd.Series)) for x in X)
    ):
        samples = [
            Sample(str(i), x.to_frame() if isinstance(x, pd.Series) else x.copy())
            for i, x in enumerate(X)
        ]
    else:
        raise TypeError(
            "X must be a pandas Series/DataFrame or a sktime Panel (nested, "
            "MultiIndex, numpy3D, df-list)"
        )
    if not samples:
        raise ValueError("X contains no samples")
    if sample_ids is not None:
        if len(sample_ids) != len(samples):
            raise ValueError("sample_ids must match the number of samples")
        for s, name in zip(samples, sample_ids):
            s.sample_id = str(name)
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("sample IDs must be unique; pass sample_ids explicitly")
    for s in samples:
        f = s.frame
        if f.empty or not f.index.is_unique or not f.index.is_monotonic_increasing:
            raise ValueError(
                f"{s.sample_id}: input must be nonempty "
                "with unique, increasing time index"
            )
        if not f.columns.is_unique or len(set(map(str, f.columns))) != len(f.columns):
            raise ValueError(
                "Channel names must be unique, including after string conversion"
            )
        if not all(pd.api.types.is_numeric_dtype(t) for t in f.dtypes):
            raise ValueError("Channel values must be numeric")
        s.frame = f.astype(float)
    return samples


def window_samples(samples, length=None, stride=None):
    """Extract complete windows without crossing samples or train/test splits."""
    if length is None:
        if stride is not None:
            raise ValueError("stride requires window_length")
        return samples
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("window_length must be a positive integer")
    stride = length if stride is None else stride
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    result = []
    for s in samples:
        if len(s.frame) < length:
            raise ValueError(f"{s.sample_id}: shorter than window_length={length}")
        for start in range(0, len(s.frame) - length + 1, stride):
            meta = dict(
                s.metadata,
                parent_id=s.sample_id,
                window_start=start,
                window_stop=start + length,
            )
            result.append(
                Sample(
                    f"{s.sample_id}/window:{start}",
                    s.frame.iloc[start : start + length].copy(),
                    meta,
                )
            )
    return result


def fingerprint(samples):
    """Hash IDs, channel names, exact numeric values, index type/values and metadata."""
    h = hashlib.sha256()
    for s in samples:
        header = [
            s.sample_id,
            list(map(str, s.frame.columns)),
            type(s.frame.index).__name__,
            str(s.frame.index.dtype),
            list(map(str, s.frame.index)),
            s.metadata,
        ]
        h.update(
            json.dumps(header, ensure_ascii=False, sort_keys=True, default=str).encode()
        )
        h.update(np.asarray(s.frame, dtype="<f8").tobytes())
    return h.hexdigest()


def load_dataset(name):
    """Load only the six repository-bundled datasets; never download data."""
    from sktime import datasets

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose {list(DATASETS)}")
    loader = getattr(datasets, DATASETS[name])
    result = []
    if name in {"Airline", "DailyDelhiClimate"}:
        result = normalize(loader())
        result[0].sample_id = name
        result[0].metadata.update(dataset=name, split="unsplit")
    else:
        for split in ("TRAIN", "TEST"):
            X, y = loader(split=split, return_X_y=True)
            for i, s in enumerate(normalize(X)):
                s.sample_id = f"{name}/{split}/{i}"
                s.metadata.update(
                    dataset=name,
                    split=split,
                    original_id=str(X.index[i]),
                    label=str(y[i]),
                )
                result.append(s)
    return result
