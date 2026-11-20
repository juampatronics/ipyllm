# Copyright 2026 Juan Pablo de la Cruz Gutiérrez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type-specific, bounded inspectors backing the ``describe_object`` tool.

The goal is a compact, structured description of a *live* object that does not
grow with its size: a 10-million-row DataFrame must not become a 100 MB prompt.
Every inspector returns short text and is size-independent by construction.

Inspection is **progressive**. ``describe_object`` accepts a ``detail`` level:

- ``summary`` (default): cheap, size-independent identity facts (type, shape,
  dtype, device, memory, fitted status, labels). No per-row work.
- ``detail``: adds per-column breakdowns, missing values, and for floating
  data descriptive statistics (mean, std, min, median, max, IQR fences) that
  help flag outliers. Per-column work is bounded to a fixed number of columns.
- ``full``: adds sample values, a small head preview, or the source code,
  depending on the type.

Optional scientific libraries (NumPy, pandas, PyTorch, scikit-learn,
matplotlib) are imported lazily and only when an object of that type is
actually inspected, so their absence never breaks the tool.
"""

from __future__ import annotations

import importlib
import inspect as pyinspect
import math
import numbers
import types

# Hard cap on any inspector's output (defense in depth on top of the bounded
# iteration limits below). 12,000 characters ≈ a few thousand tokens.
MAX_INSPECT_CHARS = 12_000
# How many DataFrame columns receive per-column (row-scanning) work.
_MAX_STAT_COLUMNS = 40
# How many largest values per numeric column are listed (detail and up), so
# the model can count how many values exceed a threshold without executing.
_TOP_K = 5
# How many names/keys/values are shown before we elide the rest.
_MAX_NAMES = 24
# How many sample values / preview items we render.
_MAX_PREVIEW = 12
# Max characters for a single inline value.
_PREVIEW_CHARS = 80
# Fallback (unknown-type) representation budget.
_FALLBACK_CHARS = 4_000

_DETAIL_LEVELS = ("summary", "detail", "full")


# --- Small formatting helpers ---------------------------------------------


def _num(value) -> str:
    """Compact, deterministic rendering of a numeric scalar.

    Integers (including NumPy integers) are shown exactly; floats are shown to
    six significant digits, with NaN/Inf spelled out.
    """
    if value is None:
        return "n/a"
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return str(int(value))
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _preview(value)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.6g}"


def _mb(nbytes) -> str:
    try:
        return f"{int(nbytes) / 1_048_576:.1f} MiB"
    except (TypeError, ValueError):
        return "n/a"


def _preview(value) -> str:
    """A bounded single-line rendering of an arbitrary value."""
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - a hostile __repr__ must not break us
        return f"<unrepresentable: {type(exc).__name__}>"
    if len(text) > _PREVIEW_CHARS:
        text = text[:_PREVIEW_CHARS] + "…"
    return text


def _cap(text: str, limit: int = MAX_INSPECT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n… (output truncated)"


def _names_line(label: str, names: list[str]) -> str:
    shown = names[:_MAX_NAMES]
    extra = len(names) - len(shown)
    text = f"{label}: {', '.join(shown)}"
    if extra > 0:
        text += f"  (+{extra} more)"
    return text


def _dtypes_groups(dtypes) -> str:
    """Collapse a pandas ``dtypes`` axis into ``dtype×count`` groups."""
    groups: dict[str, int] = {}
    for dtype in dtypes:
        key = str(dtype)
        groups[key] = groups.get(key, 0) + 1
    ordered = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name}×{count}" for name, count in ordered)


# --- Optional type classes -------------------------------------------------


def _optional_class(module: str, name: str):
    """Import ``module.name`` lazily, returning ``None`` when unavailable.

    Any import failure (missing, partially installed, or incompatible) is
    swallowed so a broken optional dependency can never prevent ``ipyllm``
    from loading or the tool from working for other types.
    """
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:  # noqa: BLE001 - optional dependency; never fatal
        return None


def _is_function(obj) -> bool:
    return isinstance(
        obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)
    )


# --- Inspectors ------------------------------------------------------------
#
# Each inspector takes ``(obj, detail)`` and returns a bounded string. They
# import their own optional dependency, so a missing library only disables the
# inspector for the object types it would have handled.


def _describe_numpy_array(arr, detail) -> str:
    import numpy as np

    lines = [
        f"ndarray shape={list(arr.shape)} dtype={arr.dtype} ndim={arr.ndim}",
        f"nbytes={arr.nbytes} ({_mb(arr.nbytes)}) strides={list(arr.strides)} "
        f"contiguous={bool(arr.flags['C_CONTIGUOUS'])} aligned={bool(arr.flags['ALIGNED'])}",
    ]
    if detail == "summary":
        return "\n".join(lines)

    flat = np.asarray(arr).ravel()
    if arr.dtype.kind in "biufc" and flat.size:
        try:
            lines.append("stats: " + _array_stats(flat))
        except Exception as exc:  # noqa: BLE001
            lines.append(f"stats: unavailable ({type(exc).__name__})")
    else:
        lines.append(f"stats: not numeric (dtype {arr.dtype!r})")
    if detail == "full":
        lines.append(f"sample: {_array_sample(flat)}")
    return "\n".join(lines)


def _array_stats(flat) -> str:
    import numpy as np

    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return "all non-finite"
    q1, q3 = (float(v) for v in np.quantile(finite, [0.25, 0.75]))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    k = min(_TOP_K, finite.size)
    top = np.sort(np.partition(finite, finite.size - k)[-k:])[::-1]
    top_rendered = ", ".join(_num(v) for v in top.tolist())
    return (
        f"min={_num(finite.min())} max={_num(finite.max())} "
        f"mean={_num(finite.mean())} std={_num(finite.std())} "
        f"median={_num(np.median(finite))} "
        f"iqr=[{_num(q1)},{_num(q3)}] outlier_fence=[{_num(lo)},{_num(hi)}] "
        f"top{_TOP_K}=[{top_rendered}]"
    )


def _array_sample(flat) -> str:
    import numpy as np

    if flat.size == 0:
        return "[]"
    count = min(_MAX_PREVIEW, flat.size)
    idx = np.linspace(0, flat.size - 1, count).astype(int)
    values = [_num(x) for x in flat[idx]]
    suffix = f" … ({flat.size} total)" if flat.size > count else ""
    return "[" + ", ".join(values) + suffix + "]"


def _describe_pandas_dataframe(df, detail) -> str:
    import pandas as pd

    nrows, ncols = df.shape
    lines = [f"DataFrame shape=({nrows}, {ncols})"]
    try:
        lines.append(f"memory≈{_mb(df.memory_usage(deep=True).sum())}")
    except Exception:  # noqa: BLE001
        pass
    lines.append(f"dtypes: {_dtypes_groups(df.dtypes)}")
    lines.append(_names_line("columns", [str(c) for c in df.columns]))
    if detail == "summary":
        return "\n".join(lines)

    lines.extend(_describe_pandas_columns(df, pd, _MAX_STAT_COLUMNS))
    if detail == "full":
        lines.append(f"head(5):\n{_frame_head(df)}")
    return "\n".join(lines)


def _describe_pandas_series(series, detail) -> str:
    import pandas as pd

    lines = [
        f"Series shape=({series.size},) dtype={series.dtype} "
        f"name={series.name!r}"
    ]
    try:
        lines.append(f"memory≈{_mb(series.memory_usage(deep=True))}")
    except Exception:  # noqa: BLE001
        pass
    if detail == "summary":
        return "\n".join(lines)

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        try:
            lines.append("stats: " + _series_stats(series))
        except Exception as exc:  # noqa: BLE001
            lines.append(f"stats: unavailable ({type(exc).__name__})")
    try:
        missing = int(series.isna().sum())
        lines.append(f"missing: {missing}")
    except Exception:  # noqa: BLE001
        lines.append("missing: unavailable")
    if detail == "full":
        lines.append(f"sample: {_array_sample(series.to_numpy())}")
    return "\n".join(lines)


def _describe_pandas_columns(df, pd, limit: int) -> list[str]:
    """Per-column breakdown over a bounded prefix of the frame's columns.

    Row-scanning work (missing counts, statistics) is confined to the first
    ``limit`` columns, so cost stays bounded even for wide frames.
    """
    lines: list[str] = []
    columns = list(df.columns)
    bounded = columns[:limit]
    extra = len(columns) - len(bounded)
    for column in bounded:
        series = df[column]
        entry = f"  {column!r} ({series.dtype})"
        try:
            missing = int(series.isna().sum())
            entry += f" missing={missing}"
        except Exception:  # noqa: BLE001
            entry += " missing=n/a"
        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            try:
                entry += " " + _series_stats(series)
            except Exception as exc:  # noqa: BLE001
                entry += f" stats_unavailable({type(exc).__name__})"
        lines.append(entry)
    if extra > 0:
        lines.append(f"  … (+{extra} more column(s) not scanned)")
    lines.insert(0, f"per-column (first {len(bounded)} of {len(columns)}):")
    return lines


def _top_values(series, k: int = _TOP_K) -> str:
    """The ``k`` largest non-missing values, in descending order.

    Lets the model reason about how many values exceed a threshold without
    executing code: entries of the list above the threshold are the answer
    (or a lower bound when the list is exhausted).
    """
    try:
        values = series.nlargest(k)
    except Exception:  # noqa: BLE001
        return f"top{k}=n/a"
    if values.empty:
        return f"top{k}=[]"
    rendered = ", ".join(_num(v) for v in values.tolist())
    return f"top{k}=[{rendered}]"


def _series_stats(series) -> str:
    if series.dropna().empty:
        return "all missing"
    mean = series.mean()
    std = series.std()
    median = series.median()
    minimum = series.min()
    maximum = series.max()
    q1, q3 = (float(v) for v in series.quantile([0.25, 0.75]))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (
        f"mean={_num(mean)} std={_num(std)} min={_num(minimum)} "
        f"median={_num(median)} max={_num(maximum)} "
        f"iqr=[{_num(q1)},{_num(q3)}] outlier_fence=[{_num(lo)},{_num(hi)}] "
        + _top_values(series)
    )


def _frame_head(df) -> str:
    try:
        text = df.head(5).to_string(max_cols=_MAX_NAMES)
    except Exception as exc:  # noqa: BLE001
        return f"(head unavailable: {type(exc).__name__})"
    return _cap(text, _FALLBACK_CHARS)


def _describe_torch_tensor(tensor, detail) -> str:
    lines = [
        f"Tensor shape={list(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device}"
    ]
    try:
        nbytes = tensor.element_size() * tensor.nelement()
        lines.append(f"nbytes={nbytes} ({_mb(nbytes)})")
    except Exception:  # noqa: BLE001
        pass
    lines.append(
        f"requires_grad={bool(tensor.requires_grad)} "
        f"grad={'set' if tensor.grad is not None else 'none'}"
    )
    if detail == "summary":
        return "\n".join(lines)
    import torch

    try:
        with torch.no_grad():
            numeric = tensor.detach().cpu().float().numpy()
            lines.append("stats: " + _array_stats(numeric.ravel()))
            if detail == "full":
                lines.append(f"sample: {_array_sample(numeric.ravel())}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"stats: unavailable ({type(exc).__name__})")
    return "\n".join(lines)


def _describe_torch_module(module, detail) -> str:
    params = list(module.named_parameters())
    total = sum(p.numel() for _, p in params)
    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    children = [type(c).__name__ for c in module.children()]
    shown = children[:_MAX_NAMES]
    extra = len(children) - len(shown)
    submodules = ", ".join(shown) + (f"  (+{extra} more)" if extra > 0 else "")
    lines = [
        f"{type(module).__name__} (nn.Module)",
        f"parameters: total={total:,} trainable={trainable:,} "
        f"mode={'training' if module.training else 'eval'}",
        f"submodules({len(children)}): {submodules}",
    ]
    if detail == "summary":
        return "\n".join(lines)
    # Per-submodule parameter counts, bounded.
    breakdown: dict[str, int] = {}
    for name, param in params:
        head = name.split(".", 1)[0]
        breakdown[head] = breakdown.get(head, 0) + param.numel()
    ordered = sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ordered[:_MAX_NAMES]
    extra = len(ordered) - len(shown)
    text = ", ".join(f"{name}={count:,}" for name, count in shown)
    if extra > 0:
        text += f"  (+{extra} more)"
    lines.append(f"parameters by block: {text}")
    if detail == "full":
        lines.append(f"structural repr:\n{_cap(str(module), _FALLBACK_CHARS)}")
    return "\n".join(lines)


def _describe_matplotlib_figure(fig, detail) -> str:
    lines = [
        f"Figure size={tuple(float(v) for v in fig.get_size_inches())}in "
        f"dpi={fig.dpi} axes={len(fig.axes)}"
    ]
    suptitle = getattr(fig, "_supptitle", None)
    if suptitle is not None and suptitle.get_text():
        lines.append(f"title: {suptitle.get_text()}")
    if detail == "summary":
        return "\n".join(lines)
    for index, ax in enumerate(fig.axes):
        lines.append(
            f"axes[{index}] title={ax.get_title()!r} "
            f"xlabel={ax.get_xlabel()!r} ylabel={ax.get_ylabel()!r}"
        )
        artists = _collect_artists(ax)
        if artists:
            ordered = sorted(artists.items(), key=lambda kv: (-kv[1], kv[0]))
            lines.append(
                "  artists: "
                + ", ".join(f"{name}×{count}" for name, count in ordered)
            )
        else:
            lines.append("  artists: none")
        if detail == "full":
            legend = ax.get_legend()
            labels = [t.get_text() for t in legend.get_texts()] if legend else []
            if labels:
                lines.append(_names_line("legend", labels))
    return "\n".join(lines)


def _collect_artists(ax) -> dict[str, int]:
    """Count a matplotlib Axes' plotted objects, skipping its background patch."""
    artists: dict[str, int] = {}

    def count(name: str) -> None:
        artists[name] = artists.get(name, 0) + 1

    for line in ax.get_lines():
        count(type(line).__name__)
    for collection in ax.collections:
        count(type(collection).__name__)
    for patch in ax.patches:
        count(type(patch).__name__)
    for image in ax.images:
        count(type(image).__name__)
    for container in getattr(ax, "containers", ()):
        count(type(container).__name__)
    return artists


def _describe_sklearn_estimator(est, detail) -> str:
    import sklearn.base
    import sklearn.utils.validation as validation

    lines = [f"{type(est).__module__}.{type(est).__name__} (sklearn estimator)"]
    try:
        validation.check_is_fitted(est)
        fitted = True
    except Exception:  # noqa: BLE001
        fitted = False
    lines.append(f"fitted: {'yes' if fitted else 'no'}")
    if detail == "summary":
        # Surface fitted attributes that reveal the model's learned structure.
        fitted_attrs = [
            name for name in vars(est) if name.endswith("_") and name not in {"_estimator_type"}
        ]
        if fitted_attrs:
            lines.append(_names_line("fitted attrs", sorted(fitted_attrs)))
        classes = getattr(est, "classes_", None)
        if classes is not None:
            lines.append(_names_line("classes_", [str(c) for c in classes]))
        return "\n".join(lines)

    try:
        params = est.get_params(deep=True)
    except Exception:  # noqa: BLE001
        params = {}
    ordered = sorted(params.items())
    shown = ordered[:_MAX_NAMES]
    extra = len(ordered) - len(shown)
    lines.append(f"params({len(ordered)}):")
    for name, value in shown:
        lines.append(f"  {name}={_preview(value)}")
    if extra > 0:
        lines.append(f"  … (+{extra} more)")
    if detail == "full":
        for name in sorted(vars(est)):
            if name.startswith("_") or not name.endswith("_"):
                continue
            value = getattr(est, name, None)
            lines.append(f"fitted {name}: {_preview(value)}")
    return "\n".join(lines)


def _describe_function(fn, detail) -> str:
    qualname = getattr(fn, "__qualname__", None) or fn.__name__
    kind = "method" if isinstance(fn, types.MethodType) else (
        "builtin" if isinstance(fn, types.BuiltinFunctionType) else "function"
    )
    lines = [f"{kind} {qualname}"]
    try:
        lines.append(f"signature: {fn.__name__}{_format_signature(fn)}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"signature: unavailable ({type(exc).__name__})")
    doc = (pyinspect.getdoc(fn) or "").strip()
    first_line = doc.splitlines()[0] if doc else ""
    if first_line:
        lines.append(f"doc: {first_line}")
    elif not doc:
        lines.append("doc: (none)")
    if detail == "summary":
        return "\n".join(lines)
    if detail == "detail" and len(doc) > len(first_line):
        lines.append(f"docstring:\n{_cap(doc, _FALLBACK_CHARS)}")
    if detail == "full":
        try:
            source = pyinspect.getsource(fn)
            lines.append(f"source:\n{_cap(source, _FALLBACK_CHARS)}")
        except (OSError, TypeError):
            lines.append("source: (unavailable; REPL-defined or builtin)")
    return "\n".join(lines)


def _format_signature(fn) -> str:
    try:
        return str(pyinspect.signature(fn))
    except (pyinspect.SignatureError, ValueError):
        return "(...)"


def _describe_mapping(mapping, detail) -> str:
    length = len(mapping)
    lines = [f"{type(mapping).__name__} length={length}"]
    if detail == "summary":
        return "\n".join(lines)
    keys = list(mapping.keys())
    shown = keys[:_MAX_NAMES]
    extra = len(keys) - len(shown)
    lines.append(_names_line("keys", [str(k) for k in shown]))
    for key in shown:
        lines.append(f"  {key!r} = {_preview(mapping[key])}")
    if extra > 0:
        lines.append(f"  … (+{extra} more)")
    return "\n".join(lines)


def _describe_sequence(seq, detail) -> str:
    length = len(seq)
    lines = [f"{type(seq).__name__} length={length}"]
    if detail == "summary":
        return "\n".join(lines)
    items = []
    for item in list(seq)[:_MAX_PREVIEW]:
        items.append(_preview(item))
    extra = length - len(items)
    text = "[" + ", ".join(items)
    if extra > 0:
        text += f"  (+{extra} more)"
    lines.append(text + "]")
    return "\n".join(lines)


def _describe_scalar(obj, detail) -> str:
    return f"{type(obj).__name__}: {_preview(obj)}"


def _describe_fallback(obj, detail) -> str:
    try:
        text = repr(obj)
    except Exception as exc:  # noqa: BLE001
        return f"{type(obj).__name__}: (unrepresentable: {type(exc).__name__})"
    if len(text) > _FALLBACK_CHARS:
        text = text[:_FALLBACK_CHARS] + "…"
    return f"{type(obj).__name__}: {text}"


# --- Dispatcher ------------------------------------------------------------


def _build_inspectors() -> list[tuple[object, object]]:
    """Ordered ``(class_or_tuple, inspector)`` pairs, most-specific first.

    Optional dependencies are imported lazily; a missing library simply omits
    its inspector(s). Functions are checked separately before this table.
    """
    pairs: list[tuple[object, object]] = []

    def add(module: str, name: str, inspector) -> None:
        cls = _optional_class(module, name)
        if cls is not None:
            pairs.append((cls, inspector))

    # Structured types (check before generic containers).
    add("matplotlib.figure", "Figure", _describe_matplotlib_figure)
    add("sklearn.base", "BaseEstimator", _describe_sklearn_estimator)
    add("torch.nn", "Module", _describe_torch_module)
    add("torch", "Tensor", _describe_torch_tensor)
    add("pandas", "DataFrame", _describe_pandas_dataframe)
    add("pandas", "Series", _describe_pandas_series)
    add("numpy", "ndarray", _describe_numpy_array)
    add("numpy", "generic", _describe_scalar)

    # Generic containers and scalars.
    pairs.append((dict, _describe_mapping))
    pairs.append(((list, tuple, set, frozenset), _describe_sequence))
    pairs.append(((str, bytes, bytearray), _describe_scalar))
    pairs.append(
        ((int, float, complex, bool, type(None)), _describe_scalar)
    )
    return pairs


_INSPECTORS = _build_inspectors()


def describe_object(obj, detail: str = "summary") -> str:
    """Return a bounded, type-specific description of ``obj``.

    ``detail`` selects the depth: ``summary`` (default), ``detail``, or
    ``full``. Unknown values fall back to ``summary``. The result is always
    bounded to ``MAX_INSPECT_CHARS`` characters regardless of the object's
    size.
    """
    if detail not in _DETAIL_LEVELS:
        detail = "summary"

    if _is_function(obj):
        text = _describe_function(obj, detail)
    else:
        text = None
        for cls, inspector in _INSPECTORS:
            if isinstance(obj, cls):
                text = inspector(obj, detail)
                break
        if text is None:
            text = _describe_fallback(obj, detail)
    return _cap(text)
