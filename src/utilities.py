"""Feature transforms used across research projects."""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

# ── Optional heavy imports — never break the workspace if one is missing ──────

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
    mdates = None

try:
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.stattools import acf, adfuller, coint, kpss, pacf
except ImportError:  # pragma: no cover
    sm = None
    adfuller = kpss = coint = None
    acf = pacf = acorr_ljungbox = None
    AutoReg = None
    plot_acf = plot_pacf = None

try:
    from scipy import optimize, stats
except ImportError:  # pragma: no cover
    optimize = stats = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    px = go = None

try:
    import sklearn
    from sklearn import linear_model, metrics, preprocessing
    from sklearn.model_selection import TimeSeriesSplit
except ImportError:  # pragma: no cover
    sklearn = linear_model = metrics = preprocessing = TimeSeriesSplit = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = nn = None

try:
    from data_profiling import ProfileReport
except ImportError:  # pragma: no cover
    ProfileReport = None

try:
    from lse import LSE
except ImportError:  # pragma: no cover
    LSE = None

# Set Themes
from theme import set_theme as _set_plot_theme

_set_plot_theme()


# ── Workspace paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("research")


def data_path(product: str, layer: str = "processed") -> Dict[str, Any]:
    """
    Resolve the data directory for a product without hardcoding paths.

    Parameters
    ----------
    product : 'energy' | 'fx' | 'crypto' (any folder at the workspace root)
    layer   : 'raw' | 'processed'

    Returns
    -------
    dict with keys: path (Path), exists (bool)
    """
    path = ROOT / product / "data" / layer
    return {"path": path, "exists": path.exists()}


def save_processed(df: pd.DataFrame, product: str, name: str) -> Dict[str, Any]:
    """
    Save a DataFrame to <product>/data/processed/<name>.parquet.

    Returns
    -------
    dict with keys: path (Path), rows (int), size_mb (float)
    """
    directory = data_path(product, "processed")["path"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    df.to_parquet(path)
    return {
        "path": path,
        "rows": len(df),
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def load_processed(product: str, name: str) -> Dict[str, Any]:
    """
    Load <product>/data/processed/<name>.parquet.

    Returns
    -------
    dict with keys: df (DataFrame), path (Path), rows (int)
    """
    path = data_path(product, "processed")["path"] / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed file: {path}")
    df = pd.read_parquet(path)
    return {"df": df, "path": path, "rows": len(df)}


# ── Notebook / display setup ───────────────────────────────────────────────────

def setup_notebook(
    figsize: Tuple[int, int] = (12, 5),
    max_rows: int = 60,
    max_columns: int = 40,
) -> Dict[str, Any]:
    """
    Apply standard pandas display options and matplotlib defaults.
    Call once at the top of every notebook.

    Returns
    -------
    dict with keys: pandas_options (dict), matplotlib (bool)
    """
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    mpl_ok = plt is not None
    if mpl_ok:
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3

    warnings.filterwarnings("ignore", category=FutureWarning)

    return {
        "pandas_options": {
            "max_rows": max_rows,
            "max_columns": max_columns,
        },
        "matplotlib": mpl_ok,
    }


# ── Frame inspection ───────────────────────────────────────────────────────────

def frame_summary(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    """
    Structured snapshot of a DataFrame: shape, per-column diagnostics, index
    health, and head/tail samples.

    Parameters
    ----------
    df : DataFrame to inspect
    n  : rows to keep in the head/tail samples

    Returns
    -------
    dict with keys: shape (tuple), memory_mb (float), columns (DataFrame),
                    index (dict), head (DataFrame), tail (DataFrame)
    """
    columns = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
    })

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        columns = columns.join(pd.DataFrame({
            "min": numeric.min(),
            "mean": numeric.mean(),
            "max": numeric.max(),
        }))

    index: Dict[str, Any] = {
        "type": type(df.index).__name__,
        "monotonic": bool(df.index.is_monotonic_increasing),
        "duplicates": int(df.index.duplicated().sum()),
    }
    if len(df.index):
        index["start"] = df.index[0]
        index["end"] = df.index[-1]
    if isinstance(df.index, pd.DatetimeIndex):
        index["tz"] = str(df.index.tz)
        steps = df.index.to_series().diff().dropna()
        if not steps.empty:
            # modal step is the bar size; anything longer is a gap in the series
            step = steps.mode().iloc[0]
            index["step"] = step
            index["gaps"] = int((steps > step).sum())

    return {
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": columns,
        "index": index,
        "head": df.head(n),
        "tail": df.tail(n),
    }


def peek(df: pd.DataFrame, n: int = 5) -> None:
    """
    Print `frame_summary` as one readable console block.

    """
    summary = frame_summary(df, n=n)
    rows, cols = summary["shape"]
    print(f"shape: {rows:,} rows x {cols} cols   ({summary['memory_mb']} MB)")

    index = summary["index"]
    line = [f"index: {index['type']}"]
    if "start" in index:
        line.append(f"{index['start']} -> {index['end']}")
    line.append(f"monotonic={index['monotonic']}  dupes={index['duplicates']}")
    if "gaps" in index:
        line.append(f"step={index['step']}  gaps={index['gaps']}")
    print("  ".join(line), end="\n\n")

    print(summary["columns"].to_string())
    print(f"\nhead({n})\n{summary['head'].to_string()}")
    print(f"\ntail({n})\n{summary['tail'].to_string()}")


# data series from LSE
# api key
api_key = os.environ.get("LSE_API_KEY", "")
client = LSE(api_key=api_key) if (LSE is not None and api_key) else None


# load data series

def load_lse_data(client, asset: str, timeframe: str, limit):
    # load
    rows = client.candles(asset, timeframe, limit=limit, order="desc")
    # build df
    df = (pd.DataFrame(rows)
          .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
          .set_index("timestamp")
          .sort_index())
    return df

# ── Common transforms ──────────────────────────────────────────────────────────
def quick_adf(series: pd.Series) -> Dict[str, Any]:
    """
    Run an ADF stationarity test with sensible defaults.

    Returns
    -------
    dict with keys: stat (float), pvalue (float), n_lags (int),
                    stationary_5pct (bool)
    """
    if adfuller is None:
        raise ImportError("statsmodels is not installed")
    clean = series.dropna()
    stat, pvalue, n_lags, *_ = adfuller(clean, autolag="AIC")
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "n_lags": int(n_lags),
        "stationary_5pct": bool(pvalue < 0.05),
    }

def log_transform(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        df[f"{col}_log_return"] = np.log(df[col]).diff()
    return df

# build stationarity test using ADfuller
def ad_fuller_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Takes in a log transformed df and outputs P-value statistics for stationarity"""
    results = adfuller(df, result_object=True)
    print(results)
    print(f"p_value: {results.pvalue:.3f}")
    return results


# lags of series
def add_lags(
    df: pd.DataFrame,
    cols: str | List[str],
    n_lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add `<col>_lag_<n>` columns for every (column, lag) pair.

    Returns a new frame; `df` is not modified.
    """
    if isinstance(cols, str):
        cols = [cols]
    n_lags = n_lags or []
    new_columns = {}
    for col_name in cols:
        for lag_value in n_lags:
            new_col_name = f"{col_name}_lag_{lag_value}"
            # bind loop vars as defaults so each lambda keeps its own pair
            new_columns[new_col_name] = (
                lambda df_inner, column=col_name, lag=lag_value: df_inner[column].shift(lag)
            )
    return df.assign(**new_columns)

# calculate sharpe ratio
def calculate_sharpe(
    df: pd.DataFrame,
    returns: str,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> Dict[str, Any]:
    """
    Annualised Sharpe ratio for a column of per-period returns.

    Sharpe = mean(excess) / std(excess) * sqrt(periods_per_year), where excess
    subtracts the risk-free rate scaled down to the bar frequency.

    Parameters
    ----------
    df               : frame holding the return series
    returns          : column name of per-period (not cumulative) returns
    periods_per_year : bars per year — 252 for daily, 252*390 for 1-minute US
                       equity bars. Getting this wrong rescales the result.
    risk_free        : annual risk-free rate as a decimal (0.05 = 5%)

    Returns
    -------
    dict with keys: sharpe (float), mean_period (float), std_period (float),
                    ann_return (float), ann_vol (float), n (int)
    """
    series = df[returns].dropna()
    if series.empty:
        raise ValueError(f"Column '{returns}' has no non-null values")

    excess = series - (risk_free / periods_per_year)
    mean = float(excess.mean())
    std = float(excess.std())
    scale = np.sqrt(periods_per_year)

    return {
        "sharpe": float(mean / std * scale) if std > 0 else float("nan"),
        "mean_period": mean,
        "std_period": std,
        "ann_return": mean * periods_per_year,
        "ann_vol": std * scale,
        "n": int(series.size),
    }

def read_df(path:Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df.sort_index()


def df_report(df: pd.DataFrame) -> pd.DataFrame:
    print(df.info())
    print(df.index.min(), df.index.max())
    print(df.isnull().sum())
    print(df.describe())

def resample(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    df = df.resample(horizon).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df

def intraday_distribution(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Aggregate a column's mean/std by hour of day and day of week.

    Returns
    -------
    dict with keys: by_hour (DataFrame), by_day (DataFrame)
    """
    clean = df.copy()
    clean.index = pd.to_datetime(clean.index)
    clean["hour"] = clean.index.hour
    clean["day_of_week"] = clean.index.day_name()

    by_hour = clean.groupby("hour").agg(
        ret_mean=(col, "mean"),
        ret_std=(col, "std"),
        count=(col, "count"),
    )
    by_day = clean.groupby("day_of_week").agg(
        ret_mean=(col, "mean"),
        ret_std=(col, "std"),
        count=(col, "count"),
    )

    return {"by_hour": by_hour, "by_day": by_day}

# add volume features
def volume_features(
    df: pd.DataFrame,
    col: str = "volume",
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add rolling volume context: the rolling mean, volume relative to it, and a
    rolling z-score.

    Returns a new frame; `df` is not modified.
    """
    windows = windows or [20]
    out = df.copy()
    for window in windows:
        rolling = out[col].rolling(window)
        mean = rolling.mean()
        std = rolling.std()
        out[f"{col}_mean_{window}"] = mean
        out[f"{col}_rel_{window}"] = out[col] / mean
        out[f"{col}_z_{window}"] = (out[col] - mean) / std
    return out

# add time based features
def calculate_time_features(df, session_hour=1):
    df["hour"] = df.index.hour
    df["minutes_since_open"] = np.where(df.index.hour >= session_hour,
    (df.index.hour - session_hour) * 60 + df.index.minute,0)
    df["day_of_week"] = df.index.dayofweek
    return df

# fit ar model test
def ols_model(df: pd.DataFrame, col: str, lags: int):
    """
    Fit an AutoReg(`lags`) model with a constant to `df[col]`.

    Returns the fitted statsmodels results object (not a dict) so callers keep
    access to `.summary()`, `.params`, `.predict()` and the rest.
    """
    if AutoReg is None:
        raise ImportError("statsmodels is not installed")
    model = AutoReg(df[col].dropna(), lags=lags, trend="c")
    return model.fit()


# ── Visualizations ──────────────────────────────────────────────────────────

# plot distribution analsis with right or left skewed data confirmation

# Line series plot
def plot_line_series(df: pd.DataFrame, title: str, cols: list[str]):
    fig, ax = plt.subplots()
    ax.plot(df[cols])
    ax.set(
        xlabel="Date",
        ylabel = "Price",
        title = title,
    )

def plot_corr_matrix(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        corr = df[col].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        corr_map = sns.diverging_palette(160, 10, s=70, l=45, center="dark", as_cmap=True)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=corr_map, ax=ax)
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        plt.show()
    return df

def plot_return_distribution(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        ax = df[col].hist(bins=80, figsize=(10, 4))
        ax.set_title(f"Return distribution: {col}")
        plt.tight_layout()
        plt.show()

def plot_holding_period_comp(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        table = df.pivot_table(index="Holding Period", columns=[col],values="return",aggfunc="mean")
        table.reset_index(inplace=True)
        ax = table.plot(kind="bar", stacked=True, figsize=(10, 7))
        ax.set_title(f"Holding Time Distribution: {col}")
        ax.set_ylabel("Return")
        ax.set_xlabel("Holding Period")
        plt.show()

# auto correlation
def pacf_plot(df, col, lags):
    if plot_pacf is None:
        raise ImportError("statsmodels is not installed")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(df[col].dropna(), lags=lags, ax=ax)
    ax.set_title("Partial Autocorrelation (PACF) — identify significant lags")
    plt.tight_layout()
    plt.show()


def ljung_box(
    series: pd.Series,
    lags: Optional[List[int]] = None,
    n_plot_lags: int = 60,
) -> Dict[str, Any]:
    """
    Formally test the joint null that a series has no autocorrelation.

    An ACF/PACF plot is not a test. Eyeballing 60 lags against a 5% band
    expects three exceedances under a true white-noise null, so "some bars
    poke out" and "there is structure" are not the same statement. Ljung-Box
    tests all lags up to k jointly, which is the claim actually being made.

    Also reports the same test on squared returns. Uncorrelated is not
    independent: a return series can pass in the mean and fail hard in the
    variance, which is the normal case for a commodity and matters for
    position sizing even when the mean has no memory.

    Parameters
    ----------
    series      : per-period returns (not levels, not cumulative)
    lags        : lag horizons to test jointly; defaults to [5, 10, 20]
    n_plot_lags : horizon for the band-exceedance count

    Returns
    -------
    dict with keys: n (int), lags (DataFrame of lb_stat/lb_pvalue for the
                    level series), squared (DataFrame, same for squared),
                    white_noise_5pct (bool), acf (ndarray), band (float),
                    n_outside_band (int), expected_outside_band (float)
    """
    if acorr_ljungbox is None or acf is None:
        raise ImportError("statsmodels is not installed")

    clean = series.dropna()
    if clean.empty:
        raise ValueError("series has no non-null values")
    lags = lags or [5, 10, 20]

    level = acorr_ljungbox(clean, lags=lags, return_df=True)
    squared = acorr_ljungbox(clean ** 2, lags=lags, return_df=True)

    n = int(clean.size)
    n_plot_lags = min(n_plot_lags, n // 2)
    correlations = acf(clean, nlags=n_plot_lags, fft=True)[1:]
    band = 1.96 / np.sqrt(n)

    return {
        "n": n,
        "lags": level,
        "squared": squared,
        "white_noise_5pct": bool((level["lb_pvalue"] > 0.05).all()),
        "acf": correlations,
        "band": float(band),
        "n_outside_band": int((np.abs(correlations) > band).sum()),
        "expected_outside_band": 0.05 * len(correlations),
    }


def print_ljung_box(series: pd.Series, label: str, **kwargs) -> Dict[str, Any]:
    """Run `ljung_box` and print it as one readable block. Returns the dict."""
    result = ljung_box(series, **kwargs)

    print(f"{label}  (n={result['n']:,})")
    table = pd.DataFrame({
        "lb_p_level": result["lags"]["lb_pvalue"],
        "lb_p_squared": result["squared"]["lb_pvalue"],
    })
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"  ACF bars outside +/-{result['band']:.4f}: "
        f"{result['n_outside_band']}/{len(result['acf'])} "
        f"(expected by chance: {result['expected_outside_band']:.0f})"
    )
    verdict = "no autocorrelation in the mean" if result["white_noise_5pct"] \
        else "AUTOCORRELATION PRESENT in the mean"
    print(f"  -> {verdict}")
    if not bool((result["squared"]["lb_pvalue"] > 0.05).all()):
        print("  -> squared returns ARE autocorrelated (volatility clustering)")
    return result

# ── Factor screening ───────────────────────────────────────────────────────────

def ic_rank_spearman(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Rank IC of each feature against one target column.

    Rank IC (Spearman) is used instead of plain correlation because it only
    cares about the ORDER of the predictions, which is what a signal is judged
    on, and it is not dragged around by outliers.

    Returns one row per feature with the IC, its p-value, and the sample size.
    """
    from scipy.stats import spearmanr

    rows = {}
    for feature in features:
        pair = df[[feature, target]].dropna()
        ic, p_value = spearmanr(pair[feature], pair[target])
        rows[feature] = {"ic": ic, "p_value": p_value, "n": len(pair)}

    return pd.DataFrame(rows).T


def nw_t_stat(df: pd.DataFrame, features: list[str], target: str, lags: int) -> pd.DataFrame:
    """Newey-West t-statistic for each feature's rank IC.

    Overlapping forward returns make the plain p-value far too optimistic: a
    10-day target re-uses 9 of its 10 days at every step, so the observations
    are not independent. Setting lags to the forecast horizon corrects for it.

    The slope of standardised ranks IS the Spearman IC, so this is the same
    number as ic_rank_spearman with an honest standard error attached.
    """
    rows = {}
    for feature in features:
        pair = df[[feature, target]].dropna()

        x = pair[feature].rank()
        y = pair[target].rank()
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows[feature] = {"t_nw": fit.tvalues.iloc[1], "p_nw": fit.pvalues.iloc[1], "n": len(pair)}

    return pd.DataFrame(rows).T


def ic_decay(df: pd.DataFrame, features: list[str], horizons: list[int],
             target_template: str = "target_{horizon}_d") -> pd.DataFrame:
    """Rank IC of each feature at several forward horizons.

    The target columns must already exist - build them with
    features.forward_horizons() first.

    Returns a table with one row per feature and one column per horizon.
    """
    columns = {}
    for horizon in horizons:
        target = target_template.format(horizon=horizon)
        result = ic_rank_spearman(df, features, target)
        columns[horizon] = result["ic"].astype(float)

    return pd.DataFrame(columns)


def plot_ic_decay(ic_table: pd.DataFrame, title: str = "IC decay by horizon"):
    """Line plot of the table returned by ic_decay().

    A real signal decays smoothly from a peak. A jagged line that jumps around
    zero is noise.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for feature in ic_table.index:
        ax.plot(ic_table.columns, ic_table.loc[feature], marker="o", label=feature)

    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Rank IC")
    ax.set_title(title)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


def feature_correlation(df: pd.DataFrame, features: list[str],
                        method: str = "spearman") -> pd.DataFrame:
    """Correlation matrix of the feature set."""
    return df[features].corr(method=method)


def plot_feature_correlation(df: pd.DataFrame, features: list[str], method: str = "spearman",
                             title: str = "Feature correlation"):
    """Heatmap of feature_correlation(), with the upper triangle hidden.

    Anything above about 0.7 is close to a duplicate: it will split importance
    in a model and make the feature set look wider than it is.
    """
    corr = feature_correlation(df, features, method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    corr_map = sns.diverging_palette(160, 10, s=70, l=45, center="dark", as_cmap=True)
    sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, cmap=corr_map, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return corr


def plot_equity_curve(returns: pd.Series, title: str = "Equity curve"):
    """Compounded growth of capital, with the drawdown underneath, both in percent.

    `returns` must be a FRACTION of capital per trade, not raw P&L - use
    model.to_simple_returns() to convert first. Percentages are what make two
    different products comparable on the same axes.

    Pass the NON-OVERLAPPING sample, or the curve counts each trade several
    times and looks far smoother than the strategy really is.
    """
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax_equity.plot(equity.index, (equity.values - 1.0) * 100, linewidth=1.6)
    ax_equity.axhline(0, color="grey", linewidth=1)
    ax_equity.set_ylabel("Cumulative return (%)")
    ax_equity.set_title(title)

    ax_drawdown.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.6)
    ax_drawdown.set_ylabel("Drawdown (%)")
    ax_drawdown.set_xlabel("Date")

    plt.tight_layout()
    plt.show()

    return equity

# ═══════════════════════════════════════════════════════════════════════
# SIGNALS — 30 formulas as plain pandas
# ═══════════════════════════════════════════════════════════════════════

def _zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# --- Standard Atlas (4 families) ---

def fade_breakout(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Short at new highs, long at new lows. Fade exhaustion moves."""
    c = df["close"]
    return_trade = (c - df['open']) / df['open']
    prev_high = c.rolling(window).max().shift(1)
    prev_low = c.rolling(window).min().shift(1)
    signal = pd.Series(0.0, index=df.index)
    signal[c > prev_high] = -1.0
    signal[c < prev_low] = 1.0
    strat_return = signal.shift(1) * return_trade
    position = signal.shift(1)
    return pd.DataFrame({"signal": signal, "strategy_return": strat_return, "bar_return": return_trade, "position": position})


def fade_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Short above MA, long below MA. Mean reversion."""
    c = df["close"]
    ma = c.rolling(window).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[c > ma] = -1.0
    signal[c < ma] = 1.0
    return signal


def cs_reversion(prices: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    """Cross-sectional reversion: fade relative strength across assets.
    Requires multiple assets. Returns dict of {ticker: signal}.
    """
    ratios = {}
    for ticker, df in prices.items():
        c = df["close"]
        ma = c.rolling(window).mean()
        ratios[ticker] = c / ma
    ratio_df = pd.DataFrame(ratios)
    mean = ratio_df.mean(axis=1)
    std = ratio_df.std(axis=1).replace(0, np.nan)
    zscores = ratio_df.sub(mean, axis=0).div(std, axis=0)
    return {ticker: -zscores[ticker] for ticker in prices}


def mean_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Negative z-score of close. Short when extended, long when depressed."""
    return -_zscore(df["close"], window)


# --- Exotic Atlas (6 families) ---

def skew_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Negative rolling skewness. Fade distributional asymmetry."""
    return -df["close"].rolling(window).skew()


def keltner_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Deviation from MA normalised by ATR. Robust mean reversion."""
    c = df["close"]
    ma = c.rolling(window).mean()
    atr = _atr(df["high"], df["low"], c, window)
    return -(c - ma) / atr.replace(0, np.nan)


def accel_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Fade the second derivative of price. Captures momentum exhaustion."""
    accel = df["close"].diff().diff()
    return -_zscore(accel, window)


def ret_z_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Z-score of 1-bar returns. Fade extreme recent moves."""
    c = df["close"]
    return -_zscore(c / c.shift(1), window)


def skewkurt_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Skewness x kurtosis. Combines direction with tail thickness."""
    c = df["close"]
    return -c.rolling(window).skew() * c.rolling(window).kurt()


def fractal_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Range efficiency x z-score. The only trend-following signal."""
    c = df["close"]
    range_ = c.rolling(window).max() - c.rolling(window).min()
    noise = c.diff().abs().rolling(window).sum()
    efficiency = range_ / noise.replace(0, np.nan)
    return efficiency * _zscore(c, window)


# --- Signal registry ---

LOOKBACKS = [5, 20, 120]

_FAMILIES = [
    (fade_breakout,     "Fade-BO",       False),
    (fade_ma,           "Fade-MA",       False),
    (cs_reversion,      "CS-Rev",        True),
    (mean_reversion,    "MR",            False),
    (skew_fade,         "Skew-Fade",     False),
    (keltner_reversion, "Keltner-Rev",   False),
    (accel_fade,        "Accel-Fade",    False),
    (ret_z_reversion,   "Ret-Z-Rev",     False),
    (skewkurt_fade,     "SkewKurt",      False),
    (fractal_trend,     "Fractal-Trend", False),
]

SIGNAL_REGISTRY = []
for _func, _family, _cs in _FAMILIES:
    for _lb in LOOKBACKS:
        SIGNAL_REGISTRY.append({
            "name": f"{_family} {_lb}", "family": _family,
            "lookback": _lb, "function": _func, "cross_sectional": _cs,
        })


def list_signals() -> pd.DataFrame:
    return pd.DataFrame([{"name": s["name"], "family": s["family"], "lookback": s["lookback"]}
                         for s in SIGNAL_REGISTRY])


# Feature sets
def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using differences, so a series that crosses zero is fine."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def moving_avg_feat(df: pd.DataFrame, col: str, windows: list[int], z_window: int = 252) -> pd.DataFrame:
    """Distance from a moving average, z-scored.

    This measures how STRETCHED price is from its own mean. It is a level,
    not a direction - see ma_momentum_feat for direction.
    """
    for window in windows:
        moving_average = df[col].rolling(window).mean()
        df[f"{col}_ma_{window}"] = moving_average

        distance = df[col] - moving_average
        df[f"{col}_ma_dist_z_{window}"] = zscore(distance, z_window)

    return df


def ma_momentum_feat(df: pd.DataFrame, col: str, cross_pairs: list[tuple], above_windows: list[int],
                     macd_spans: tuple = (12, 26), z_window: int = 252) -> pd.DataFrame:
    """Direction and persistence of the trend, rather than distance from it.

    cross_pairs   - list of (fast, slow) window pairs, e.g. [(10, 50)]
    above_windows - share of the last N days spent above the N-day average
    macd_spans    - (fast, slow) spans for the EMA difference
    """
    for fast, slow in cross_pairs:
        fast_average = df[col].rolling(fast).mean()
        slow_average = df[col].rolling(slow).mean()
        df[f"{col}_macross_{fast}_{slow}"] = zscore(fast_average - slow_average, z_window)

    for window in above_windows:
        is_above = df[col] > df[col].rolling(window).mean()
        df[f"{col}_above_ma_{window}"] = is_above.rolling(window).mean()

    fast_span, slow_span = macd_spans
    macd = df[col].ewm(span=fast_span).mean() - df[col].ewm(span=slow_span).mean()
    df[f"{col}_macd"] = zscore(macd, z_window)

    return df


def seasonality_feat(df: pd.DataFrame) -> pd.DataFrame:
    """Day of year as a sine/cosine pair, so December and January sit next to each other."""
    day_of_year = df.index.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def vol_regime_feat(df: pd.DataFrame, col: str, windows: list[int],
                    regime_window: int = 252) -> pd.DataFrame:
    """Realised volatility over several windows, plus where the last one sits historically.

    vol_regime is a percentile rank, so 0.9 means today is more volatile than
    90 percent of the trailing window.
    """
    returns = df[col].diff()

    for window in windows:
        df[f"vol_{window}"] = returns.rolling(window).std()

    last_window = windows[-1]
    df["vol_regime"] = df[f"vol_{last_window}"].rolling(regime_window).rank(pct=True)

    return df


def volume_feat(df: pd.DataFrame, col: str, ratio_window: int = 20, short_window: int = 5,
                long_window: int = 20) -> pd.DataFrame:
    """Volume against its own recent average, computed within each contract.

    Volume is grouped by expiry because a contract's volume ramps up while it is
    the front month and falls away into delivery - that lifecycle is contract
    specific, so it must not be carried across a roll.

    Keep long_window well under the typical contract length or most rows come
    back empty. The median contract here is about 98 rows.
    """
    grouped = df.groupby("expiry")[col]

    df["volume_ratio"] = df[col] / grouped.transform(lambda s: s.rolling(ratio_window).mean())

    short_average = grouped.transform(lambda s: s.rolling(short_window).mean())
    long_average = grouped.transform(lambda s: s.rolling(long_window).mean())
    df["volume_trend"] = short_average / long_average

    return df


def days_to_harvest_feat(df: pd.DataFrame, harvest_month: int = 4) -> pd.DataFrame:
    """Calendar days until the next harvest window (roughly April for SA maize).

    Dates already past this year's harvest point at next year's.
    """
    year = df.index.year + (df.index.month > harvest_month).astype(int)
    next_harvest = pd.to_datetime(dict(year=year, month=harvest_month, day=1))

    days = (next_harvest.values - df.index.values) / np.timedelta64(1, "D")
    df["days_to_harvest"] = days

    return df


def forward_horizons(df: pd.DataFrame, col: str, horizons: list[int],
                     vol_window: int = 20) -> pd.DataFrame:
    """Forward returns, scaled by expected volatility, one column per horizon.

    `col` must be a LOG PRICE LEVEL (a clog_ column), not a return series -
    the forward return is a difference of two points on that level.

    Dividing by expected vol puts every horizon on the same scale, so a 1-day
    and a 10-day target are comparable.
    """
    returns = df[col].diff()

    for horizon in horizons:
        forward_return = df[col].shift(-horizon) - df[col]
        expected_vol = returns.rolling(vol_window).std() * np.sqrt(horizon)
        df[f"target_{horizon}_d"] = forward_return / expected_vol

    return df

# Hedged beta spread

def rolling_hedge_ratio_naive(df, base_col, quote_col, window=120, rebalance_days=5):
    """Estimate beta by rolling OLS, then freeze it between rebalances.

    A beta of 1 assumes SAFEX and CBOT move one-for-one. They don't -- freight,
    local basis and FX pass-through mean the true ratio drifts. Freezing between
    rebalances stops the hedge jittering on daily noise.
    """
    base = df[base_col]
    quote = df[quote_col]

    # Rolling OLS slope: cov(base, quote) / var(base).
    covariance = quote.rolling(window).cov(base)
    variance = base.rolling(window).var()
    beta_raw = covariance / variance

    # Freeze: only let beta update every `rebalance_days` bars.
    beta_frozen = beta_raw.copy()
    beta_frozen[:] = np.nan
    beta_frozen.iloc[::rebalance_days] = beta_raw.iloc[::rebalance_days]
    return beta_frozen.ffill()

def hedged_spread(df, base_col, quote_col, window=120, rebalance_days=5):
    """Continuous hedged spread built from returns, not level differences.

    Differencing the level across a beta change measures the revaluation from
    re-hedging, not a market move -- the same trap as differencing across a
    roll. So take the hedged return each bar using the beta already in force,
    then sum back up into a continuous series.
    """
    base_ret = df[base_col].diff()
    quote_ret = df[quote_col].diff()

    # Estimate on returns: regressing log levels on log levels is spurious.
    covariance = quote_ret.rolling(window).cov(base_ret)
    variance = base_ret.rolling(window).var()
    beta_raw = covariance / variance

    # Freeze between rebalances.
    beta_frozen = pd.Series(np.nan, index=df.index)
    beta_frozen.iloc[::rebalance_days] = beta_raw.iloc[::rebalance_days]
    beta_frozen = beta_frozen.ffill()

    # Apply the beta that was already in force at the start of the bar, so the
    # hedge ratio is never set with knowledge of the bar it prices.
    hedged_return = quote_ret - beta_frozen.shift(1) * base_ret

    return beta_frozen, hedged_return.cumsum()

def ou_half_life(series, window=60):
    """Half-life of mean reversion, from an AR(1) fit on the rolling window.

    Regress the change on the lagged level: delta_t = a + b * level_{t-1}.
    A negative b means reversion; half-life = -ln(2)/ln(1+b). Short half-life
    means reversion trades make sense, long or undefined means they don't.
    """
    def fit(x):
        level = x[:-1]
        delta = np.diff(x)
        if np.var(level) == 0:
            return np.nan
        b = np.cov(level, delta)[0, 1] / np.var(level)
        if b >= 0 or (1 + b) <= 0:
            return np.nan          # not reverting over this window
        return -np.log(2) / np.log(1 + b)

    return series.rolling(window).apply(fit, raw=True)


def variance_ratio(series, window=60, lag=5):
    """Variance of k-period returns divided by k times variance of 1-period.

    Above 1 means trending (moves compound), below 1 means mean-reverting
    (moves offset), around 1 means random walk.
    """
    returns_1 = series.diff()
    returns_k = series.diff(lag)
    var_1 = returns_1.rolling(window).var()
    var_k = returns_k.rolling(window).var()
    return var_k / (lag * var_1)


def cross_leg_correlation(df, col_a, col_b, window=60):
    """Rolling correlation between the two legs' returns.

    A direct measure of whether the arb relationship is holding. Falling
    correlation is the decoupling regime where fading the spread is dangerous.
    """
    return df[col_a].diff().rolling(window).corr(df[col_b].diff())


def relative_volatility(df, col_a, col_b, window=60):
    """Which leg is driving. Ratio of the two legs' realised vol."""
    vol_a = df[col_a].diff().rolling(window).std()
    vol_b = df[col_b].diff().rolling(window).std()
    return vol_a / vol_b