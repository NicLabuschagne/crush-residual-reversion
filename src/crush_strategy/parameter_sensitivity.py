"""
parameter_sensitivity.py
------------------------
Heatmaps of Sharpe and total return across the strategy's main settings,
on the DEVELOPMENT period only (Jan 2021 - Nov 2024).

How to read them: these are sensitivity maps, not an optimisation. A good setting
sits on a broad plateau where neighbouring cells look similar. The final settings (outlined)
were fixed before the holdout was run.

  1. Exit grid      : profit barrier x stop barrier   (|z| >= 2, 12-hour time stop)
  2. Threshold/time : entry |z| x time stop            (profit 1 sigma, stop 3 sigma)
  3. Threshold/cost : entry |z| x cost per leg         (final exits)
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_ingest import build_bars
from kalman_filter import add_kalman_residual
from features import add_session_zscore, detect_signals
from strategy import StrategyConfig, simulate_trades, performance, DEVELOPMENT_START, HOLDOUT_START

FINAL = StrategyConfig()          # threshold 2.0, profit 1, stop 3, 12 hours, 1 tick


def development_result(bars, signals_by_threshold, config, dates):
    """Sharpe and total return (% of bean notional) on development signals for one config."""
    trades = simulate_trades(bars, signals_by_threshold[config.threshold], config)
    signal_dates = pd.to_datetime(trades["signal_date"])
    trades = trades[(signal_dates >= DEVELOPMENT_START) & (signal_dates < HOLDOUT_START)]
    summary, _ = performance(trades, dates, config.cost_ticks)
    return summary["sharpe"], summary["total_return_pct"], summary["trades"]


def grid(bars, signals_by_threshold, dates, row_name, row_values, column_name, column_values):
    """Evaluate every (row, column) pair, all other settings at their final values."""
    sharpe = pd.DataFrame(index=row_values, columns=column_values, dtype=float)
    total_return = sharpe.copy()
    trades = sharpe.copy()
    for row in row_values:
        for column in column_values:
            config = replace(FINAL, **{row_name: row, column_name: column})
            sharpe.loc[row, column], total_return.loc[row, column], trades.loc[row, column] = \
                development_result(bars, signals_by_threshold, config, dates)
    return sharpe, total_return, trades


def draw(axis, values, trades, title, row_label, column_label, final_cell, value_format):
    """One heatmap, annotated with the value and trade count, final setting outlined."""
    limit = np.nanmax(np.abs(values.values))
    image = axis.imshow(values.values.astype(float), cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(values.columns)), [str(c) for c in values.columns])
    axis.set_yticks(range(len(values.index)), [str(r) for r in values.index])
    axis.set_xlabel(column_label)
    axis.set_ylabel(row_label)
    axis.set_title(title)
    for i, row in enumerate(values.index):
        for j, column in enumerate(values.columns):
            axis.text(j, i, f"{values.loc[row, column]:{value_format}}\n{int(trades.loc[row, column])} tr",
                      ha="center", va="center", fontsize=9)
    i, j = list(values.index).index(final_cell[0]), list(values.columns).index(final_cell[1])
    axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
    plt.colorbar(image, ax=axis, fraction=0.046)


if __name__ == "__main__":
    bars = build_bars(Path("data/latest_df_backup.parquet"), FINAL.bar_minutes)
    bars = add_kalman_residual(bars, FINAL.bar_minutes)
    bars = add_session_zscore(bars)
    thresholds = [1.5, 2.0, 2.5]
    signals_by_threshold = {k: detect_signals(bars, k) for k in thresholds}
    all_dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
    dates = all_dates[(all_dates >= DEVELOPMENT_START) & (all_dates < HOLDOUT_START)]

    panels = [
        ("Exit grid (|z| >= 2, 12h time stop)", "profit_sigma", [1.0, 2.0, 3.0], "Profit barrier (sigma)",
         "stop_sigma", [1.0, 2.0, 3.0], "Stop barrier (sigma)", (FINAL.profit_sigma, FINAL.stop_sigma)),
        ("Entry threshold x time stop (exits 1 / 3 sigma)", "threshold", thresholds, "Entry |z|",
         "holding_hours", [6, 12, 24], "Time stop (hours)", (FINAL.threshold, FINAL.holding_hours)),
        ("Entry threshold x cost", "threshold", thresholds, "Entry |z|",
         "cost_ticks", [0.5, 1.0, 1.5], "Cost (ticks per leg, round trip)", (FINAL.threshold, FINAL.cost_ticks)),
    ]
    figure, axes = plt.subplots(len(panels), 2, figsize=(12, 13))
    for row_axes, (title, row_name, rows, row_label, column_name, columns, column_label, final_cell) in zip(axes, panels):
        sharpe, total_return, trades = grid(bars, signals_by_threshold, dates, row_name, rows, column_name, columns)
        draw(row_axes[0], sharpe, trades, f"{title}\nSharpe", row_label, column_label, final_cell, ".2f")
        draw(row_axes[1], total_return, trades, f"{title}\nTotal return, % of bean notional",
             row_label, column_label, final_cell, ".1f")
        print(f"\n{title}\nSharpe:\n{sharpe.round(2)}\nTotal return %:\n{total_return.round(1)}")
    figure.suptitle("Parameter sensitivity, development Jan 2021 - Nov 2024 (final spec outlined)", fontsize=13)
    figure.tight_layout()
    Path("output").mkdir(exist_ok=True)
    figure.savefig("output/parameter_sensitivity.png", dpi=130)
