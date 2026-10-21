"""
strategy.py
-----------
Overnight crush-residual reversion with triple-barrier exits (final specification).

Signal : evening session, Kalman residual z-score (features.py) crosses |z| >= 2.
Trade rules
  Entry  : next bar's open after the signal bar closes; hedge (betas) frozen at the signal bar.
  Exits  : first barrier touched, checked on bar closes
             profit : favourable move >= 1 sigma   (about half the gap to the mean;
                      part of every dislocation is permanent, so waiting for the full
                      mean often fails)
             stop   : adverse move     >= 3 sigma
             time   : 12 hours
           sigma = residual std at the signal (z_std), so barriers scale with the regime.
  Sizing : 1 bushel of beans against beta-weighted meal and oil. The residual change is
           a log return on the beans notional; x bean price gives cents per bushel.
  Costs  : 1 tick per leg, round trip (= 0.58 c/bu on the spread).
  One trade at a time, every signal taken (no filter / meta-model).

Reporting: returns as % of bean notional (independent of contract count), USD per 5,000 bu.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theme import set_theme
set_theme()

from data_ingest import build_bars
from kalman_filter import add_kalman_residual
from features import add_session_zscore, detect_signals

TICK_COST_CENTS = 0.58          # ZS 0.25 + ZM 0.22 + ZL 0.11 c/bu, one tick each, round trip
BUSHELS_PER_CONTRACT = 5000
HOLDOUT_START = pd.Timestamp("2024-12-01")
DEVELOPMENT_START = pd.Timestamp("2021-01-01")


@dataclass
class StrategyConfig:
    threshold: float = 1.5      # entry |z|
    profit_sigma: float = 1.0
    stop_sigma: float = 3.0
    holding_hours: int = 12
    bar_minutes: int = 5
    cost_ticks: float = 1.0      # per leg, round trip


def simulate_trades(bars: pd.DataFrame, signals: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Walk each signal forward with the hedge frozen at the signal bar."""
    time_bars = config.holding_hours * 60 // config.bar_minutes
    log_open = {leg: np.log(bars[f"{leg}_adj_open"].values) for leg in ["ZS", "ZM", "ZL"]}
    log_close = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}
    session = bars["session_id"].values

    trades, next_free_bar = [], -1
    for signal in signals.itertuples():
        signal_bar, entry_bar = signal.signal_position, signal.signal_position + 1
        if entry_bar <= next_free_bar:                     # still in the previous trade
            continue
        if entry_bar + time_bars >= len(bars) or session[entry_bar] != session[signal_bar]:
            continue                                       # no fill in the same session / not enough data

        alpha = bars["frozen_alpha"].values[signal_bar]
        beta_meal = bars["frozen_beta_meal"].values[signal_bar]
        beta_oil = bars["frozen_beta_oil"].values[signal_bar]

        def hedged_residual(prices, index):
            return prices["ZS"][index] - (alpha + beta_meal * prices["ZM"][index] + beta_oil * prices["ZL"][index])

        entry_residual = hedged_residual(log_open, entry_bar)
        path = np.arange(entry_bar, entry_bar + time_bars + 1)
        favourable = signal.direction * (hedged_residual(log_close, path) - entry_residual)

        sigma = bars["z_std"].values[signal_bar]
        profit_hits = np.flatnonzero(favourable >= config.profit_sigma * sigma)
        stop_hits = np.flatnonzero(favourable <= -config.stop_sigma * sigma)
        first_profit = profit_hits[0] if len(profit_hits) else np.inf
        first_stop = stop_hits[0] if len(stop_hits) else np.inf
        exit_step = int(min(first_profit, first_stop, len(path) - 1))
        exit_reason = "profit" if exit_step == first_profit else "stop" if exit_step == first_stop else "time"

        bean_price = bars["ZS_adj_close"].values[signal_bar]
        trades.append(dict(
            signal_time=bars.index[signal_bar], exit_time=bars.index[path[exit_step]],
            signal_date=bars["trading_date"].values[signal_bar],
            exit_date=bars["trading_date"].values[path[exit_step]],
            entry_bar=entry_bar, exit_bar=path[exit_step],
            side="short beans / long products" if signal.direction < 0 else "long beans / short products",
            entry_z=signal.entry_z, beta_meal=beta_meal, beta_oil=beta_oil,
            sigma_cents=sigma * bean_price, bean_price_cents=bean_price,
            exit_reason=exit_reason, holding_minutes=(exit_step + 1) * config.bar_minutes,
            gross_cents=favourable[exit_step] * bean_price,
        ))
        next_free_bar = path[exit_step]
    return pd.DataFrame(trades)


def performance(trades: pd.DataFrame, trading_dates: pd.DatetimeIndex, cost_ticks: float) -> tuple[dict, pd.Series]:
    """Net results after costs; daily series over every trading day (zero when flat) for Sharpe and drawdown."""
    net_cents = trades["gross_cents"] - cost_ticks * TICK_COST_CENTS
    return_pct = net_cents / trades["bean_price_cents"] * 100
    exit_dates = pd.to_datetime(trades["exit_date"])
    # include exit dates past the sample end, so a trade that closes after the cut-off still counts
    all_days = trading_dates.union(pd.DatetimeIndex(exit_dates.unique()))
    daily = return_pct.groupby(exit_dates).sum().reindex(all_days, fill_value=0.0)
    equity = daily.cumsum()
    summary = dict(
        trades=len(trades), trades_per_year=len(trades) / (len(trading_dates) / 252),
        hit_rate=(net_cents > 0).mean(), mean_net_cents=net_cents.mean(),
        total_return_pct=equity.iloc[-1], max_drawdown_pct=(equity - equity.cummax()).min(),
        sharpe=daily.mean() / daily.std() * np.sqrt(252),
        total_usd=net_cents.sum() / 100 * BUSHELS_PER_CONTRACT,
    )
    return summary, equity


def run(raw_path: Path, config: StrategyConfig,
        start: pd.Timestamp = DEVELOPMENT_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: ingest -> Kalman residual -> z-score -> signals -> trades (signals from `start`)."""
    bars = build_bars(raw_path, config.bar_minutes)
    bars = add_kalman_residual(bars, config.bar_minutes)
    bars = add_session_zscore(bars)
    signals = detect_signals(bars, config.threshold)
    trades = simulate_trades(bars, signals, config)
    trades = trades[pd.to_datetime(trades["signal_date"]) >= start].reset_index(drop=True)
    trades["sample"] = np.where(pd.to_datetime(trades["signal_date"]) >= HOLDOUT_START, "holdout", "development")
    return bars, trades


if __name__ == "__main__":
    config = StrategyConfig()
    bars, trades = run(Path(r"C:\Users\Amin\PycharmProjects\PythonProject1\data\raw\latest_df_backup.parquet"), config)
    all_dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
    samples = {"development": all_dates[(all_dates >= DEVELOPMENT_START) & (all_dates < HOLDOUT_START)],
               "holdout": all_dates[all_dates >= HOLDOUT_START]}

    rows, curves = [], {}
    for sample, dates in samples.items():
        summary, equity = performance(trades[trades["sample"] == sample], dates, config.cost_ticks)
        rows.append(dict(sample=sample, **summary))
        curves[sample] = equity
    report = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(report.round(2).to_string(index=False))
    print("exit mix:", trades["exit_reason"].value_counts(normalize=True).round(2).to_dict())

    Path("output").mkdir(exist_ok=True)
    trades.to_csv("output/trades.csv", index=False)
    report.to_csv("output/performance.csv", index=False)

    # equity and drawdown in % of bean notional; development and holdout as separate curves
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for sample, equity in curves.items():
        axes[0].plot(equity.index, equity.values, label=sample)
        axes[1].plot(equity.index, (equity - equity.cummax()).values, label=sample)
    axes[0].set_ylabel("Cumulative return, % of bean notional")
    axes[0].set_title(f"Overnight crush-residual reversion, |z| >= {config.threshold}, {config.cost_ticks:g} tick per leg")
    axes[1].set_ylabel("Drawdown, %")
    for axis in axes:
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.legend()
    figure.tight_layout()
    figure.savefig("output/equity_drawdown.png", dpi=130)
