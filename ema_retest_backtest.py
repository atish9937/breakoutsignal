"""
20 EMA breakout-retest intraday scalping backtest (5-minute timeframe).

Strategy (long side; short is the mirror):
  Step 1 (Breakout):  Green candle that opens below the 20 EMA and closes above it.
  Step 2 (Retest):    Within the next 2 candles, at least one red candle must close
                      above the 20 EMA. The first qualifying red candle = pullback bar.
  Step 3 (Entry):     As soon as any subsequent candle's high breaks the pullback
                      bar's high, enter at that high (stop-buy fill).

Exits:
  - Target: entry + 3 * (entry - EMA_at_entry)      (1:3 risk-to-reward)
  - Stop:   any 5m candle closes below the 20 EMA -> exit at that close
  - EOD:    square off at 15:20 session close

Invalidation (no entry taken):
  - No red bar within 2 candles after the breakout
  - Pullback bar itself closes below the EMA
  - Before trigger, a bar closes back below the EMA

Reads cached 5m parquet produced by smc_data.py.  Writes trades CSV + metrics
JSON under smc_results/<SYMBOL>_ema_retest/.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import smc_data

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("ema_retest")

Side = Literal["LONG", "SHORT"]

SESSION_FLAT = dt.time(15, 20)
SLIPPAGE_POINTS = 1.0
EMA_LEN = 20
RR = 3.0
MAX_RETEST_BARS = 2    # look at most N bars after breakout for the retest candle


@dc.dataclass
class Filters:
    """Optional gates applied before taking a setup."""
    htf_trend: int = 0           # 0 = off; N = resample to Nmin, require close vs 20 EMA alignment
    skip_opening_bars: int = 0   # skip first N 5m bars of the day (0..6)
    skip_lunch: bool = False     # skip 13:00..14:00 setups
    min_risk_pts: float = 0.0    # require EMA-to-entry distance >= this (points)
    max_risk_pts: float = 0.0    # 0 = no cap; otherwise skip if distance > cap
    breakeven_after_r: float = 0.0
    allow_long: bool = True
    allow_short: bool = True

    # Trader-knowledge additions
    ema_slope_bars: int = 0      # require EMA[i] - EMA[i-N] > 0 for longs (and < 0 for shorts)
    body_pct_min: float = 0.0    # require breakout bar |close-open| / (high-low) >= this (0..1)
    scale_out_r: float = 0.0     # 0 = off; else book half at entry + R*risk, trail rest with EMA
    trail_only: bool = False     # ignore fixed 1:3 target; exit only on EMA close or EOD


@dc.dataclass
class Trade:
    side: Side
    setup_bar: pd.Timestamp       # index of breakout candle
    retest_bar: pd.Timestamp      # index of pullback candle
    entry_bar: pd.Timestamp       # index of candle that triggered entry
    entry_price: float
    ema_at_entry: float
    stop_price: float             # == ema_at_entry (used only for RR math)
    target_price: float
    exit_bar: pd.Timestamp
    exit_price: float
    exit_reason: str              # TARGET | EMA_CLOSE | EOD
    pnl_points: float
    bars_held: int


def compute_ema(df: pd.DataFrame, length: int | None = None) -> pd.Series:
    span = int(length) if length else EMA_LEN
    return df["close"].ewm(span=span, adjust=False).mean()


def _is_green(bar: pd.Series) -> bool:
    return bar["close"] > bar["open"]


def _is_red(bar: pd.Series) -> bool:
    return bar["close"] < bar["open"]


def htf_trend_series(df: pd.DataFrame, minutes: int, length: int = EMA_LEN) -> pd.Series:
    """Resample closes to `minutes`m bars, compute 20-EMA, forward-fill back onto
    the 5m index. Value = +1 if HTF close > HTF EMA, -1 if below, 0 if unknown."""
    rule = f"{minutes}min"
    bars = df["close"].resample(rule, label="right", closed="right").last().dropna()
    ema_htf = bars.ewm(span=length, adjust=False).mean()
    trend = np.where(bars > ema_htf, 1, np.where(bars < ema_htf, -1, 0))
    s = pd.Series(trend, index=bars.index).reindex(df.index, method="ffill").fillna(0)
    return s.astype(int)


def _simulate_one(
    df: pd.DataFrame,
    ema: pd.Series,
    side: Side,
    breakout_idx: int,
    filters: Filters,
    htf: pd.Series | None = None,
) -> Trade | None:
    """Try to form a complete trade starting from breakout bar index `breakout_idx`.

    Returns None if the setup invalidates at any step before an exit."""
    n = len(df)
    retest_idx: int | None = None

    # Step 2: find pullback bar within next MAX_RETEST_BARS bars
    for j in range(breakout_idx + 1, min(breakout_idx + 1 + MAX_RETEST_BARS, n)):
        bar = df.iloc[j]
        e = ema.iloc[j]
        if side == "LONG":
            # Need a red bar that closes ABOVE the EMA.
            if _is_red(bar) and bar["close"] > e:
                retest_idx = j
                break
            # If a bar CLOSES back below EMA before a red-retest appears, cancel.
            if bar["close"] < e:
                return None
        else:  # SHORT
            if _is_green(bar) and bar["close"] < e:
                retest_idx = j
                break
            if bar["close"] > e:
                return None

    if retest_idx is None:
        return None

    # Entry level is the retest bar's extreme
    retest_bar = df.iloc[retest_idx]
    retest_extreme = retest_bar["high"] if side == "LONG" else retest_bar["low"]

    # Step 3: walk forward looking for a candle that trades through the extreme.
    for k in range(retest_idx + 1, n):
        bar = df.iloc[k]
        e = ema.iloc[k]

        # Must still be on the right side of the EMA at bar close before triggering
        # (if the bar closes wrong-side without triggering, setup is dead).
        triggered = (
            bar["high"] > retest_extreme if side == "LONG"
            else bar["low"] < retest_extreme
        )

        if triggered:
            # Fill at the retest extreme + slippage (stop-order style)
            entry = (retest_extreme + SLIPPAGE_POINTS) if side == "LONG" else (retest_extreme - SLIPPAGE_POINTS)
            ema_at_entry = e

            # Risk = distance from entry to the EMA at entry bar.
            risk = (entry - ema_at_entry) if side == "LONG" else (ema_at_entry - entry)
            if risk <= 0:
                return None
            if filters.min_risk_pts and risk < filters.min_risk_pts:
                return None
            if filters.max_risk_pts and risk > filters.max_risk_pts:
                return None
            target = entry + RR * risk if side == "LONG" else entry - RR * risk

            return _manage_exit(
                df, ema, side, breakout_idx, retest_idx, k, entry, ema_at_entry, target,
                filters,
            )

        # No trigger: did this bar close on the wrong side of EMA? If so, kill setup.
        if side == "LONG" and bar["close"] < e:
            return None
        if side == "SHORT" and bar["close"] > e:
            return None

    return None


def _manage_exit(
    df: pd.DataFrame,
    ema: pd.Series,
    side: Side,
    breakout_idx: int,
    retest_idx: int,
    entry_idx: int,
    entry: float,
    ema_at_entry: float,
    target: float,
    filters: Filters,
) -> Trade:
    n = len(df)
    risk = abs(entry - ema_at_entry)
    be_level = None
    if filters.breakeven_after_r > 0:
        be_level = entry + filters.breakeven_after_r * risk if side == "LONG" else entry - filters.breakeven_after_r * risk
    be_active = False

    # Scale-out: book 0.5 units at scale_level, let 0.5 trail with EMA.
    scale_level = None
    scale_filled = False
    scaled_pnl = 0.0
    if filters.scale_out_r > 0:
        scale_level = entry + filters.scale_out_r * risk if side == "LONG" else entry - filters.scale_out_r * risk

    for m in range(entry_idx, n):
        bar = df.iloc[m]
        e = ema.iloc[m]
        ts = df.index[m]

        if ts.time() >= SESSION_FLAT:
            # EOD flat on remaining position.
            exit_price = bar["close"]
            final = _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                                 entry, ema_at_entry, target, m, exit_price, "EOD")
            if scale_filled:
                final.pnl_points = round(scaled_pnl + 0.5 * ((exit_price - entry) if side == "LONG" else (entry - exit_price)) - 0.5 * SLIPPAGE_POINTS, 2)
                final.exit_reason = "EOD_RUNNER"
            return final

        hit_target = bar["high"] >= target if side == "LONG" else bar["low"] <= target
        stop_triggered = bar["close"] < e if side == "LONG" else bar["close"] > e

        # Scale-out fill check (half the position comes off, rest trails)
        if scale_level is not None and not scale_filled:
            scaled_in_bar = (
                bar["high"] >= scale_level if side == "LONG" else bar["low"] <= scale_level
            )
            if scaled_in_bar:
                scale_filled = True
                scaled_pnl = 0.5 * ((scale_level - entry) if side == "LONG" else (entry - scale_level)) - 0.5 * SLIPPAGE_POINTS

        if be_level is not None and not be_active:
            if side == "LONG" and bar["high"] >= be_level:
                be_active = True
            elif side == "SHORT" and bar["low"] <= be_level:
                be_active = True
        be_hit = be_active and (
            (side == "LONG" and bar["low"] <= entry) or (side == "SHORT" and bar["high"] >= entry)
        )

        # TRAIL_ONLY mode: ignore the fixed 1:3 target; exit solely on EMA close.
        if filters.trail_only:
            if stop_triggered:
                px = bar["close"]
                reason = "EMA_CLOSE"
                return _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                                    entry, ema_at_entry, target, m, px, reason)
            if be_hit:
                return _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                                    entry, ema_at_entry, target, m, entry, "BREAKEVEN")
            continue  # keep trailing

        if hit_target and not (stop_triggered and m == entry_idx):
            t = _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                             entry, ema_at_entry, target, m, target, "TARGET")
            if scale_filled:
                # Half already booked at scale_level; remainder exits at target.
                remainder = 0.5 * ((target - entry) if side == "LONG" else (entry - target)) - 0.5 * SLIPPAGE_POINTS
                t.pnl_points = round(scaled_pnl + remainder, 2)
                t.exit_reason = "SCALE+TARGET"
            return t

        if stop_triggered:
            t = _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                             entry, ema_at_entry, target, m, bar["close"], "EMA_CLOSE")
            if scale_filled:
                remainder = 0.5 * ((bar["close"] - entry) if side == "LONG" else (entry - bar["close"])) - 0.5 * SLIPPAGE_POINTS
                t.pnl_points = round(scaled_pnl + remainder, 2)
                t.exit_reason = "SCALE+EMA"
            return t

        if be_hit:
            return _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                                entry, ema_at_entry, target, m, entry, "BREAKEVEN")

    # Ran off the end of data — flat at last close.
    m = n - 1
    return _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                        entry, ema_at_entry, target, m, df.iloc[m]["close"], "EOD")


def _build_trade(df, ema, side, breakout_idx, retest_idx, entry_idx,
                 entry, ema_at_entry, target, exit_idx, exit_price, reason) -> Trade:
    pnl = (exit_price - entry) if side == "LONG" else (entry - exit_price)
    pnl -= SLIPPAGE_POINTS  # exit-side slippage
    return Trade(
        side=side,
        setup_bar=df.index[breakout_idx],
        retest_bar=df.index[retest_idx],
        entry_bar=df.index[entry_idx],
        entry_price=round(entry, 2),
        ema_at_entry=round(ema_at_entry, 2),
        stop_price=round(ema_at_entry, 2),
        target_price=round(target, 2),
        exit_bar=df.index[exit_idx],
        exit_price=round(exit_price, 2),
        exit_reason=reason,
        pnl_points=round(pnl, 2),
        bars_held=exit_idx - entry_idx + 1,
    )


def _in_lunch(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return dt.time(13, 0) <= t < dt.time(14, 0)


def scan_trades(df: pd.DataFrame, filters: Filters | None = None,
                ema_len: int | None = None) -> list[Trade]:
    filters = filters or Filters()
    ema = compute_ema(df, ema_len)
    htf = htf_trend_series(df, filters.htf_trend) if filters.htf_trend else None

    trades: list[Trade] = []
    for day, day_df in df.groupby(df.index.date):
        day_ema = ema.loc[day_df.index]
        day_htf = htf.loc[day_df.index] if htf is not None else None
        taken_until_idx = -1
        for i in range(1, len(day_df)):
            if i <= taken_until_idx:
                continue
            if i < filters.skip_opening_bars:
                continue
            bar = day_df.iloc[i]
            e = day_ema.iloc[i]
            ts = day_df.index[i]
            if filters.skip_lunch and _in_lunch(ts):
                continue

            # Pre-compute common trader-knowledge gates
            rng = bar["high"] - bar["low"]
            body_ok = True
            if filters.body_pct_min and rng > 0:
                body_ok = (abs(bar["close"] - bar["open"]) / rng) >= filters.body_pct_min

            slope_ok_long = True
            slope_ok_short = True
            if filters.ema_slope_bars:
                k = filters.ema_slope_bars
                if i - k >= 0:
                    slope = day_ema.iloc[i] - day_ema.iloc[i - k]
                    slope_ok_long = slope > 0
                    slope_ok_short = slope < 0
                else:
                    slope_ok_long = slope_ok_short = False

            # LONG breakout
            if (filters.allow_long and _is_green(bar) and bar["open"] < e and bar["close"] > e
                    and body_ok and slope_ok_long):
                if day_htf is None or day_htf.iloc[i] >= 0:
                    t = _simulate_one(day_df, day_ema, "LONG", i, filters, day_htf)
                    if t is not None:
                        trades.append(t)
                        taken_until_idx = day_df.index.get_loc(t.exit_bar)
                        continue

            # SHORT breakdown
            if (filters.allow_short and _is_red(bar) and bar["open"] > e and bar["close"] < e
                    and body_ok and slope_ok_short):
                if day_htf is None or day_htf.iloc[i] <= 0:
                    t = _simulate_one(day_df, day_ema, "SHORT", i, filters, day_htf)
                    if t is not None:
                        trades.append(t)
                        taken_until_idx = day_df.index.get_loc(t.exit_bar)
    return trades


# ---------------------------------------------------------------------------
# Metrics / reporting
# ---------------------------------------------------------------------------
def summarise(trades: list[Trade], lot_size: int) -> dict:
    if not trades:
        return {"n_trades": 0}

    dfm = pd.DataFrame([dc.asdict(t) for t in trades])
    wins = dfm[dfm["pnl_points"] > 0]
    losses = dfm[dfm["pnl_points"] <= 0]

    gross_points = dfm["pnl_points"].sum()
    gross_win = wins["pnl_points"].sum()
    gross_loss = losses["pnl_points"].sum()

    # Daily aggregation
    dfm["date"] = pd.to_datetime(dfm["entry_bar"]).dt.date
    daily = dfm.groupby("date")["pnl_points"].sum()

    # Drawdown on cumulative points
    curve = dfm["pnl_points"].cumsum()
    peak = curve.cummax()
    dd = (curve - peak).min()

    out = {
        "n_trades": int(len(dfm)),
        "n_long": int((dfm["side"] == "LONG").sum()),
        "n_short": int((dfm["side"] == "SHORT").sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100 * len(wins) / len(dfm), 2),
        "gross_points": round(gross_points, 2),
        "avg_win_pts": round(wins["pnl_points"].mean(), 2) if len(wins) else 0.0,
        "avg_loss_pts": round(losses["pnl_points"].mean(), 2) if len(losses) else 0.0,
        "expectancy_pts": round(gross_points / len(dfm), 2),
        "profit_factor": round(gross_win / abs(gross_loss), 2) if gross_loss < 0 else float("inf"),
        "max_drawdown_pts": round(dd, 2),
        "trading_days": int(daily.shape[0]),
        "best_day_pts": round(daily.max(), 2),
        "worst_day_pts": round(daily.min(), 2),
        "exit_breakdown": dfm["exit_reason"].value_counts().to_dict(),
        # Rupee proxy using index spot points * lot_size (1 lot per trade).
        "gross_rupees_1lot": round(gross_points * lot_size, 2),
        "max_drawdown_rupees_1lot": round(dd * lot_size, 2),
        "lot_size": lot_size,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--refresh", action="store_true",
                   help="Force refetch from Dhan API before backtesting")
    p.add_argument("--lot-size", type=int, default=65,
                   help="NIFTY lot size (SEBI Nov-2025 revision = 65).")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--variant", default="base",
                   help="Tag written into output dir name (base|htf|time|combo|...)")
    p.add_argument("--htf-trend", type=int, default=0,
                   help="HTF EMA alignment (minutes). 0=off, 15 or 30 typical.")
    p.add_argument("--skip-opening-bars", type=int, default=0,
                   help="Skip first N 5m bars of each day (e.g. 3 = skip 09:15..09:30).")
    p.add_argument("--skip-lunch", action="store_true",
                   help="Skip 13:00..14:00 setups (low-volatility hour).")
    p.add_argument("--min-risk-pts", type=float, default=0.0,
                   help="Skip setup if EMA-to-entry distance < X points.")
    p.add_argument("--max-risk-pts", type=float, default=0.0,
                   help="Skip setup if EMA-to-entry distance > X points (0 = no cap).")
    p.add_argument("--breakeven-after-r", type=float, default=0.0,
                   help="Move stop to entry once price reaches entry + N*risk (0=off).")
    p.add_argument("--no-long", action="store_true", help="Disable long setups")
    p.add_argument("--no-short", action="store_true", help="Disable short setups")
    p.add_argument("--interval", type=int, default=5, choices=(5, 15),
                   help="Bar timeframe in minutes (reads NIFTY_5m or NIFTY_15m cache).")
    p.add_argument("--ema-slope-bars", type=int, default=0,
                   help="Require EMA[i]-EMA[i-N]>0 for longs (and <0 for shorts).")
    p.add_argument("--body-pct-min", type=float, default=0.0,
                   help="Reject breakout bars with body/range < this (e.g. 0.5).")
    p.add_argument("--scale-out-r", type=float, default=0.0,
                   help="Book 50% at entry +/- R*risk, trail the rest with EMA.")
    p.add_argument("--trail-only", action="store_true",
                   help="Remove the 1:3 target; exit only on EMA-close/EOD.")
    p.add_argument("--ema-len", type=int, default=20,
                   help="EMA period (default 20 — the video's value).")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = smc_data.load_config()
    if args.refresh:
        df = smc_data.fetch_and_cache(args.symbol, args.days, args.interval, refresh=True, cfg=cfg)
    else:
        df = smc_data.read_cache(cfg, args.symbol, args.interval)
        if df.empty:
            df = smc_data.fetch_and_cache(args.symbol, args.days, args.interval, refresh=False, cfg=cfg)
        else:
            cutoff = dt.datetime.now(smc_data.IST) - dt.timedelta(days=args.days)
            df = df[df.index >= cutoff]

    if df.empty:
        print(f"No 5m bars for {args.symbol}", file=sys.stderr)
        return 1

    print(f"[data] {args.symbol} 5m rows={len(df)} "
          f"from={df.index.min()} to={df.index.max()}")

    filters = Filters(
        htf_trend=args.htf_trend,
        skip_opening_bars=args.skip_opening_bars,
        skip_lunch=args.skip_lunch,
        min_risk_pts=args.min_risk_pts,
        max_risk_pts=args.max_risk_pts,
        breakeven_after_r=args.breakeven_after_r,
        allow_long=not args.no_long,
        allow_short=not args.no_short,
        ema_slope_bars=args.ema_slope_bars,
        body_pct_min=args.body_pct_min,
        scale_out_r=args.scale_out_r,
        trail_only=args.trail_only,
    )
    print(f"[filters] {filters}")

    trades = scan_trades(df, filters, ema_len=args.ema_len)
    metrics = summarise(trades, args.lot_size)
    metrics["filters"] = dc.asdict(filters)
    metrics["variant"] = args.variant

    out_dir = ROOT / cfg["backtest"]["results_dir"] / f"{args.symbol}_ema_retest_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame([dc.asdict(t) for t in trades])
    trades_csv = out_dir / "trades.csv"
    trades_df.to_csv(trades_csv, index=False)
    metrics_json = out_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    print(f"[out] trades -> {trades_csv}")
    print(f"[out] metrics -> {metrics_json}")
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
