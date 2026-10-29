#!/usr/bin/env python3
"""Backtest the liquidity_sweep pattern detection against the last N trading days.

Replays the exact `_find_pivots`, `_detect_sweep`, `_confirms` and `adx` logic
from liquidity_sweep.py on completed 5-minute candles pulled from Dhan. For
each detected sweep + confirmation, simulates the hypothetical underlying-side
outcome: did the stop or T1 (1.5R) or T2 (2.5R) hit first in the remaining
bars of the session?

Does NOT test the option-chain gates (DTE / delta / premium / OI / spread)
because Dhan does not serve historical option chains. Those gates mostly
affect whether the signal becomes tradeable, not whether the sweep pattern
itself existed. On Nifty50 names and index options, the gates almost always
pass.

Usage:
  .venv/bin/python sweep_backtest.py [--days 5] [--universe nifty50]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from collections import Counter

import pandas as pd

import breakout_scanner as core
import liquidity_sweep as ls


def trading_dates(n: int) -> list[dt.date]:
    """Last n weekdays ending yesterday (today's partial data is excluded)."""
    today = dt.datetime.now(core.IST).date()
    out: list[dt.date] = []
    d = today - dt.timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d -= dt.timedelta(days=1)
    return list(reversed(out))


def fetch_candles(dh, sid: int, seg: str, inst: str, day: dt.date) -> pd.DataFrame:
    frm = dt.datetime.combine(day, dt.time(9, 15), core.IST)
    to = dt.datetime.combine(day, dt.time(15, 30), core.IST)
    try:
        return dh.intraday(sid, frm, to, seg=seg, inst=inst)
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] intraday fetch failed for {sid} on {day}: "
              f"{type(exc).__name__}", file=sys.stderr)
        return pd.DataFrame()


_daily_cache: dict[tuple[int, str], pd.DataFrame] = {}


def prev_day_high_low(dh, sid: int, seg: str, inst: str,
                      day: dt.date) -> tuple[float | None, float | None]:
    """Pull daily history once per symbol; derive prev-session H/L per target day."""
    cache_key = (sid, seg)
    if cache_key not in _daily_cache:
        try:
            if seg == "IDX_I":
                df = dh.daily(sid, seg=seg, inst=inst, days=30)
            else:
                df = dh.daily(sid, days=30)
        except Exception:  # noqa: BLE001
            _daily_cache[cache_key] = pd.DataFrame()
            return None, None
        _daily_cache[cache_key] = df if df is not None else pd.DataFrame()
    df = _daily_cache[cache_key]
    if df.empty:
        return None, None
    prior = df[df.index.date < day]
    if prior.empty:
        return None, None
    return float(prior["high"].iloc[-1]), float(prior["low"].iloc[-1])


def hypothetical_outcome(bars_after: pd.DataFrame, side: str, entry: float,
                         stop: float, t1: float, t2: float) -> str:
    """Scan bars after the confirmation close for the first-hit outcome.
    When a bar straddles both stop and target, stop counts first (conservative)."""
    sign = 1 if side == "CE" else -1
    for _, bar in bars_after.iterrows():
        high, low = float(bar["high"]), float(bar["low"])
        if sign > 0:
            stop_hit = low <= stop
            t1_hit = high >= t1
            t2_hit = high >= t2
        else:
            stop_hit = high >= stop
            t1_hit = low <= t1
            t2_hit = low <= t2
        if stop_hit:
            return "stop_before_t1"
        if t2_hit:
            return "t2_hit"
        if t1_hit:
            return "t1_hit_still_open"
    return "unresolved_at_close"


def backtest_symbol(dh, symbol: str, sid: int, seg: str, inst: str,
                    day: dt.date) -> list[dict]:
    candles = fetch_candles(dh, sid, seg, inst, day)
    if candles.empty or not isinstance(candles.index, pd.DatetimeIndex):
        return []
    today = candles[candles.index.date == day]
    today = today[today.index.time >= ls.RUN_START]
    if len(today) < ls.PIVOT_LOOKBACK + 3:
        return []
    prev_high, prev_low = prev_day_high_low(dh, sid, seg, inst, day)

    hits: list[dict] = []
    for i in range(ls.PIVOT_LOOKBACK + 2, len(today)):
        sweep_bar = today.iloc[i - 1]
        confirm_bar = today.iloc[i]
        prev_window = today.iloc[: i - 1].tail(ls.PIVOT_LOOKBACK)
        atr_value = core.atr(today.iloc[: i], min(14, i))
        if atr_value <= 0:
            continue
        pivot_highs, pivot_lows = ls._find_pivots(prev_window, ls.PIVOT_WIDTH)
        pools_high = pivot_highs + ([prev_high] if prev_high else [])
        pools_low = pivot_lows + ([prev_low] if prev_low else [])
        sweep = ls._detect_sweep(sweep_bar, prev_window, pools_high, pools_low, atr_value)
        if not sweep:
            continue
        side = sweep["side"]
        ok, confirm_info = ls._confirms(confirm_bar, sweep_bar, side, atr_value)
        if not ok:
            continue
        adx_value = core.adx(today.iloc[: i + 1])
        adx_ok = (adx_value is not None
                  and ls.MIN_ADX <= adx_value <= ls.MAX_ADX)
        if not adx_ok:
            continue

        entry = float(confirm_bar["close"])
        sweep_extreme = float(sweep["sweep_extreme"])
        if side == "CE":
            stop = sweep_extreme - ls.STOP_WICK_BUFFER_ATR * atr_value
            risk = entry - stop
        else:
            stop = sweep_extreme + ls.STOP_WICK_BUFFER_ATR * atr_value
            risk = stop - entry
        if risk <= 0:
            continue
        sign = 1 if side == "CE" else -1
        t1 = entry + sign * ls.T1_R * risk
        t2 = entry + sign * ls.T2_R * risk
        outcome = hypothetical_outcome(today.iloc[i + 1:], side, entry, stop, t1, t2)
        hits.append({
            "date": day.isoformat(),
            "symbol": symbol,
            "side": side,
            "sweep_time": sweep_bar.name.strftime("%H:%M"),
            "confirm_time": confirm_bar.name.strftime("%H:%M"),
            "pool_level": round(sweep["pool_level"], 2),
            "sweep_extreme": round(sweep["sweep_extreme"], 2),
            "penetration_atr": round(sweep["penetration_atr"], 3),
            "wick_ratio": round(sweep["wick_ratio"], 3),
            "volume_ratio": round(sweep["volume_ratio"], 2),
            "confirmation_body_atr": confirm_info.get("body_atr_mult"),
            "adx": round(adx_value, 1),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "t1": round(t1, 2),
            "t2": round(t2, 2),
            "outcome": outcome,
        })
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=5,
                        help="trading days to replay, ending yesterday (default 5)")
    parser.add_argument("--universe", default="nifty50",
                        help="universe spec (nifty50, nifty100, fno, etc.)")
    parser.add_argument("--out", default="scanner_out/sweep_backtest.json",
                        help="where to save per-hit detail (default scanner_out/sweep_backtest.json)")
    args = parser.parse_args()

    days = trading_dates(args.days)
    print(f"Backtesting {args.days} days: {days[0]} -> {days[-1]}")

    dh = core.Dhan(*core.credentials())
    today = dt.datetime.now(core.IST).date()
    eq, master = core.load_master(today)
    universe = core.build_universe(args.universe, today, eq, master)
    print(f"Universe: {len(universe)} stocks + {len(ls.INDICES)} indices")

    all_hits: list[dict] = []
    for day in days:
        print(f"\n=== {day} ({day.strftime('%a')}) ===")
        day_hits: list[dict] = []
        for symbol, sid in universe.items():
            hits = backtest_symbol(dh, symbol, sid, "NSE_EQ", "EQUITY", day)
            day_hits.extend(hits)
        for index_name, cfg in ls.INDICES.items():
            hits = backtest_symbol(dh, index_name, cfg["sid"],
                                   cfg["underlying_seg"], "INDEX", day)
            day_hits.extend(hits)
        if not day_hits:
            print("  no sweep + confirmation detected")
        for hit in sorted(day_hits, key=lambda h: h["confirm_time"]):
            emoji = {
                "t2_hit": "++",
                "t1_hit_still_open": " +",
                "unresolved_at_close": " .",
                "stop_before_t1": " -",
            }.get(hit["outcome"], " ?")
            print(f"  {emoji}  {hit['confirm_time']} {hit['symbol']:>12s} {hit['side']} "
                  f"pool={hit['pool_level']:>9.2f} pen={hit['penetration_atr']:.2f}R "
                  f"wick={hit['wick_ratio']:.0%} adx={hit['adx']} "
                  f"entry={hit['entry']:.2f} stop={hit['stop']:.2f} "
                  f"T2={hit['t2']:.2f} -> {hit['outcome']}")
        all_hits.extend(day_hits)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(all_hits)} liquidity sweeps across {len(days)} day(s)")
    print(f"{'='*80}")
    sides = Counter(h["side"] for h in all_hits)
    print(f"By side:  CE {sides.get('CE', 0)}   PE {sides.get('PE', 0)}")
    outcomes = Counter(h["outcome"] for h in all_hits)
    for name in ("t2_hit", "t1_hit_still_open", "unresolved_at_close", "stop_before_t1"):
        count = outcomes.get(name, 0)
        pct = (100 * count / len(all_hits)) if all_hits else 0
        print(f"  {name:28s} {count:>3d}  ({pct:5.1f}%)")
    resolved = outcomes.get("t2_hit", 0) + outcomes.get("stop_before_t1", 0)
    wins = outcomes.get("t2_hit", 0)
    print(f"\nResolved win rate (T2 vs stop): "
          f"{wins}/{resolved} = {100 * wins / resolved:.1f}%"
          if resolved else "\nResolved win rate: n/a (no fully resolved trades)")
    partial_wins = outcomes.get("t1_hit_still_open", 0)
    print(f"Partial wins (hit T1 but not T2 before close): {partial_wins}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_hits, indent=2))
    print(f"\nDetailed hits saved to {out_path}")


if __name__ == "__main__":
    main()
