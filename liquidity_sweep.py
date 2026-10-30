#!/usr/bin/env python3
"""Recommendations-only intraday liquidity-sweep options scanner.

Pattern (two-candle requirement):

  SWEEP bar (candle N-1):
    bearish sweep: high > pool_high + 0.10*ATR, close < pool_high,
                   upper wick >= 50% of range, volume >= 1.3x 20-bar avg
    bullish sweep: low  < pool_low  - 0.10*ATR, close > pool_low,
                   lower wick >= 50% of range, volume >= 1.3x 20-bar avg

  CONFIRMATION bar (candle N, just closed):
    body       >= 0.5 * ATR          (displacement)
    close in extreme 30% of range    (conviction)
    direction  opposite to the swept side
    close      beyond sweep bar's close (continuation)

Liquidity pools considered:
  * Prior-day high / prior-day low (from the shared prep cache)
  * Swing extremes - pivot highs / pivot lows on 5-min bars (lookback 20, width 2)

Trade plan:
  entry = confirmation bar's close
  stop  = sweep bar's extreme +/- 0.10 * ATR (just beyond the wick, per ICT spec)
  T1    = entry +/- 1.5R, T2 = entry +/- 2.5R (capped at opposite liquidity pool)

Options:
  reuses breakout_scanner's pick_option + strike-walker + liquidity gates;
  participates in cross-strategy fired-set dedupe via strategy_runner.

References:
  - zeiierman.com "How to identify and trade Liquidity Sweeps like a pro"
  - pineconnector.com "SMC/ICT concepts as testable rules"
  - trinitytrading.io "Liquidity Sweeps ICT trading guide"
This strategy never submits live orders.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

import breakout_scanner as core


STRATEGY = "liquidity_sweep"
RUN_START = dt.time(9, 30)        # skip the first 15 minutes (opening-range noise)
RUN_END = dt.time(14, 30)
ENTRY_START = dt.time(9, 45)      # need two completed bars + opening noise to settle
ENTRY_END = dt.time(14, 0)        # Commit F: tightened from 14:30 - sweeps need 90+ min
                                   # to work a 2R move; 5-day backtest showed 10/11 of
                                   # the 14:30-15:10 entries ended "unresolved_at_close"

# Pool identification
PIVOT_LOOKBACK = 20                # bars scanned for swing extremes
PIVOT_WIDTH = 2                    # a pivot is a local extreme over +/- this many bars

# Sweep bar
SWEEP_PENETRATION_ATR = 0.30       # Commit F: raised from 0.10 - 7 of 11 stops in the
                                   # 5-day backtest were shallow pokes (pen < 0.5 ATR)
                                   # during ADX>25 regimes, which are usually trend
                                   # continuation, not stop-hunt reversals
SWEEP_WICK_RATIO = 0.50            # wick in sweep direction >= this fraction of total range
SWEEP_VOLUME_RATIO = 1.3           # sweep bar volume >= this * 20-bar avg

# Confirmation bar
CONFIRMATION_BODY_ATR = 0.50       # body >= this * ATR (displacement)
CONFIRMATION_CLOSE_FRAC = 0.30     # close in top/bottom this fraction of its own range

# Risk
STOP_WICK_BUFFER_ATR = 0.10        # stop placed this * ATR beyond the sweep wick
T1_R = 1.5
T2_R = 2.0                         # Commit F: dropped from 2.5R - backtest showed 0 T2
                                   # hits across 11 resolved trades. 2.0R is realistic
                                   # intraday on Nifty50 names; still a 2:1 R/R vs the stop

# Market regime
MIN_ADX = 20.0
MAX_ADX = 55.0                     # Commit F: NEW - sweeps don't reverse in runaway
                                   # trends. Backtest showed stops cluster at ADX>55
                                   # (strong trend keeps going, no stop-hunt reversal)
MAX_SIGNALS_PER_CYCLE = 2

# Option gates (keep in lock-step with the other strategies)
OPTION_MIN_PREMIUM = 2.0
OPTION_MIN_OI = 1000
OPTION_MIN_DELTA = 0.25
OPTION_MIN_DTE_DAYS = 5
OPTION_REQUIRE_BID = True
MAX_OPTION_SPREAD_PCT = 5.0
OPTION_DELTA_FALLBACK = 0.5

VIX_SPIKE_MULTIPLIER = 1.30

# Index underlyings scanned alongside the stock universe. "underlying_seg" is
# the Dhan segment for the index itself (IDX_I); "option_seg" is the segment
# the options sit on (NSE_FNO for NSE indices). Sensex / BankEx require the
# BSE scrip master and are deferred to a follow-up commit.
INDICES = {
    "NIFTY":     {"sid": 13, "underlying_seg": "IDX_I", "option_seg": "NSE_FNO"},
    "BANKNIFTY": {"sid": 25, "underlying_seg": "IDX_I", "option_seg": "NSE_FNO"},
}

OUT = core.OUT


def _record(checks: list, name: str, current: str, criterion: str, passed: bool) -> bool:
    checks.append({
        "gate": name,
        "current": current,
        "criterion": criterion,
        "passed": bool(passed),
    })
    return bool(passed)


def _find_pivots(bars: pd.DataFrame, width: int) -> tuple[list[float], list[float]]:
    """Return (pivot_highs, pivot_lows) within `bars`.

    A pivot high is a bar whose high strictly exceeds the high of every bar in
    the +/-`width` window around it. Same for pivot lows on the low column.
    Edge bars (within `width` of the start or end) can't be pivots.
    """
    pivot_highs: list[float] = []
    pivot_lows: list[float] = []
    if len(bars) < 2 * width + 1:
        return pivot_highs, pivot_lows
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    n = len(bars)
    for i in range(width, n - width):
        window_high = highs[i - width:i + width + 1]
        window_low = lows[i - width:i + width + 1]
        if highs[i] == window_high.max() and (window_high == highs[i]).sum() == 1:
            pivot_highs.append(float(highs[i]))
        if lows[i] == window_low.min() and (window_low == lows[i]).sum() == 1:
            pivot_lows.append(float(lows[i]))
    return pivot_highs, pivot_lows


def _nearest_pool_above(pools: list[float], price: float) -> float | None:
    above = [level for level in pools if level >= price]
    return min(above) if above else None


def _nearest_pool_below(pools: list[float], price: float) -> float | None:
    below = [level for level in pools if level <= price]
    return max(below) if below else None


def _wick_ratios(bar: pd.Series) -> tuple[float, float]:
    """Return (upper_wick_ratio, lower_wick_ratio) relative to total bar range."""
    high = float(bar["high"])
    low = float(bar["low"])
    bar_range = high - low
    if bar_range <= 0:
        return 0.0, 0.0
    open_ = float(bar["open"])
    close = float(bar["close"])
    body_high = max(open_, close)
    body_low = min(open_, close)
    return (high - body_high) / bar_range, (body_low - low) / bar_range


def _detect_sweep(sweep_bar: pd.Series, prev_bars: pd.DataFrame,
                  pool_highs: list[float], pool_lows: list[float],
                  atr_value: float) -> dict | None:
    """Return sweep dict (side, pool_level, wick_ratio, volume_ratio) or None."""
    if atr_value <= 0 or len(prev_bars) < PIVOT_LOOKBACK // 2:
        return None
    high = float(sweep_bar["high"])
    low = float(sweep_bar["low"])
    close = float(sweep_bar["close"])
    volume = float(sweep_bar["volume"])
    avg_volume = float(prev_bars["volume"].tail(PIVOT_LOOKBACK).mean() or 0)
    volume_ratio = (volume / avg_volume) if avg_volume > 0 else 0
    upper_wick_ratio, lower_wick_ratio = _wick_ratios(sweep_bar)
    pool_high = _nearest_pool_above(pool_highs, close)
    pool_low = _nearest_pool_below(pool_lows, close)

    # Bearish sweep: high pierces the nearest pool_high above close
    if pool_high is not None:
        penetration = high - pool_high
        if (penetration >= SWEEP_PENETRATION_ATR * atr_value
                and close < pool_high
                and upper_wick_ratio >= SWEEP_WICK_RATIO
                and volume_ratio >= SWEEP_VOLUME_RATIO):
            return {
                "side": "PE",   # price swept a resistance -> trade down -> BUY PE
                "pool_level": pool_high,
                "sweep_extreme": high,
                "penetration_atr": penetration / atr_value,
                "wick_ratio": upper_wick_ratio,
                "volume_ratio": volume_ratio,
            }
    # Bullish sweep: low pierces the nearest pool_low below close
    if pool_low is not None:
        penetration = pool_low - low
        if (penetration >= SWEEP_PENETRATION_ATR * atr_value
                and close > pool_low
                and lower_wick_ratio >= SWEEP_WICK_RATIO
                and volume_ratio >= SWEEP_VOLUME_RATIO):
            return {
                "side": "CE",   # price swept a support -> trade up -> BUY CE
                "pool_level": pool_low,
                "sweep_extreme": low,
                "penetration_atr": penetration / atr_value,
                "wick_ratio": lower_wick_ratio,
                "volume_ratio": volume_ratio,
            }
    return None


def _confirms(confirmation_bar: pd.Series, sweep_bar: pd.Series,
              side: str, atr_value: float) -> tuple[bool, dict]:
    """Return (ok, measurements) for the confirmation bar."""
    high = float(confirmation_bar["high"])
    low = float(confirmation_bar["low"])
    open_ = float(confirmation_bar["open"])
    close = float(confirmation_bar["close"])
    sweep_close = float(sweep_bar["close"])
    bar_range = high - low
    body = abs(close - open_)
    body_atr_mult = (body / atr_value) if atr_value > 0 else 0

    measurements = {
        "body_atr_mult": round(body_atr_mult, 3),
        "range": round(bar_range, 4),
        "close_vs_sweep_close": round(close - sweep_close, 4),
    }
    if bar_range <= 0:
        measurements["reason"] = "zero-range confirmation bar"
        return False, measurements

    if body_atr_mult < CONFIRMATION_BODY_ATR:
        measurements["reason"] = (
            f"body {body_atr_mult:.2f}x ATR < {CONFIRMATION_BODY_ATR:.2f}x required"
        )
        return False, measurements

    close_pos_from_low = (close - low) / bar_range  # 0 = closed at low, 1 = closed at high

    if side == "CE":
        # Bullish confirmation: strong close near the high, continuation above sweep close
        if close_pos_from_low < (1 - CONFIRMATION_CLOSE_FRAC):
            measurements["reason"] = (
                f"close {close_pos_from_low:.2f} not in top "
                f"{CONFIRMATION_CLOSE_FRAC:.0%} of bar range"
            )
            return False, measurements
        if close <= sweep_close:
            measurements["reason"] = f"close {close:.2f} not above sweep close {sweep_close:.2f}"
            return False, measurements
    else:
        # Bearish confirmation: strong close near the low, continuation below sweep close
        if close_pos_from_low > CONFIRMATION_CLOSE_FRAC:
            measurements["reason"] = (
                f"close {close_pos_from_low:.2f} not in bottom "
                f"{CONFIRMATION_CLOSE_FRAC:.0%} of bar range"
            )
            return False, measurements
        if close >= sweep_close:
            measurements["reason"] = f"close {close:.2f} not below sweep close {sweep_close:.2f}"
            return False, measurements
    measurements["close_pos_from_low"] = round(close_pos_from_low, 3)
    return True, measurements


def evaluate_setup(symbol: str, candles: pd.DataFrame, now: dt.datetime,
                   prev_high: float | None, prev_low: float | None,
                   nifty_bias: dict, vix: dict | None = None,
                   prior_week_high: float | None = None,
                   prior_week_low: float | None = None,
                   prior_month_high: float | None = None,
                   prior_month_low: float | None = None) -> dict:
    """Evaluate the last-closed 5-min bar as a potential sweep + confirmation entry.

    `candles` must be the 5-min candle series (completed bars only). The
    second-to-last bar is tested as the SWEEP, the last bar as the CONFIRMATION.

    Pools considered (any of these levels count as liquidity to sweep):
      * 5-minute swing pivots (last 20 bars this session)
      * prior session's high and low (classic session levels)
      * prior ISO week's high and low (where swing stops cluster)   [Commit H]
      * prior calendar month's high and low (institutional pivots)  [Commit H]
    """
    result = {
        "strategy": STRATEGY,
        "symbol": symbol,
        "side": None,
        "time": now.strftime("%H:%M:%S"),
        "mechanical_checks": [],
        "fails": [],
    }
    checks = result["mechanical_checks"]
    if not isinstance(candles.index, pd.DatetimeIndex):
        _record(checks, "Candle history", "no timestamped candles",
                "two completed 5-min bars + swing lookback", False)
        result["verdict"] = "SKIP"
        result["fails"].append("no timestamped candle data")
        return result
    today = candles[candles.index.date == now.date()]
    today = today[today.index.time >= RUN_START]
    enough = len(candles) >= PIVOT_LOOKBACK + 2 and len(today) >= 2
    _record(checks, "Candle history",
            f"{len(candles)} total; {len(today)} today after {RUN_START}",
            f"at least {PIVOT_LOOKBACK + 2} total and 2 today", enough)
    if not enough:
        result["verdict"] = "SKIP"
        result["fails"].append("not enough completed candles after opening-range window")
        return result

    atr_value = core.atr(today, min(14, len(today)))
    if atr_value <= 0:
        _record(checks, "ATR", "non-positive", "ATR > 0", False)
        result["verdict"] = "SKIP"
        result["fails"].append("ATR unavailable")
        return result

    sweep_bar = today.iloc[-2]
    confirmation_bar = today.iloc[-1]
    prev_window = today.iloc[:-1].tail(PIVOT_LOOKBACK)
    pivot_highs, pivot_lows = _find_pivots(prev_window, PIVOT_WIDTH)
    pool_highs = list(pivot_highs)
    pool_lows = list(pivot_lows)
    pool_sources: dict[float, str] = {}
    for level, label in (
        (prev_high, "prev_session_high"),
        (prior_week_high, "prior_week_high"),
        (prior_month_high, "prior_month_high"),
    ):
        if level is not None:
            pool_highs.append(float(level))
            pool_sources[float(level)] = label
    for level, label in (
        (prev_low, "prev_session_low"),
        (prior_week_low, "prior_week_low"),
        (prior_month_low, "prior_month_low"),
    ):
        if level is not None:
            pool_lows.append(float(level))
            pool_sources[float(level)] = label

    sweep = _detect_sweep(sweep_bar, prev_window, pool_highs, pool_lows, atr_value)
    sweep_passed = sweep is not None
    if sweep_passed:
        # Tag which pool was swept so the Telegram alert / report shows context
        sweep["pool_source"] = pool_sources.get(float(sweep["pool_level"]), "swing_pivot")
    _record(
        checks,
        "Liquidity sweep bar",
        (f"side={sweep['side']}; pool={sweep['pool_level']:.2f} "
         f"({sweep['pool_source']}); pen={sweep['penetration_atr']:.2f}x ATR; "
         f"wick={sweep['wick_ratio']:.2f}; vol={sweep['volume_ratio']:.2f}x avg")
        if sweep_passed else
        f"no sweep (pools above={len(pool_highs)}, below={len(pool_lows)})",
        (f"wick >= {SWEEP_WICK_RATIO:.0%} of range; "
         f"penetration >= {SWEEP_PENETRATION_ATR}x ATR; "
         f"volume >= {SWEEP_VOLUME_RATIO}x 20-bar avg; close back inside pool"),
        sweep_passed,
    )
    if not sweep_passed:
        result["verdict"] = "SKIP"
        result["fails"].append("no sweep bar")
        return result

    side = sweep["side"]
    sign = 1 if side == "CE" else -1
    ok, confirmation = _confirms(confirmation_bar, sweep_bar, side, atr_value)
    _record(
        checks,
        "Displacement confirmation",
        (f"body {confirmation['body_atr_mult']:.2f}x ATR; "
         f"close_pos={confirmation.get('close_pos_from_low')}; "
         f"vs_sweep_close={confirmation['close_vs_sweep_close']}")
        if ok else
        confirmation.get("reason", "confirmation failed"),
        (f"body >= {CONFIRMATION_BODY_ATR}x ATR; "
         f"close in extreme {CONFIRMATION_CLOSE_FRAC:.0%} of range; "
         "close beyond sweep close in trade direction"),
        ok,
    )
    if not ok:
        result["side"] = side
        result["verdict"] = f"WATCH_{side}"
        result["fails"].append("confirmation failed")
        return result

    # ADX trend-strength band: too-weak = chop (no clear structure to sweep).
    # Too-strong = runaway trend (sweep is usually a continuation flag, not a reversal).
    adx_value = core.adx(today)
    adx_ok = (adx_value is not None
              and MIN_ADX <= adx_value <= MAX_ADX)
    _record(checks, "ADX trend strength",
            "n/a" if adx_value is None else f"ADX14 {adx_value:.1f}",
            f"between {MIN_ADX:.0f} and {MAX_ADX:.0f}", adx_ok)

    # Nifty alignment: sweeps against Nifty bias are lower probability.
    aligned = ((side == "CE" and nifty_bias.get("bias") != "DOWN")
               or (side == "PE" and nifty_bias.get("bias") != "UP"))
    _record(checks, "Nifty alignment", nifty_bias.get("bias", "unknown"),
            "not opposed to the trade direction", aligned)

    # VIX regime
    vix_ltp = (vix or {}).get("ltp")
    vix_baseline = (vix or {}).get("ema20")
    if vix_ltp is None or vix_baseline is None or vix_baseline <= 0:
        vix_ok = True
        vix_current = "VIX regime n/a"
    else:
        multiplier = vix_ltp / vix_baseline
        vix_ok = multiplier <= VIX_SPIKE_MULTIPLIER
        vix_current = (f"VIX {vix_ltp:.2f}/EMA20 {vix_baseline:.2f}="
                       f"{multiplier:.2f}x; cap {VIX_SPIKE_MULTIPLIER:.2f}x")
    _record(checks, "India VIX regime", vix_current,
            f"VIX / 20d-EMA <= {VIX_SPIKE_MULTIPLIER:.2f}", vix_ok)

    # Entry time window
    entry_time_ok = ENTRY_START <= now.time() <= ENTRY_END
    _record(checks, "Entry time", now.strftime("%H:%M"),
            f"{ENTRY_START.strftime('%H:%M')}-{ENTRY_END.strftime('%H:%M')} IST",
            entry_time_ok)

    # Build the plan
    entry = float(confirmation_bar["close"])
    stop_extreme = float(sweep["sweep_extreme"])
    stop = stop_extreme + sign * (STOP_WICK_BUFFER_ATR * atr_value)  # stop is OPPOSITE direction
    # Note: for CE (sign=+1) the sweep_extreme is a LOW; stop is below it (sign is still +1 but
    # stop should be *below* entry). Fix with explicit sign direction:
    if side == "CE":
        stop = stop_extreme - STOP_WICK_BUFFER_ATR * atr_value
    else:
        stop = stop_extreme + STOP_WICK_BUFFER_ATR * atr_value
    risk = abs(entry - stop)
    if risk <= 0:
        _record(checks, "Risk per share", "zero", "> 0", False)
        result["side"] = side
        result["verdict"] = f"WATCH_{side}"
        result["fails"].append("zero risk per share")
        return result

    t1 = entry + sign * T1_R * risk
    t2 = entry + sign * T2_R * risk

    # Cap targets at the opposite-side pool when it sits inside the raw target.
    # Must preserve ordering (entry < t1 <= t2 for CE; entry > t1 >= t2 for PE)
    # or the ledger insert in record_buy_recommendations rejects the record.
    opposite_pool = (_nearest_pool_above(pool_highs, entry) if side == "CE"
                     else _nearest_pool_below(pool_lows, entry))
    if opposite_pool is not None:
        if side == "CE":
            t2 = min(t2, opposite_pool)
            if t1 >= t2:
                # Cap compressed T2 below the un-capped T1 - place T1 at 60% of
                # the entry-to-T2 distance so entry < t1 < t2 is preserved.
                t1 = entry + 0.6 * (t2 - entry)
        else:
            t2 = max(t2, opposite_pool)
            if t1 <= t2:
                t1 = entry + 0.6 * (t2 - entry)
    # Final sanity: if the cap made T1-T2 collapse too tightly, drop the signal
    # rather than ship an un-tradeable R:R.
    min_t_gap = 0.1 * risk
    if abs(t2 - t1) < min_t_gap:
        _record(checks, "Target spacing",
                f"T1 {t1:.2f} and T2 {t2:.2f} too close after opposite-pool cap",
                f">= {min_t_gap:.3f} points apart", False)
        result["side"] = side
        result["verdict"] = f"WATCH_{side}"
        result["fails"].append("targets collapsed by opposite-pool cap")
        return result

    result.update({
        "side": side,
        "ltp": core.r2(entry),
        "last_candle_time": today.index[-1].strftime("%H:%M"),
        "atr_5m": core.r2(atr_value),
        "adx14_5m": core.r2(adx_value),
        "sweep": {
            "pool_level": core.r2(sweep["pool_level"]),
            "pool_source": sweep.get("pool_source", "swing_pivot"),
            "sweep_extreme": core.r2(sweep["sweep_extreme"]),
            "penetration_atr": round(sweep["penetration_atr"], 3),
            "wick_ratio": round(sweep["wick_ratio"], 3),
            "volume_ratio": round(sweep["volume_ratio"], 3),
        },
        "confirmation": confirmation,
        "plan": {
            "entry": core.r2(entry),
            "stop": core.r2(stop),
            "risk_per_share": core.r2(risk),
            "t1": core.r2(t1),
            "t2": core.r2(t2),
            "opposite_pool": core.r2(opposite_pool) if opposite_pool is not None else None,
        },
    })
    result["fails"] = [check["gate"] for check in checks if not check["passed"]]
    result["verdict"] = f"SETUP_{side}" if not result["fails"] else f"WATCH_{side}"
    return result


def attach_option_check(result: dict, master, expiry: str, chain: dict,
                        day: dt.date | None = None) -> dict:
    """Add option contract, OI-runway, spread/liquidity gates, strike selection.
    Mirrors trend_pullback's option gate; applies the strike-walker for budget."""
    side = result["side"]
    ltp = float(result["ltp"])
    option_info = core.pick_option(chain, side, ltp, float(result["plan"]["entry"]))
    checks = result["mechanical_checks"]
    if not option_info:
        _record(checks, "Option chain", "unavailable", "ATM option data available", False)
        result["fails"].append("option chain unavailable")
        result["verdict"] = f"WATCH_{side}"
        return result

    # Try the ATM first, then walk OTM if needed to fit the risk budget.
    chosen_strike = option_info["strike"]
    chosen_opt = option_info["opt"] or {}
    strike_selection = "ATM"
    risk_per_share = float(result["plan"]["risk_per_share"])

    def _loss_pct(strike_val, opt_quote):
        delta_raw_local = ((opt_quote or {}).get("greeks") or {}).get("delta")
        if delta_raw_local is None:
            return None
        lot_local, _ = core.lot_and_id(master, result["symbol"], expiry, strike_val, side)
        if not lot_local or risk_per_share <= 0:
            return None
        loss = abs(float(delta_raw_local)) * risk_per_share * lot_local
        return loss / core.CONFIG["capital"] * 100

    budget_pct = core.CONFIG["max_risk_pct"]
    atm_loss_pct = _loss_pct(chosen_strike, chosen_opt)
    if (core.CONFIG.get("walk_strike_on_over_risk", True)
            and atm_loss_pct is not None and atm_loss_pct > budget_pct):
        for walk_idx, (walk_strike, walk_opt) in enumerate(
            option_info.get("walk_candidates") or []
        ):
            if walk_idx >= core.CONFIG["max_strike_walks"]:
                break
            walk_delta_raw = ((walk_opt or {}).get("greeks") or {}).get("delta")
            walk_delta = abs(float(walk_delta_raw)) if walk_delta_raw is not None else None
            if walk_delta is None or walk_delta < OPTION_MIN_DELTA:
                continue
            walk_loss_pct = _loss_pct(walk_strike, walk_opt)
            if walk_loss_pct is None:
                continue
            if walk_loss_pct <= budget_pct:
                chosen_strike = walk_strike
                chosen_opt = walk_opt
                strike_selection = f"OTM+{walk_idx + 1}"
                break

    premium = float(chosen_opt.get("last_price") or 0)
    bid = float(chosen_opt.get("top_bid_price") or 0)
    ask = float(chosen_opt.get("top_ask_price") or 0)
    oi_value = int(chosen_opt.get("oi") or 0)
    delta_raw = (chosen_opt.get("greeks") or {}).get("delta")
    delta_abs = abs(float(delta_raw)) if delta_raw is not None else None
    spread = ((ask - bid) / premium * 100) if (premium > 0 and ask > 0 and bid > 0) else None

    dte_days = None
    if day is not None and expiry:
        try:
            dte_days = (dt.date.fromisoformat(expiry) - day).days
        except (TypeError, ValueError):
            dte_days = None

    reasons = []
    if premium < OPTION_MIN_PREMIUM:
        reasons.append(f"premium {premium:.2f} < {OPTION_MIN_PREMIUM:.2f}")
    if oi_value < OPTION_MIN_OI:
        reasons.append(f"OI {oi_value} < {OPTION_MIN_OI}")
    if OPTION_REQUIRE_BID and bid <= 0:
        reasons.append("bid unavailable")
    if spread is not None and spread > MAX_OPTION_SPREAD_PCT:
        reasons.append(f"spread {spread:.2f}% > {MAX_OPTION_SPREAD_PCT:.2f}%")
    if delta_abs is None or delta_abs < OPTION_MIN_DELTA:
        reasons.append(
            f"delta {'missing' if delta_abs is None else f'{delta_abs:.2f}'} "
            f"< {OPTION_MIN_DELTA:.2f}"
        )
    if dte_days is None or dte_days < OPTION_MIN_DTE_DAYS:
        reasons.append(
            f"DTE {'missing' if dte_days is None else dte_days} < {OPTION_MIN_DTE_DAYS}d"
        )
    liquid_ok = not reasons
    _record(
        checks, "Option liquidity",
        (f"premium {premium:.2f}; OI {oi_value}; "
         f"spread {f'{spread:.2f}%' if spread is not None else 'n/a'}; "
         f"delta {f'{delta_abs:.2f}' if delta_abs is not None else 'missing'}; "
         f"DTE {f'{dte_days}d' if dte_days is not None else 'missing'}"),
        (f"premium >= {OPTION_MIN_PREMIUM}; OI >= {OPTION_MIN_OI}; "
         f"bid > 0; spread <= {MAX_OPTION_SPREAD_PCT}%; "
         f"|delta| >= {OPTION_MIN_DELTA}; DTE >= {OPTION_MIN_DTE_DAYS}d"),
        liquid_ok,
    )

    lot, option_sid = core.lot_and_id(master, result["symbol"], expiry, chosen_strike, side)
    delta_use = delta_abs if delta_abs is not None else OPTION_DELTA_FALLBACK
    cost_per_lot = premium * lot if lot else None
    estimated_loss_per_lot = delta_use * risk_per_share * lot if lot else None

    result["option"] = {
        "contract": f"{result['symbol']} {expiry} {chosen_strike:g} {side}",
        "security_id": option_sid,
        "expiry": expiry,
        "strike": chosen_strike,
        "type": side,
        "selection": strike_selection,
        "premium": core.r2(premium),
        "bid": core.r2(bid),
        "ask": core.r2(ask),
        "oi": chosen_opt.get("oi"),
        "delta_used": core.r2(delta_use),
        "lot_size": lot,
        "cost_per_lot": core.r2(cost_per_lot),
        "loss_per_lot_at_stop": core.r2(estimated_loss_per_lot),
        "loss_pct_of_capital": core.r2(
            estimated_loss_per_lot / core.CONFIG["capital"] * 100
        ) if estimated_loss_per_lot else None,
    }
    result["plan"]["estimated_loss_per_lot"] = core.r2(estimated_loss_per_lot)

    result["fails"] = [check["gate"] for check in checks if not check["passed"]]
    if not result["fails"] and lot:
        result["verdict"] = f"BUY_{side}"
    else:
        result["verdict"] = f"WATCH_{side}"
        if not lot:
            result["fails"].append("option lot size unavailable")
    return result


def _notify(text: str) -> None:
    core.load_env()
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        core.send_telegram_message(token, chat_id, text)


def notify_signal(result: dict) -> None:
    option, plan, sweep = result["option"], result["plan"], result["sweep"]
    _notify(
        f"[Liquidity Sweep] {result['verdict']} {result['symbol']} underlying {result['ltp']}\n"
        f"Swept {sweep['pool_level']} [{sweep.get('pool_source', 'swing_pivot')}] "
        f"({sweep['penetration_atr']:.2f}x ATR, wick {sweep['wick_ratio']:.0%}) "
        "+ displacement confirmed\n"
        f"{option['contract']} [{option['selection']}] ~Rs{option['premium']} | "
        f"lot {option['lot_size']}\n"
        f"Entry {plan['entry']} | stop {plan['stop']} | T1 {plan['t1']} | T2 {plan['t2']}\n"
        f"Approx. loss/lot Rs{plan['estimated_loss_per_lot']}\n"
        f"Checks passed: {sum(c['passed'] for c in result['mechanical_checks'])}/"
        f"{len(result['mechanical_checks'])}\n"
        "Recommendations only; no order was placed. Validate news and risk before trading."
    )


_vix_baseline_cache: dict = {}


def _fetch_vix(dh, day):
    """Shared VIX helper. Fail silently if Dhan returns nothing."""
    global _vix_baseline_cache
    try:
        snapshot = dh.ohlc({"IDX_I": [core.INDIA_VIX_ID]})
        vix_quote = (snapshot.get("IDX_I") or {}).get(str(core.INDIA_VIX_ID), {})
        ltp = vix_quote.get("last_price")
    except Exception:  # noqa: BLE001
        return {}
    if _vix_baseline_cache.get("day") != day:
        try:
            vix_daily = dh.daily(core.INDIA_VIX_ID, seg="IDX_I", inst="INDEX", days=40)
            if vix_daily is not None and not vix_daily.empty:
                _vix_baseline_cache = {
                    "day": day,
                    "ema20": round(float(core.ema(vix_daily["close"], 20).iloc[-1]), 2),
                }
            else:
                _vix_baseline_cache = {"day": day, "ema20": None}
        except Exception:  # noqa: BLE001
            _vix_baseline_cache = {"day": day, "ema20": None}
    return {
        "ltp": float(ltp) if ltp is not None else None,
        "ema20": _vix_baseline_cache.get("ema20"),
    }


_index_hl_cache: dict[int, tuple[dt.date, dict]] = {}


def _index_hl_pools(dh, sid: int, seg: str, day: dt.date) -> dict:
    """Fetch all liquidity-pool levels for an index: prior-session, prior-week,
    prior-month highs and lows. Cached per-day so one dh.daily per index per day."""
    cached = _index_hl_cache.get(sid)
    if cached and cached[0] == day:
        return cached[1]
    try:
        # 40 trading days = ~2 calendar months, enough for prior-month calc
        daily = dh.daily(sid, seg=seg, inst="INDEX", days=60)
    except Exception:  # noqa: BLE001
        _index_hl_cache[sid] = (day, {})
        return {}
    if daily is None or daily.empty:
        _index_hl_cache[sid] = (day, {})
        return {}
    prior = daily[daily.index.date < day]
    if prior.empty:
        _index_hl_cache[sid] = (day, {})
        return {}
    prev_high = float(prior["high"].iloc[-1])
    prev_low = float(prior["low"].iloc[-1])
    pw_high, pw_low = core.prior_week_high_low(prior, day)
    pm_high, pm_low = core.prior_month_high_low(prior, day)
    pools = {
        "prev_high": prev_high, "prev_low": prev_low,
        "prior_week_high": pw_high, "prior_week_low": pw_low,
        "prior_month_high": pm_high, "prior_month_low": pm_low,
    }
    _index_hl_cache[sid] = (day, pools)
    return pools


def _index_prev_high_low(dh, sid: int, seg: str, day: dt.date) -> tuple[float | None, float | None]:
    """Back-compat shim for tests - returns just (prev_high, prev_low) from the
    full pool dict. New code should use _index_hl_pools."""
    pools = _index_hl_pools(dh, sid, seg, day)
    return pools.get("prev_high"), pools.get("prev_low")


# Legacy name kept for backwards-compatible tests
_index_prev_hl_cache = _index_hl_cache


def _pick_index_expiry(dh, sid: int, seg: str, day: dt.date,
                       min_dte_days: int) -> str | None:
    """Pick the nearest index expiry at least `min_dte_days` out (weekly for Nifty/BN)."""
    try:
        expiries = dh.expiries(sid, seg=seg)
    except Exception:  # noqa: BLE001
        return None
    future = []
    for raw in expiries or []:
        try:
            exp_date = dt.date.fromisoformat(str(raw)[:10])
        except (TypeError, ValueError):
            continue
        if (exp_date - day).days >= min_dte_days:
            future.append(str(raw)[:10])
    return min(future) if future else None


def scan(dh, master, prep: dict, universe: dict[str, int], day: dt.date,
         now: dt.datetime, fired: set[str]) -> dict:
    """One scan cycle. Only fires on the 5-minute boundary; the main runner
    schedules five-minute wake-ups so the second-to-last bar is a candidate
    sweep and the last bar is the confirmation.

    Scans the full equity universe PLUS the configured INDICES (Nifty, BankNifty).
    """
    quotes = dh.ohlc({"NSE_EQ": list(universe.values()), "IDX_I": [core.NIFTY_ID]})
    nifty = core.nifty_bias(dh, quotes, day, now)
    vix = _fetch_vix(dh, day)
    stock_expiry = prep.get("expiry")
    frm = dt.datetime.combine(day - dt.timedelta(days=4), dt.time(9, 15), core.IST)

    results: list[dict] = []
    stocks = prep.get("stocks") or {}
    signal_path = OUT / f"{STRATEGY}_signals_{day}.jsonl"

    def _fire_if_ready(result: dict) -> None:
        """Immediate dedupe + file-write + Telegram when a BUY is confirmed.
        Called inside the per-symbol loop so Telegram arrives sub-second after
        the gates pass, not after the whole 50-symbol scan completes.

        Order matters: file-write first, then fired.add, then notify. If the
        notify POST crashes, we still have the signal row + dedupe (no re-fire
        next cycle). If the file write fails, we don't add to fired (next
        cycle retries).
        """
        if not result.get("verdict", "").startswith("BUY_"):
            return
        key = f"{result['symbol']}_{result['side']}"
        if key in fired:
            result["verdict"] = f"WATCH_{result['side']}"
            result.setdefault("fails", []).append(
                "already recommended today by another scanner (cross-strategy dedupe)"
            )
            return
        try:
            with open(signal_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            print(f"liquidity_sweep signal file write failed: {exc}", file=sys.stderr)
            return
        fired.add(key)
        try:
            notify_signal(result)
        except Exception as exc:  # noqa: BLE001
            print(f"liquidity_sweep Telegram notify failed: {exc}", file=sys.stderr)

    # ---- Equity universe -----------------------------------------------------
    for symbol, sid in universe.items():
        try:
            candles = dh.intraday(sid, frm, now)
            candles = candles[candles.index <= pd.Timestamp(now) - pd.Timedelta(minutes=5)]
            prep_entry = stocks.get(symbol) or {}
            result = evaluate_setup(
                symbol, candles, now,
                prep_entry.get("prev_high"), prep_entry.get("prev_low"),
                nifty, vix,
                prior_week_high=prep_entry.get("prior_week_high"),
                prior_week_low=prep_entry.get("prior_week_low"),
                prior_month_high=prep_entry.get("prior_month_high"),
                prior_month_low=prep_entry.get("prior_month_low"),
            )
            if result.get("verdict") == f"SETUP_{result.get('side')}":
                if not stock_expiry:
                    _record(result["mechanical_checks"], "Option expiry",
                            "unavailable", "valid expiry available", False)
                    result["fails"].append("option expiry unavailable")
                    result["verdict"] = f"WATCH_{result['side']}"
                else:
                    chain = dh.option_chain(sid, stock_expiry)
                    result = attach_option_check(result, master, stock_expiry, chain, day=day)
            _fire_if_ready(result)
            results.append(result)
        except core.DhanAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            results.append({
                "strategy": STRATEGY,
                "symbol": symbol,
                "side": None,
                "verdict": "SKIP",
                "fails": [f"{type(exc).__name__}: {exc}"],
                "mechanical_checks": [],
            })

    # ---- Index universe (Nifty 50, Bank Nifty) ------------------------------
    for index_name, cfg in INDICES.items():
        sid = cfg["sid"]
        seg = cfg["underlying_seg"]
        try:
            pools = _index_hl_pools(dh, sid, seg, day)
            candles = dh.intraday(sid, frm, now, seg=seg, inst="INDEX")
            candles = candles[candles.index <= pd.Timestamp(now) - pd.Timedelta(minutes=5)]
            result = evaluate_setup(
                index_name, candles, now,
                pools.get("prev_high"), pools.get("prev_low"),
                nifty, vix,
                prior_week_high=pools.get("prior_week_high"),
                prior_week_low=pools.get("prior_week_low"),
                prior_month_high=pools.get("prior_month_high"),
                prior_month_low=pools.get("prior_month_low"),
            )
            if result.get("verdict") == f"SETUP_{result.get('side')}":
                idx_expiry = _pick_index_expiry(dh, sid, seg, day, OPTION_MIN_DTE_DAYS)
                if not idx_expiry:
                    _record(result["mechanical_checks"], "Option expiry",
                            "no expiry at DTE threshold", "valid expiry available", False)
                    result["fails"].append("index option expiry unavailable")
                    result["verdict"] = f"WATCH_{result['side']}"
                else:
                    chain = dh.option_chain(sid, idx_expiry, seg=seg)
                    result = attach_option_check(result, master, idx_expiry, chain, day=day)
            result["underlying_segment"] = seg
            # Index symbols aren't in the equity `universe` dict passed to
            # record_buy_recommendations, so expose the sid on the result for
            # the ledger insert to find.
            result["underlying_security_id"] = sid
            _fire_if_ready(result)
            results.append(result)
        except core.DhanAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            results.append({
                "strategy": STRATEGY,
                "symbol": index_name,
                "side": None,
                "verdict": "SKIP",
                "fails": [f"{type(exc).__name__}: {exc}"],
                "mechanical_checks": [],
                "underlying_segment": seg,
            })

    snapshot = {
        "strategy": STRATEGY,
        "scan_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "expiry": stock_expiry,
        "candidates_checked": len(results),
        "nifty": nifty,
        "vix": vix,
        "results": results,
    }

    OUT.mkdir(exist_ok=True)
    (OUT / f"{STRATEGY}_latest.json").write_text(json.dumps(snapshot, indent=2))
    with open(OUT / f"{STRATEGY}_scan_{day}.jsonl", "a", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps({"nifty_bias": nifty.get("bias"), **result}) + "\n")

    # File writes + Telegram happened INSIDE the loop (see _fire_if_ready above).
    # Here we just emit the end-of-cycle summary. MAX_SIGNALS_PER_CYCLE is no
    # longer applied as a cap - gates are strict enough that >2 simultaneous
    # sweeps in one cycle is a strong signal and operator should see all.
    counts = {
        prefix: sum(r["verdict"].startswith(prefix) for r in results)
        for prefix in ("BUY_", "WATCH_", "SKIP")
    }
    print(f"\n[{snapshot['scan_time']}] Liquidity Sweep | Nifty {nifty.get('bias')} | "
          f"stock expiry {stock_expiry or 'unavailable'} | candidates {len(results)}")
    print(f"  BUY: {counts['BUY_']} | WATCH: {counts['WATCH_']} | SKIP: {counts['SKIP']}")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Standalone liquidity-sweep options recommender")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--universe", default="nifty50")
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()
    exclude = {symbol.strip().upper() for symbol in args.exclude.split(",") if symbol.strip()}
    dh = core.Dhan(*core.credentials())
    day = dt.datetime.now(core.IST).date()
    OUT.mkdir(exist_ok=True)
    eq, master = core.load_master(day)
    universe_full = core.build_universe(args.universe, day, eq, master)
    universe = {s: sid for s, sid in universe_full.items() if s not in exclude}
    prep = core.prepare(dh, day, universe)
    fired: set[str] = set()
    if args.once:
        scan(dh, master, prep, universe, day, dt.datetime.now(core.IST), fired)
        return
    print(f"Running liquidity-sweep scanner every 5 minutes "
          f"from {ENTRY_START.strftime('%H:%M')}-{ENTRY_END.strftime('%H:%M')} IST. "
          "Ctrl+C to stop.")
    while True:
        now = dt.datetime.now(core.IST)
        next_run = (
            now.replace(second=0, microsecond=0)
            + dt.timedelta(minutes=5 - now.minute % 5)
            + dt.timedelta(seconds=20)
        )
        time.sleep(max(1, (next_run - dt.datetime.now(core.IST)).total_seconds()))
        current = dt.datetime.now(core.IST)
        if not (ENTRY_START <= current.time() <= ENTRY_END):
            continue
        try:
            scan(dh, master, prep, universe, day, current, fired)
        except core.DhanAuthError as exc:
            sys.exit(f"Dhan rejected the token ({exc}).")
        except KeyboardInterrupt:
            print("\nStopped.")
            return


if __name__ == "__main__":
    main()
