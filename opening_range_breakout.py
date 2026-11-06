#!/usr/bin/env python3
"""Opening Range Breakout - targets the gap-and-grind pattern.

Problem the strategy solves:
  On days where the broad market gaps up (or down) at open and grinds in
  the direction of the gap through the session, our other strategies tend
  to be silent:
    * breakout_scanner needs a 5-min bar to CLOSE beyond an N-day box edge
      with fresh volume - but the move happened pre-market/at-open
    * trend_pullback needs a pullback-and-reversal structure - but a trend
      day without pullbacks produces none
  Nov 2-3 2026 was a textbook example: Nifty +0.75% + 1.18% back-to-back,
  our scanner fired zero breakouts + zero trend-pullbacks across 2 days.

Pattern (ACD / Crabel opening-range method, standard in intraday literature):
  Opening Range (OR)    = the first OR_PERIOD_MINS of the session (09:15-09:45)
                          OR_high = max high of those bars
                          OR_low  = min low  of those bars
  Entry window          = ENTRY_START to ENTRY_END (09:45-11:00)
  Breakout bar          = a completed 5-min bar inside the entry window whose
                          CLOSE is beyond the OR boundary by >= MIN_PENETRATION_ATR
                          and whose volume is >= MIN_VOLUME_RATIO x avg

Gates (same stack as breakout / trend_pullback):
  * Prior-day H/L alignment  (price must have cleared yesterday's extreme too,
                              not just intraday opening range)
  * Nifty intraday-trend     (symmetric: blocks PE on bullish, CE on bearish)
  * ADX >= MIN_ADX           (trending-only)
  * MACD momentum            (sign agrees with trade direction)
  * Market breadth           (internals aligned)
  * India VIX regime         (no entries on vol spikes)
  * All option gates         (DTE, |delta|, premium, OI, bid, spread)
  * Strike walker            (walk OTM when ATM breaches risk budget)
  * Dynamic target cap       (cap T2 at nearest round-number / opposite-OR,
                              preserve ordering)
  * Max 2 fires per day      (one CE + one PE max, so a mid-day failed
                              breakout can't produce a second attempt)

Trade plan:
  entry  = breakout bar's close
  stop   = breakout bar's extreme +/- STOP_WICK_BUFFER_ATR * ATR
  T1     = 1.0R, T2 = 2.0R
  Target cap: at nearest round-number resistance (CE) / support (PE), or
              the opposite OR boundary if closer. Preserves entry < t1 < t2.

Never submits live orders.
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


STRATEGY = "opening_range_breakout"
# OR period: first N minutes of the session
OR_PERIOD_MINS = 30                 # 09:15-09:45
OR_START = dt.time(9, 15)
OR_END = dt.time(9, 45)
# Entry window: after OR period, before mid-morning drift
ENTRY_START = dt.time(9, 45)
ENTRY_END = dt.time(11, 0)

# Breakout bar criteria
MIN_PENETRATION_ATR = 0.20          # close beyond OR boundary by >= this * ATR
MIN_VOLUME_RATIO = 1.5              # breakout bar volume >= this * 20-bar avg
FADE_WICK_RATIO = 0.40              # Commit P: if the breakout bar leaves a wick >= this
                                    # fraction of its range in its OWN breakout direction,
                                    # the move is an exhaustion stop-run. Flip the trade to
                                    # the opposite side and anchor the stop at the rejected
                                    # extreme. Nov 4 HDFCAMC CE fired at 2519.10 within
                                    # 0.22% of the day's high 2524.70 - a classic chase; a
                                    # dominant upper wick on that bar would have flipped
                                    # the setup to PE.

# Risk
STOP_WICK_BUFFER_ATR = 0.10
T1_R = 1.0
T2_R = 2.0

# Regime gates
MIN_ADX = 20.0
MAX_ADX = 55.0
NIFTY_TREND_VETO_PCT = 0.30         # same as breakout + trend_pullback
BREADTH_MIN_PCT_FOR_CE = 55.0
BREADTH_MAX_PCT_FOR_PE = 45.0
VIX_SPIKE_MULTIPLIER = 1.30

# Option gates (match the other strategies after Commits A-G)
OPTION_MIN_PREMIUM = 2.0
OPTION_MIN_OI = 1000
OPTION_MIN_DELTA = 0.25
OPTION_MIN_DTE_DAYS = 5
OPTION_REQUIRE_BID = True
MAX_OPTION_SPREAD_PCT = 5.0
OPTION_DELTA_FALLBACK = 0.5
MIN_T1_DISTANCE_PCT = 0.40

MAX_FIRES_PER_DAY = 2               # at most 1 CE + 1 PE per day

OUT = core.OUT


def _record(checks: list, name: str, current: str, criterion: str, passed: bool) -> bool:
    checks.append({
        "gate": name,
        "current": current,
        "criterion": criterion,
        "passed": bool(passed),
    })
    return bool(passed)


def _opening_range(today_bars: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (OR_high, OR_low) for the first OR_PERIOD_MINS of the session."""
    if today_bars.empty:
        return None, None
    or_mask = today_bars.index.map(lambda ts: OR_START <= ts.time() < OR_END)
    or_bars = today_bars[or_mask]
    if or_bars.empty:
        return None, None
    return float(or_bars["high"].max()), float(or_bars["low"].min())


def evaluate_setup(symbol: str, candles: pd.DataFrame, now: dt.datetime,
                   prev_high: float | None, prev_low: float | None,
                   nifty_bias: dict, vix: dict | None = None,
                   breadth: dict | None = None) -> dict:
    """Evaluate the latest completed 5-min bar as a potential OR breakout."""
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
                "complete OR + entry-window bars", False)
        result["verdict"] = "SKIP"
        result["fails"].append("no timestamped candle data")
        return result

    today = candles[candles.index.date == now.date()]
    if len(today) < 7:   # 6 OR bars + at least one post-OR bar
        _record(checks, "Candle history", f"{len(today)} today",
                "OR complete + >=1 post-OR bar", False)
        result["verdict"] = "SKIP"
        result["fails"].append("OR not yet complete or no post-OR bar")
        return result

    or_high, or_low = _opening_range(today)
    if or_high is None or or_low is None:
        _record(checks, "Opening range", "unavailable",
                "OR high/low computable from 09:15-09:45 bars", False)
        result["verdict"] = "SKIP"
        result["fails"].append("OR not computable")
        return result

    # Latest completed bar
    bar = today.iloc[-1]
    bar_time = today.index[-1].time()

    # Entry-window gate
    entry_time_ok = ENTRY_START <= bar_time <= ENTRY_END
    _record(checks, "Entry window", f"{bar_time.strftime('%H:%M')}",
            f"{ENTRY_START.strftime('%H:%M')}-{ENTRY_END.strftime('%H:%M')}",
            entry_time_ok)
    if not entry_time_ok:
        result["verdict"] = "SKIP"
        result["fails"].append("outside entry window")
        return result

    # Breakout detection against OR
    atr = core.atr(today, min(14, len(today)))
    if atr <= 0:
        _record(checks, "ATR", "non-positive", "ATR > 0", False)
        result["verdict"] = "SKIP"
        result["fails"].append("ATR unavailable")
        return result

    close = float(bar["close"])
    high = float(bar["high"])
    low = float(bar["low"])
    volume = float(bar["volume"])
    prev_window = today.iloc[:-1].tail(20)
    avg_vol = float(prev_window["volume"].mean() or 0)
    vol_ratio = volume / avg_vol if avg_vol > 0 else 0

    # CE breakout: close > OR_high by at least MIN_PENETRATION_ATR * ATR
    pen_up = (close - or_high) / atr if atr > 0 else 0
    pen_dn = (or_low - close) / atr if atr > 0 else 0

    if close > or_high and pen_up >= MIN_PENETRATION_ATR:
        side = "CE"
        boundary = or_high
        penetration = pen_up
        bar_extreme = low  # stop reference is breakout bar's low
    elif close < or_low and pen_dn >= MIN_PENETRATION_ATR:
        side = "PE"
        boundary = or_low
        penetration = pen_dn
        bar_extreme = high
    else:
        _record(checks, "OR breakout",
                f"close {close:.2f} vs OR [{or_low:.2f}, {or_high:.2f}]; "
                f"pen_up {pen_up:.2f}R, pen_dn {pen_dn:.2f}R",
                f"close beyond OR boundary by >= {MIN_PENETRATION_ATR}x ATR",
                False)
        result["verdict"] = "SKIP"
        result["fails"].append("no OR breakout")
        return result

    # Commit P: breakout-bar exhaustion flip. If the bar that broke the OR
    # leaves a dominant wick on its own breakout side, the bar is a stop-run,
    # not continuation. Flip to the opposite side; the downstream direction
    # gates (MACD, breadth, prior-day H/L, Nifty bias) then re-evaluate
    # against the fade direction and will veto if the broader regime still
    # favors the original move (which is the correct conservative behavior).
    bar_open = float(bar["open"])
    bar_range = high - low
    upper_wick_ratio = (high - max(close, bar_open)) / bar_range if bar_range > 0 else 0.0
    lower_wick_ratio = (min(close, bar_open) - low) / bar_range if bar_range > 0 else 0.0
    rejection_wick = upper_wick_ratio if side == "CE" else lower_wick_ratio
    flipped_from_exhaustion = False
    original_side = side
    if rejection_wick >= FADE_WICK_RATIO:
        flipped_from_exhaustion = True
        if side == "CE":
            side = "PE"
            bar_extreme = high
        else:
            side = "CE"
            bar_extreme = low

    sign = 1 if side == "CE" else -1
    flip_note = (f" -> flipped to {side} (exhaustion wick)"
                 if flipped_from_exhaustion else "")
    _record(checks, "OR breakout",
            f"{original_side}: close {close:.2f} beyond {boundary:.2f} by "
            f"{penetration:.2f}x ATR{flip_note}",
            f"close beyond OR boundary by >= {MIN_PENETRATION_ATR}x ATR",
            True)
    _record(checks, "Breakout bar rejection",
            f"{'upper' if original_side == 'CE' else 'lower'} wick "
            f"{rejection_wick:.2f} of range"
            + (" -> side flipped" if flipped_from_exhaustion else " -> no flip"),
            f"< {FADE_WICK_RATIO:.2f} else flip (either way is a valid state)",
            True)

    volume_ok = vol_ratio >= MIN_VOLUME_RATIO
    _record(checks, "Breakout volume",
            f"{vol_ratio:.2f}x 20-bar avg",
            f">= {MIN_VOLUME_RATIO:.2f}x", volume_ok)

    # Prior-day H/L alignment
    prev_level = prev_high if side == "CE" else prev_low
    if prev_level is not None:
        prior_ok = (close >= prev_level) if side == "CE" else (close <= prev_level)
        _record(checks, "Prior-day H/L alignment",
                f"close {close:.2f} vs prev-day {('high' if side == 'CE' else 'low')} {prev_level:.2f}",
                f"close at or beyond prev-day {('high' if side == 'CE' else 'low')}",
                prior_ok)
    else:
        prior_ok = True
        _record(checks, "Prior-day H/L alignment", "no prev-day H/L cached",
                "prev-day level unavailable", True)

    # ADX band
    adx_value = core.adx(today)
    adx_ok = (adx_value is not None and MIN_ADX <= adx_value <= MAX_ADX)
    _record(checks, "ADX trend strength",
            "n/a" if adx_value is None else f"ADX14 {adx_value:.1f}",
            f"between {MIN_ADX:.0f} and {MAX_ADX:.0f}", adx_ok)

    # MACD momentum
    closes = today["close"]
    macd = core.macd_hist(closes)
    import math
    mh_nan = isinstance(macd, float) and math.isnan(macd)
    if mh_nan:
        macd_ok = False
        macd_text = "histogram unavailable (insufficient history)"
    else:
        macd_ok = (macd > 0) if side == "CE" else (macd < 0)
        macd_text = f"histogram {macd:+.3f}"
    _record(checks, "MACD momentum", macd_text,
            "> 0 for CE / < 0 for PE", macd_ok)

    # Nifty intraday-trend (symmetric)
    nifty_chg = (nifty_bias or {}).get("chg_pct")
    if nifty_chg is None:
        nifty_trend_ok = True
        nifty_trend_text = "Nifty chg_pct unavailable"
    elif side == "CE" and nifty_chg < -NIFTY_TREND_VETO_PCT:
        nifty_trend_ok = False
        nifty_trend_text = f"Nifty {nifty_chg:+.2f}% - too bearish for CE"
    elif side == "PE" and nifty_chg > NIFTY_TREND_VETO_PCT:
        nifty_trend_ok = False
        nifty_trend_text = f"Nifty {nifty_chg:+.2f}% - too bullish for PE"
    else:
        nifty_trend_ok = True
        nifty_trend_text = f"Nifty {nifty_chg:+.2f}% aligned with {side}"
    _record(checks, "Nifty intraday trend", nifty_trend_text,
            f"within +/-{NIFTY_TREND_VETO_PCT:.2f}% or aligned", nifty_trend_ok)

    # Market breadth
    breadth_pct = (breadth or {}).get("pct_above_prev_close")
    if breadth_pct is None:
        breadth_ok = True
        breadth_text = "breadth n/a"
    elif side == "CE":
        breadth_ok = breadth_pct >= BREADTH_MIN_PCT_FOR_CE
        breadth_text = f"{breadth_pct:.1f}% above prev-close; need >= {BREADTH_MIN_PCT_FOR_CE:.0f}%"
    else:
        breadth_ok = breadth_pct <= BREADTH_MAX_PCT_FOR_PE
        breadth_text = f"{breadth_pct:.1f}% above prev-close; need <= {BREADTH_MAX_PCT_FOR_PE:.0f}%"
    _record(checks, "Market breadth", breadth_text,
            "breadth aligned with trade direction", breadth_ok)

    # VIX regime
    vix_ltp = (vix or {}).get("ltp")
    vix_ema = (vix or {}).get("ema20")
    if vix_ltp is None or vix_ema is None or vix_ema <= 0:
        vix_ok = True
        vix_text = "VIX regime n/a"
    else:
        multiplier = vix_ltp / vix_ema
        vix_ok = multiplier <= VIX_SPIKE_MULTIPLIER
        vix_text = f"VIX {vix_ltp:.2f}/EMA20 {vix_ema:.2f} = {multiplier:.2f}x"
    _record(checks, "India VIX regime", vix_text,
            f"VIX / 20d-EMA <= {VIX_SPIKE_MULTIPLIER:.2f}", vix_ok)

    # Plan
    stop = bar_extreme - sign * STOP_WICK_BUFFER_ATR * atr
    risk = sign * (close - stop)
    if risk <= 0:
        _record(checks, "Risk per share", "zero or negative", "> 0", False)
        result["verdict"] = f"WATCH_{side}"
        result["fails"].append("non-positive risk")
        return result

    t1 = close + sign * T1_R * risk
    t2 = close + sign * T2_R * risk

    # T1-distance gate (per cent of entry)
    t1_distance_pct = abs(t1 - close) / close * 100 if close else 0
    t1_distance_ok = t1_distance_pct >= MIN_T1_DISTANCE_PCT
    _record(checks, "T1 distance",
            f"{t1_distance_pct:.2f}% of underlying",
            f">= {MIN_T1_DISTANCE_PCT:.2f}%", t1_distance_ok)

    # Dynamic target cap at opposite OR boundary or nearest round number
    opposite_or = or_low if side == "CE" else or_high
    rn_above, rn_below = _nearest_round_numbers(close)
    cap_candidate = None
    if side == "CE":
        for level in (rn_above,):   # only consider resistances above entry
            if close < level < t2:
                cap_candidate = level if cap_candidate is None else min(cap_candidate, level)
        if cap_candidate is not None:
            t2 = cap_candidate
            if t1 >= t2:
                t1 = close + 0.5 * (t2 - close)
    else:
        for level in (rn_below,):
            if t2 < level < close:
                cap_candidate = level if cap_candidate is None else max(cap_candidate, level)
        if cap_candidate is not None:
            t2 = cap_candidate
            if t1 <= t2:
                t1 = close + 0.5 * (t2 - close)
    target_capped = cap_candidate is not None

    result.update({
        "side": side,
        "ltp": core.r2(close),
        "last_candle_time": today.index[-1].strftime("%H:%M"),
        "atr_5m": core.r2(atr),
        "adx14_5m": core.r2(adx_value),
        "or_high": core.r2(or_high),
        "or_low": core.r2(or_low),
        "breakout_bar_volume_ratio": round(vol_ratio, 2),
        "penetration_atr": round(penetration, 3),
        "breakout_bar_upper_wick_ratio": round(upper_wick_ratio, 3),
        "breakout_bar_lower_wick_ratio": round(lower_wick_ratio, 3),
        "flipped_from_exhaustion": flipped_from_exhaustion,
        "plan": {
            "entry": core.r2(close),
            "stop": core.r2(stop),
            "risk_per_share": core.r2(risk),
            "t1": core.r2(t1),
            "t2": core.r2(t2),
            "target_capped_at_round_number": target_capped,
            "opposite_or": core.r2(opposite_or),
        },
    })
    result["fails"] = [c["gate"] for c in checks if not c["passed"]]
    result["verdict"] = f"SETUP_{side}" if not result["fails"] else f"WATCH_{side}"
    return result


def _nearest_round_numbers(price: float) -> tuple[float, float]:
    """Nearest round numbers (above, below). Step: 10 <500, 50 500-5000, 100 >5000."""
    if price < 500:
        step = 10
    elif price < 5000:
        step = 50
    else:
        step = 100
    below = (int(price) // step) * step
    above = below + step
    return float(above), float(below)


def attach_option_check(result: dict, master, expiry: str, chain: dict,
                        day: dt.date | None = None) -> dict:
    """Reuses breakout_scanner.pick_option + strike walker + option gates."""
    side = result["side"]
    ltp = float(result["ltp"])
    opt_info = core.pick_option(chain, side, ltp, float(result["plan"]["entry"]))
    checks = result["mechanical_checks"]
    if not opt_info:
        _record(checks, "Option chain", "unavailable", "chain data available", False)
        result["fails"].append("option chain unavailable")
        result["verdict"] = f"WATCH_{side}"
        return result

    chosen_strike = opt_info["strike"]
    chosen_opt = opt_info["opt"] or {}
    strike_selection = "ATM"
    risk_per_share = float(result["plan"]["risk_per_share"])

    def _loss_pct(strike_val, opt_quote):
        delta_raw = ((opt_quote or {}).get("greeks") or {}).get("delta")
        if delta_raw is None:
            return None
        lot_local, _ = core.lot_and_id(master, result["symbol"], expiry, strike_val, side)
        if not lot_local or risk_per_share <= 0:
            return None
        loss = abs(float(delta_raw)) * risk_per_share * lot_local
        return loss / core.CONFIG["capital"] * 100

    budget_pct = core.CONFIG["max_risk_pct"]
    atm_loss = _loss_pct(chosen_strike, chosen_opt)
    if (core.CONFIG.get("walk_strike_on_over_risk", True)
            and atm_loss is not None and atm_loss > budget_pct):
        for walk_idx, (ws, wo) in enumerate(opt_info.get("walk_candidates") or []):
            if walk_idx >= core.CONFIG["max_strike_walks"]:
                break
            wd_raw = ((wo or {}).get("greeks") or {}).get("delta")
            wd = abs(float(wd_raw)) if wd_raw is not None else None
            if wd is None or wd < OPTION_MIN_DELTA:
                continue
            wloss = _loss_pct(ws, wo)
            if wloss is None:
                continue
            if wloss <= budget_pct:
                chosen_strike = ws
                chosen_opt = wo
                strike_selection = f"OTM+{walk_idx + 1}"
                break

    premium = float(chosen_opt.get("last_price") or 0)
    bid = float(chosen_opt.get("top_bid_price") or 0)
    ask = float(chosen_opt.get("top_ask_price") or 0)
    oi = int(chosen_opt.get("oi") or 0)
    delta_raw = (chosen_opt.get("greeks") or {}).get("delta")
    delta_abs = abs(float(delta_raw)) if delta_raw is not None else None
    spread = ((ask - bid) / premium * 100) if (premium > 0 and ask > 0 and bid > 0) else None
    dte = None
    if day is not None and expiry:
        try:
            dte = (dt.date.fromisoformat(expiry) - day).days
        except (TypeError, ValueError):
            dte = None

    reasons = []
    if premium < OPTION_MIN_PREMIUM:
        reasons.append(f"premium {premium:.2f} < {OPTION_MIN_PREMIUM:.2f}")
    if oi < OPTION_MIN_OI:
        reasons.append(f"OI {oi} < {OPTION_MIN_OI}")
    if OPTION_REQUIRE_BID and bid <= 0:
        reasons.append("bid unavailable")
    if spread is not None and spread > MAX_OPTION_SPREAD_PCT:
        reasons.append(f"spread {spread:.2f}% > {MAX_OPTION_SPREAD_PCT:.2f}%")
    if delta_abs is None or delta_abs < OPTION_MIN_DELTA:
        reasons.append(f"delta {delta_abs} < {OPTION_MIN_DELTA}")
    if dte is None or dte < OPTION_MIN_DTE_DAYS:
        reasons.append(f"DTE {dte} < {OPTION_MIN_DTE_DAYS}")
    liquid_ok = not reasons
    _record(checks, "Option liquidity",
            f"premium {premium:.2f}; OI {oi}; spread {spread}; delta {delta_abs}; DTE {dte}",
            f"premium>={OPTION_MIN_PREMIUM}; OI>={OPTION_MIN_OI}; bid>0; "
            f"spread<={MAX_OPTION_SPREAD_PCT}%; |delta|>={OPTION_MIN_DELTA}; "
            f"DTE>={OPTION_MIN_DTE_DAYS}", liquid_ok)

    lot, option_sid = core.lot_and_id(master, result["symbol"], expiry, chosen_strike, side)
    delta_use = delta_abs if delta_abs is not None else OPTION_DELTA_FALLBACK
    cost_per_lot = premium * lot if lot else None
    loss_per_lot = delta_use * risk_per_share * lot if lot else None

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
        "loss_per_lot_at_stop": core.r2(loss_per_lot),
        "loss_pct_of_capital": core.r2(
            loss_per_lot / core.CONFIG["capital"] * 100) if loss_per_lot else None,
    }
    result["plan"]["estimated_loss_per_lot"] = core.r2(loss_per_lot)

    result["fails"] = [c["gate"] for c in checks if not c["passed"]]
    if not result["fails"] and lot:
        result["verdict"] = f"BUY_{side}"
    else:
        result["verdict"] = f"WATCH_{side}"
        if not lot:
            result["fails"].append("option lot size unavailable")
    return result


def _notify(text: str) -> None:
    core.load_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        core.send_telegram_message(token, chat_id, text)


def notify_signal(result: dict) -> None:
    opt, plan = result["option"], result["plan"]
    flip_tag = " [EXHAUSTION FLIP]" if result.get("flipped_from_exhaustion") else ""
    _notify(
        f"[OR Breakout]{flip_tag} {result['verdict']} {result['symbol']} "
        f"underlying {result['ltp']}\n"
        f"OR [{result['or_low']}, {result['or_high']}] breakout "
        f"({result['penetration_atr']:.2f}x ATR, vol {result['breakout_bar_volume_ratio']:.2f}x)\n"
        f"{opt['contract']} [{opt['selection']}] ~Rs{opt['premium']} | lot {opt['lot_size']}\n"
        f"Entry {plan['entry']} | stop {plan['stop']} | T1 {plan['t1']} | T2 {plan['t2']}"
        f"{' (round-number capped)' if plan.get('target_capped_at_round_number') else ''}\n"
        f"Approx. loss/lot Rs{plan['estimated_loss_per_lot']}\n"
        f"Checks passed: {sum(c['passed'] for c in result['mechanical_checks'])}/"
        f"{len(result['mechanical_checks'])}\n"
        "Recommendations only; no order was placed."
    )


def scan(dh, master, prep: dict, universe: dict[str, int], day: dt.date,
         now: dt.datetime, fired: set[str]) -> dict:
    """One scan cycle. Only runs during the entry window (09:45-11:00)."""
    current = now.astimezone(core.IST)
    if not (ENTRY_START <= current.time() <= ENTRY_END):
        return {"strategy": STRATEGY, "scan_time": current.strftime("%Y-%m-%d %H:%M:%S"),
                "candidates_checked": 0, "results": [], "note": "outside entry window"}

    # Per-day fire limit - count by reading the signal file directly so a
    # server restart mid-session doesn't reset the counter. (Previous
    # implementation stashed prefix keys in the shared `fired` set, but those
    # don't persist across process restarts - a crash + restart could let ORB
    # over-fire past MAX_FIRES_PER_DAY.)
    signal_path = OUT / f"{STRATEGY}_signals_{day}.jsonl"
    fired_today = 0
    if signal_path.exists():
        try:
            fired_today = sum(
                1 for line in signal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            fired_today = 0
    if fired_today >= MAX_FIRES_PER_DAY:
        return {"strategy": STRATEGY, "scan_time": current.strftime("%Y-%m-%d %H:%M:%S"),
                "candidates_checked": 0, "results": [],
                "note": f"MAX_FIRES_PER_DAY={MAX_FIRES_PER_DAY} reached"}

    quotes = dh.ohlc({"NSE_EQ": list(universe.values()), "IDX_I": [core.NIFTY_ID]})
    nifty = core.nifty_bias(dh, quotes, day, now)
    vix = _fetch_vix(dh, day)
    breadth = _compute_breadth(prep, quotes, list(universe.keys()))
    stock_expiry = prep.get("expiry")
    frm = dt.datetime.combine(day - dt.timedelta(days=4), dt.time(9, 15), core.IST)

    results: list[dict] = []
    stocks = prep.get("stocks") or {}

    def _fire_if_ready(result: dict) -> None:
        nonlocal fired_today
        if not result.get("verdict", "").startswith("BUY_"):
            return
        key = f"{result['symbol']}_{result['side']}"
        if key in fired:
            result["verdict"] = f"WATCH_{result['side']}"
            result.setdefault("fails", []).append(
                "already recommended today by another scanner (cross-strategy dedupe)"
            )
            return
        if fired_today >= MAX_FIRES_PER_DAY:
            result["verdict"] = f"WATCH_{result['side']}"
            result.setdefault("fails", []).append(
                f"MAX_FIRES_PER_DAY={MAX_FIRES_PER_DAY} reached"
            )
            return
        try:
            with open(signal_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            print(f"ORB signal file write failed: {exc}", file=sys.stderr)
            return
        fired.add(key)
        fired_today += 1
        try:
            notify_signal(result)
        except Exception as exc:   # noqa: BLE001
            print(f"ORB Telegram notify failed: {exc}", file=sys.stderr)

    # ---- Equity universe -----------------------------------------------------
    for symbol, sid in universe.items():
        try:
            candles = dh.intraday(sid, frm, now)
            candles = candles[candles.index <= pd.Timestamp(now) - pd.Timedelta(minutes=5)]
            prep_entry = stocks.get(symbol) or {}
            result = evaluate_setup(
                symbol, candles, now,
                prep_entry.get("prev_high"), prep_entry.get("prev_low"),
                nifty, vix, breadth,
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
        except Exception as exc:   # noqa: BLE001
            results.append({
                "strategy": STRATEGY, "symbol": symbol, "side": None,
                "verdict": "SKIP", "fails": [f"{type(exc).__name__}: {exc}"],
                "mechanical_checks": [],
            })

    # ---- Index universe (Nifty 50, Bank Nifty) ------------------------------
    # Commit N: ORB on indices. Index gap days are the cleanest expression of
    # the pattern - deepest option liquidity, direct macro bet. Reuses
    # liquidity_sweep's INDICES config and helpers so index pool/expiry
    # resolution stays consistent across both strategies.
    import liquidity_sweep as _ls
    for index_name, cfg in _ls.INDICES.items():
        sid = cfg["sid"]
        seg = cfg["underlying_seg"]
        try:
            pools = _ls._index_hl_pools(dh, sid, seg, day)
            candles = dh.intraday(sid, frm, now, seg=seg, inst="INDEX")
            candles = candles[candles.index <= pd.Timestamp(now) - pd.Timedelta(minutes=5)]
            result = evaluate_setup(
                index_name, candles, now,
                pools.get("prev_high"), pools.get("prev_low"),
                nifty, vix, breadth,
            )
            if result.get("verdict") == f"SETUP_{result.get('side')}":
                idx_expiry = _ls._pick_index_expiry(dh, sid, seg, day, OPTION_MIN_DTE_DAYS)
                if not idx_expiry:
                    _record(result["mechanical_checks"], "Option expiry",
                            "no expiry at DTE threshold", "valid expiry available", False)
                    result["fails"].append("index option expiry unavailable")
                    result["verdict"] = f"WATCH_{result['side']}"
                else:
                    chain = dh.option_chain(sid, idx_expiry, seg=seg)
                    result = attach_option_check(result, master, idx_expiry, chain, day=day)
            result["underlying_segment"] = seg
            # Index symbols aren't in the equity universe - stamp the sid so
            # record_buy_recommendations can resolve it (same pattern as
            # Commit I fix for liquidity_sweep indices).
            result["underlying_security_id"] = sid
            _fire_if_ready(result)
            results.append(result)
        except core.DhanAuthError:
            raise
        except Exception as exc:   # noqa: BLE001
            results.append({
                "strategy": STRATEGY, "symbol": index_name, "side": None,
                "verdict": "SKIP", "fails": [f"{type(exc).__name__}: {exc}"],
                "mechanical_checks": [],
                "underlying_segment": seg,
            })

    snapshot = {
        "strategy": STRATEGY,
        "scan_time": current.strftime("%Y-%m-%d %H:%M:%S"),
        "expiry": stock_expiry,
        "candidates_checked": len(results),
        "nifty": nifty,
        "vix": vix,
        "results": results,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{STRATEGY}_latest.json").write_text(json.dumps(snapshot, indent=2))
    with open(OUT / f"{STRATEGY}_scan_{day}.jsonl", "a", encoding="utf-8") as stream:
        for r in results:
            stream.write(json.dumps({"nifty_bias": nifty.get("bias"), **r}) + "\n")

    from collections import Counter
    counts = Counter(r.get("verdict", "?") for r in results)
    print(f"\n[{snapshot['scan_time']}] ORB | Nifty {nifty.get('bias')} | "
          f"candidates {len(results)}")
    print(f"  BUY: {sum(v for k, v in counts.items() if k.startswith('BUY_'))} | "
          f"WATCH: {sum(v for k, v in counts.items() if k.startswith('WATCH_'))} | "
          f"SKIP: {counts.get('SKIP', 0)}")
    return snapshot


_vix_baseline_cache: dict = {}


def _fetch_vix(dh, day):
    """Best-effort VIX fetch (shared pattern across strategies)."""
    global _vix_baseline_cache
    try:
        snapshot = dh.ohlc({"IDX_I": [core.INDIA_VIX_ID]})
        vix_quote = (snapshot.get("IDX_I") or {}).get(str(core.INDIA_VIX_ID), {})
        ltp = vix_quote.get("last_price")
    except Exception:   # noqa: BLE001
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
        except Exception:   # noqa: BLE001
            _vix_baseline_cache = {"day": day, "ema20": None}
    return {
        "ltp": float(ltp) if ltp is not None else None,
        "ema20": _vix_baseline_cache.get("ema20"),
    }


def _compute_breadth(prep: dict, quotes: dict, symbols: list[str]) -> dict:
    """% of universe trading above its prev-close (shared pattern)."""
    eq_quotes = (quotes or {}).get("NSE_EQ") or {}
    total = above = 0
    stocks = prep.get("stocks") or {}
    for sym in symbols:
        stock = stocks.get(sym) or {}
        sid = stock.get("sid")
        quote = eq_quotes.get(str(sid)) if sid else None
        if not quote:
            continue
        ltp = quote.get("last_price")
        prev_close = stock.get("prev_close")
        if ltp is None or prev_close is None or prev_close <= 0:
            continue
        total += 1
        if float(ltp) > float(prev_close):
            above += 1
    if total == 0:
        return {"pct_above_prev_close": None, "sample": 0}
    return {"pct_above_prev_close": round(100 * above / total, 2), "sample": total}


def main():
    parser = argparse.ArgumentParser(description="Standalone ORB options recommender")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--universe", default="nifty50")
    parser.add_argument("--exclude", default="")
    args = parser.parse_args()
    exclude = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}
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
    print(f"Running ORB scanner from {ENTRY_START.strftime('%H:%M')}-"
          f"{ENTRY_END.strftime('%H:%M')} IST. Ctrl+C to stop.")
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
