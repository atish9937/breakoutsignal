#!/usr/bin/env python3
"""Standalone, recommendations-only intraday trend-pullback options scanner.

Run ``python trend_pullback.py --once`` for one scan or ``python trend_pullback.py``
for an independent five-minute loop. Use ``python trend_pullback_backtest.py --start
YYYY-MM-DD --end YYYY-MM-DD --symbols RELIANCE,HDFCBANK`` to evaluate historical
signals. BUY results require trend, pullback, confirmation, VWAP, RSI, MACD, volume,
Nifty, timing, option-liquidity, and OI-runway checks.
This strategy never submits live orders.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

import breakout_scanner as core


STRATEGY = "trend_pullback"
RUN_START = dt.time(9, 20)
RUN_END = dt.time(12, 35)
ENTRY_START = dt.time(9, 30)
ENTRY_END = dt.time(12, 30)
PULLBACK_BARS = 4
PULLBACK_ATR_TOLERANCE = 0.25
MAX_EXTENSION_ATR = 0.5
MIN_VOLUME_RATIO = 0.8
MIN_RUNWAY_R = 1.0
MAX_OPTION_SPREAD_PCT = 5.0
OPTION_MIN_PREMIUM = 2.0
OPTION_MIN_OI = 1000
OPTION_MIN_DELTA = 0.25
OPTION_MIN_DTE_DAYS = 5
OPTION_REQUIRE_BID = True
OPTION_DELTA_FALLBACK = 0.5
MAX_SIGNALS_PER_CYCLE = 3
MIN_ADX = 20.0
DAILY_SLOPE_LOOKBACK = 5          # bars used for the daily-EMA20 slope check
BREADTH_MIN_PCT_FOR_CE = 55.0
BREADTH_MAX_PCT_FOR_PE = 45.0
VIX_SPIKE_MULTIPLIER = 1.30
OUT = core.OUT


def record_check(checks, name, current, criterion, passed):
    checks.append({
        "gate": name,
        "current": current,
        "criterion": criterion,
        "passed": bool(passed),
    })


def evaluate_setup(symbol, candles, now, nifty_bias, daily_ctx=None,
                   breadth=None, vix=None):
    """Evaluate a completed-bar trend-pullback setup without making any API calls.

    daily_ctx: optional dict with keys 'ema20_series' or 'slope' - used by the
    multi-timeframe (daily) alignment gate. When absent the gate is skipped.
    breadth: {'pct_above_prev_close': float|None, ...}; skipped if value is None.
    vix:     {'ltp': float|None, 'ema20': float|None}; skipped if either is None.
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
        record_check(checks, "Candle history", "no timestamped candles",
                     "at least 50 total and 3 completed today", False)
        result["verdict"] = "SKIP"
        result["fails"].append("no timestamped candle data")
        return result
    today = candles[candles.index.date == now.date()]
    enough_bars = len(candles) >= 50 and len(today) >= 3
    record_check(checks, "Candle history", f"{len(candles)} total; {len(today)} today",
                 "at least 50 total and 3 completed today", enough_bars)
    if not enough_bars:
        result["verdict"] = "SKIP"
        result["fails"].append("not enough completed candles")
        return result

    closes = candles["close"]
    ema9 = core.ema(closes, 9)
    ema21 = core.ema(closes, 21)
    ema50 = core.ema(closes, 50)
    last = today.iloc[-1]
    previous = today.iloc[-2]
    e9, e21, e50 = float(ema9.iloc[-1]), float(ema21.iloc[-1]), float(ema50.iloc[-1])

    if e9 > e21 > e50:
        side, sign = "CE", 1
        trend_ok = True
        trend_text = f"EMA9 {e9:.2f} > EMA21 {e21:.2f} > EMA50 {e50:.2f}"
    elif e9 < e21 < e50:
        side, sign = "PE", -1
        trend_ok = True
        trend_text = f"EMA9 {e9:.2f} < EMA21 {e21:.2f} < EMA50 {e50:.2f}"
    else:
        side, sign = None, 0
        trend_ok = False
        trend_text = f"EMA9 {e9:.2f}; EMA21 {e21:.2f}; EMA50 {e50:.2f}"

    result.update({
        "side": side,
        "ltp": core.r2(float(last["close"])),
        "last_candle_time": today.index[-1].strftime("%H:%M"),
        "ema9": core.r2(e9),
        "ema21": core.r2(e21),
        "ema50": core.r2(e50),
    })
    record_check(checks, "Trend alignment", trend_text,
                 "EMA9 > EMA21 > EMA50 for CE; reverse order for PE", trend_ok)
    if side is None:
        result["verdict"] = "SKIP"
        result["fails"].append("no aligned 5-minute trend")
        return result

    tolerance = max(abs(e21) * 0.0005, core.atr(today, 14) * PULLBACK_ATR_TOLERANCE)
    recent = today.tail(PULLBACK_BARS)
    if side == "CE":
        touched = bool((recent["low"] <= ema21.loc[recent.index] + tolerance).any())
    else:
        touched = bool((recent["high"] >= ema21.loc[recent.index] - tolerance).any())
    record_check(checks, "EMA21 pullback",
                 f"recent {PULLBACK_BARS} bars; tolerance {tolerance:.2f}",
                 "price retraced to within 0.25 ATR of EMA21", touched)

    if side == "CE":
        confirmation = float(last["close"]) > float(previous["high"]) and float(last["close"]) > e9
    else:
        confirmation = float(last["close"]) < float(previous["low"]) and float(last["close"]) < e9
    record_check(checks, "Pullback confirmation",
                 f"close {last['close']:.2f}; previous high {previous['high']:.2f}; "
                 f"previous low {previous['low']:.2f}",
                 "completed candle closes beyond previous candle in trend direction and EMA9",
                 confirmation)

    vw = core.vwap(today)
    vwap_ok = float(last["close"]) > vw if side == "CE" else float(last["close"]) < vw
    record_check(checks, "VWAP alignment", f"close {last['close']:.2f}; VWAP {vw:.2f}",
                 "close above VWAP for CE; below VWAP for PE", vwap_ok)

    rs = core.rsi(closes)
    rsi_ok = 50 <= rs <= 70 if side == "CE" else 30 <= rs <= 50
    record_check(checks, "RSI momentum", f"{rs:.1f}",
                 "50–70 for CE; 30–50 for PE", rsi_ok)

    macd = core.macd_hist(closes)
    macd_ok = macd > 0 if side == "CE" else macd < 0
    record_check(checks, "MACD momentum", f"histogram {macd:.4f}",
                 "positive for CE; negative for PE", macd_ok)

    previous_volume = today["volume"].iloc[-4:-1]
    volume_ratio = (float(last["volume"]) / float(previous_volume.mean())
                    if len(previous_volume) == 3 and previous_volume.mean() > 0 else 0.0)
    volume_ok = volume_ratio >= MIN_VOLUME_RATIO
    result["volume_ratio"] = core.r2(volume_ratio)
    record_check(checks, "Confirmation volume", f"{volume_ratio:.2f}x previous 3-bar average",
                 f">= {MIN_VOLUME_RATIO:.2f}x", volume_ok)

    aligned = ((side == "CE" and nifty_bias["bias"] != "DOWN")
               or (side == "PE" and nifty_bias["bias"] != "UP"))
    record_check(checks, "Nifty alignment", nifty_bias["bias"],
                 "not opposed to the trade direction", aligned)

    atr_value = core.atr(today, 14)
    extension = sign * (float(last["close"]) - e9)
    extension_ok = atr_value > 0 and extension <= MAX_EXTENSION_ATR * atr_value
    record_check(checks, "Entry extension", f"{extension:.2f}; ATR {atr_value:.2f}",
                 f"not more than {MAX_EXTENSION_ATR:.2f} ATR beyond EMA9", extension_ok)

    entry_allowed = ENTRY_START <= now.time() <= ENTRY_END
    record_check(checks, "Entry time", now.strftime("%H:%M"),
                 f"{ENTRY_START.strftime('%H:%M')}–{ENTRY_END.strftime('%H:%M')} IST",
                 entry_allowed)

    # ADX trend strength on the 5-minute series - trending-only strategy.
    adx_value = core.adx(today)
    adx_ok = adx_value is not None and adx_value >= MIN_ADX
    record_check(checks, "ADX trend strength",
                 "n/a" if adx_value is None else f"ADX14 {adx_value:.1f}",
                 f">= {MIN_ADX:.1f}", adx_ok)
    result["adx14_5m"] = core.r2(adx_value)

    # Daily EMA20 slope must agree with the intraday trend direction (multi-timeframe gate).
    daily_slope = None
    if daily_ctx is not None:
        if "slope" in daily_ctx and daily_ctx["slope"] is not None:
            daily_slope = float(daily_ctx["slope"])
        else:
            ema20_series = daily_ctx.get("ema20_series")
            if ema20_series is not None and len(ema20_series) >= DAILY_SLOPE_LOOKBACK + 1:
                recent = ema20_series.tail(DAILY_SLOPE_LOOKBACK + 1)
                daily_slope = float(recent.iloc[-1] - recent.iloc[0])
    if daily_slope is None:
        daily_ok = True
        daily_slope_text = "daily EMA20 slope unavailable"
    elif side == "CE":
        daily_ok = daily_slope > 0
        daily_slope_text = f"daily EMA20 slope {daily_slope:+.2f} over {DAILY_SLOPE_LOOKBACK} sessions"
    else:
        daily_ok = daily_slope < 0
        daily_slope_text = f"daily EMA20 slope {daily_slope:+.2f} over {DAILY_SLOPE_LOOKBACK} sessions"
    record_check(checks, "Daily trend", daily_slope_text,
                 "slope in trade direction", daily_ok)

    # Market breadth: % of universe above its prev close - internals must agree.
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
    record_check(checks, "Market breadth", breadth_text,
                 "breadth aligned with trade direction", breadth_ok)
    result["breadth_pct"] = core.r2(breadth_pct) if breadth_pct is not None else None

    # India VIX regime: veto when VIX is spiking.
    vix_ltp = (vix or {}).get("ltp")
    vix_baseline = (vix or {}).get("ema20")
    if vix_ltp is None or vix_baseline is None or vix_baseline <= 0:
        vix_ok = True
        vix_text = "VIX regime n/a"
    else:
        multiplier = vix_ltp / vix_baseline
        vix_ok = multiplier <= VIX_SPIKE_MULTIPLIER
        vix_text = (f"VIX {vix_ltp:.2f} / 20d-EMA {vix_baseline:.2f} = "
                    f"{multiplier:.2f}x; cap {VIX_SPIKE_MULTIPLIER:.2f}x")
    record_check(checks, "India VIX regime", vix_text,
                 f"VIX / 20d-EMA <= {VIX_SPIKE_MULTIPLIER:.2f}", vix_ok)
    result["vix"] = {"ltp": core.r2(vix_ltp), "ema20": core.r2(vix_baseline)} if vix_ltp is not None else None

    # Momentum divergence veto (opposite of a fresh trend-pullback entry).
    divergence = core.momentum_divergence(today, side)
    divergence_ok = divergence is None
    record_check(checks, "Momentum divergence",
                 divergence or "no divergence detected",
                 "no bearish RSI divergence for CE / no bullish for PE",
                 divergence_ok)

    risk = atr_value
    result.update({
        "vwap": core.r2(vw),
        "rsi14_5m": core.r2(rs),
        "macd_hist_5m": core.r2(macd),
        "atr_5m": core.r2(atr_value),
        "plan": {
            "entry": core.r2(float(last["close"])),
            "stop": core.r2(float(last["close"]) - sign * risk),
            "risk_per_share": core.r2(risk),
            "t1": core.r2(float(last["close"]) + sign * risk),
            "t2": core.r2(float(last["close"]) + sign * 2 * risk),
        },
    })
    result["fails"] = [check["gate"] for check in checks if not check["passed"]]
    result["verdict"] = f"WATCH_{side}" if result["fails"] else f"SETUP_{side}"
    return result


def attach_option_check(result, master, expiry, chain, day=None):
    """Add ATM contract, OI-runway, and spread/liquidity gates to a valid setup."""
    side = result["side"]
    ltp = float(result["ltp"])
    option_info = core.pick_option(chain, side, ltp, float(result["plan"]["entry"]))
    checks = result["mechanical_checks"]
    if not option_info:
        record_check(checks, "Option chain", "unavailable", "ATM option data available", False)
        result["fails"].append("option chain unavailable")
        result["verdict"] = f"WATCH_{side}"
        return result

    wall = option_info["wall"]
    risk = float(result["plan"]["risk_per_share"])
    sign = 1 if side == "CE" else -1
    runway = sign * (wall - float(result["plan"]["entry"])) / risk if wall is not None and risk else None
    runway_ok = runway is not None and runway >= MIN_RUNWAY_R
    record_check(checks, "OI-wall runway",
                 f"{runway:.2f}R" if runway is not None else "no directional OI wall",
                 f">= {MIN_RUNWAY_R:.2f}R of room", runway_ok)

    option = option_info["opt"] or {}
    premium = float(option.get("last_price") or 0)
    bid = float(option.get("top_bid_price") or 0)
    ask = float(option.get("top_ask_price") or 0)
    spread = (ask - bid) / premium * 100 if premium > 0 and ask > 0 and bid > 0 else None
    oi_value = int(option.get("oi") or 0)
    delta_raw = (option.get("greeks") or {}).get("delta")
    delta_abs = abs(float(delta_raw)) if delta_raw is not None else None
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
        reasons.append("bid unavailable (no live quote)")
    if spread is not None and spread > MAX_OPTION_SPREAD_PCT:
        reasons.append(f"spread {spread:.2f}% > {MAX_OPTION_SPREAD_PCT:.1f}%")
    if delta_abs is None or delta_abs < OPTION_MIN_DELTA:
        reasons.append(
            f"delta {'missing' if delta_abs is None else f'{delta_abs:.2f}'} "
            f"< {OPTION_MIN_DELTA:.2f}"
        )
    if dte_days is None or dte_days < OPTION_MIN_DTE_DAYS:
        reasons.append(
            f"DTE {'missing' if dte_days is None else dte_days} "
            f"< {OPTION_MIN_DTE_DAYS} days"
        )
    liquid = not reasons
    record_check(
        checks, "Option liquidity",
        f"premium {premium:.2f}; OI {oi_value}; "
        f"spread {f'{spread:.2f}%' if spread is not None else 'unavailable'}; "
        f"delta {f'{delta_abs:.2f}' if delta_abs is not None else 'missing'}; "
        f"DTE {f'{dte_days}d' if dte_days is not None else 'missing'}",
        f"premium >= {OPTION_MIN_PREMIUM:.2f}; OI >= {OPTION_MIN_OI}; "
        f"bid > 0; spread <= {MAX_OPTION_SPREAD_PCT:.1f}%; "
        f"|delta| >= {OPTION_MIN_DELTA:.2f}; DTE >= {OPTION_MIN_DTE_DAYS}d",
        liquid,
    )

    lot, option_sid = core.lot_and_id(
        master, result["symbol"], expiry, option_info["strike"], side
    )
    delta = delta_abs if delta_abs is not None else OPTION_DELTA_FALLBACK
    cost = premium * lot if lot else None
    capital_ok = cost is not None and cost <= core.CONFIG["capital"]
    record_check(checks, "One-lot premium budget",
                 f"Rs{cost:.2f}" if cost is not None else "lot size unavailable",
                 f"<= Rs{core.CONFIG['capital']:.2f}", capital_ok)
    result["option"] = {
        "contract": f"{result['symbol']} {expiry} {option_info['strike']:g} {side}",
        "security_id": option_sid,
        "expiry": expiry,
        "strike": option_info["strike"],
        "type": side,
        "premium": core.r2(premium),
        "bid": core.r2(bid),
        "ask": core.r2(ask),
        "oi": option.get("oi"),
        "delta_used": core.r2(delta),
        "lot_size": lot,
        "cost_per_lot": core.r2(cost),
    }
    result["plan"].update({
        "next_oi_wall": wall,
        "runway_r": core.r2(runway),
        "estimated_loss_per_lot": core.r2(delta * risk * lot) if lot else None,
    })
    result["fails"] = [check["gate"] for check in checks if not check["passed"]]
    result["verdict"] = f"BUY_{side}" if not result["fails"] and lot else f"WATCH_{side}"
    if not lot:
        result["fails"].append("option lot size unavailable")
    return result


def _notify(text):
    core.load_env()
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        core.send_telegram_message(token, chat_id, text)


def notify_signal(result):
    option, plan = result["option"], result["plan"]
    _notify(
        f"[Trend Pullback] {result['verdict']} {result['symbol']} "
        f"underlying {result['ltp']}\n"
        f"{option['contract']} ~Rs{option['premium']} | lot {option['lot_size']}\n"
        f"Underlying entry {plan['entry']} | stop {plan['stop']} | "
        f"T1 {plan['t1']} | T2 {plan['t2']}\n"
        f"Approx. delta-based loss/lot Rs{plan['estimated_loss_per_lot']}\n"
        f"Checks passed: {sum(c['passed'] for c in result['mechanical_checks'])}/"
        f"{len(result['mechanical_checks'])} | runway {plan['runway_r']}R\n"
        "Recommendations only; no order was placed. Validate news and risk before trading."
    )


def _breadth_from_quotes(quotes: dict, symbols: dict[str, int]) -> dict:
    """Fraction of universe trading above prev-close; prev-close read from the ohlc payload."""
    eq_quotes = (quotes or {}).get("NSE_EQ") or {}
    total = above = 0
    for sid in symbols.values():
        quote = eq_quotes.get(str(sid)) or {}
        ltp = quote.get("last_price")
        prev_close = (quote.get("ohlc") or {}).get("close")
        if ltp is None or prev_close is None or prev_close <= 0:
            continue
        total += 1
        if float(ltp) > float(prev_close):
            above += 1
    if total == 0:
        return {"pct_above_prev_close": None, "sample": 0}
    return {"pct_above_prev_close": round(100 * above / total, 2), "sample": total}


_vix_baseline_cache: dict = {}


def _fetch_vix(dh, day):
    """Best-effort VIX fetch; errors zero the gate rather than aborting the scan."""
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


_daily_cache: dict[tuple[str, int], tuple[dt.date, pd.Series]] = {}


def _daily_context(dh, symbol: str, sid: int, day: dt.date) -> dict:
    """Return a cached dict with the daily EMA20 series (for slope checks)."""
    key = (symbol, int(sid))
    cached = _daily_cache.get(key)
    if cached and cached[0] == day:
        return {"ema20_series": cached[1]}
    try:
        daily = dh.daily(sid, days=80)
        if daily is None or daily.empty:
            return {}
        ema20_series = core.ema(daily["close"], 20)
    except Exception:  # noqa: BLE001
        return {}
    _daily_cache[key] = (day, ema20_series)
    return {"ema20_series": ema20_series}


def scan(dh, master, universe, exclude, expiry, day, now, fired):
    symbols = {symbol: sid for symbol, sid in universe.items() if symbol not in exclude}
    quotes = dh.ohlc({
        "NSE_EQ": list(symbols.values()),
        "IDX_I": [core.NIFTY_ID],
    })
    nifty = core.nifty_bias(dh, quotes, day, now)
    breadth = _breadth_from_quotes(quotes, symbols)
    vix = _fetch_vix(dh, day)
    results = []
    frm = dt.datetime.combine(day - dt.timedelta(days=4), dt.time(9, 15), core.IST)
    for symbol, sid in symbols.items():
        try:
            daily_ctx = _daily_context(dh, symbol, sid, day)
            candles = dh.intraday(sid, frm, now)
            candles = candles[candles.index <= pd.Timestamp(now) - pd.Timedelta(minutes=5)]
            result = evaluate_setup(
                symbol, candles, now, nifty,
                daily_ctx=daily_ctx, breadth=breadth, vix=vix,
            )
            if result.get("verdict") == f"SETUP_{result.get('side')}":
                if not expiry:
                    record_check(result["mechanical_checks"], "Option expiry",
                                 "unavailable", "valid expiry available", False)
                    result["fails"].append("option expiry unavailable")
                    result["verdict"] = f"WATCH_{result['side']}"
                else:
                    chain = dh.option_chain(sid, expiry)
                    result = attach_option_check(result, master, expiry, chain, day=day)
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

    snapshot = {
        "strategy": STRATEGY,
        "scan_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "expiry": expiry,
        "candidates_checked": len(results),
        "nifty": nifty,
        "results": results,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "trend_pullback_latest.json").write_text(json.dumps(snapshot, indent=2))
    with open(OUT / f"trend_pullback_scan_{day}.jsonl", "a", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps({"nifty_bias": nifty["bias"], **result}) + "\n")

    buys = [result for result in results if result["verdict"].startswith("BUY_")]
    buys.sort(
        key=lambda r: (
            float(((r.get("plan") or {}).get("runway_r")) or 0),
            float(r.get("volume_ratio") or 0),
        ),
        reverse=True,
    )
    if len(buys) > MAX_SIGNALS_PER_CYCLE:
        for result in buys[MAX_SIGNALS_PER_CYCLE:]:
            result["verdict"] = f"WATCH_{result['side']}"
            result["fails"].append(
                f"exceeds MAX_SIGNALS_PER_CYCLE={MAX_SIGNALS_PER_CYCLE} this cycle"
            )
        buys = buys[:MAX_SIGNALS_PER_CYCLE]
    signal_path = OUT / f"trend_pullback_signals_{day}.jsonl"
    for result in buys:
        key = f"{result['symbol']}_{result['side']}"
        if key in fired:
            continue
        fired.add(key)
        with open(signal_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        notify_signal(result)

    counts = {
        prefix: sum(result["verdict"].startswith(prefix) for result in results)
        for prefix in ("BUY_", "WATCH_", "SKIP")
    }
    print(f"\n[{snapshot['scan_time']}] Trend Pullback | Nifty {nifty['bias']} | "
          f"expiry {expiry or 'unavailable'} | candidates {len(results)}")
    print(f"  BUY: {counts['BUY_']} | WATCH: {counts['WATCH_']} | SKIP: {counts['SKIP']}")
    for result in results:
        if result["verdict"].startswith(("BUY_", "WATCH_")):
            failed = ", ".join(result["fails"]) or "all checks passed"
            print(f"  {result['verdict']:8} {result['symbol']:12} "
                  f"{result.get('ltp', ''):>9} | {failed}")
    return snapshot


def get_expiry(dh, universe, day):
    for symbol in ("RELIANCE", "HDFCBANK", *universe):
        sid = universe.get(symbol)
        if sid:
            expiries = [value for value in dh.expiries(sid) if value >= str(day)]
            if expiries:
                return sorted(expiries)[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Standalone trend-pullback options recommender")
    parser.add_argument("--once", action="store_true", help="run one scan now and exit")
    parser.add_argument("--universe", default="nifty50",
                        help="nifty50, niftynext50, nifty100, nifty200, fno, or watchlist")
    parser.add_argument("--exclude", default="", help="comma-separated symbols to skip")
    args = parser.parse_args()
    exclude = {symbol.strip().upper() for symbol in args.exclude.split(",") if symbol.strip()}
    dh = core.Dhan(*core.credentials())
    day = dt.datetime.now(core.IST).date()
    OUT.mkdir(exist_ok=True)
    try:
        eq, master = core.load_master(day)
        universe = core.build_universe(args.universe, day, eq, master)
        expiry = get_expiry(dh, universe, day)
        if not expiry:
            print("No valid option expiry was found; scans will report option expiry as unavailable.",
                  file=sys.stderr)
        fired = set()
        signal_path = OUT / f"trend_pullback_signals_{day}.jsonl"
        if signal_path.exists():
            for line in signal_path.read_text(encoding="utf-8").splitlines():
                signal = json.loads(line)
                fired.add(f"{signal['symbol']}_{signal['side']}")

        if args.once:
            scan(dh, master, universe, exclude, expiry, day,
                 dt.datetime.now(core.IST), fired)
            return

        print("Trend Pullback scanner runs independently, every 5 minutes from "
              "09:20-12:35 IST; recommendations only. Ctrl+C to stop.")
        while True:
            now = dt.datetime.now(core.IST)
            if now.time() > RUN_END:
                print("Trend Pullback scanner finished for the day.")
                return
            next_run = (now.replace(second=0, microsecond=0)
                        + dt.timedelta(minutes=5 - now.minute % 5)
                        + dt.timedelta(seconds=20))
            if now.time() < RUN_START:
                next_run = max(next_run, dt.datetime.combine(day, RUN_START, core.IST))
            time.sleep(max(1, (next_run - dt.datetime.now(core.IST)).total_seconds()))
            now = dt.datetime.now(core.IST)
            if now.time() > RUN_END:
                continue
            try:
                scan(dh, master, universe, exclude, expiry, day, now, fired)
            except core.DhanAuthError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"Trend Pullback scan failed; will retry next candle: {exc}",
                      file=sys.stderr)
    except core.DhanAuthError as exc:
        sys.exit(f"Dhan rejected the token ({exc}). Check Dhan credentials and retry.")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
