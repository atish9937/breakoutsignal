#!/usr/bin/env python3
"""Historical breakout strategy backtest using Dhan underlying and rolling option data.

Historical option bid/ask spreads are not provided by the rolling-options endpoint, and the
historical strike window is limited to ATM +/-3 for stock options.

Run, for example:
    backtest.bat --start 2025-01-01 --end 2025-03-31 --symbols RELIANCE,HDFCBANK --disable breakout_volume
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd
import requests

import breakout_scanner as scanner


IST = scanner.IST
OUT = scanner.OUT
FILTERS = (
    "atr_room",
    "fresh_breakout",
    "chase_limit",
    "breakout_volume",
    "entry_time",
    "quality_score",
    "oi_wall_runway",
    "option_liquidity",
)
ENTRY_END = dt.time(15, 10)
EXIT_LAST_BAR_START = dt.time(15, 10)
INTRADAY_CHUNK_DAYS = 30
DAILY_CHUNK_DAYS = 365


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def parse_symbols(value: str) -> list[str]:
    symbols = list(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))
    if not symbols:
        raise argparse.ArgumentTypeError("provide at least one comma-separated symbol")
    return symbols


def parse_filters(value: str) -> set[str]:
    selected = {part.strip().lower() for part in value.split(",") if part.strip()}
    invalid = selected.difference(FILTERS)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown filter(s): {', '.join(sorted(invalid))}; choose from {', '.join(FILTERS)}"
        )
    return selected


def _cache_path(kind: str, symbol: str, sid: int, start: dt.date, end: dt.date) -> Path:
    cache_dir = OUT / "backtest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{kind}_{symbol}_{sid}_{start}_{end}.csv"


def _read_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(IST)
    else:
        frame.index = frame.index.tz_convert(IST)
    return frame.astype(float).sort_index()


def _write_cached(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index_label="timestamp")


def _rolling_option_frame(response: dict, option_type: str) -> pd.DataFrame:
    data = response.get("data") or {}
    leg = data.get("ce" if option_type == "CALL" else "pe") or {}
    timestamps = leg.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"])
    columns = ("open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot")
    lengths = {name: len(leg.get(name) or []) for name in columns}
    lengths["timestamp"] = len(timestamps)
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Dhan rolling-option response has mismatched array lengths: {lengths}")
    frame = pd.DataFrame({name: leg.get(name) for name in columns})
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(IST)
    )
    return frame.astype(float).sort_index()


def fetch_rolling_options(dh: scanner.Dhan, symbol: str, sid: int,
                          start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
    """Fetch nearest-month rolling options around ATM in 30-day API windows."""
    offsets = {
        f"{option_type}_ATM{offset:+d}" if offset else f"{option_type}_ATM":
            (option_type, f"ATM{offset:+d}" if offset else "ATM")
        for option_type in ("CALL", "PUT")
        for offset in range(-3, 4)
    }
    gathered: dict[str, list[pd.DataFrame]] = {key: [] for key in offsets}
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=INTRADAY_CHUNK_DAYS - 1))
        for key, (option_type, strike) in offsets.items():
            path = _cache_path(f"rolling_{option_type}_{strike}", symbol, sid, cursor, chunk_end)
            frame = _read_cached(path)
            if frame is None:
                body = {
                    "exchangeSegment": "NSE_FNO",
                    "interval": "5",
                    "securityId": str(sid),
                    "instrument": "OPTSTK",
                    "expiryFlag": "MONTH",
                    "expiryCode": 1,
                    "strike": strike,
                    "drvOptionType": option_type,
                    "requiredData": ["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
                    "fromDate": str(cursor),
                    "toDate": str(chunk_end + dt.timedelta(days=1)),
                }
                try:
                    response = dh.rolling_option(body)
                except requests.RequestException as exc:
                    raise RuntimeError(
                        f"Dhan rolling-options request failed for {symbol} {option_type} {strike} "
                        f"({cursor} through {chunk_end}) after retries. Successful cached requests "
                        f"will be reused when you rerun. Details: {exc}"
                    ) from exc
                frame = _rolling_option_frame(response, option_type)
                if not frame.empty:
                    _write_cached(frame, path)
            if not frame.empty:
                gathered[key].append(frame)
        cursor = chunk_end + dt.timedelta(days=1)

    result = {}
    for key, frames in gathered.items():
        if frames:
            combined = pd.concat(frames).sort_index()
            result[key] = combined.loc[~combined.index.duplicated(keep="last")]
        else:
            result[key] = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
                index=pd.DatetimeIndex([], tz=IST),
            )
    return result


def fetch_daily(dh: scanner.Dhan, symbol: str, sid: int,
                start: dt.date, end: dt.date) -> pd.DataFrame:
    parts = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=DAILY_CHUNK_DAYS - 1))
        path = _cache_path("daily", symbol, sid, cursor, chunk_end)
        part = _read_cached(path)
        if part is None:
            part = dh.daily_range(sid, cursor, chunk_end + dt.timedelta(days=1))
            if not part.empty:
                _write_cached(part, path)
        if not part.empty:
            parts.append(part)
        cursor = chunk_end + dt.timedelta(days=1)
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(parts).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


def fetch_intraday(dh: scanner.Dhan, symbol: str, sid: int,
                   start: dt.date, end: dt.date, seg="NSE_EQ", inst="EQUITY") -> pd.DataFrame:
    parts = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=INTRADAY_CHUNK_DAYS - 1))
        path = _cache_path(f"intraday_{seg}", symbol, sid, cursor, chunk_end)
        part = _read_cached(path)
        if part is None:
            frm = dt.datetime.combine(cursor, dt.time(0), IST)
            to = dt.datetime.combine(chunk_end + dt.timedelta(days=1), dt.time(0), IST)
            part = dh.intraday(sid, frm, to, seg=seg, inst=inst, interval=5)
            if not part.empty:
                _write_cached(part, path)
        if not part.empty:
            parts.append(part)
        cursor = chunk_end + dt.timedelta(days=1)
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    combined = pd.concat(parts).sort_index()
    return combined.loc[~combined.index.duplicated(keep="last")]


def resolve_symbols(symbols: list[str], eq: pd.DataFrame | None) -> dict[str, int]:
    ids = {}
    if eq is not None and {"SYMBOL", "SID"}.issubset(eq.columns):
        ids = dict(zip(eq["SYMBOL"].astype(str).str.upper(), eq["SID"].astype(int)))
    missing = [symbol for symbol in symbols if symbol not in ids and symbol not in scanner.WATCHLIST]
    if missing:
        raise ValueError(f"Could not resolve Dhan security ID(s): {', '.join(missing)}")
    return {symbol: ids.get(symbol, scanner.WATCHLIST.get(symbol)) for symbol in symbols}


def _day_key(index: pd.DatetimeIndex, day: dt.date) -> pd.Series:
    return pd.Series(index.date == day, index=index)


def _nifty_bias(index_bars: pd.DataFrame, day: dt.date, timestamp: pd.Timestamp) -> str:
    bars = index_bars.loc[index_bars.index <= timestamp]
    bars = bars.loc[bars.index >= pd.Timestamp(day - dt.timedelta(days=4), tz=IST)]
    day_bars = bars.loc[bars.index.date == day]
    if day_bars.empty:
        return "FLAT"
    ltp, opening = float(day_bars["close"].iloc[-1]), float(day_bars["open"].iloc[0])
    e9 = scanner.ema(bars["close"], 9).iloc[-1] if len(bars) > 21 else None
    e21 = scanner.ema(bars["close"], 21).iloc[-1] if len(bars) > 21 else None
    if ltp > opening and (e9 is None or e9 > e21):
        return "UP"
    if ltp < opening and (e9 is None or e9 < e21):
        return "DOWN"
    return "FLAT"


def _within_entry_window(timestamp: pd.Timestamp) -> bool:
    current = timestamp.timetz().replace(tzinfo=None)
    return dt.time(9, 30) <= current <= dt.time(12, 30)


def _row_at(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    if timestamp not in frame.index:
        return None
    row = frame.loc[timestamp]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _option_snapshot(options: dict[str, pd.DataFrame], side: str, timestamp: pd.Timestamp,
                     entry: float, atr: float, underlying_spot: float) -> dict:
    option_type = "CALL" if side == "CE" else "PUT"
    atm_key = f"{option_type}_ATM"
    atm = _row_at(options.get(atm_key, pd.DataFrame()), timestamp)
    if atm is None:
        return {"available": False, "wall": None, "runway_r": None,
                "premium": None, "oi": None, "strike": None, "iv": None}

    option_spot = float(atm["spot"])
    if not math.isfinite(option_spot) or option_spot <= 0 or underlying_spot <= 0:
        return {"available": False, "wall": None, "runway_r": None,
                "premium": None, "oi": None, "strike": None, "iv": None}
    underlying_scale = underlying_spot / option_spot
    sign = 1 if side == "CE" else -1
    candidates = []
    offsets = range(4) if side == "CE" else range(-3, 1)
    keys = [f"{option_type}_ATM{offset:+d}" if offset else f"{option_type}_ATM"
            for offset in offsets]
    for key in keys:
        row = _row_at(options.get(key, pd.DataFrame()), timestamp)
        if row is None:
            continue
        strike, oi = float(row["strike"]) * underlying_scale, float(row["oi"])
        if oi <= 0 or (side == "CE" and strike <= entry) or (side == "PE" and strike >= entry):
            continue
        candidates.append((strike, oi))

    wall = None
    runway_r = None
    if candidates:
        max_oi = max(oi for _, oi in candidates)
        walls = [(strike, oi) for strike, oi in candidates
                 if oi >= scanner.CONFIG["wall_frac"] * max_oi]
        if walls:
            wall = min(walls, key=lambda pair: abs(pair[0] - entry))[0]
            risk = scanner.CONFIG["stop_atr_mult"] * atr
            runway_r = sign * (wall - entry) / risk if risk > 0 else None
    premium = float(atm["close"])
    oi = float(atm["oi"])
    return {
        "available": True,
        "strike": float(atm["strike"]),
        "underlying_scale": underlying_scale,
        "option_spot": option_spot,
        "premium": premium,
        "oi": oi,
        "iv": float(atm["iv"]),
        "wall": wall,
        "runway_r": runway_r,
        "option_liquid": premium > 0 and oi > 0,
    }


def _option_price_at(options: dict[str, pd.DataFrame], option_type: str, strike: float,
                     timestamp: pd.Timestamp, field: str) -> float | None:
    for offset in range(-3, 4):
        key = f"{option_type}_ATM{offset:+d}" if offset else f"{option_type}_ATM"
        row = _row_at(options.get(key, pd.DataFrame()), timestamp)
        if row is not None and math.isclose(float(row["strike"]), strike, abs_tol=0.01):
            value = float(row[field])
            return value if math.isfinite(value) and value > 0 else None
    return None


def _option_pnl(signal: dict, trade: dict, options: dict[str, pd.DataFrame],
                slippage_bps: float, cost_bps: float) -> dict:
    option = signal["option"]
    if not option.get("available"):
        return {"available": False, "reason": "historical ATM option data unavailable at signal"}
    option_type = "CALL" if signal["side"] == "CE" else "PUT"
    strike = option["strike"]
    entry_time = pd.Timestamp(trade["entry_bar_time"]).tz_localize(IST)
    raw_entry = _option_price_at(options, option_type, strike, entry_time, "open")
    if raw_entry is None:
        return {"available": False, "reason": "fixed-strike option entry candle unavailable"}
    slip = slippage_bps / 10000
    entry = raw_entry * (1 + slip)
    costs = entry * cost_bps / 10000
    pnl = 0.0
    for fill in trade["exit_fills"]:
        exit_time = pd.Timestamp(fill["bar_time"]).tz_localize(IST)
        raw_exit = _option_price_at(options, option_type, strike, exit_time, "close")
        if raw_exit is None:
            return {"available": False,
                    "reason": f"fixed-strike option exit candle unavailable at {fill['bar_time']}"}
        exit_price = raw_exit * (1 - slip)
        fraction = fill["fraction"]
        pnl += fraction * (exit_price - entry)
        costs += fraction * exit_price * cost_bps / 10000
    net = pnl - costs
    return {
        "available": True, "strike": strike, "entry_premium": round(entry, 4),
        "pnl_per_option_unit": round(net, 4),
        "return_pct": round(net / entry * 100, 4) if entry else None,
        "slippage_bps_per_fill": slippage_bps, "cost_bps_per_fill": cost_bps,
        "fill_assumption": "next underlying candle open for entry; option candle close for each underlying-triggered exit",
    }


def _daily_setup(daily: pd.DataFrame, day: dt.date) -> dict | None:
    if daily.empty:
        return None
    daily_before = daily.loc[daily.index.date < day]
    if len(daily_before) < max(scanner.CONFIG["atr_period"] + 2, 20):
        return None
    a = scanner.atr(daily_before, scanner.CONFIG["atr_period"])
    if not math.isfinite(a) or a <= 0:
        return None
    box = daily_before.tail(scanner.CONFIG["box_days"])
    return {
        "atr": a,
        "box_high": float(box["high"].max()),
        "box_low": float(box["low"].min()),
        "res_levels": sorted({float(daily_before["high"].tail(n).max())
                              for n in range(1, scanner.CONFIG["box_days"] + 1)}),
        "sup_levels": sorted({float(daily_before["low"].tail(n).min())
                              for n in range(1, scanner.CONFIG["box_days"] + 1)}, reverse=True),
        "sma20": float(daily_before["close"].tail(20).mean()),
    }


def _signal_record(symbol: str, side: str, day: dt.date, decision_time: pd.Timestamp,
                   bars: pd.DataFrame, i: int, daily_setup: dict, warm: pd.Series,
                   nifty_bias: str, disabled: set[str],
                   options: dict[str, pd.DataFrame] | None = None) -> dict | None:
    sign = 1 if side == "CE" else -1
    close = float(bars["close"].iloc[i])
    previous_close = float(bars["close"].iloc[i - 1])
    a = daily_setup["atr"]
    prep = daily_setup
    level = scanner.nearest_level(prep, side, previous_close)
    beyond = sign * (close - level) > 0
    live_price_beyond = scanner._live_price_beyond_trigger(side, close, level)
    crossed_recent = any(sign * (float(value) - level) <= 0
                         for value in bars["close"].iloc[max(0, i - 2):i])
    fresh = beyond and live_price_beyond and crossed_recent
    if not beyond or (not fresh and "fresh_breakout" not in disabled):
        return None

    opening = float(bars["open"].iloc[0])
    atr_used = abs(close - opening) / a
    chase = sign * (close - level) / level * 100
    volume_base = float(bars["volume"].iloc[i - 3:i].mean())
    volume_ratio = float(bars["volume"].iloc[i] / volume_base) if volume_base else 0.0
    candle_history = pd.concat([warm, bars["close"].iloc[:i + 1].reset_index(drop=True)],
                               ignore_index=True)
    e9, e21 = float(scanner.ema(candle_history, 9).iloc[-1]), float(scanner.ema(candle_history, 21).iloc[-1])
    rs, mh = scanner.rsi(candle_history), scanner.macd_hist(candle_history)
    current_bars = bars.iloc[:i + 1]
    current_vwap = scanner.vwap(current_bars)
    vwap_distance = sign * (close - current_vwap) / current_vwap * 100 if current_vwap else -1
    factors = {
        "daily_trend": close > daily_setup["sma20"] if side == "CE"
                       else close < daily_setup["sma20"],
        "intraday_trend": (close > current_vwap and e9 > e21) if side == "CE"
                          else (close < current_vwap and e9 < e21),
        "momentum": (55 <= rs <= 75 or mh > 0) if side == "CE" else (25 <= rs <= 45 or mh < 0),
        "vwap_aligned": 0 <= vwap_distance <= scanner.CONFIG["max_vwap_dist_pct"],
        "nifty_aligned": (side == "CE" and nifty_bias != "DOWN")
                         or (side == "PE" and nifty_bias != "UP"),
    }
    quality = sum(factors.values())
    checks = {
        "atr_room": atr_used <= scanner.CONFIG["max_atr_used"],
        "fresh_breakout": fresh,
        "chase_limit": 0 <= chase <= scanner.CONFIG["max_chase_pct"],
        "breakout_volume": volume_ratio > 1.0,
        "entry_time": _within_entry_window(decision_time),
        "quality_score": quality >= scanner.CONFIG["min_recommend_score"],
    }
    if any(not passed for name, passed in checks.items() if name not in disabled):
        return None

    # Match the live scanner's cheap prefilter where that prefilter is not the filter under test.
    distance_to_level = sign * (level - close) / close * 100
    if "chase_limit" not in disabled and not (
        -scanner.CONFIG["max_chase_pct"] <= distance_to_level <= scanner.CONFIG["watch_dist_pct"]
    ):
        return None
    if "atr_room" not in disabled and atr_used > scanner.CONFIG["max_atr_used"] + 0.1:
        return None
    if "entry_time" not in disabled and not _within_entry_window(decision_time):
        return None

    entry = max(close, level) if side == "CE" else min(close, level)
    option = _option_snapshot(options or {}, side, bars.index[i], entry, a, close)
    runway = option["runway_r"]
    checks["oi_wall_runway"] = runway is not None and runway >= scanner.CONFIG["min_runway_r"]
    checks["option_liquidity"] = bool(option.get("option_liquid"))
    if any(not passed for name, passed in checks.items()
           if name not in disabled and name in {"oi_wall_runway", "option_liquidity"}):
        return None

    target_cap = None
    if option["wall"] is not None:
        target_cap = option["wall"] - sign * 0.1 * a

    return {
        "symbol": symbol, "side": side, "date": str(day),
        "signal_time": decision_time.strftime("%H:%M:%S"),
        "live_price_proxy": "completed 5-minute candle close",
        "signal_index": i, "level": level, "signal_close": close,
        "atr": a, "atr_used": atr_used, "chase_pct": chase,
        "volume_ratio": volume_ratio, "quality": quality, "factors": factors,
        "checks": checks, "option": option, "target_cap": target_cap,
    }


DEFAULT_EXIT = {"t1_r": None, "t2_r": None, "t1_fraction": 0.5,
                "be_trigger_r": None, "trail_r": None}

# Research-only exit rules compared against the live 50% T1 / T2 / breakeven plan.
EXIT_VARIANTS = {
    "live_plan": {},
    "full_exit_at_t1": {"t1_fraction": 1.0},
    "be_after_1r": {"be_trigger_r": 1.0},
    "t1_1r_t2_2r": {"t1_r": 1.0, "t2_r": 2.0},
    "runner_to_close": {"t2_r": float("inf")},
    "trail_1r_after_t1": {"t2_r": float("inf"), "trail_r": 1.0},
}


def _execute_trade(signal: dict, bars: pd.DataFrame, slippage_bps: float,
                   cost_bps: float, disable_chase: bool = False,
                   exit_rules: dict | None = None) -> dict | None:
    rules = {**DEFAULT_EXIT, **(exit_rules or {})}
    entry_index = signal["signal_index"] + 1
    if entry_index >= len(bars):
        return None
    sign = 1 if signal["side"] == "CE" else -1
    level = signal["level"]
    raw_entry = float(bars["open"].iloc[entry_index])
    chase_at_fill = sign * (raw_entry - level) / level * 100
    if chase_at_fill > scanner.CONFIG["max_chase_pct"] and not disable_chase:
        return None
    slip = slippage_bps / 10000
    entry = raw_entry * (1 + sign * slip)
    risk = scanner.CONFIG["stop_atr_mult"] * signal["atr"]
    stop = entry - sign * risk
    t1_r = rules["t1_r"] if rules["t1_r"] is not None else scanner.CONFIG["t1_r"]
    t2_r = rules["t2_r"] if rules["t2_r"] is not None else scanner.CONFIG["t2_r"]
    t1_fraction = rules["t1_fraction"]
    t1 = entry + sign * t1_r * risk
    t2 = entry + sign * t2_r * risk if math.isfinite(t2_r) else None
    if signal.get("target_cap") is not None:
        cap = signal["target_cap"]
        t1 = min(t1, cap) if sign == 1 else max(t1, cap)
        if t2 is not None:
            t2 = min(t2, cap) if sign == 1 else max(t2, cap)
    future = bars.iloc[entry_index:].loc[lambda frame: frame.index.time <= EXIT_LAST_BAR_START]
    if future.empty:
        return None

    remaining = 1.0
    realized = 0.0
    costs = entry * cost_bps / 10000
    t1_taken = False
    be_armed = False
    best = entry
    exit_reason = "eod"
    exit_fills = []
    last_close = float(future["close"].iloc[-1])

    def close_piece(fraction: float, raw_exit: float, timestamp: pd.Timestamp, reason: str) -> float:
        nonlocal costs
        fill = raw_exit * (1 - sign * slip)
        costs += fraction * fill * cost_bps / 10000
        exit_fills.append({
            "bar_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "fraction": fraction, "reason": reason,
        })
        return fraction * (fill - entry) * sign

    for timestamp, bar in future.iterrows():
        high, low = float(bar["high"]), float(bar["low"])
        stop_hit = low <= stop if sign == 1 else high >= stop
        if stop_hit:
            opening = float(bar["open"])
            stop_fill = min(stop, opening) if sign == 1 else max(stop, opening)
            stop_reason = (
                "trail" if t1_taken and rules["trail_r"] else
                "breakeven" if t1_taken or be_armed else "stop"
            )
            realized += close_piece(remaining, stop_fill, timestamp, stop_reason)
            remaining = 0
            exit_reason = stop_reason
            break
        if not t1_taken:
            t1_hit = high >= t1 if sign == 1 else low <= t1
            if t1_hit:
                realized += close_piece(t1_fraction, t1, timestamp, "t1")
                remaining = 1.0 - t1_fraction
                t1_taken = True
                if remaining <= 0:
                    remaining = 0
                    exit_reason = "t1"
                    break
                stop = entry
                # With OHLC bars, conservatively assume breakeven preceded T2 if both were touched.
                be_hit = low <= stop if sign == 1 else high >= stop
                if be_hit:
                    exit_reason = "t1_then_breakeven_same_bar"
                    opening = float(bar["open"])
                    be_fill = min(stop, opening) if sign == 1 else max(stop, opening)
                    realized += close_piece(remaining, be_fill, timestamp, "breakeven")
                    remaining = 0
                    break
        if t1_taken and remaining and t2 is not None and (
                high >= t2 if sign == 1 else low <= t2):
            realized += close_piece(remaining, t2, timestamp, "t2")
            remaining = 0
            exit_reason = "t2"
            break
        # Stop changes take effect from the next candle, never the one that triggered them.
        favorable = high if sign == 1 else low
        best = max(best, favorable) if sign == 1 else min(best, favorable)
        if (rules["be_trigger_r"] is not None and not t1_taken and not be_armed
                and sign * (best - entry) >= rules["be_trigger_r"] * risk):
            be_armed = True
            stop = entry
        if t1_taken and rules["trail_r"]:
            trailed = best - sign * rules["trail_r"] * risk
            stop = max(stop, trailed) if sign == 1 else min(stop, trailed)
    if remaining:
        realized += close_piece(remaining, last_close, future.index[-1], "eod")

    net_pnl = realized - costs
    return {
        **{key: signal[key] for key in ("symbol", "side", "date", "signal_time")},
        "entry_time": bars.index[entry_index].strftime("%H:%M:%S"),
        "entry_bar_time": bars.index[entry_index].strftime("%Y-%m-%d %H:%M:%S"),
        "entry": round(entry, 4), "exit_reason": exit_reason,
        "pnl_per_share": round(net_pnl, 4), "r_multiple": round(net_pnl / risk, 4),
        "quality": signal["quality"], "trigger_level": round(level, 4),
        "checks": signal["checks"], "historical_option": signal.get("option"),
        "target_cap": signal.get("target_cap"), "exit_fills": exit_fills,
    }


def _scenario_trades(data: dict, start: dt.date, end: dt.date,
                     disabled: set[str], slippage_bps: float, cost_bps: float,
                     exit_rules: dict | None = None) -> list[dict]:
    events: dict[tuple[dt.date, pd.Timestamp], list[dict]] = {}
    for symbol, bundle in data.items():
        daily, intraday = bundle["daily"], bundle["intraday"]
        days = sorted({day for day in intraday.index.date if start <= day <= end})
        nifty_bias_cache = {}
        for day in days:
            day_mask = _day_key(intraday.index, day)
            bars = intraday.loc[day_mask].between_time("09:15", "15:30")
            if len(bars) < 4:
                continue
            daily_setup = _daily_setup(daily, day)
            if daily_setup is None:
                continue
            warm_start = pd.Timestamp(day - dt.timedelta(days=16), tz=IST)
            warm = intraday.loc[(intraday.index < pd.Timestamp(day, tz=IST))
                                & (intraday.index >= warm_start), "close"].tail(60).reset_index(drop=True)
            for i in range(3, len(bars)):
                timestamp = bars.index[i] + dt.timedelta(minutes=5, seconds=20)
                if timestamp.time() > ENTRY_END:
                    continue
                bar_timestamp = bars.index[i]
                if bar_timestamp not in nifty_bias_cache:
                    nifty_bias_cache[bar_timestamp] = _nifty_bias(
                        data["NIFTY"]["intraday"], day, bar_timestamp
                    )
                bias = nifty_bias_cache[bar_timestamp]
                for side in ("CE", "PE"):
                    signal = _signal_record(symbol, side, day, timestamp, bars, i, daily_setup,
                                            warm, bias, disabled, bundle.get("options"))
                    if signal:
                        events.setdefault((day, timestamp), []).append(signal)

    trades = []
    fired: set[tuple[str, str, dt.date]] = set()
    for (_, _), candidates in sorted(events.items(), key=lambda item: item[0]):
        candidates.sort(key=lambda signal: abs(signal["chase_pct"]))
        for signal in candidates[:scanner.CONFIG["max_candidates_per_scan"]]:
            key = (signal["symbol"], signal["side"], dt.date.fromisoformat(signal["date"]))
            if key in fired:
                continue
            fired.add(key)
            bars = data[signal["symbol"]]["intraday"]
            day_bars = bars.loc[_day_key(bars.index, dt.date.fromisoformat(signal["date"]))]
            trade = _execute_trade(signal, day_bars, slippage_bps, cost_bps,
                                   disable_chase="chase_limit" in disabled,
                                   exit_rules=exit_rules)
            if trade:
                trade["option_pnl"] = _option_pnl(
                    signal, trade, data[signal["symbol"]].get("options", {}),
                    slippage_bps, cost_bps
                )
                trades.append(trade)
    return trades


def _metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "total_r": 0.0, "average_r": 0.0, "profit_factor": None, "max_drawdown_r": 0.0}
    results = [float(trade["r_multiple"]) for trade in trades]
    gains = sum(value for value in results if value > 0)
    losses = -sum(value for value in results if value < 0)
    equity = peak = max_drawdown = 0.0
    for result in results:
        equity += result
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(results), "wins": sum(value > 0 for value in results),
        "losses": sum(value <= 0 for value in results),
        "win_rate_pct": round(100 * sum(value > 0 for value in results) / len(results), 2),
        "total_r": round(sum(results), 4), "average_r": round(sum(results) / len(results), 4),
        "profit_factor": round(gains / losses, 4) if losses else (None if not gains else "inf"),
        "max_drawdown_r": round(max_drawdown, 4),
    }


def _option_metrics(trades: list[dict]) -> dict:
    results = [trade["option_pnl"]["return_pct"] for trade in trades
               if trade.get("option_pnl", {}).get("available")]
    wins = sum(value > 0 for value in results)
    return {
        "trades_with_option_prices": len(results),
        "trades_without_complete_option_prices": len(trades) - len(results),
        "win_rate_pct": round(wins * 100 / len(results), 2) if results else 0.0,
        "average_return_pct_per_option_unit": round(sum(results) / len(results), 4) if results else None,
        "median_return_pct_per_option_unit": round(float(pd.Series(results).median()), 4) if results else None,
        "scope_note": "Per option unit, no historical lot size applied; percentages are not rupee P&L.",
    }


def run(args: argparse.Namespace) -> dict:
    today = dt.datetime.now(IST).date()
    if args.end >= today:
        raise ValueError(f"end date must be before today ({today}) so incomplete candles are not included")
    if args.start > args.end:
        raise ValueError("--start must be on or before --end")
    if args.slippage_bps < 0 or args.cost_bps < 0:
        raise ValueError("slippage and cost basis points cannot be negative")

    dh = scanner.Dhan(*scanner.credentials())
    eq, _ = scanner.load_master(today)
    symbols = resolve_symbols(args.symbols, eq)
    fetch_start = args.start - dt.timedelta(days=120)
    data = {}
    for symbol, sid in symbols.items():
        print(f"Loading {symbol} historical candles...")
        data[symbol] = {
            "daily": fetch_daily(dh, symbol, sid, fetch_start, args.end),
            "intraday": fetch_intraday(dh, symbol, sid, fetch_start, args.end),
        }
        print(f"Loading {symbol} historical option OI and premiums...")
        data[symbol]["options"] = fetch_rolling_options(dh, symbol, sid, args.start, args.end)
    data["NIFTY"] = {
        "daily": pd.DataFrame(),
        "intraday": fetch_intraday(dh, "NIFTY", scanner.NIFTY_ID,
                                   args.start - dt.timedelta(days=5), args.end,
                                   seg="IDX_I", inst="INDEX"),
    }
    if data["NIFTY"]["intraday"].empty:
        raise RuntimeError("Dhan returned no Nifty intraday candles; cannot evaluate Nifty alignment")
    for symbol in symbols:
        if data[symbol]["daily"].empty or data[symbol]["intraday"].empty:
            raise RuntimeError(f"Dhan returned insufficient historical candles for {symbol}")

    baseline = _scenario_trades(data, args.start, args.end, set(),
                                args.slippage_bps, args.cost_bps)
    variant = _scenario_trades(data, args.start, args.end, args.disabled,
                               args.slippage_bps, args.cost_bps)
    baseline_keys = {(t["symbol"], t["side"], t["date"], t["signal_time"]) for t in baseline}
    added = [t for t in variant if (t["symbol"], t["side"], t["date"], t["signal_time"]) not in baseline_keys]
    result = {
        "period": {"start": str(args.start), "end": str(args.end)},
        "symbols": list(symbols), "disabled_filters": sorted(args.disabled),
        "assumptions": {
            "data_scope": "Underlying 5-minute and daily candles plus Dhan rolling 5-minute stock-option data (nearest monthly expiry, ATM +/-3) for historical OI-wall and premium/OI gates.",
            "not_tested": ["historical bid/ask spread (not returned by rolling-option endpoint)",
                           "historical lot size and exact rupee-per-lot option P&L",
                           "point-in-time index constituents"],
            "option_data_limit": "OI walls are reconstructed only within the documented stock-option ATM +/-3 strike range. The live scanner can see a wider option chain; farther walls may be missed.",
            "expiry_assumption": "Dhan rolling-option expiryFlag=MONTH, expiryCode=1 is used to approximate the nearest monthly stock-option expiry.",
            "liquidity_assumption": "Historical option premium close > 0 and OI > 0 are required. Historical spread cannot be verified; live scanner allows missing spread data.",
            "pnl_scope": "Underlying R-multiples are reported alongside option premium return per unit. Option exits use the fixed ATM strike selected at signal, but historical lot sizes are unavailable so no rupee-per-lot P&L is reported.",
            "option_exit_assumption": "Enter the selected fixed-strike option at the next 5-minute candle open; exit option at the 5-minute close of each underlying stop/target/EOD exit bar. Missing fixed-strike candles make option P&L unavailable for that trade.",
            "fills": "Signal uses a completed 5-minute close; entry is next 5-minute candle open.",
            "live_trigger_limit": "Historical 5-minute candle close is used as a proxy for live price. Intrabar live-price reclaims cannot be reconstructed from this dataset.",
            "exits": "50% at T1 (1.5R); remaining 50% stop moves to entry, then T2 (2.5R), breakeven, or 15:15 close.",
            "ambiguous_candles": "Stop is prioritized over target; after T1, breakeven is prioritized over T2 within the same candle.",
            "slippage_bps_per_fill": args.slippage_bps,
            "cost_bps_per_fill": args.cost_bps,
            "universe_warning": "Only user-selected symbols are evaluated; historical membership/delisted symbols are not reconstructed.",
            "execution_difference": "The simulator applies the top-8 cap among simulated breakout candidates; the live scanner ranks all near-level candidates, and REST scan timing gaps are not replayed.",
        },
        "baseline": {"metrics": _metrics(baseline), "option_metrics": _option_metrics(baseline),
                     "trades": baseline},
        "variant": {"metrics": _metrics(variant), "option_metrics": _option_metrics(variant),
                    "trades": variant},
        "variant_trades_not_in_baseline": {"count": len(added), "trades": added},
    }
    OUT.mkdir(exist_ok=True)
    if getattr(args, "exit_variants", False):
        comparison = {}
        for name, rules in EXIT_VARIANTS.items():
            trades = _scenario_trades(data, args.start, args.end, set(),
                                      args.slippage_bps, args.cost_bps, rules)
            comparison[name] = {"rules": {k: (None if v == float("inf") else v)
                                          for k, v in rules.items()},
                                "metrics": _metrics(trades),
                                "option_metrics": _option_metrics(trades),
                                "exit_reasons": {
                                    reason: sum(t["exit_reason"] == reason for t in trades)
                                    for reason in sorted({t["exit_reason"] for t in trades})},
                                "trades": trades}
        result["exit_variants"] = comparison
    output_path = OUT / f"backtest_{args.start}_{args.end}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "exit_variants" in result:
        print(f"\nExit-rule comparison (same entries) | {args.start} through {args.end}")
        print(f"{'Exit rule':20} {'Trades':>7} {'Win%':>8} {'Total R':>9} {'Avg R':>9} {'PF':>8} {'Max DD R':>10}")
        for name, row in result["exit_variants"].items():
            m = row["metrics"]
            print(f"{name:20} {m['trades']:7} {m['win_rate_pct']:8.2f} {m['total_r']:9.2f} "
                  f"{m['average_r']:9.3f} {str(m['profit_factor']):>8} {m['max_drawdown_r']:10.2f}")
    print(f"\nHistorical scanner-filter backtest | {args.start} through {args.end}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Disabled filters: {', '.join(sorted(args.disabled)) or 'none (baseline only)'}")
    print(f"{'Scenario':22} {'Trades':>7} {'Win%':>8} {'Total R':>9} {'Avg R':>9} {'PF':>8} {'Max DD R':>10}")
    for name, scenario in (("Baseline", result["baseline"]), ("Selected filters off", result["variant"])):
        metrics = scenario["metrics"]
        print(f"{name:22} {metrics['trades']:7} {metrics['win_rate_pct']:8.2f} "
              f"{metrics['total_r']:9.2f} {metrics['average_r']:9.3f} "
              f"{str(metrics['profit_factor']):>8} {metrics['max_drawdown_r']:10.2f}")
        option_metrics = scenario["option_metrics"]
        print(f"  option-priced: {option_metrics['trades_with_option_prices']}/"
              f"{metrics['trades']} trades, win rate {option_metrics['win_rate_pct']:.2f}%, "
              f"average return {option_metrics['average_return_pct_per_option_unit']}% per unit")
    print(f"Additional variant trades: {len(added)}")
    print(f"Saved detailed results: {output_path}")
    print("OI-wall and premium/OI gates use Dhan rolling options. Bid/ask spread and historical lot sizes are unavailable.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scanner filters using historical underlying candles (not option P&L)."
    )
    parser.add_argument("--start", required=True, type=parse_date, help="first session date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=parse_date, help="last session date, YYYY-MM-DD (before today)")
    parser.add_argument("--symbols", required=True, type=parse_symbols,
                        help="comma-separated symbols, e.g. RELIANCE,HDFCBANK")
    parser.add_argument("--disable", dest="disabled", type=parse_filters, default=set(),
                        help=f"comma-separated filters to remove: {', '.join(FILTERS)}")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="adverse slippage per fill in basis points of each instrument's price (default: 5)")
    parser.add_argument("--cost-bps", type=float, default=0.0,
                        help="estimated cost per fill in basis points of each instrument's price (default: 0)")
    parser.add_argument("--exit-variants", action="store_true",
                        help="also replay the same entries under alternative exit rules (research only)")
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, scanner.DhanAuthError) as exc:
        parser.exit(1, f"Backtest failed: {exc}\n")


if __name__ == "__main__":
    main()
