"""Paper-only daily/hourly trend filter with completed 15-minute option entries."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import pandas as pd

import breakout_scanner as core


STRATEGY = "daily_hourly_15m"
ENTRY_CUTOFF = dt.time(14, 0)
FORCED_EXIT = dt.time(15, 15)
MAX_SIGNALS_PER_DAY = 4
MAX_OPTION_CANDIDATES = 12
MAX_OPTION_SPREAD_PCT = 3.0
OPTION_MIN_PREMIUM = 2.0
OPTION_MIN_OI = 1000
OPTION_MIN_DELTA = 0.25
OPTION_MIN_DTE_DAYS = 5
OPTION_REQUIRE_BID = True
# Exit the option if its live mid-price has dropped below PREMIUM_STOP_FRAC of the
# entry mid. Caps the gamma / vega bleed when the underlying wanders without hitting
# the stop level. Independent of the underlying-stop and underlying-target rules.
PREMIUM_STOP_FRAC = 0.60
RISK_REWARD = 2.0
PULLBACK_ATR = 0.35
_hourly_cache: dict[tuple[str, int], tuple[pd.Timestamp, pd.DataFrame]] = {}


def _completed(frame: pd.DataFrame, interval: int, now: dt.datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    close_through = (
        pd.Timestamp(now).floor(f"{interval}min")
        - dt.timedelta(minutes=interval)
    )
    return frame.loc[frame.index <= close_through]


def _trend_side(daily: dict, hourly: pd.DataFrame, now: dt.datetime) -> str | None:
    daily_ema20 = daily.get("ema20")
    daily_ema50 = daily.get("ema50")
    hourly = _completed(hourly, 60, now)
    if daily_ema20 is None or daily_ema50 is None or len(hourly) < 50:
        return None
    closes = hourly["close"].astype(float)
    ema20 = core.ema(closes, 20)
    ema50 = core.ema(closes, 50)
    slope = float(ema20.iloc[-1] - ema20.iloc[-6])
    if daily_ema20 > daily_ema50 and ema20.iloc[-1] > ema50.iloc[-1] and slope > 0:
        return "CE"
    if daily_ema20 < daily_ema50 and ema20.iloc[-1] < ema50.iloc[-1] and slope < 0:
        return "PE"
    return None


def _setup(symbol: str, side: str, candles: pd.DataFrame, now: dt.datetime) -> dict | None:
    candles = _completed(candles, 15, now)
    if len(candles) < 30:
        return None
    sign = 1 if side == "CE" else -1
    close = candles["close"].astype(float)
    volume = candles["volume"].astype(float)
    e9 = core.ema(close, 9)
    e21 = core.ema(close, 21)
    atr = core.atr(candles, 14)
    if atr <= 0:
        return None
    previous = candles.iloc[-2]
    last = candles.iloc[-1]
    recent = candles.tail(5)
    if side == "CE":
        pullback = bool((recent["low"].astype(float) <=
                         e21.loc[recent.index] + PULLBACK_ATR * atr).any())
        confirmed = float(last["close"]) > float(previous["high"]) and float(last["close"]) > e9.iloc[-1]
        stop = float(recent["low"].min()) - 0.1 * atr
    else:
        pullback = bool((recent["high"].astype(float) >=
                         e21.loc[recent.index] - PULLBACK_ATR * atr).any())
        confirmed = float(last["close"]) < float(previous["low"]) and float(last["close"]) < e9.iloc[-1]
        stop = float(recent["high"].max()) + 0.1 * atr
    trend = float(e9.iloc[-1]) > float(e21.iloc[-1]) if side == "CE" else float(e9.iloc[-1]) < float(e21.iloc[-1])
    volume_base = float(volume.iloc[-4:-1].mean())
    volume_ratio = float(last["volume"]) / volume_base if volume_base > 0 else 0
    vwap = core.vwap(candles.loc[candles.index.date == now.date()])
    vwap_ok = float(last["close"]) > vwap if side == "CE" else float(last["close"]) < vwap
    risk = sign * (float(last["close"]) - stop)
    if not (trend and pullback and confirmed and volume_ratio >= 1.0 and vwap_ok and risk > 0):
        return None
    target = float(last["close"]) + sign * RISK_REWARD * risk
    return {
        "strategy": STRATEGY,
        "symbol": symbol,
        "side": side,
        "candle_time": candles.index[-1].isoformat(),
        "signal_time": now.isoformat(),
        "underlying_entry": round(float(last["close"]), 2),
        "underlying_stop": round(stop, 2),
        "underlying_target": round(target, 2),
        "risk_reward": f"1:{RISK_REWARD:g}",
        "atr_15m": round(float(atr), 3),
        "relative_volume": round(volume_ratio, 2),
        "vwap": round(float(vwap), 2),
        "forced_exit_ist": FORCED_EXIT.strftime("%H:%M"),
        "exit_rule": "Underlying stop or 2R target; otherwise exit option by 15:15 IST. No overnight holding.",
    }


def _select_option(chain: dict, side: str, spot: float, symbol: str,
                   expiry: str, master: pd.DataFrame,
                   day: dt.date | None = None) -> dict | None:
    leg_name = "ce" if side == "CE" else "pe"
    contracts = []
    for raw_strike, legs in (chain.get("oc") or {}).items():
        try:
            strike = float(raw_strike)
        except (TypeError, ValueError):
            continue
        itm = strike < spot if side == "CE" else strike > spot
        distance = abs(spot - strike)
        contracts.append((distance, itm, strike, (legs or {}).get(leg_name) or {}))
    if not contracts:
        return None
    atm = min(contracts, key=lambda row: row[0])
    itm_options = [row for row in contracts if row[1]]
    choices = [("ATM", atm)]
    if itm_options:
        nearest_itm = min(itm_options, key=lambda row: row[0])
        if nearest_itm[2] != atm[2]:
            choices.append(("slightly_ITM", nearest_itm))
    dte_days = None
    if day is not None and expiry:
        try:
            dte_days = (dt.date.fromisoformat(expiry) - day).days
        except (TypeError, ValueError):
            dte_days = None
    if dte_days is not None and dte_days < OPTION_MIN_DTE_DAYS:
        return None
    eligible = []
    for label, (_, _, strike, quote) in choices:
        bid = float(quote.get("top_bid_price") or 0)
        ask = float(quote.get("top_ask_price") or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask >= bid else 0
        spread = (100 * (ask - bid) / mid) if mid > 0 else None
        premium = float(quote.get("last_price") or 0)
        oi_value = int(float(quote.get("oi") or 0))
        volume_value = float(quote.get("volume") or 0)
        delta_raw = (quote.get("greeks") or {}).get("delta")
        delta_abs = abs(float(delta_raw)) if delta_raw is not None else None
        lot, security_id = core.lot_and_id(master, symbol, expiry, strike, side)
        if not (
            premium >= OPTION_MIN_PREMIUM
            and oi_value >= OPTION_MIN_OI
            and volume_value > 0
            and (not OPTION_REQUIRE_BID or bid > 0)
            and spread is not None
            and spread <= MAX_OPTION_SPREAD_PCT
            and delta_abs is not None
            and delta_abs >= OPTION_MIN_DELTA
            and lot and security_id
        ):
            continue
        eligible.append({
            "selection": label,
            "strike": strike,
            "security_id": security_id,
            "premium": premium,
            "bid": bid,
            "ask": ask,
            "spread_pct": round(spread, 3),
            "oi": quote.get("oi"),
            "volume": quote.get("volume"),
            "delta": round(delta_abs, 3),
            "dte_days": dte_days,
            "lot_size": lot,
            "cost_per_lot": round(premium * lot, 2),
        })
    return min(eligible, key=lambda row: (row["spread_pct"], -float(row["oi"]), -float(row["volume"])),
               default=None)


def scan(dh, master: pd.DataFrame, prep: dict, universe: dict[str, int],
         day: dt.date, now: dt.datetime, fired: set[str]) -> list[dict]:
    """Check the most recently completed 15-minute bar and notify only new signals."""
    current = now.astimezone(core.IST)
    if current.time() > ENTRY_CUTOFF or current.minute % 15 != 5:
        return []
    if len(fired) >= MAX_SIGNALS_PER_DAY:
        return []
    candidates = []
    daily_stocks = prep.get("stocks") or {}
    for symbol, security_id in universe.items():
        daily = daily_stocks.get(symbol)
        if not daily:
            continue
        cache_key = (symbol, int(security_id))
        cached = _hourly_cache.get(cache_key)
        completed_hour = pd.Timestamp(current).floor("60min") - pd.Timedelta(minutes=60)
        if cached is None or cached[0] != completed_hour:
            start = current - dt.timedelta(days=20)
            hourly = dh.intraday(
                security_id, start, current, interval=60,
            )
            _hourly_cache[cache_key] = (completed_hour, hourly)
        else:
            hourly = cached[1]
        side = _trend_side(daily, hourly, current)
        if side is None:
            continue
        start_15m = current - dt.timedelta(days=15)
        candles = dh.intraday(security_id, start_15m, current, interval=15)
        signal = _setup(symbol, side, candles, current)
        if not signal:
            continue
        # Commit K: 5-min sub-bar confirmation. The 15m scanner wakes at :05,
        # :20, :35, :50 of each hour - 5 minutes after each 15-min bar closed.
        # The just-closed 5-min bar (e.g. 11:30-11:35 at an 11:35 wake) is a
        # free confirmation sample: it must close in signal direction, otherwise
        # the 11:30-11:45 15-min bar is already reversing within its first 5 min.
        start_5m = current - dt.timedelta(hours=2)
        candles_5m = dh.intraday(security_id, start_5m, current, interval=5)
        candles_5m = _completed(candles_5m, 5, current)
        if candles_5m.empty:
            # If we can't fetch a 5-min bar, don't block - fall back to raw signal
            signal["sub_bar_confirmation"] = "unavailable"
            candidates.append(signal)
            continue
        sub_bar = candles_5m.iloc[-1]
        sub_close = float(sub_bar["close"])
        signal_entry = signal["underlying_entry"]
        if side == "CE":
            sub_confirmed = sub_close > signal_entry
        else:
            sub_confirmed = sub_close < signal_entry
        signal["sub_bar_time"] = candles_5m.index[-1].strftime("%H:%M")
        signal["sub_bar_close"] = round(sub_close, 2)
        signal["sub_bar_confirmation"] = "confirmed" if sub_confirmed else "reversed"
        if not sub_confirmed:
            # 5-min bar after the 15-min signal bar is already going against us.
            # Skip - this is the "immediate reversal at entry" pattern the user
            # flagged on 2026-10-30.
            print(f"  15m {symbol} {side} REJECTED: 5-min sub-bar close {sub_close:.2f} "
                  f"vs signal entry {signal_entry:.2f} (not confirmed)",
                  file=sys.stderr)
            continue
        candidates.append(signal)

    candidates.sort(key=lambda item: (-item["relative_volume"], item["symbol"]))
    emitted = []
    for signal in candidates[:MAX_OPTION_CANDIDATES]:
        key = f"{signal['symbol']}_{signal['side']}"
        if key in fired:
            continue
        if len(fired) >= MAX_SIGNALS_PER_DAY:
            break
        expiry_list = dh.expiries(universe[signal["symbol"]])
        expiry = next((value for value in sorted(expiry_list)
                       if dt.date.fromisoformat(str(value)) >= day + dt.timedelta(days=7)), None)
        if not expiry:
            continue
        chain = dh.option_chain(universe[signal["symbol"]], expiry)
        option = _select_option(
            chain, signal["side"], signal["underlying_entry"],
            signal["symbol"], expiry, master, day=day,
        )
        if not option:
            continue
        signal["expiry"] = expiry
        signal["option"] = option
        signal["underlying_security_id"] = int(universe[signal["symbol"]])
        signal["header"] = "[15M SAME-DAY LONG-OPTION PAPER SIGNAL]"
        signal["status"] = "paper_entry_signal_sent"
        signal["live_orders_placed"] = False
        fired.add(key)
        signal_path = core.OUT / f"{STRATEGY}_signals_{day}.jsonl"
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        with signal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(signal) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        notify_signal(signal)
        emitted.append(signal)
    return emitted


def notify_signal(signal: dict) -> None:
    option = signal["option"]
    text = (
        f"{signal['header']}\n"
        f"{signal['symbol']} {signal['side']} | {signal['expiry']} "
        f"{option['selection']} {option['strike']}\n"
        f"Option bid/ask: {option['bid']:.2f}/{option['ask']:.2f}; "
        f"spread {option['spread_pct']:.2f}% | premium {option['premium']:.2f}\n"
        f"Underlying entry reference {signal['underlying_entry']:.2f} | "
        f"stop {signal['underlying_stop']:.2f} | target {signal['underlying_target']:.2f} "
        f"(1:{RISK_REWARD:g})\n"
        f"15m candle {signal['candle_time']} | relative volume {signal['relative_volume']:.2f}x\n"
        f"Exit by {FORCED_EXIT:%H:%M} IST; no overnight holding. "
        "Paper signal only; refresh quotes and validate before any decision."
    )
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        core.send_telegram_message(token, chat_id, text)
    else:
        print("15m option signal was saved but Telegram credentials are not configured.", file=sys.stderr)


def _contract_quote(chain: dict, side: str, strike: float) -> dict | None:
    key = "ce" if side == "CE" else "pe"
    for raw_strike, legs in (chain.get("oc") or {}).items():
        try:
            if abs(float(raw_strike) - strike) < 0.01:
                quote = (legs or {}).get(key) or {}
                return {
                    "bid": float(quote.get("top_bid_price") or 0),
                    "ask": float(quote.get("top_ask_price") or 0),
                    "premium": float(quote.get("last_price") or 0),
                }
        except (TypeError, ValueError):
            continue
    return None


def monitor_positions(dh, day: dt.date, now: dt.datetime) -> list[dict]:
    """Paper-monitor same-day underlying levels using completed 5-minute candles."""
    current = now.astimezone(core.IST)
    path = core.OUT / f"{STRATEGY}_signals_{day}.jsonl"
    if not path.exists() or current.minute % 5 != 0:
        return []
    try:
        signals = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path.name}: {exc}") from exc

    exited = []
    changed = False
    for signal in signals:
        if signal.get("status") != "paper_entry_signal_sent":
            continue
        security_id = int(signal["underlying_security_id"])
        start = dt.datetime.combine(day, dt.time(9, 15), core.IST)
        candles = dh.intraday(security_id, start, current, interval=5)
        candles = _completed(candles, 5, current)
        if candles.empty:
            print(
                f"No completed 5-minute candles available to monitor {signal['symbol']}.",
                file=sys.stderr,
            )
            continue
        signal_time = pd.Timestamp(signal["signal_time"])
        if signal_time.tzinfo is None:
            signal_time = signal_time.tz_localize(core.IST)
        first_monitor_bar = signal_time.ceil("5min")
        future = candles.loc[
            (candles.index.date == day) & (candles.index >= first_monitor_bar)
        ]
        if future.empty:
            if current.time() >= FORCED_EXIT:
                print(
                    f"No post-signal completed 5-minute candles available to exit "
                    f"{signal['symbol']}.",
                    file=sys.stderr,
                )
            continue
        sign = 1 if signal["side"] == "CE" else -1
        stop = float(signal["underlying_stop"])
        target = float(signal["underlying_target"])
        reason = None
        exit_price = None
        exit_time = None

        # Premium-side stop takes precedence - caps the loss size for the whole
        # strategy. Fetch the current option quote once per monitor pass.
        option_quote = None
        try:
            chain = dh.option_chain(security_id, signal["expiry"])
            option_quote = _contract_quote(
                chain, signal["side"], float(signal["option"]["strike"])
            )
        except core.DhanAuthError:
            raise
        except Exception as exc:
            print(
                f"Could not refresh option quote for {signal['symbol']} "
                f"({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
        entry_quote = signal.get("option") or {}
        entry_ask = float(entry_quote.get("ask") or 0)
        current_bid = float((option_quote or {}).get("bid") or 0)
        if entry_ask > 0 and current_bid > 0:
            premium_ratio = current_bid / entry_ask
            if premium_ratio <= PREMIUM_STOP_FRAC:
                reason = "premium stop"
                exit_price = float(future.iloc[0]["close"])  # underlying ref at first post-signal 5m bar
                exit_time = future.index[0]
                signal["premium_stop_ratio"] = round(premium_ratio, 3)

        if reason is None:
            for timestamp, candle in future.iterrows():
                high, low = float(candle["high"]), float(candle["low"])
                stop_hit = low <= stop if sign > 0 else high >= stop
                target_hit = high >= target if sign > 0 else low <= target
                if stop_hit:
                    reason = "underlying stop"
                    exit_price = min(stop, float(candle["open"])) if sign > 0 else max(stop, float(candle["open"]))
                elif target_hit:
                    reason = "underlying 2R target"
                    exit_price = target
                if reason:
                    exit_time = timestamp
                    break
        if reason is None and current.time() >= FORCED_EXIT:
            reason = "mandatory same-day exit"
            exit_time = future.index[-1]
            exit_price = float(future.iloc[-1]["close"])
        if reason is None:
            continue

        signal["status"] = "paper_exit_signal_sent"
        signal["exit_reason"] = reason
        signal["exit_time"] = exit_time.isoformat() if exit_time is not None else None
        signal["underlying_exit_reference"] = round(float(exit_price), 2)
        signal["option_exit_quote"] = option_quote
        changed = True
        token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            bid_ask = (
                f"{option_quote['bid']:.2f}/{option_quote['ask']:.2f}"
                if option_quote and option_quote["bid"] and option_quote["ask"]
                else "unavailable"
            )
            core.send_telegram_message(
                token,
                chat_id,
                f"[15M SAME-DAY LONG-OPTION PAPER EXIT]\n"
                f"{signal['symbol']} {signal['side']} | {reason} at "
                f"{signal['exit_time']}\n"
                f"Underlying exit reference {signal['underlying_exit_reference']:.2f} | "
                f"option bid/ask {bid_ask}\n"
                "Paper exit estimate only; no order was placed and this is not a verified fill.",
            )
        exited.append(signal)

    if changed:
        path.write_text(
            "".join(json.dumps(signal) + "\n" for signal in signals),
            encoding="utf-8",
        )
    return exited
