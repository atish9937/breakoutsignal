"""Paper-only daily swing candidates for long stock options."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path

import pandas as pd

import breakout_scanner as core


MAX_HOLD_SESSIONS = 3
MIN_EXPIRY_DAYS = 7
MAX_OPTION_SPREAD_PCT = 5.0
MAX_CHAIN_CANDIDATES = 3
PULLBACK_ATR_TOLERANCE = 0.5
SWING_STOP_ATR_BUFFER = 0.15
MIN_PULLBACK_VOLUME_RATIO = 1.0
MIN_BREAKOUT_VOLUME_RATIO = 1.5
MIN_SCORE = 5


def _completed(candles: pd.DataFrame, interval: int, now: dt.datetime) -> pd.DataFrame:
    if candles.empty:
        return candles
    complete_through = pd.Timestamp(now).floor(f"{interval}min") - pd.Timedelta(minutes=interval)
    return candles.loc[candles.index <= complete_through]


def _check(checks: list[dict], name: str, current: str, criterion: str, passed: bool) -> None:
    checks.append({
        "gate": name,
        "current": current,
        "criterion": criterion,
        "passed": bool(passed),
    })


def evaluate_setups(
    symbol: str,
    daily: pd.DataFrame,
    nifty_daily: pd.DataFrame,
    as_of: dt.date,
) -> list[dict]:
    """Build long-call/put candidates using only daily bars through ``as_of``."""
    frame = daily.loc[daily.index.date <= as_of].sort_index()
    index = nifty_daily.loc[nifty_daily.index.date <= as_of].sort_index()
    if len(frame) < 60 or len(index) < 50:
        return []

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    ema20 = core.ema(close, 20)
    ema50 = core.ema(close, 50)
    atr = core.atr(frame, 14)
    if not math.isfinite(atr) or atr <= 0:
        return []
    index_ema20 = core.ema(index["close"].astype(float), 20).iloc[-1]
    index_close = float(index["close"].iloc[-1])
    nifty_bias = "UP" if index_close > index_ema20 else "DOWN"

    last_close = float(close.iloc[-1])
    previous_high = float(high.iloc[-2])
    previous_low = float(low.iloc[-2])
    average_volume = float(volume.iloc[-21:-1].mean())
    volume_ratio = float(volume.iloc[-1] / average_volume) if average_volume > 0 else 0.0
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    e20_slope = e20 - float(ema20.iloc[-6])
    rsi = core.rsi(close)
    last_date = frame.index[-1].date()
    recent = frame.tail(5)
    candidates = []

    for side, sign in (("CE", 1), ("PE", -1)):
        directional_trend = e20 > e50 if side == "CE" else e20 < e50
        slope_ok = e20_slope > 0 if side == "CE" else e20_slope < 0
        index_ok = nifty_bias == ("UP" if side == "CE" else "DOWN")
        rsi_ok = 50 <= rsi <= 75 if side == "CE" else 25 <= rsi <= 50

        checks: list[dict] = []
        _check(checks, "Daily EMA trend", f"EMA20 {e20:.2f}; EMA50 {e50:.2f}",
               "EMA20 above EMA50 for calls; below for puts", directional_trend)
        _check(checks, "EMA20 slope", f"{e20_slope:+.2f} over 5 sessions",
               "slope in trade direction", slope_ok)
        _check(checks, "Nifty daily trend", f"{index_close:.2f} vs EMA20 {index_ema20:.2f}",
               "index aligned with option direction", index_ok)
        _check(checks, "RSI14", f"{rsi:.1f}",
               "50–75 calls; 25–50 puts", rsi_ok)

        if side == "CE":
            pullback_touched = bool((recent["low"].astype(float) <=
                                     ema20.loc[recent.index] + PULLBACK_ATR_TOLERANCE * atr).any())
            pullback_confirmed = last_close > previous_high and last_close > e20
            breakout_confirmed = last_close > float(high.iloc[-21:-1].max())
        else:
            pullback_touched = bool((recent["high"].astype(float) >=
                                     ema20.loc[recent.index] - PULLBACK_ATR_TOLERANCE * atr).any())
            pullback_confirmed = last_close < previous_low and last_close < e20
            breakout_confirmed = last_close < float(low.iloc[-21:-1].min())

        extension_atr = sign * (last_close - e20) / atr
        pullback_extension_ok = extension_atr <= 1.5
        breakout_extension_ok = extension_atr <= 2.0
        pattern_checks = (
            ("Daily pullback and reversal", pullback_touched and pullback_confirmed,
             "EMA20 touched within 0.5 ATR in prior 5 sessions; close reverses beyond prior bar"),
            ("Daily range breakout", breakout_confirmed,
             "close beyond the previous 20-session high/low"),
        )
        for setup_type, pattern_ok, criterion in pattern_checks:
            local_checks = list(checks)
            _check(local_checks, setup_type, f"close {last_close:.2f}", criterion, pattern_ok)
            volume_minimum = (MIN_PULLBACK_VOLUME_RATIO if setup_type.startswith("Daily pullback")
                              else MIN_BREAKOUT_VOLUME_RATIO)
            volume_ok = volume_ratio >= volume_minimum
            _check(local_checks, "Relative volume", f"{volume_ratio:.2f}x 20-session average",
                   f">= {volume_minimum:.2f}x", volume_ok)
            extension_ok = (pullback_extension_ok if setup_type.startswith("Daily pullback")
                            else breakout_extension_ok)
            _check(local_checks, "Entry extension", f"{extension_atr:.2f} ATR beyond EMA20",
                   f"<= {1.5 if setup_type.startswith('Daily pullback') else 2.0:.1f} ATR",
                   extension_ok)
            passed_count = sum(check["passed"] for check in local_checks)
            _check(local_checks, "Candidate score", f"{passed_count}/{len(local_checks) - 1}",
                   f">= {MIN_SCORE} checks and all structural gates pass",
                   passed_count >= MIN_SCORE)
            structural = directional_trend and slope_ok and index_ok and rsi_ok
            if not (pattern_ok and structural and volume_ok and extension_ok
                    and passed_count >= MIN_SCORE):
                continue

            swing = recent.tail(5)
            if side == "CE":
                stop = float(swing["low"].astype(float).min()) - SWING_STOP_ATR_BUFFER * atr
                entry_trigger = float(high.iloc[-1])
                risk = entry_trigger - stop
            else:
                stop = float(swing["high"].astype(float).max()) + SWING_STOP_ATR_BUFFER * atr
                entry_trigger = float(low.iloc[-1])
                risk = stop - entry_trigger
            if risk <= 0:
                continue
            candidates.append({
                "symbol": symbol,
                "side": side,
                "setup": setup_type,
                "signal_date": last_date.isoformat(),
                "score": passed_count,
                "checks": local_checks,
                "nifty_bias": nifty_bias,
                "underlying": {
                    "close": round(last_close, 2),
                    "entry_trigger": round(entry_trigger, 2),
                    "stop": round(stop, 2),
                    "target_2r": round(entry_trigger + sign * 2 * risk, 2),
                    "risk": round(risk, 2),
                    "atr14": round(atr, 2),
                },
                "holding": {
                    "max_trading_sessions": MAX_HOLD_SESSIONS,
                    "weekend_carry": False,
                    "entry_validity": "Next trading session only; confirmation beyond the displayed trigger is required.",
                },
            })
    return candidates


def _expiry_at_least(expiries: list[str], as_of: dt.date) -> str | None:
    eligible = []
    for value in expiries:
        try:
            expiry = dt.date.fromisoformat(str(value))
        except ValueError:
            continue
        if (expiry - as_of).days >= MIN_EXPIRY_DAYS:
            eligible.append(expiry)
    return str(min(eligible)) if eligible else None


def _option_candidates(
    chain: dict,
    side: str,
    underlying_price: float,
    symbol: str,
    expiry: str,
    master: pd.DataFrame,
) -> list[dict]:
    option_key = "ce" if side == "CE" else "pe"
    rows = []
    for raw_strike, legs in (chain.get("oc") or {}).items():
        try:
            strike = float(raw_strike)
        except (TypeError, ValueError):
            continue
        contract = (legs or {}).get(option_key) or {}
        if not contract:
            continue
        is_itm = strike < underlying_price if side == "CE" else strike > underlying_price
        rows.append((strike, is_itm, contract))
    if not rows:
        return []

    atm = min(rows, key=lambda row: abs(row[0] - underlying_price))
    itm_rows = [row for row in rows if row[1]]
    itm = min(itm_rows, key=lambda row: abs(row[0] - underlying_price)) if itm_rows else None
    chosen = [("ATM", atm)]
    if itm and itm[0] != atm[0]:
        chosen.append(("slightly_ITM", itm))

    candidates = []
    for label, (strike, _is_itm, quote) in chosen:
        bid = float(quote.get("top_bid_price") or 0)
        ask = float(quote.get("top_ask_price") or 0)
        premium = float(quote.get("last_price") or 0)
        spread_pct = (100 * (ask - bid) / ((ask + bid) / 2)
                      if bid > 0 and ask >= bid else None)
        lot, security_id = core.lot_and_id(master, symbol, expiry, strike, side)
        checks = {
            "premium_positive": premium > 0,
            "open_interest_positive": float(quote.get("oi") or 0) > 0,
            "volume_positive": float(quote.get("volume") or 0) > 0,
            "bid_ask_available": spread_pct is not None,
            "spread_within_limit": spread_pct is not None and spread_pct <= MAX_OPTION_SPREAD_PCT,
            "security_id_available": bool(security_id),
            "lot_size_available": bool(lot),
        }
        candidates.append({
            "selection": label,
            "strike": strike,
            "security_id": security_id,
            "premium": premium,
            "bid": bid or None,
            "ask": ask or None,
            "spread_pct": round(spread_pct, 3) if spread_pct is not None else None,
            "open_interest": quote.get("oi"),
            "volume": quote.get("volume"),
            "lot_size": lot,
            "checks": checks,
            "passes": all(checks.values()),
        })
    return candidates


def scan_daily_candidates(dh, master: pd.DataFrame, universe: dict[str, int],
                          as_of: dt.date, output_dir: Path) -> dict:
    """Fetch daily data, rank candidates, validate option liquidity, and save paper results."""
    nifty = dh.daily(core.NIFTY_ID, seg="IDX_I", inst="INDEX", days=180)
    if nifty.empty:
        raise RuntimeError("Dhan returned no daily Nifty data; swing candidates cannot be checked")

    candidates = []
    errors = []
    for symbol, security_id in universe.items():
        try:
            daily = dh.daily(security_id, days=180)
            if daily.empty:
                errors.append(f"{symbol}: no daily underlying candles")
                continue
            candidates.extend(evaluate_setups(symbol, daily, nifty, as_of))
        except core.DhanAuthError:
            raise
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
    candidates.sort(key=lambda item: (-item["score"], item["symbol"], item["side"]))
    deduplicated = {}
    for candidate in candidates:
        deduplicated.setdefault((candidate["symbol"], candidate["side"]), candidate)

    finalized = []
    for candidate in list(deduplicated.values())[:MAX_CHAIN_CANDIDATES]:
        candidate["underlying_security_id"] = int(universe[candidate["symbol"]])
        try:
            expiry_list = dh.expiries(universe[candidate["symbol"]])
            expiry = _expiry_at_least(expiry_list, as_of)
            if not expiry:
                candidate["option_error"] = (
                    f"No stock-option expiry at least {MIN_EXPIRY_DAYS} calendar days away"
                )
                finalized.append(candidate)
                continue
            chain = dh.option_chain(universe[candidate["symbol"]], expiry)
            options = _option_candidates(
                chain, candidate["side"], candidate["underlying"]["close"],
                candidate["symbol"], expiry, master,
            )
            passing = [option for option in options if option["passes"]]
            selected = min(
                passing,
                key=lambda option: (
                    option["spread_pct"],
                    -float(option["open_interest"] or 0),
                    -float(option["volume"] or 0),
                ),
                default=None,
            )
            candidate["expiry"] = expiry
            candidate["option_candidates"] = options
            candidate["selected_option"] = selected
            if selected:
                candidate["entry_option_limit_reference"] = selected["ask"]
                candidate["quote_note"] = (
                    "Option-chain quotes are a snapshot; refresh bid/ask and revalidate "
                    "the underlying trigger before any decision."
                )
            else:
                candidate["option_error"] = (
                    "Neither ATM nor slightly ITM contract passed all premium, OI, "
                    "volume, spread, security-ID, and lot-size checks"
                )
            finalized.append(candidate)
        except core.DhanAuthError:
            raise
        except Exception as exc:
            candidate["option_error"] = f"{type(exc).__name__}: {exc}"
            finalized.append(candidate)

    report = {
        "strategy": "multi_day_long_options_paper",
        "as_of": as_of.isoformat(),
        "max_hold_trading_sessions": MAX_HOLD_SESSIONS,
        "weekend_carry": False,
        "report_after_ist": "15:35",
        "candidate_count": len(candidates),
        "checked_option_chains": len(finalized),
        "candidates": finalized,
        "scan_errors": errors,
        "live_orders_placed": False,
        "limitations": [
            "Paper candidates only; no orders are submitted.",
            "A daily signal is not an entry fill; wait for the displayed underlying trigger.",
            "Candidate setup expires if the trigger is not confirmed during the next trading session.",
            "Backtests must compare underlying-based and option-premium-based exits before this is treated as a validated strategy.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"swing_option_candidates_{as_of.isoformat()}.json"
    report["output_path"] = str(path)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def send_daily_report(report: dict) -> None:
    """Send one headed paper notification for each option-ready daily candidate."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Swing candidates were saved but Telegram credentials are not configured.")
        return
    viable = [
        row for row in report["candidates"]
        if row.get("selected_option") is not None
    ]
    for row in viable:
        option = row["selected_option"]
        plan = row["underlying"]
        core.send_telegram_message(
            token,
            chat_id,
            f"[SWING NEXT-SESSION PAPER CANDIDATE]\n"
            f"{row['symbol']} {row['side']} | {row['setup']} | "
            f"score {row['score']} | signal date {row['signal_date']}\n"
            f"Next-session trigger {plan['entry_trigger']} | invalidation stop "
            f"{plan['stop']} | underlying 2R {plan['target_2r']}\n"
            f"{row['expiry']} {option['selection']} {option['strike']} | "
            f"bid/ask {option['bid']}/{option['ask']} | spread {option['spread_pct']}%\n"
            "Confirmation is required next trading session; candidate expires if not "
            "triggered. Maximum hold 3 sessions; exit by Friday close. Paper only; "
            "refresh live quotes and risk before any decision."
        )


def monitor_next_session(dh, master: pd.DataFrame, day: dt.date,
                         now: dt.datetime, output_dir: Path) -> list[dict]:
    """Confirm saved swing candidates only on their next available trading session."""
    current = now.astimezone(core.IST)
    if current.minute % 15 != 5 or current.time() > dt.time(15, 5):
        return []
    fired = []
    for path in sorted(output_dir.glob("swing_option_candidates_*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            signal_day = dt.date.fromisoformat(report["as_of"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Could not read swing candidate file {path.name}: {exc}", file=sys.stderr)
            continue
        if signal_day >= day or (day - signal_day).days > 10:
            continue

        changed = False
        for candidate in report.get("candidates", []):
            if candidate.get("next_session_status", "pending") != "pending":
                continue
            selected = candidate.get("selected_option")
            if not selected:
                candidate["next_session_status"] = "not_option_qualified"
                changed = True
                continue
            try:
                candles = dh.intraday(
                    int(candidate["underlying_security_id"]),
                    dt.datetime.combine(signal_day + dt.timedelta(days=1), dt.time(9, 15), core.IST),
                    current,
                    interval=15,
                )
            except core.DhanAuthError:
                raise
            except Exception as exc:
                candidate["next_session_error"] = f"{type(exc).__name__}: {exc}"
                changed = True
                continue
            candles = _completed(candles, 15, current)
            sessions = sorted({timestamp.date() for timestamp in candles.index
                               if timestamp.date() > signal_day})
            if not sessions:
                continue
            first_session = sessions[0]
            if first_session < day:
                candidate["next_session_status"] = "expired_untriggered"
                changed = True
                continue
            if first_session != day:
                continue

            sign = 1 if candidate["side"] == "CE" else -1
            trigger = float(candidate["underlying"]["entry_trigger"])
            eligible = candles.loc[candles.index.date == day]
            after_signal = eligible.loc[eligible.index > pd.Timestamp(
                signal_day, tz=core.IST
            )]
            hit = after_signal[
                after_signal["close"].astype(float).apply(
                    lambda close: close >= trigger if sign > 0 else close <= trigger
                )
            ]
            if hit.empty:
                if current.time() >= dt.time(15, 5):
                    candidate["next_session_status"] = "expired_untriggered"
                    changed = True
                continue

            try:
                security_id = int(candidate["underlying_security_id"])
                chain = dh.option_chain(security_id, candidate["expiry"])
                options = _option_candidates(
                    chain, candidate["side"], float(candidate["underlying"]["close"]),
                    candidate["symbol"], candidate["expiry"], master,
                )
                refreshed = next(
                    (
                        option for option in options
                        if option["strike"] == selected["strike"] and option["passes"]
                    ),
                    None,
                )
                if not refreshed:
                    candidate["next_session_status"] = "triggered_option_recheck_failed"
                    candidate["next_session_error"] = (
                        "Trigger confirmed, but selected contract no longer passed live "
                        "spread/volume/OI/contract checks"
                    )
                    changed = True
                    continue
                candidate["next_session_status"] = "trigger_confirmed"
                candidate["confirmed_at"] = hit.index[0].isoformat()
                candidate["refreshed_option"] = refreshed
                changed = True
                token = os.getenv("TELEGRAM_BOT_TOKEN")
                chat_id = os.getenv("TELEGRAM_CHAT_ID")
                if token and chat_id:
                    core.send_telegram_message(
                        token,
                        chat_id,
                        f"[SWING NEXT-SESSION ENTRY CONFIRMATION — PAPER]\n"
                        f"{candidate['symbol']} {candidate['side']} | "
                        f"{candidate['setup']} | {candidate['expiry']} "
                        f"{refreshed['selection']} {refreshed['strike']}\n"
                        f"15m trigger confirmed at {candidate['confirmed_at']} "
                        f"(underlying close {float(hit.iloc[0]['close']):.2f}; "
                        f"trigger {trigger:.2f}).\n"
                        f"Current option bid/ask {refreshed['bid']}/{refreshed['ask']}; "
                        f"spread {refreshed['spread_pct']}%.\n"
                        "This is a paper confirmation, not an order/fill. Recheck live "
                        "quote, premium risk and remaining holding window before acting."
                    )
                fired.append(candidate)
            except core.DhanAuthError:
                raise
            except Exception as exc:
                candidate["next_session_error"] = f"{type(exc).__name__}: {exc}"
                changed = True
        if changed:
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return fired
