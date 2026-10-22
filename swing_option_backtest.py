#!/usr/bin/env python3
"""Compare paper-only daily swing option exits using Dhan historical candles.

Example:
    python swing_option_backtest.py --start 2025-01-01 --end 2025-06-30 \
        --symbols RELIANCE,HDFCBANK

Historical rolling-option candles do not contain bid/ask spreads or verified fills.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math

import pandas as pd

import backtest as historical
import breakout_scanner as core
import swing_option_strategy as strategy


IST = core.IST
OPTION_TYPES = {"CE": "CALL", "PE": "PUT"}
EXIT_MODELS = ("underlying_atr", "option_premium")
CONTRACT_TYPES = ("ATM", "slightly_ITM")


def _row_at_or_before(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    if frame.empty:
        return None
    available = frame.loc[frame.index <= timestamp]
    if available.empty:
        return None
    row = available.iloc[-1]
    return row if row.name.date() == timestamp.date() else None


def _option_choices(
    options: dict[str, pd.DataFrame],
    side: str,
    timestamp: pd.Timestamp,
    underlying_price: float,
) -> dict[str, dict]:
    option_type = OPTION_TYPES[side]
    by_strike: dict[float, dict] = {}
    for offset in range(-3, 4):
        key = f"{option_type}_ATM{offset:+d}" if offset else f"{option_type}_ATM"
        row = _row_at_or_before(options.get(key, pd.DataFrame()), timestamp)
        if row is None:
            continue
        strike = float(row["strike"])
        premium = float(row["close"])
        oi = float(row.get("oi", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        if premium <= 0 or oi <= 0 or volume <= 0:
            continue
        by_strike[strike] = {
            "strike": strike,
            "premium": premium,
            "oi": oi,
            "volume": volume,
            "option_type": option_type,
        }
    if not by_strike:
        return {}

    atm = min(by_strike.values(), key=lambda item: abs(item["strike"] - underlying_price))
    atm["selection"] = "ATM"
    result = {"ATM": atm}
    itm = [
        item for item in by_strike.values()
        if (item["strike"] < underlying_price if side == "CE"
            else item["strike"] > underlying_price)
    ]
    if itm:
        result["slightly_ITM"] = min(
            itm, key=lambda item: abs(item["strike"] - underlying_price)
        )
        result["slightly_ITM"]["selection"] = "slightly_ITM"
    return result


def _price_at(
    options: dict[str, pd.DataFrame],
    option_type: str,
    strike: float,
    timestamp: pd.Timestamp,
) -> pd.Series | None:
    for offset in range(-3, 4):
        key = f"{option_type}_ATM{offset:+d}" if offset else f"{option_type}_ATM"
        row = options.get(key, pd.DataFrame()).loc[
            options.get(key, pd.DataFrame()).index == timestamp
        ]
        if row.empty:
            continue
        quote = row.iloc[-1]
        if math.isclose(float(quote["strike"]), strike, abs_tol=0.01):
            return quote
    return None


def _future_bars(
    bars: pd.DataFrame,
    entry_timestamp: pd.Timestamp,
    max_sessions: int = strategy.MAX_HOLD_SESSIONS,
) -> pd.DataFrame:
    future = bars.loc[bars.index >= entry_timestamp].sort_index()
    session_dates = list(dict.fromkeys(future.index.date))
    friday = next(
        (index for index, session_day in enumerate(session_dates)
         if session_day.weekday() == 4),
        None,
    )
    if friday is not None:
        session_dates = session_dates[:friday + 1]
    allowed = set(session_dates[:max_sessions])
    return future.loc[[timestamp.date() in allowed for timestamp in future.index]]


def _execute_candidate(
    candidate: dict,
    underlying: pd.DataFrame,
    options: dict[str, pd.DataFrame],
    contract: dict,
    exit_model: str,
    slippage_bps: float = 5.0,
    cost_bps: float = 0.0,
) -> dict | None:
    if exit_model not in EXIT_MODELS:
        raise ValueError(f"unknown exit model: {exit_model}")
    signal_day = dt.date.fromisoformat(candidate["signal_date"])
    post_signal = underlying.loc[underlying.index.date > signal_day].sort_index()
    if post_signal.empty:
        return None
    entry_day = post_signal.index[0].date()
    entry_bars = post_signal.loc[post_signal.index.date == entry_day]
    if entry_bars.empty:
        return None
    trigger = float(candidate["underlying"]["entry_trigger"])
    sign = 1 if candidate["side"] == "CE" else -1
    entry_index = None
    for position in range(len(entry_bars) - 1):
        bar = entry_bars.iloc[position]
        confirmed = (
            float(bar["close"]) >= trigger if sign > 0
            else float(bar["close"]) <= trigger
        )
        if confirmed:
            entry_index = position + 1
            break
    if entry_index is None:
        return None
    entry_timestamp = entry_bars.index[entry_index]
    entry_underlying = float(entry_bars["open"].iloc[entry_index])
    stop = float(candidate["underlying"]["stop"])
    if sign * (entry_underlying - stop) <= 0:
        return None

    option_entry = _price_at(
        options, contract["option_type"], contract["strike"], entry_timestamp
    )
    if option_entry is None or float(option_entry["open"]) <= 0:
        return None
    slip = slippage_bps / 10000
    entry_premium = float(option_entry["open"]) * (1 + slip)
    option_stop = entry_premium * 0.70
    option_target = entry_premium * 1.60

    underlying_risk = abs(entry_underlying - stop)
    underlying_target = entry_underlying + sign * 2 * underlying_risk
    future = _future_bars(underlying, entry_timestamp)
    if future.empty:
        return None

    exit_reason = "max_holding_period"
    raw_exit = None
    exit_time = None
    for timestamp, bar in future.iterrows():
        quote = _price_at(options, contract["option_type"], contract["strike"], timestamp)
        if quote is None:
            return None
        option_open = float(quote["open"])
        option_high = float(quote["high"])
        option_low = float(quote["low"])
        option_close = float(quote["close"])
        if exit_model == "option_premium":
            stop_hit = option_low <= option_stop
            target_hit = option_high >= option_target
            if stop_hit:
                raw_exit = min(option_stop, option_open)
                exit_reason = "premium_stop"
            elif target_hit:
                raw_exit = option_target
                exit_reason = "premium_target"
        else:
            high = float(bar["high"])
            low = float(bar["low"])
            stop_hit = low <= stop if sign > 0 else high >= stop
            target_hit = high >= underlying_target if sign > 0 else low <= underlying_target
            if stop_hit:
                open_price = float(bar["open"])
                raw_exit = option_close
                exit_reason = "underlying_stop"
                if (open_price < stop if sign > 0 else open_price > stop):
                    exit_reason = "underlying_stop_gap"
            elif target_hit:
                raw_exit = option_close
                exit_reason = "underlying_2r_target"
        if raw_exit is not None:
            exit_time = timestamp
            break
    if raw_exit is None:
        timestamp = future.index[-1]
        quote = _price_at(options, contract["option_type"], contract["strike"], timestamp)
        if quote is None:
            return None
        raw_exit = float(quote["close"])
        exit_time = timestamp

    exit_premium = raw_exit * (1 - slip)
    gross_return = (exit_premium - entry_premium) / entry_premium
    net_return = gross_return - 2 * cost_bps / 10000
    return {
        "symbol": candidate["symbol"],
        "side": candidate["side"],
        "setup": candidate["setup"],
        "signal_date": candidate["signal_date"],
        "entry_time": entry_timestamp.isoformat(),
        "exit_time": exit_time.isoformat(),
        "contract_selection": contract["selection"],
        "strike": contract["strike"],
        "exit_model": exit_model,
        "exit_reason": exit_reason,
        "entry_premium": round(entry_premium, 4),
        "exit_premium": round(exit_premium, 4),
        "return_pct": round(net_return * 100, 4),
        "holding_sessions": len(set(future.index.date[:future.index.get_loc(exit_time) + 1])),
    }


def _metrics(trades: list[dict]) -> dict:
    values = [float(trade["return_pct"]) for trade in trades]
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": len(values),
        "wins": sum(value > 0 for value in values),
        "win_rate_pct": round(100 * sum(value > 0 for value in values) / len(values), 2)
        if values else None,
        "average_return_pct": round(sum(values) / len(values), 4) if values else None,
        "total_return_pct_sum": round(sum(values), 4),
        "profit_factor": round(gains / losses, 4) if losses else ("inf" if gains else None),
        "max_drawdown_pct_sum": round(drawdown, 4),
    }


def _scenario_trades(
    data: dict,
    start: dt.date,
    end: dt.date,
    setup_type: str,
    contract_type: str,
    exit_model: str,
    slippage_bps: float,
    cost_bps: float,
) -> list[dict]:
    signals = []
    for symbol, bundle in data.items():
        if symbol == "NIFTY":
            continue
        daily = bundle["daily"].sort_index()
        nifty = data["NIFTY"]["daily"].sort_index()
        for day in sorted({value for value in daily.index.date if start <= value <= end}):
            setups = strategy.evaluate_setups(symbol, daily, nifty, day)
            for candidate in setups:
                if candidate["setup"] != setup_type:
                    continue
                timestamp = pd.Timestamp(day, tz=IST) + pd.Timedelta(hours=15, minutes=25)
                contracts = _option_choices(
                    bundle.get("options", {}), candidate["side"], timestamp,
                    float(candidate["underlying"]["close"]),
                )
                contract = contracts.get(contract_type)
                if not contract:
                    continue
                signals.append((timestamp, candidate, contract))

    trades = []
    skipped = 0
    next_available: dict[tuple[str, str], dt.date] = {}
    for _, candidate, contract in sorted(signals, key=lambda item: item[0]):
        key = (candidate["symbol"], candidate["side"])
        if next_available.get(key, dt.date.min) > dt.date.fromisoformat(candidate["signal_date"]):
            skipped += 1
            continue
        trade = _execute_candidate(
            candidate,
            data[candidate["symbol"]]["intraday"],
            data[candidate["symbol"]].get("options", {}),
            contract,
            exit_model,
            slippage_bps,
            cost_bps,
        )
        if not trade:
            skipped += 1
            continue
        trades.append(trade)
        next_available[key] = dt.date.fromisoformat(trade["exit_time"][:10])
    diagnostics = {
        "eligible_signals": len(signals),
        "simulated_trades": len(trades),
        "skipped_due_to_overlap_or_missing_trigger_or_option_bars": skipped,
    }
    return trades, diagnostics


def run(args: argparse.Namespace) -> dict:
    today = dt.datetime.now(IST).date()
    if args.end >= today:
        raise ValueError(f"end date must be before today ({today})")
    if args.start > args.end:
        raise ValueError("--start must be on or before --end")
    if args.slippage_bps < 0 or args.cost_bps < 0:
        raise ValueError("slippage and cost bps cannot be negative")

    dh = core.Dhan(*core.credentials())
    eq, _ = core.load_master(today)
    symbols = historical.resolve_symbols(args.symbols, eq)
    history_start = args.start - dt.timedelta(days=180)
    history_end = args.end + dt.timedelta(days=10)
    data = {}
    for symbol, security_id in symbols.items():
        print(f"Loading {symbol} daily, 5-minute, and rolling option history...")
        data[symbol] = {
            "daily": historical.fetch_daily(dh, symbol, security_id, history_start, args.end),
            "intraday": historical.fetch_intraday(
                dh, symbol, security_id, args.start, history_end
            ),
            "options": historical.fetch_rolling_options(
                dh, symbol, security_id, args.start, history_end
            ),
        }
        if data[symbol]["daily"].empty or data[symbol]["intraday"].empty:
            raise RuntimeError(f"Dhan returned insufficient historical candles for {symbol}")
    data["NIFTY"] = {
        "daily": historical.fetch_daily(
            dh, "NIFTY", core.NIFTY_ID, history_start, args.end,
            seg="IDX_I", inst="INDEX",
        ),
    }
    if data["NIFTY"]["daily"].empty:
        raise RuntimeError("Dhan returned no historical daily Nifty candles")

    comparisons = []
    for setup_type in ("Daily pullback and reversal", "Daily range breakout"):
        for contract_type in CONTRACT_TYPES:
            for exit_model in EXIT_MODELS:
                trades, diagnostics = _scenario_trades(
                    data, args.start, args.end, setup_type, contract_type,
                    exit_model, args.slippage_bps, args.cost_bps,
                )
                comparisons.append({
                    "setup": setup_type,
                    "contract": contract_type,
                    "exit_model": exit_model,
                    "metrics": _metrics(trades),
                    "coverage": diagnostics,
                    "trades": trades,
                })
    result = {
        "strategy": "multi_day_long_options_paper",
        "period": {"start": str(args.start), "end": str(args.end)},
        "symbols": list(symbols),
        "assumptions": {
            "signal": "Daily EMA20/EMA50 trend and five-session EMA slope; daily pullback/reversal or 20-session close breakout; daily RSI, relative volume, extension, and Nifty trend gates.",
            "entry": "Only the next trading session is eligible: require a 5-minute close beyond the signal-day high/low trigger, then enter at the next 5-minute bar open. If that session has no trigger, the candidate expires.",
            "max_holding": "Up to three trading sessions including entry; if holding reaches Friday, exit at Friday close and never carry over the weekend.",
            "underlying_exit": "Underlying stop at the signal's five-session swing extreme plus/minus 0.15 daily ATR; target is 2R. Option is marked at the same 5-minute candle close when underlying level is reached.",
            "premium_exit": "Long option stop at -30% and target at +60% from entry premium. If one bar touches both, stop is assumed first.",
            "contract": "ATM and nearest one-strike ITM evaluated separately; ATM/ITM choice uses prior-session rolling-option OI and volume.",
            "data_limitations": [
                "Dhan rolling option candles have no historical bid/ask, so spread-based contract choice and fill checks cannot be backtested.",
                "Rolling-option historical expiry selection does not prove that every sample had at least seven calendar days to expiry.",
                "Five-minute OHLC cannot establish intrabar sequence or actual option fills; underlying-trigger exits use that bar's option close.",
                "Results are historical paper simulations and do not establish future profitability.",
            ],
            "slippage_bps_per_fill": args.slippage_bps,
            "cost_bps_per_fill": args.cost_bps,
        },
        "comparisons": comparisons,
    }
    output = core.OUT / f"swing_option_backtest_{args.start}_{args.end}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved swing-options comparison to {output}")
    for item in comparisons:
        metrics = item["metrics"]
        print(
            f"{item['setup']} | {item['contract']} | {item['exit_model']} | "
            f"{metrics['trades']} trades | win {metrics['win_rate_pct']}% | "
            f"average {metrics['average_return_pct']}%"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare daily swing long-option exit models using historical Dhan data."
    )
    parser.add_argument("--start", required=True, type=historical.parse_date)
    parser.add_argument("--end", required=True, type=historical.parse_date)
    parser.add_argument("--symbols", required=True, type=historical.parse_symbols)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, core.DhanAuthError) as exc:
        parser.exit(1, f"Backtest failed: {exc}\n")


if __name__ == "__main__":
    main()
