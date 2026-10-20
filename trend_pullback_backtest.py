#!/usr/bin/env python3
"""Historical trend-pullback backtest using Dhan 5-minute candles and rolling options.

Example:
    python trend_pullback_backtest.py --start 2025-01-01 --end 2025-03-31 --symbols RELIANCE,HDFCBANK

Signals use completed candles and enter at the next candle open. Historical option
filters use rolling ATM option premium/OI and an approximate OI wall; historical
bid/ask spreads and lot-budget checks are unavailable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

import backtest as historical
import breakout_scanner as core
import trend_pullback as strategy


IST = core.IST
OUT = core.OUT


def _historical_option_check(result: dict, options: dict[str, pd.DataFrame],
                             timestamp: pd.Timestamp) -> dict:
    """Apply the trend strategy's historical premium, OI, and runway gates."""
    side = result["side"]
    entry = float(result["plan"]["entry"])
    risk = float(result["plan"]["risk_per_share"])
    spot = float(result["ltp"])
    option = historical._option_snapshot(
        options, side, timestamp, entry, risk, spot
    )
    if not option["available"]:
        return {**option, "runway_r": None, "passes": False}

    sign = 1 if side == "CE" else -1
    wall = option["wall"]
    runway = sign * (wall - entry) / risk if wall is not None and risk > 0 else None
    return {
        **option,
        "runway_r": round(runway, 4) if runway is not None else None,
        "passes": bool(
            runway is not None
            and runway >= strategy.MIN_RUNWAY_R
            and option["option_liquid"]
        ),
    }


def _execute_trade(signal: dict, bars: pd.DataFrame, slippage_bps: float,
                   cost_bps: float) -> dict | None:
    """Simulate next-open entry, partial T1, breakeven stop, T2, and 15:15 exit."""
    entry_index = signal["signal_index"] + 1
    if entry_index >= len(bars):
        return None

    sign = 1 if signal["side"] == "CE" else -1
    slip = slippage_bps / 10000
    entry_bar = bars.iloc[entry_index]
    raw_entry = float(entry_bar["open"])
    entry = raw_entry * (1 + sign * slip)
    stop = float(signal["plan"]["stop"])
    t1 = float(signal["plan"]["t1"])
    t2 = float(signal["plan"]["t2"])
    risk = sign * (entry - stop)
    if risk <= 0 or sign * (t1 - entry) <= 0 or sign * (t2 - t1) <= 0:
        return None

    future = bars.iloc[entry_index:].loc[
        lambda frame: frame.index.time <= historical.EXIT_LAST_BAR_START
    ]
    if future.empty:
        return None

    remaining = 1.0
    realized = 0.0
    costs = entry * cost_bps / 10000
    t1_taken = False
    exit_reason = "eod"
    exit_fills = []

    def close_piece(fraction: float, raw_exit: float, timestamp: pd.Timestamp,
                    reason: str) -> float:
        nonlocal costs
        fill = raw_exit * (1 - sign * slip)
        costs += fraction * fill * cost_bps / 10000
        exit_fills.append({
            "bar_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "fraction": fraction,
            "reason": reason,
        })
        return fraction * (fill - entry) * sign

    for timestamp, bar in future.iterrows():
        high, low = float(bar["high"]), float(bar["low"])
        stop_hit = low <= stop if sign == 1 else high >= stop
        if stop_hit:
            opening = float(bar["open"])
            fill = min(stop, opening) if sign == 1 else max(stop, opening)
            exit_reason = "breakeven" if t1_taken else "stop"
            realized += close_piece(remaining, fill, timestamp, exit_reason)
            remaining = 0
            break

        if not t1_taken and (high >= t1 if sign == 1 else low <= t1):
            realized += close_piece(0.5, t1, timestamp, "t1")
            remaining = 0.5
            t1_taken = True
            stop = entry
            breakeven_hit = low <= stop if sign == 1 else high >= stop
            if breakeven_hit:
                opening = float(bar["open"])
                fill = min(stop, opening) if sign == 1 else max(stop, opening)
                realized += close_piece(remaining, fill, timestamp, "breakeven")
                remaining = 0
                exit_reason = "t1_then_breakeven_same_bar"
                break

        if remaining and t1_taken and (high >= t2 if sign == 1 else low <= t2):
            realized += close_piece(remaining, t2, timestamp, "t2")
            remaining = 0
            exit_reason = "t2"
            break

    if remaining:
        timestamp = future.index[-1]
        realized += close_piece(remaining, float(future["close"].iloc[-1]),
                                timestamp, "eod")

    net_pnl = realized - costs
    return {
        "symbol": signal["symbol"],
        "side": signal["side"],
        "date": signal["date"],
        "signal_time": signal["signal_time"],
        "entry_time": entry_bar.name.strftime("%H:%M:%S"),
        "entry_bar_time": entry_bar.name.strftime("%Y-%m-%d %H:%M:%S"),
        "entry": round(entry, 4),
        "stop": round(float(signal["plan"]["stop"]), 4),
        "t1": round(t1, 4),
        "t2": round(t2, 4),
        "exit_reason": exit_reason,
        "pnl_per_share": round(net_pnl, 4),
        "r_multiple": round(net_pnl / risk, 4),
        "risk_per_share": round(risk, 4),
        "historical_option": signal["option"],
        "exit_fills": exit_fills,
    }


def _scenario_trades(data: dict, start: dt.date, end: dt.date,
                     slippage_bps: float, cost_bps: float) -> list[dict]:
    signals: list[tuple[pd.Timestamp, dict]] = []
    for symbol, bundle in data.items():
        if symbol == "NIFTY":
            continue
        intraday = bundle["intraday"].sort_index()
        options = bundle.get("options", {})
        session_days = sorted({day for day in intraday.index.date if start <= day <= end})
        for day in session_days:
            day_bars = intraday.loc[historical._day_key(intraday.index, day)].between_time(
                "09:15", "15:30"
            )
            if len(day_bars) < 4:
                continue

            for index in range(3, len(day_bars)):
                candle_time = day_bars.index[index]
                decision_time = candle_time + dt.timedelta(minutes=5, seconds=20)
                if decision_time.time() > strategy.ENTRY_END:
                    continue

                nifty_bias = historical._nifty_bias(
                    data["NIFTY"]["intraday"], day, candle_time
                )
                history = intraday.loc[intraday.index <= candle_time]
                setup = strategy.evaluate_setup(
                    symbol,
                    history,
                    decision_time.to_pydatetime(),
                    {"bias": nifty_bias},
                )
                if setup["verdict"] != f"SETUP_{setup.get('side')}":
                    continue

                option = _historical_option_check(setup, options, candle_time)
                if not option["passes"]:
                    continue
                setup = {
                    **setup,
                    "date": str(day),
                    "signal_time": decision_time.strftime("%H:%M:%S"),
                    "signal_index": index,
                    "option": option,
                }
                signals.append((candle_time, setup))

    trades = []
    fired: set[tuple[str, str, str]] = set()
    for _, signal in sorted(signals, key=lambda item: (item[0], item[1]["symbol"])):
        key = (signal["symbol"], signal["side"], signal["date"])
        if key in fired:
            continue
        fired.add(key)
        bars = data[signal["symbol"]]["intraday"]
        day_bars = bars.loc[historical._day_key(
            bars.index, dt.date.fromisoformat(signal["date"])
        )]
        trade = _execute_trade(signal, day_bars, slippage_bps, cost_bps)
        if trade is None:
            continue
        trade["option_pnl"] = historical._option_pnl(
            signal, trade, data[signal["symbol"]].get("options", {}),
            slippage_bps, cost_bps,
        )
        trades.append(trade)
    return trades


def run(args: argparse.Namespace) -> dict:
    today = dt.datetime.now(IST).date()
    if args.end >= today:
        raise ValueError(f"end date must be before today ({today}) so incomplete candles are not included")
    if args.start > args.end:
        raise ValueError("--start must be on or before --end")
    if args.slippage_bps < 0 or args.cost_bps < 0:
        raise ValueError("slippage and cost basis points cannot be negative")

    dh = core.Dhan(*core.credentials())
    eq, _ = core.load_master(today)
    symbols = historical.resolve_symbols(args.symbols, eq)
    fetch_start = args.start - dt.timedelta(days=120)
    data = {}
    for symbol, sid in symbols.items():
        print(f"Loading {symbol} historical 5-minute candles...")
        data[symbol] = {
            "intraday": historical.fetch_intraday(
                dh, symbol, sid, fetch_start, args.end
            ),
        }
        print(f"Loading {symbol} historical option OI and premiums...")
        data[symbol]["options"] = historical.fetch_rolling_options(
            dh, symbol, sid, args.start, args.end
        )
        if data[symbol]["intraday"].empty:
            raise RuntimeError(f"Dhan returned no historical candles for {symbol}")

    data["NIFTY"] = {
        "intraday": historical.fetch_intraday(
            dh, "NIFTY", core.NIFTY_ID,
            args.start - dt.timedelta(days=5), args.end,
            seg="IDX_I", inst="INDEX",
        ),
    }
    if data["NIFTY"]["intraday"].empty:
        raise RuntimeError("Dhan returned no Nifty intraday candles; cannot evaluate Nifty alignment")

    trades = _scenario_trades(data, args.start, args.end, args.slippage_bps, args.cost_bps)
    result = {
        "strategy": strategy.STRATEGY,
        "period": {"start": str(args.start), "end": str(args.end)},
        "symbols": list(symbols),
        "assumptions": {
            "signal": "The live trend-pullback setup is evaluated on completed 5-minute candles only.",
            "entry": "Entry is at the next 5-minute candle open; trades with a gap invalidating the planned stop or target order are skipped.",
            "exits": "50% exits at the planned T1 (1R); the remaining stop moves to breakeven and exits at the planned T2 (2R), breakeven, or the 15:15 IST candle close.",
            "ambiguous_candles": "Stop is prioritized over target; after T1, breakeven is prioritized over T2 within the same candle.",
            "option_filters": "Signals require historical ATM premium > 0, OI > 0, and an OI wall at least 1R away. OI walls are estimated from rolling ATM +/-3 strikes.",
            "not_tested": [
                "historical bid/ask spread and live spread gate",
                "historical lot sizes and one-lot capital budget gate",
                "point-in-time index constituents",
            ],
            "option_pnl": "Fixed-strike option P&L is reported when option candles are available; underlying R multiples remain the primary result.",
            "universe_warning": "Only user-selected symbols are evaluated; historical membership/delisted symbols are not reconstructed.",
            "slippage_bps_per_fill": args.slippage_bps,
            "cost_bps_per_fill": args.cost_bps,
        },
        "metrics": historical._metrics(trades),
        "option_metrics": historical._option_metrics(trades),
        "trades": trades,
    }

    OUT.mkdir(exist_ok=True)
    output_path = OUT / f"trend_pullback_backtest_{args.start}_{args.end}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["metrics"]
    print(f"\nHistorical trend-pullback backtest | {args.start} through {args.end}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Trades: {metrics['trades']} | Win rate: {metrics['win_rate_pct']:.2f}% | "
          f"Total R: {metrics['total_r']:.2f} | Average R: {metrics['average_r']:.3f} | "
          f"Profit factor: {metrics['profit_factor']} | Max drawdown: {metrics['max_drawdown_r']:.2f}R")
    print(f"Saved detailed results: {output_path}")
    print("Historical spread and lot-budget gates are unavailable; option OI walls are limited to ATM +/-3.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest trend-pullback signals using historical Dhan candles and rolling options."
    )
    parser.add_argument("--start", required=True, type=historical.parse_date,
                        help="first session date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=historical.parse_date,
                        help="last session date, YYYY-MM-DD (before today)")
    parser.add_argument("--symbols", required=True, type=historical.parse_symbols,
                        help="comma-separated symbols, e.g. RELIANCE,HDFCBANK")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="adverse slippage per fill in basis points (default: 5)")
    parser.add_argument("--cost-bps", type=float, default=0.0,
                        help="estimated cost per fill in basis points (default: 0)")
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, core.DhanAuthError) as exc:
        parser.exit(1, f"Backtest failed: {exc}\n")


if __name__ == "__main__":
    main()
