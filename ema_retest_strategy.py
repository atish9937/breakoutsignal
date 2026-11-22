#!/usr/bin/env python3
"""Recommendations-only intraday 20 EMA breakout-retest scanner for NIFTY index.

Pattern (5-minute, long-only — the only combination with a positive-expectancy
option-trading backtest; see ema_retest_backtest.py / smc_results/).

  BREAKOUT bar (candle B):
    green (close > open), opens below the 20 EMA, closes above the 20 EMA,
    body/range >= 0.60 (reject doji-ish crosses)

  RETEST bar (candle B+1 or B+2):
    red (close < open), closes above the 20 EMA
    No intermediate bar between breakout and retest closes below the EMA

  TRIGGER bar (candle after the retest — the just-closed scan bar):
    high > retest bar's high AND no bar between retest and trigger closed
    below the EMA

  Risk-band filter:
    8 <= (entry - EMA_at_trigger) <= 25 (points). Rejects both the chop-tight
    and lottery-wide extremes that killed the raw video rules on options.

  EMA slope filter:
    (EMA[trigger] - EMA[trigger - 10 bars]) > 0. The 20 EMA must be rising
    over the previous 50 minutes. Rejects flat/sideways regimes where the
    EMA gets re-tagged repeatedly without follow-through. Lifted the backtest
    PF from 1.82 to 2.27 and cut max drawdown by ~60% at the cost of taking
    ~60% fewer trades.

Trade plan:
  entry  = retest bar's high + 1 point slippage
  stop   = EMA value at trigger bar (soft stop: exit on 5m close below EMA)
  t1     = entry + 1.5R  (scale out 50% here)
  t2     = entry + 3R    (ride the rest; or EMA-close trail)

Options:
  Reuses breakout_scanner's pick_option + strike-walker + liquidity gates
  via `attach_option_check` (same gates as liquidity_sweep). CE-only.

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


STRATEGY = "ema_retest"
RUN_START = dt.time(9, 25)
RUN_END = dt.time(14, 30)        # need ~50 min for 1:3 target before 15:20 flat
ENTRY_START = dt.time(9, 25)
ENTRY_END = dt.time(14, 30)
SESSION_FLAT = dt.time(15, 20)

# --- Strategy parameters (from ema_retest_backtest.py `best_combo`) -----------
EMA_LEN = 20
RR = 3.0
SCALE_R = 1.5
MAX_RETEST_BARS = 2              # retest must be within 2 bars after breakout
BODY_PCT_MIN = 0.60              # breakout bar body/range gate
MIN_RISK_POINTS = 8.0            # risk band lower bound (points to EMA)
MAX_RISK_POINTS = 25.0           # risk band upper bound
SLIPPAGE_POINTS = 1.0            # entry slippage above retest high
EMA_SLOPE_BARS = 10              # EMA slope lookback (bars) — slope must be > 0

# --- Option liquidity gates (mirror liquidity_sweep / trend_pullback) --------
OPTION_MIN_PREMIUM = 2.0
OPTION_MIN_OI = 1000
OPTION_MIN_DELTA = 0.25
OPTION_MIN_DTE_DAYS = 1          # weekly options — allow near-expiry
MAX_OPTION_SPREAD_PCT = 5.0
OPTION_REQUIRE_BID = True
OPTION_DELTA_FALLBACK = 0.5

# Universe: NIFTY index only. BANKNIFTY can be added once a live-cycle sample
# validates the setup on it.
INDICES = {
    "NIFTY": {"sid": core.NIFTY_ID, "underlying_seg": "IDX_I", "option_seg": "NSE_FNO"},
}

OUT = core.OUT


# ---------------------------------------------------------------------------
# Setup evaluation — pure function, no API calls. Covered by test_ema_retest.
# ---------------------------------------------------------------------------
def _record(checks: list, name: str, current, criterion, passed: bool) -> None:
    checks.append({"gate": name, "current": current,
                   "criterion": criterion, "passed": bool(passed)})


def _is_green(bar: pd.Series) -> bool:
    return bool(bar["close"] > bar["open"])


def _is_red(bar: pd.Series) -> bool:
    return bool(bar["close"] < bar["open"])


def _body_pct(bar: pd.Series) -> float:
    rng = float(bar["high"] - bar["low"])
    if rng <= 0:
        return 0.0
    return abs(float(bar["close"] - bar["open"])) / rng


def evaluate_setup(symbol: str, candles: pd.DataFrame, now: dt.datetime) -> dict:
    """Decide whether the just-closed 5m bar is a valid trigger for a long setup.

    Returns a result dict following the repo convention:
      verdict  -> SETUP_CE | WATCH_CE | SKIP
      side     -> "CE" | None
      plan     -> {entry, stop, t1, t2, risk_per_share, ...} when SETUP_CE
      mechanical_checks, fails -> transparent pass/fail list
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
                "at least EMA_LEN completed bars today", False)
        result["verdict"] = "SKIP"
        result["fails"].append("no timestamped candle data")
        return result

    today = candles[candles.index.date == now.date()]
    # Need at least: breakout bar + up to 2 retest-window bars + trigger bar +
    # enough history for the 20 EMA to be meaningful.
    enough = len(candles) >= EMA_LEN + 3 and len(today) >= 5
    _record(checks, "Candle history",
            f"{len(candles)} total; {len(today)} today",
            f">={EMA_LEN + 3} total and >=5 today", enough)
    if not enough:
        result["verdict"] = "SKIP"
        result["fails"].append("not enough completed candles")
        return result

    closes = candles["close"]
    ema = core.ema(closes, EMA_LEN)

    # Index of the just-closed (trigger) bar inside the day slice.
    today_indexer = candles.index.get_indexer(today.index)
    trigger_i = int(today_indexer[-1])      # position in full `candles`
    trigger_bar = candles.iloc[trigger_i]
    ema_at_trigger = float(ema.iloc[trigger_i])

    result["ltp"] = core.r2(float(trigger_bar["close"]))
    result["ema_at_trigger"] = core.r2(ema_at_trigger)

    # Walk backwards: retest bar is 1 or 2 positions back; breakout bar is 1 or
    # 2 positions before the retest. Scan all four (retest_offset, bo_offset)
    # combinations and take the first that produces a valid setup.
    valid: dict | None = None
    for retest_offset in (1, 2):
        retest_i = trigger_i - retest_offset
        if retest_i - 2 < 0:
            continue
        retest_bar = candles.iloc[retest_i]
        retest_ema = float(ema.iloc[retest_i])
        if not (_is_red(retest_bar) and retest_bar["close"] > retest_ema):
            continue
        # Trigger must take out the retest high
        if float(trigger_bar["high"]) <= float(retest_bar["high"]):
            continue
        for bo_offset in (1, 2):
            bo_i = retest_i - bo_offset
            if bo_i < 0:
                continue
            bo_bar = candles.iloc[bo_i]
            bo_ema = float(ema.iloc[bo_i])
            if not (_is_green(bo_bar) and bo_bar["open"] < bo_ema
                    and bo_bar["close"] > bo_ema):
                continue
            # Body quality (reject weak/doji breakouts)
            body = _body_pct(bo_bar)
            if body < BODY_PCT_MIN:
                continue
            # No intermediate bar (between breakout and retest, or between
            # retest and trigger) may close back below the EMA.
            invalid = False
            for mid in range(bo_i + 1, retest_i):
                if float(candles.iloc[mid]["close"]) < float(ema.iloc[mid]):
                    invalid = True
                    break
            if invalid:
                continue
            for mid in range(retest_i + 1, trigger_i):
                if float(candles.iloc[mid]["close"]) < float(ema.iloc[mid]):
                    invalid = True
                    break
            if invalid:
                continue
            valid = {
                "breakout_bar": bo_bar.name.isoformat(),
                "breakout_body_pct": core.r2(body),
                "retest_bar": retest_bar.name.isoformat(),
                "retest_high": core.r2(float(retest_bar["high"])),
                "trigger_bar": trigger_bar.name.isoformat(),
            }
            break
        if valid is not None:
            break

    _record(checks, "Breakout + retest + trigger",
            valid or "no matching 3-bar sequence",
            f"green breakout (body>={BODY_PCT_MIN:.0%}) -> red retest above EMA -> "
            "current bar breaks retest high",
            valid is not None)
    if valid is None:
        result["verdict"] = "SKIP"
        return result

    # EMA slope gate: reject sideways/declining-EMA regimes.
    if trigger_i - EMA_SLOPE_BARS >= 0:
        ema_lookback = float(ema.iloc[trigger_i - EMA_SLOPE_BARS])
        slope = ema_at_trigger - ema_lookback
    else:
        slope = None
    _record(checks, "EMA slope",
            f"{slope:+.2f} pts over {EMA_SLOPE_BARS} bars"
            if slope is not None else "insufficient history",
            f"slope > 0 over last {EMA_SLOPE_BARS} bars (EMA rising)",
            slope is not None and slope > 0)
    if slope is None or slope <= 0:
        result["side"] = "CE"
        result["fails"] = [c["gate"] for c in checks if not c["passed"]]
        result["verdict"] = "WATCH_CE"
        return result

    entry = float(valid["retest_high"]) + SLIPPAGE_POINTS
    risk = entry - ema_at_trigger
    _record(checks, "Risk band",
            f"entry-EMA = {risk:.2f} pts",
            f"{MIN_RISK_POINTS} <= risk <= {MAX_RISK_POINTS}",
            MIN_RISK_POINTS <= risk <= MAX_RISK_POINTS)
    if not (MIN_RISK_POINTS <= risk <= MAX_RISK_POINTS):
        result["side"] = "CE"
        result["fails"] = [c["gate"] for c in checks if not c["passed"]]
        result["verdict"] = "WATCH_CE"
        return result

    # Session cutoff
    entry_time_ok = ENTRY_START <= now.time() <= ENTRY_END
    _record(checks, "Entry window",
            now.strftime("%H:%M"),
            f"{ENTRY_START.strftime('%H:%M')}..{ENTRY_END.strftime('%H:%M')}",
            entry_time_ok)
    if not entry_time_ok:
        result["side"] = "CE"
        result["fails"] = [c["gate"] for c in checks if not c["passed"]]
        result["verdict"] = "WATCH_CE"
        return result

    plan = {
        "entry": core.r2(entry),
        "stop": core.r2(ema_at_trigger),
        "t1": core.r2(entry + SCALE_R * risk),
        "t2": core.r2(entry + RR * risk),
        "risk_per_share": core.r2(risk),
        "runway_r": RR,
        "scale_r": SCALE_R,
        "ema_slope_pts": core.r2(slope),
        "ema_slope_bars": EMA_SLOPE_BARS,
        **valid,
    }
    result.update({
        "side": "CE",
        "plan": plan,
        "verdict": "SETUP_CE",
    })
    return result


# ---------------------------------------------------------------------------
# Option attachment + Telegram
# ---------------------------------------------------------------------------
def attach_option_check(result: dict, master, expiry: str, chain: dict,
                        day: dt.date | None = None) -> dict:
    """Attach ATM-CE contract, liquidity gates, and lot-size for the signal.

    Delegates gate math to liquidity_sweep.attach_option_check so the gate
    thresholds stay consistent across strategies. Trading constants
    (MIN_PREMIUM, MIN_OI, etc.) are imported from liquidity_sweep so a tuning
    change there propagates to this strategy automatically.
    """
    import liquidity_sweep as ls
    # Forward to the shared implementation. Its attach_option_check reads the
    # module-level constants of its OWN module, which is what we want (one
    # place to tune option gates across all strategies).
    return ls.attach_option_check(result, master, expiry, chain, day=day)


def _notify(text: str) -> None:
    core.load_env()
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        core.send_telegram_message(token, chat_id, text)


def notify_signal(result: dict) -> None:
    plan = result["plan"]
    opt = result.get("option") or {}
    rr_t1 = SCALE_R          # 1:1.5 planned
    rr_t2 = RR               # 1:3 planned
    _notify(
        f"[EMA Retest] {result['verdict']} {result['symbol']} spot {result['ltp']}\n"
        f"20 EMA breakout + retest confirmed (body "
        f"{plan.get('breakout_body_pct', 0):.0%}, risk {plan['risk_per_share']} pts, "
        f"EMA slope {plan.get('ema_slope_pts', 0):+.1f}/{plan.get('ema_slope_bars', 0)}b)\n"
        f"{opt.get('contract', 'option tbd')} [{opt.get('selection', '?')}] "
        f"~Rs{opt.get('premium', '?')} | lot {opt.get('lot_size', '?')}\n"
        f"Entry {plan['entry']} | stop {plan['stop']} | "
        f"T1 {plan['t1']} (1:{rr_t1:g} scale) | T2 {plan['t2']} (1:{rr_t2:g})\n"
        f"Planned R:R 1:{rr_t2:g} | approx. loss/lot Rs{plan.get('estimated_loss_per_lot', '?')}\n"
        f"Checks passed: {sum(c['passed'] for c in result['mechanical_checks'])}/"
        f"{len(result['mechanical_checks'])}\n"
        "Recommendations only; no order was placed. Scale 50% at T1; "
        "exit remainder on EMA close or T2."
    )


# ---------------------------------------------------------------------------
# Exit monitor — sends per-exit Telegram alerts when a tracked trade
# transitions out of 'tracking' status. Mirrors intraday_multiframe's
# monitor_positions() but reads from paper_trades (SQLite) rather than JSONL.
# ---------------------------------------------------------------------------
STATUS_LABELS = {
    "t1_hit": "T1 reached (scale level)",
    "t2_hit": "T2 reached",
    "stop_before_t1": "Stopped before T1",
    "t1_then_breakeven": "T1 reached; breakeven exit",
    "unresolved": "Unresolved at cutoff",
}


def _exit_notified_file(day: dt.date):
    return OUT / f"{STRATEGY}_exit_notified_{day}.jsonl"


def _load_notified_ids(day: dt.date) -> set[int]:
    path = _exit_notified_file(day)
    ids: set[int] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    ids.add(int(line))
                except ValueError:
                    continue
    return ids


def _mark_notified(day: dt.date, trade_id: int) -> None:
    path = _exit_notified_file(day)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{trade_id}\n")
        fh.flush()
        os.fsync(fh.fileno())


def notify_exit(row: dict) -> None:
    """Build and send the per-exit Telegram message for one paper_trades row."""
    entry = float(row["reference_entry"])
    stop = float(row["stop_price"])
    t1 = float(row["target_1"])
    t2 = float(row["target_2"])
    last_price = row.get("last_price")
    sign = 1 if row["side"] == "CE" else -1
    risk = (entry - stop) * sign
    status = row.get("status", "unresolved")
    status_label = STATUS_LABELS.get(status, status)

    # Realized R-multiple (signed): >0 if moved toward target, <0 if toward stop.
    realized_r = None
    if last_price is not None and risk > 0:
        move = (float(last_price) - entry) * sign
        realized_r = move / risk

    option_tag = ""
    if row.get("option_security_id"):
        option_tag = (
            f"{row.get('option_expiry') or ''} {row['side']} "
            f"@entry bid/ask {row.get('option_entry_bid') or 0:.2f}/"
            f"{row.get('option_entry_ask') or 0:.2f} | lot {row.get('option_lot_size') or '?'}\n"
        )

    lines = [
        f"[EMA Retest EXIT] {row['symbol']} {row['side']} | {status_label}",
        option_tag.rstrip() if option_tag else None,
        f"Entry {entry:.2f} | stop {stop:.2f} | T1 {t1:.2f} | T2 {t2:.2f}",
        (f"Last {float(last_price):.2f} | realized {realized_r:+.2f}R (risk {risk:.1f} pts)"
         if last_price is not None and realized_r is not None
         else f"Last price unavailable | risk {risk:.1f} pts"),
        "Paper exit; no order was placed and this is not a verified fill.",
    ]
    _notify("\n".join(line for line in lines if line))


def monitor_exits(ledger, day: dt.date, now: dt.datetime) -> list[dict]:
    """Send Telegram alerts for ema_retest paper_trades that just left
    'tracking' status. Idempotent — each trade is notified at most once
    per session via the on-disk notified-id log."""
    if ledger is None:
        return []
    # Dedupe file stops duplicate alerts if the runner is restarted mid-day.
    already_notified = _load_notified_ids(day)
    try:
        rows = ledger.execute(
            """
            SELECT * FROM paper_trades
            WHERE strategy = ? AND session_date = ? AND status != 'tracking'
            """,
            (STRATEGY, day.isoformat()),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        print(f"{STRATEGY} exit monitor DB read failed: {exc}", file=sys.stderr)
        return []

    newly_notified: list[dict] = []
    for row in rows:
        trade_id = int(row["id"])
        if trade_id in already_notified:
            continue
        row_dict = dict(row)
        try:
            notify_exit(row_dict)
        except Exception as exc:  # noqa: BLE001
            print(f"{STRATEGY} exit Telegram failed for id={trade_id}: {exc}",
                  file=sys.stderr)
            # Still mark notified so a broken Telegram doesn't flood on retry.
        try:
            _mark_notified(day, trade_id)
        except OSError as exc:
            print(f"{STRATEGY} exit-notified file write failed: {exc}",
                  file=sys.stderr)
            continue
        newly_notified.append(row_dict)
    return newly_notified


# ---------------------------------------------------------------------------
# Main scan — called by strategy_runner every 5-minute boundary
# ---------------------------------------------------------------------------
def scan(dh, master, prep: dict, universe: dict[str, int], day: dt.date,
         now: dt.datetime, fired: set[str]) -> dict:
    """One scan cycle. Matches the signature used by liquidity_sweep/ORB so
    strategy_runner can register this scanner alongside them."""
    results: list[dict] = []
    signal_path = OUT / f"{STRATEGY}_signals_{day}.jsonl"
    frm = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(9, 15), core.IST)

    def _fire_if_ready(result: dict) -> None:
        """Order: file-write -> fired.add -> notify (same as liquidity_sweep)."""
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
            print(f"{STRATEGY} signal file write failed: {exc}", file=sys.stderr)
            return
        fired.add(key)
        try:
            notify_signal(result)
        except Exception as exc:  # noqa: BLE001
            print(f"{STRATEGY} Telegram notify failed: {exc}", file=sys.stderr)

    for index_name, cfg in INDICES.items():
        sid = cfg["sid"]
        seg = cfg["underlying_seg"]
        try:
            candles = dh.intraday(sid, frm, now, seg=seg, inst="INDEX")
            candles = candles[candles.index <= pd.Timestamp(now) - pd.Timedelta(minutes=5)]
            result = evaluate_setup(index_name, candles, now)
            if result.get("verdict") == f"SETUP_{result.get('side')}":
                idx_expiry = _pick_index_expiry(dh, sid, seg, day, OPTION_MIN_DTE_DAYS)
                if not idx_expiry:
                    _record(result["mechanical_checks"], "Option expiry",
                            "no expiry at DTE threshold",
                            "valid weekly expiry available", False)
                    result["fails"].append("index option expiry unavailable")
                    result["verdict"] = f"WATCH_{result['side']}"
                else:
                    chain = dh.option_chain(sid, idx_expiry, seg=seg)
                    result = attach_option_check(result, master, idx_expiry, chain, day=day)
            result["underlying_segment"] = seg
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
        "candidates_checked": len(results),
        "results": results,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{STRATEGY}_latest.json").write_text(json.dumps(snapshot, indent=2))
    with open(OUT / f"{STRATEGY}_scan_{day}.jsonl", "a", encoding="utf-8") as stream:
        for r in results:
            stream.write(json.dumps(r) + "\n")

    counts = {prefix: sum(r["verdict"].startswith(prefix) for r in results)
              for prefix in ("BUY_", "WATCH_", "SETUP_", "SKIP")}
    print(f"\n[{snapshot['scan_time']}] EMA Retest | "
          f"BUY {counts['BUY_']} | WATCH {counts['WATCH_']} | "
          f"SETUP {counts['SETUP_']} | SKIP {counts['SKIP']}")
    for r in results:
        if r["verdict"].startswith(("BUY_", "WATCH_", "SETUP_")):
            fails = ", ".join(r["fails"]) or "all checks passed"
            print(f"  {r['verdict']:8} {r['symbol']:10} {r.get('ltp', ''):>9} | {fails}")
    return snapshot


def _pick_index_expiry(dh, sid: int, seg: str, day: dt.date,
                       min_dte_days: int) -> str | None:
    """Reuses liquidity_sweep's helper for weekly index-expiry selection."""
    import liquidity_sweep as ls
    return ls._pick_index_expiry(dh, sid, seg, day, min_dte_days)


# ---------------------------------------------------------------------------
# Standalone CLI (same shape as liquidity_sweep / trend_pullback)
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="EMA Retest NIFTY-index scanner")
    parser.add_argument("--once", action="store_true",
                        help="run one scan now and exit")
    args = parser.parse_args()

    dh = core.Dhan(*core.credentials())
    day = dt.datetime.now(core.IST).date()
    OUT.mkdir(exist_ok=True)
    eq, master = core.load_master(day)
    universe = {}  # index-only strategy; no equity universe needed
    prep: dict = {}

    fired: set[str] = set()
    signal_path = OUT / f"{STRATEGY}_signals_{day}.jsonl"
    if signal_path.exists():
        for line in signal_path.read_text(encoding="utf-8").splitlines():
            signal = json.loads(line)
            fired.add(f"{signal['symbol']}_{signal['side']}")

    if args.once:
        scan(dh, master, prep, universe, day,
             dt.datetime.now(core.IST), fired)
        return

    print(f"{STRATEGY} scanner runs every 5 minutes "
          f"{RUN_START.strftime('%H:%M')}..{RUN_END.strftime('%H:%M')} IST; "
          "recommendations only. Ctrl+C to stop.")
    while True:
        now = dt.datetime.now(core.IST)
        if now.time() > RUN_END:
            print(f"{STRATEGY} scanner finished for the day.")
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
            scan(dh, master, prep, universe, day, now, fired)
        except core.DhanAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"{STRATEGY} scan failed; will retry next candle: {exc}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
