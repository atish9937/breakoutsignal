"""Forward-paper validation summary read from the SQLite paper ledger.

Usage:
    python validation_report.py              # text summary
    python validation_report.py --json       # machine-readable summary
    python validation_report.py --backup     # also snapshot the ledger (keeps 14)
Research only; it never places orders.
"""
import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path

import option_positions
import trade_performance

OUT = Path(__file__).resolve().parent / "scanner_out"
LEDGER = OUT / "paper_trade_ledger.sqlite3"
MIN_SESSIONS = 60
MIN_TRADES = 100
MIN_PROFIT_FACTOR = 1.3
BACKUPS_KEPT = 14


def _r_multiple(row) -> float | None:
    """Underlying R using the 50% T1 / 50% T2-or-breakeven plan; None while still open."""
    ref, stop = row["reference_entry"], row["stop_price"]
    risk = abs(ref - stop)
    if risk <= 0:
        return None
    r1 = abs(row["target_1"] - ref) / risk
    r2 = abs(row["target_2"] - ref) / risk
    return {"stop_before_t1": -1.0, "t1_then_breakeven": 0.5 * r1,
            "t2_hit": 0.5 * r1 + 0.5 * r2}.get(row["status"])


def _stats(values: list[float]) -> dict:
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    gross_loss = abs(sum(losses))
    return {
        "n": len(values),
        "total": round(sum(values), 2),
        "average": round(sum(values) / len(values), 3) if values else None,
        "win_rate_pct": round(len(wins) / len(values) * 100, 1) if values else None,
        "profit_factor": round(sum(wins) / gross_loss, 2) if gross_loss else None,
        "max_drawdown": round(drawdown, 2),
        "worst": round(min(values), 2) if values else None,
    }


def _tracked(connection: sqlite3.Connection, strategy: str) -> dict:
    days = [r[0] for r in connection.execute(
        "SELECT DISTINCT session_date FROM paper_trades WHERE strategy = ? ORDER BY 1",
        (strategy,))]
    rows = []
    for day in days:
        rows += [r for r in trade_performance._report_rows(connection, dt.date.fromisoformat(day))
                 if r["strategy"] == strategy]
    filled = [r for r in rows if r["entry_at"]]
    resolved = [r for r in filled if r["status"] in trade_performance.RESOLVED_STATUSES]
    r_values = [v for v in (_r_multiple(r) for r in resolved) if v is not None]
    pnl = [p for p in (trade_performance._trade_premium_pnl(r) for r in resolved) if p is not None]
    return {
        "sessions_with_signals": len(days),
        "signals": len(rows),
        "filled": len(filled),
        "resolved": len(resolved),
        "open_or_unresolved": sum(r["status"] in ("tracking", "t1_hit", "unresolved") for r in filled),
        "missing_entry_bid_ask": sum(not (r["option_entry_bid"] and r["option_entry_ask"]) for r in rows),
        "capture_errors": sum(bool(r["data_capture_error"] or r["tracking_error"]) for r in rows),
        "no_option_ticks": sum(not r["option_quote_samples"] for r in filled),
        "underlying_R": _stats(r_values),
        "premium_pnl_per_lot_rupees": _stats(pnl),
        "premium_priced": len(pnl),
    }


def _positions(connection: sqlite3.Connection, strategy: str) -> dict:
    rows = connection.execute(
        "SELECT * FROM paper_option_positions WHERE strategy = ? ORDER BY entry_date, id",
        (strategy,)).fetchall()
    closed = [r for r in rows if r["status"] not in ("open", "unresolved")]
    pnl = [r["premium_pnl_per_lot"] for r in closed if r["premium_pnl_per_lot"] is not None]
    return {
        "sessions_with_signals": len({r["entry_date"] for r in rows}),
        "signals": len(rows),
        "closed": len(closed),
        "open": sum(r["status"] == "open" for r in rows),
        "unresolved": sum(r["status"] == "unresolved" for r in rows),
        "missing_entry_bid_ask": sum(not (r["entry_bid"] and r["entry_ask"]) for r in rows),
        "capture_errors": sum(bool(r["tracking_error"]) for r in rows),
        "premium_pnl_per_lot_rupees": _stats(pnl),
        "premium_priced": len(pnl),
    }


def _gate(summary: dict, pnl_key: str, count_key: str) -> dict:
    stats = summary[pnl_key]
    profit_factor = stats["profit_factor"]
    checks = {
        f"sessions >= {MIN_SESSIONS}": summary["sessions_with_signals"] >= MIN_SESSIONS,
        f"{MIN_TRADES}+ premium-priced trades": summary["premium_priced"] >= MIN_TRADES,
        "net premium P&L > 0": stats["total"] > 0,
        f"profit factor >= {MIN_PROFIT_FACTOR}": bool(profit_factor and profit_factor >= MIN_PROFIT_FACTOR),
        "no capture errors": summary["capture_errors"] == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def build(connection: sqlite3.Connection) -> dict:
    result = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "strategies": {}}
    for name in ("breakout", "trend_pullback"):
        summary = _tracked(connection, name)
        summary["gate"] = _gate(summary, "premium_pnl_per_lot_rupees", "premium_priced")
        result["strategies"][name] = summary
    for name in ("multiframe_15m", "swing_option"):
        summary = _positions(connection, name)
        summary["gate"] = _gate(summary, "premium_pnl_per_lot_rupees", "premium_priced")
        result["strategies"][name] = summary
    first, last, ticks = connection.execute(
        "SELECT MIN(session_date), MAX(session_date), COUNT(*) FROM paper_market_ticks"
    ).fetchone()
    per_day = connection.execute(
        "SELECT session_date, COUNT(*) FROM paper_market_ticks GROUP BY session_date ORDER BY 1"
    ).fetchall()
    reports = {r[0]: r[1] for r in connection.execute(
        "SELECT status, COUNT(*) FROM performance_reports GROUP BY status")}
    result["data_capture"] = {
        "tick_first_day": first, "tick_last_day": last, "tick_rows": ticks,
        "tick_days": len(per_day),
        "days_with_few_ticks": [d for d, n in per_day if n < 1000],
        "one_minute_candle_rows": connection.execute(
            "SELECT COUNT(*) FROM paper_instrument_candles").fetchone()[0],
        "reports": reports,
    }
    return result


def render(result: dict) -> str:
    lines = [f"Forward paper validation ({result['generated_at']}) - research only"]
    for name, s in result["strategies"].items():
        lines.append(f"\n[{name}] sessions={s['sessions_with_signals']} signals={s['signals']} "
                     f"priced={s['premium_priced']} errors={s['capture_errors']} "
                     f"missing bid/ask={s['missing_entry_bid_ask']}")
        if "underlying_R" in s:
            lines.append(f"  underlying R: {s['underlying_R']}")
        lines.append(f"  premium P&L/lot: {s['premium_pnl_per_lot_rupees']}")
        for check, ok in s["gate"]["checks"].items():
            lines.append(f"  [{'PASS' if ok else 'fail'}] {check}")
    lines.append(f"\nData capture: {result['data_capture']}")
    lines.append("Gates are necessary, not sufficient: also review sample days manually.")
    return "\n".join(lines)


def backup(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"ledger_{dt.date.today().isoformat()}.sqlite3"
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    for old in sorted(target_dir.glob("ledger_*.sqlite3"))[:-BACKUPS_KEPT]:
        old.unlink()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()
    if not args.ledger.exists():
        raise SystemExit(f"Ledger not found: {args.ledger}")
    if args.backup:
        print(f"Backup saved: {backup(args.ledger, args.ledger.parent / 'backups')}")
    connection = trade_performance.connect(args.ledger)
    result = build(connection)
    print(json.dumps(result, indent=2) if args.json else render(result))


if __name__ == "__main__":
    main()
