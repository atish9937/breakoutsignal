#!/usr/bin/env python3
"""Run all registered recommendation strategies on one shared Dhan client and schedule."""

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

import requests

import breakout_scanner as breakout
import intraday_multiframe_strategy
import trend_pullback
import trade_performance
import swing_option_strategy
from paper_quote_feed import DhanQuoteRecorder


STRATEGY_START = dt.time(9, 20)
STRATEGY_END = dt.time(12, 35)
REPORT_TIME = dt.time(15, 10)
SWING_REPORT_TIME = dt.time(15, 35)


def cycle_end(ignore_entry_cutoff):
    return REPORT_TIME


def _at_or_before(now, cutoff):
    return now.strftime("%H:%M") <= cutoff.strftime("%H:%M")


def report_due(now):
    return now.time().replace(tzinfo=None) >= REPORT_TIME


def load_fired_signals(path):
    fired = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            signal = json.loads(line)
            fired.add(f"{signal['symbol']}_{signal['side']}")
    return fired


def run_cycle(dh, master, prep, universe, exclude, trend_expiry, day, now, fired,
              ignore_entry_cutoff=False, ledger=None):
    """Run eligible strategies for one candle-close using the same API client."""
    breakout_end = (
        breakout.hm(breakout.CONFIG["extended_entry_cutoff"])
        if ignore_entry_cutoff
        else breakout.hm(breakout.CONFIG["run_end"])
    )
    scanners = []
    if _at_or_before(now, breakout_end):
        scanners.append((
            "breakout",
            breakout.scan,
            (dh, master, prep, day, exclude, fired["breakout"], universe),
            {"now": now, "ignore_entry_cutoff": ignore_entry_cutoff},
        ))
    if _at_or_before(now, STRATEGY_END):
        scanners.append((
            "trend_pullback",
            trend_pullback.scan,
            (dh, master, universe, exclude, trend_expiry, day, now,
             fired["trend_pullback"]),
            {},
        ))
    results = {}
    for name, scanner, args, kwargs in scanners:
        try:
            results[name] = scanner(*args, **kwargs)
        except breakout.DhanAuthError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"{name} scan failed; other strategies will still run: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
    if ledger is not None:
        for name, snapshot in results.items():
            try:
                trade_performance.record_buy_recommendations(
                    ledger, name, snapshot, universe, day
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Paper tracker could not save {name} recommendations: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
        try:
            trade_performance.update_open_trades(
                ledger, dh, day, dt.datetime.now(breakout.IST)
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Paper tracker candle update failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
    try:
        signals = intraday_multiframe_strategy.scan(
            dh,
            master,
            prep,
            universe,
            day,
            now,
            fired.setdefault("daily_hourly_15m", set()),
        )
        if signals:
            results["daily_hourly_15m"] = signals
    except breakout.DhanAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"{intraday_multiframe_strategy.STRATEGY} scan failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
    try:
        exited = intraday_multiframe_strategy.monitor_positions(
            dh, day, now
        )
        if exited:
            print(f"Sent {len(exited)} same-day 15m paper exit notification(s).")
    except breakout.DhanAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"{intraday_multiframe_strategy.STRATEGY} exit monitor failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
    return results


def run_swing_next_session_checks(dh, master, day, now):
    try:
        signals = swing_option_strategy.monitor_next_session(
            dh, master, day, now, breakout.OUT
        )
        if signals:
            print(f"Confirmed {len(signals)} next-session swing paper signal(s).")
    except breakout.DhanAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"Next-session swing check failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()


def run_swing_candidate_report(dh, master, universe, day):
    try:
        report = swing_option_strategy.scan_daily_candidates(
            dh, master, universe, day, breakout.OUT
        )
        swing_option_strategy.send_daily_report(report)
        print(
            f"Saved {report['candidate_count']} daily swing setup candidate(s) "
            f"to {report['output_path']}; option chains checked: "
            f"{report['checked_option_chains']}."
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Daily swing paper-candidate scan failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run breakout and trend-pullback recommendations on one shared Dhan client"
    )
    parser.add_argument("--once", action="store_true", help="run one cycle of every strategy")
    parser.add_argument(
        "--ignore-entry-cutoff",
        action="store_true",
        help="extend breakout BUY recommendations through 15:00 IST; trend-pullback keeps its normal cutoff",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="retry/generate today's saved HTML performance report and send it to Telegram",
    )
    parser.add_argument("--universe", default="nifty50",
                        help="supported NSE indices, fno (all F&O stocks), or watchlist; "
                             "combine with commas, e.g. nifty50,niftymidcap150,niftybank")
    parser.add_argument("--exclude", default="", help="comma-separated symbols to skip")
    args = parser.parse_args()
    exclude = {symbol.strip().upper() for symbol in args.exclude.split(",") if symbol.strip()}

    breakout.load_env()
    day = dt.datetime.now(breakout.IST).date()
    breakout.OUT.mkdir(exist_ok=True)
    ledger = trade_performance.connect(
        breakout.OUT / "paper_trade_ledger.sqlite3"
    )
    if args.report_only:
        report_now = dt.datetime.now(breakout.IST)
        if report_now.time() < REPORT_TIME:
            sys.exit("--report-only is available after 15:10 IST so the day's tracking is complete.")
        try:
            report_path = trade_performance.finalize_day(
                ledger,
                day,
                breakout.OUT,
                os.getenv("TELEGRAM_BOT_TOKEN"),
                os.getenv("TELEGRAM_CHAT_ID"),
                report_now,
            )
        except RuntimeError as exc:
            sys.exit(str(exc))
        delivery = ledger.execute(
            "SELECT status FROM performance_reports WHERE session_date = ?",
            (day.isoformat(),),
        ).fetchone()
        if not delivery or delivery["status"] != "sent":
            sys.exit(
                f"Report saved to {report_path}, but Telegram delivery did not complete; "
                "check the error in the runner log and retry later."
            )
        print(f"Performance report saved to {report_path}")
        return

    client_id, access_token = breakout.credentials()
    dh = breakout.Dhan(client_id, access_token)
    eq, master = breakout.load_master(day)
    universe = breakout.build_universe(args.universe, day, eq, master)
    prep = breakout.prepare(dh, day, universe)
    signal_files = (
        ("breakout", breakout.OUT / f"signals_{day}.jsonl"),
        ("trend_pullback", breakout.OUT / f"trend_pullback_signals_{day}.jsonl"),
    )
    saved_signal_ids = dict(universe)
    if eq is not None:
        saved_signal_ids.update(
            zip(eq["SYMBOL"], eq["SID"].astype(int))
        )
    for name, signal_file in signal_files:
        try:
            imported = trade_performance.import_saved_signals(
                ledger, name, signal_file, saved_signal_ids, day
            )
            if imported:
                print(f"Recovered {imported} saved {name} recommendation(s) into the paper ledger.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"Paper tracker could not recover {name} signal history from "
                f"{signal_file}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
    fired = {
        "breakout": load_fired_signals(breakout.OUT / f"signals_{day}.jsonl"),
        "trend_pullback": load_fired_signals(
            breakout.OUT / f"trend_pullback_signals_{day}.jsonl"
        ),
        "daily_hourly_15m": load_fired_signals(
            breakout.OUT / f"daily_hourly_15m_signals_{day}.jsonl"
        ),
    }

    if args.once:
        run_cycle(dh, master, prep, universe, exclude, prep.get("expiry"), day,
                  dt.datetime.now(breakout.IST), fired, args.ignore_entry_cutoff, ledger)
        return

    end = cycle_end(args.ignore_entry_cutoff)
    if args.ignore_entry_cutoff:
        print("Running breakout and trend-pullback strategies every 5 minutes from "
              "09:20-12:35 IST; breakout BUY cutoff is 15:00 IST. The runner tracks "
              "open recommendations and sends the daily HTML report by 15:10 IST. "
              "No live orders are placed. Ctrl+C to stop.")
    else:
        print("Running breakout and trend-pullback strategies every 5 minutes from "
              f"09:20-12:35 IST, then tracking recommendations until the "
              f"{end.strftime('%H:%M')} IST report. No live orders are placed. Ctrl+C to stop.")

    def finish_report():
        quote_recorder.stop()
        report_now = dt.datetime.now(breakout.IST)
        trade_performance.update_open_trades(ledger, dh, day, report_now)
        report_path = trade_performance.finalize_day(
            ledger,
            day,
            breakout.OUT,
            os.getenv("TELEGRAM_BOT_TOKEN"),
            os.getenv("TELEGRAM_CHAT_ID"),
            report_now,
        )
        try:
            archived = trade_performance.archive_one_minute_candles(
                ledger, dh, day, dt.datetime.now(breakout.IST)
            )
            print(
                f"Archived {archived} post-signal one-minute instrument candles "
                "after report delivery."
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Post-report one-minute candle archive failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
        swing_scan_time = dt.datetime.combine(
            day, SWING_REPORT_TIME, breakout.IST
        )
        if dt.datetime.now(breakout.IST) < swing_scan_time:
            print(
                "Waiting until 15:35 IST so the daily market candles are complete "
                "before generating swing candidates."
            )
            while True:
                current = dt.datetime.now(breakout.IST)
                wait_seconds = (swing_scan_time - current).total_seconds()
                if wait_seconds <= 0:
                    break
                next_run = (
                    current.replace(second=0, microsecond=0)
                    + dt.timedelta(minutes=5 - current.minute % 5)
                    + dt.timedelta(seconds=20)
                )
                time.sleep(
                    min(wait_seconds, max(1, (next_run - current).total_seconds()))
                )
                current = dt.datetime.now(breakout.IST)
                try:
                    exited = intraday_multiframe_strategy.monitor_positions(
                        dh, day, current
                    )
                    if exited:
                        print(
                            f"Sent {len(exited)} same-day 15m "
                            "paper exit notification(s)."
                        )
                except breakout.DhanAuthError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"Post-report same-day exit monitor failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
        run_swing_candidate_report(dh, master, universe, day)
        return report_path

    quote_recorder = DhanQuoteRecorder(
        breakout.OUT / "paper_trade_ledger.sqlite3",
        client_id,
        access_token,
        day,
    )
    quote_recorder.start()
    while True:
        now = dt.datetime.now(breakout.IST)
        if report_due(now):
            try:
                report_path = finish_report()
                print(f"Daily performance report saved to {report_path}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Daily performance report failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
            return

        next_run = (
            now.replace(second=0, microsecond=0)
            + dt.timedelta(minutes=5 - now.minute % 5)
            + dt.timedelta(seconds=20)
        )
        if now.time() < STRATEGY_START:
            next_run = max(
                next_run,
                dt.datetime.combine(day, STRATEGY_START, breakout.IST),
            )
        time.sleep(max(1, (next_run - dt.datetime.now(breakout.IST)).total_seconds()))
        cycle_time = dt.datetime.now(breakout.IST)
        if report_due(cycle_time):
            continue
        if not _at_or_before(cycle_time, end):
            continue
        try:
            run_cycle(dh, master, prep, universe, exclude, prep.get("expiry"),
                      day, cycle_time, fired, args.ignore_entry_cutoff, ledger)
            run_swing_next_session_checks(
                dh, master, day, cycle_time
            )
        except breakout.DhanAuthError as exc:
            sys.exit(f"Dhan rejected the token ({exc}). Check Dhan credentials and retry.")
        except requests.RequestException as exc:
            print(f"Shared Dhan request failed; strategies will retry next cycle: {exc}",
                  file=sys.stderr)
        except KeyboardInterrupt:
            quote_recorder.stop()
            print("\nStopped.")
            return


if __name__ == "__main__":
    main()
