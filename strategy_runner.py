#!/usr/bin/env python3
"""Run all registered recommendation strategies on one shared Dhan client and schedule."""

import argparse
import datetime as dt
import json
import sys
import time
import traceback

import requests

import breakout_scanner as breakout
import trend_pullback


STRATEGY_START = dt.time(9, 20)
STRATEGY_END = dt.time(12, 35)


def cycle_end(ignore_entry_cutoff):
    if ignore_entry_cutoff:
        return breakout.hm(breakout.CONFIG["extended_run_end"])
    return max(STRATEGY_END, breakout.hm(breakout.CONFIG["run_end"]))


def load_fired_signals(path):
    fired = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            signal = json.loads(line)
            fired.add(f"{signal['symbol']}_{signal['side']}")
    return fired


def run_cycle(dh, master, prep, universe, exclude, trend_expiry, day, now, fired,
              ignore_entry_cutoff=False):
    """Run eligible strategies for one candle-close using the same API client."""
    scanners = [(
        "breakout",
        breakout.scan,
        (dh, master, prep, day, exclude, fired["breakout"], universe),
        {"now": now, "ignore_entry_cutoff": ignore_entry_cutoff},
    )]
    if now.time() <= STRATEGY_END:
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
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run breakout and trend-pullback recommendations on one shared Dhan client"
    )
    parser.add_argument("--once", action="store_true", help="run one cycle of every strategy")
    parser.add_argument(
        "--ignore-entry-cutoff",
        action="store_true",
        help="extend breakout entries to 15:25 IST and keep breakout scans running until 15:30; trend-pullback keeps its normal cutoff",
    )
    parser.add_argument("--universe", default="nifty50",
                        help="supported NSE indices, fno (all F&O stocks), or watchlist; "
                             "combine with commas, e.g. nifty50,niftymidcap150,niftybank")
    parser.add_argument("--exclude", default="", help="comma-separated symbols to skip")
    args = parser.parse_args()
    exclude = {symbol.strip().upper() for symbol in args.exclude.split(",") if symbol.strip()}

    client_id, access_token = breakout.credentials()
    dh = breakout.Dhan(client_id, access_token)
    day = dt.datetime.now(breakout.IST).date()
    breakout.OUT.mkdir(exist_ok=True)
    eq, master = breakout.load_master(day)
    universe = breakout.build_universe(args.universe, day, eq, master)
    prep = breakout.prepare(dh, day, universe)
    fired = {
        "breakout": load_fired_signals(breakout.OUT / f"signals_{day}.jsonl"),
        "trend_pullback": load_fired_signals(
            breakout.OUT / f"trend_pullback_signals_{day}.jsonl"
        ),
    }

    if args.once:
        run_cycle(dh, master, prep, universe, exclude, prep.get("expiry"), day,
                  dt.datetime.now(breakout.IST), fired, args.ignore_entry_cutoff)
        return

    end = cycle_end(args.ignore_entry_cutoff)
    if args.ignore_entry_cutoff:
        print("Running breakout and trend-pullback strategies every 5 minutes from "
              "09:20-12:35 IST; breakout only continues until 15:30 IST with its "
              "entry cutoff extended to 15:25. Strategies share one Dhan client and "
              "request limiter; no live orders are placed. Ctrl+C to stop.")
    else:
        print("Running breakout and trend-pullback strategies every 5 minutes from "
              f"09:20-{end.strftime('%H:%M')} IST. Strategies share one Dhan client "
              "and request limiter; no live orders are placed. Ctrl+C to stop.")
    while True:
        now = dt.datetime.now(breakout.IST)
        if now.time() > end:
            print("All strategies finished for the day.")
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
        if cycle_time.time() > end:
            continue
        try:
            run_cycle(dh, master, prep, universe, exclude, prep.get("expiry"),
                      day, cycle_time, fired, args.ignore_entry_cutoff)
        except breakout.DhanAuthError as exc:
            sys.exit(f"Dhan rejected the token ({exc}). Check Dhan credentials and retry.")
        except requests.RequestException as exc:
            print(f"Shared Dhan request failed; strategies will retry next cycle: {exc}",
                  file=sys.stderr)
        except KeyboardInterrupt:
            print("\nStopped.")
            return


if __name__ == "__main__":
    main()
