#!/usr/bin/env python3
"""
breakout_scanner.py - Intraday "box breakout" scanner for NSE F&O stocks (Dhan API v2)
====================================================================================

Finds stocks that break out of their 5-day box with volume, still have room to run,
and have a clear runway to the next option-OI wall. Same rules as the
"Intraday Breakout Checklist" (JSW Steel, 15 Oct 2026).

It ONLY READS data and prints/saves signals. It never places orders.

SETUP (one time)
    pip install requests pandas
    Set your Dhan credentials (Dhan web -> My Profile -> DhanHQ Trading APIs):
      Windows (PowerShell):  $env:DHAN_CLIENT_ID="1100xxxxxx"; $env:DHAN_ACCESS_TOKEN="eyJ..."
      Mac/Linux:             export DHAN_CLIENT_ID=1100xxxxxx; export DHAN_ACCESS_TOKEN=eyJ...
    (or put them in config.json next to this file: {"client_id": "...", "access_token": "..."})
    The access token expires every 24 h - paste a fresh one each morning.

RUN
    python breakout_scanner.py                  # runs 09:20-12:35 IST, scans every 5 min
    python breakout_scanner.py --once           # one scan now (for testing)
    python breakout_scanner.py --exclude TECHM,WIPRO   # skip stocks (results today / F&O ban)
    python breakout_scanner.py --universe nifty100      # which stocks to scan (default nifty50)

UNIVERSE (pulled fresh every day, cached in scanner_out/)
    nifty50 | niftynext50 | nifty100 | nifty200  - official NSE constituent lists (niftyindices.com)
    fno        - every stock that has stock options on NSE (from the Dhan scrip master, ~200)
    watchlist  - the built-in WATCHLIST below
    Combine with commas (nifty50,watchlist). Stocks without stock options are dropped automatically.
    Symbols are mapped to Dhan security ids from the Dhan scrip master.
    Each successful download is saved as scanner_out/index_<name>_<date>.json. If NSE can't be
    reached, the most recent saved list is used. The built-in WATCHLIST is used only if no list
    has ever been downloaded. The start-up lines always say which source was used.

OUTPUT (folder ./scanner_out/)
    prep_YYYY-MM-DD.json   daily levels (box, ATR, prev day) - built once per day
    scan_YYYY-MM-DD.jsonl  every evaluation, one JSON per line
    latest.json            the most recent scan (market + all candidates)
    signals_YYYY-MM-DD.jsonl  only BUY_CE / BUY_PE signals
    On screen: a table each scan, and a "SIGNAL FOR CLAUDE" JSON block on every BUY.
    Copy that block into Claude for a second opinion.

THE CONCLUSIVE FIELD: "verdict"
    BUY_CE / BUY_PE   all checks passed - a trade you can consider now
    WATCH_CE / WATCH_PE  near its trigger but not triggered yet - watch the level
    SKIP_CE / SKIP_PE    failed one or more checks ("fails" lists why)
    plus "risk_warning" when one lot's loss at the stop is above your max risk %.

Research tool, not SEBI-registered investment advice. Verify every level on your broker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# SETTINGS - change these to suit you
# --------------------------------------------------------------------------------------
CONFIG = {
    "capital": 72000,            # your trading capital in Rs
    "max_risk_pct": 2.0,         # max loss per trade as % of capital
    "block_over_risk": False,    # True = downgrade BUY to SKIP if one lot risks more than max_risk_pct
    "atr_period": 14,            # daily ATR length
    "box_days": 5,               # box = high/low of the last N completed days
    "max_atr_used": 0.60,        # skip if |price - open| > 60% of ATR (move already done)
    "watch_dist_pct": 0.60,      # list as WATCH when within 0.6% of the box edge
    "max_chase_pct": 0.50,       # don't buy if already >0.5% beyond the box edge
    "max_vwap_dist_pct": 1.0,    # price must be within 1% of VWAP
    "stop_atr_mult": 0.35,       # stop distance = 0.35 x ATR (JSW: 0.35 x 30 = ~Rs 11)
    "min_runway_r": 2.0,         # next OI wall must be >= 2R away
    "t1_r": 1.5,                 # target 1 = 1.5R
    "t2_r": 2.5,                 # target 2 = 2.5R (capped just before the wall)
    "wall_frac": 0.5,            # a strike counts as a "wall" if its OI >= 50% of the biggest
    "min_auto_score": 6,         # of 7 automatic factors (catalyst/news is left to you/Claude)
    "max_option_spread_pct": 3.0,  # bid-ask spread of the option, % of its price
    "entry_start": "09:30",
    "entry_cutoff": "12:30",
    "run_start": "09:20",
    "run_end": "12:35",
    "max_candidates_per_scan": 8,  # option-chain calls are slow (1 per 3 s)
}

# Built-in list. NOT used by default any more (the default is the live Nifty 50 list from NSE).
# It is used only if you pass --universe watchlist, or as a last resort if NSE has never been
# reachable and no saved list exists. NSE_EQ security ids verified 15 Oct 2026.
WATCHLIST = {
    "RELIANCE": 2885, "HDFCBANK": 1333, "ICICIBANK": 4963, "SBIN": 3045, "AXISBANK": 5900,
    "INFY": 1594, "TCS": 11536, "KOTAKBANK": 1922, "LT": 11483, "BHARTIARTL": 10604,
    "ITC": 1660, "HINDUNILVR": 1394, "MARUTI": 10999, "M&M": 2031, "BAJFINANCE": 317,
    "BAJAJFINSV": 16675, "HCLTECH": 7229, "SUNPHARMA": 3351, "CIPLA": 694, "DRREDDY": 881,
    "LUPIN": 10440, "TATASTEEL": 3499, "JSWSTEEL": 11723, "HINDALCO": 1363, "VEDL": 3063,
    "ONGC": 2475, "BPCL": 526, "IOC": 1624, "NTPC": 11630, "POWERGRID": 14977,
    "TATAPOWER": 3426, "COALINDIA": 20374, "ADANIENT": 25, "ADANIPORTS": 15083, "TITAN": 3506,
    "ASIANPAINT": 236, "ULTRACEMCO": 11532, "GRASIM": 1232, "EICHERMOT": 910, "HEROMOTOCO": 1348,
    "BEL": 383, "HAL": 2303, "BHEL": 438, "TRENT": 1964, "SHRIRAMFIN": 4306, "INDUSINDBK": 5258,
    "PNB": 10666, "BANKBARODA": 4668, "CANBK": 10794, "IDFCFIRSTB": 11184, "DLF": 14732,
    "ETERNAL": 5097, "INDIGO": 11195, "DIXON": 21690, "BSE": 19585, "APOLLOHOSP": 157,
    "TATACONSUM": 3432,
    # unverified ids - check before trusting: "TECHM": 13538, "WIPRO": 3787,
    # "PERSISTENT": 18365, "POLYCAB": 9590,
}

# Used only if the Dhan scrip master can't be downloaded. Verify on Dhan before trading.
LOT_FALLBACK = {"JSWSTEEL": 675, "SHRIRAMFIN": 825, "VEDL": 1150, "HDFCBANK": 550}

NIFTY_ID = 13  # IDX_I
API = "https://api.dhan.co/v2"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OUT = Path(__file__).resolve().parent / "scanner_out"


# --------------------------------------------------------------------------------------
# Dhan REST client (read-only)
# --------------------------------------------------------------------------------------
class DhanAuthError(RuntimeError):
    pass


class Dhan:
    """Minimal read-only Dhan v2 client with simple rate limiting."""

    def __init__(self, client_id: str, access_token: str):
        self.s = requests.Session()
        self.s.headers.update({
            "access-token": access_token,
            "client-id": str(client_id),
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._last: dict[str, float] = {}

    def _post(self, path: str, body: dict, min_gap: float = 0.25) -> dict:
        wait = min_gap - (time.time() - self._last.get(path, 0.0))
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            r = self.s.post(API + path, json=body, timeout=30)
            self._last[path] = time.time()
            if r.status_code == 429:
                time.sleep(2 + 2 * attempt)
                continue
            if r.status_code in (401, 403):
                raise DhanAuthError(f"{r.status_code} {r.text[:200]}")
            if r.status_code >= 500 and attempt < 3:
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    def daily(self, sid, seg="NSE_EQ", inst="EQUITY", days=120) -> pd.DataFrame:
        today = dt.datetime.now(IST).date()
        body = {"securityId": str(sid), "exchangeSegment": seg, "instrument": inst,
                "expiryCode": 0, "oi": False,
                "fromDate": str(today - dt.timedelta(days=days)),
                "toDate": str(today + dt.timedelta(days=1))}
        return candles_df(self._post("/charts/historical", body))

    def intraday(self, sid, frm: dt.datetime, to: dt.datetime, seg="NSE_EQ", inst="EQUITY",
                 interval=5) -> pd.DataFrame:
        body = {"securityId": str(sid), "exchangeSegment": seg, "instrument": inst,
                "interval": str(interval), "oi": False,
                "fromDate": frm.strftime("%Y-%m-%d %H:%M:%S"),
                "toDate": to.strftime("%Y-%m-%d %H:%M:%S")}
        return candles_df(self._post("/charts/intraday", body))

    def ohlc(self, segments: dict) -> dict:
        return self._post("/marketfeed/ohlc", segments, min_gap=1.1).get("data", {})

    def expiries(self, sid, seg="NSE_EQ") -> list[str]:
        j = self._post("/optionchain/expirylist", {"UnderlyingScrip": int(sid), "UnderlyingSeg": seg},
                       min_gap=3.1)
        return list(j.get("data", []))

    def option_chain(self, sid, expiry: str, seg="NSE_EQ") -> dict:
        j = self._post("/optionchain", {"UnderlyingScrip": int(sid), "UnderlyingSeg": seg,
                                        "Expiry": expiry}, min_gap=3.1)
        return j.get("data", {})


def candles_df(j: dict) -> pd.DataFrame:
    """Dhan chart response {open:[],high:[],...,timestamp:[]} -> DataFrame indexed by IST time."""
    if not j or not j.get("timestamp"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame({k: j[k] for k in ("open", "high", "low", "close", "volume")})
    idx = pd.DatetimeIndex(pd.to_datetime(j["timestamp"], unit="s", utc=True)).tz_convert(IST)
    # Safety: intraday bars should sit between 09:00 and 16:00 IST. If most don't (and they are not
    # midnight-stamped daily bars), the epoch was IST wall-clock - re-read it that way.
    mins = idx.hour * 60 + idx.minute
    if len(idx) > 3 and (idx.hour != 0).any() and ((mins < 540) | (mins > 960)).mean() > 0.5:
        idx = pd.DatetimeIndex(pd.to_datetime(j["timestamp"], unit="s")).tz_localize(IST)
    df.index = idx
    return df.astype(float).sort_index()


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] else math.inf
    return float(100 - 100 / (1 + rs))


def macd_hist(s: pd.Series) -> float:
    m = ema(s, 12) - ema(s, 26)
    return float((m - ema(m, 9)).iloc[-1])


def atr(d: pd.DataFrame, n: int = 14) -> float:
    pc = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def vwap(d: pd.DataFrame) -> float:
    tp = (d["high"] + d["low"] + d["close"]) / 3
    v = d["volume"].sum()
    return float((tp * d["volume"]).sum() / v) if v else float(d["close"].iloc[-1])


def r2(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)


# --------------------------------------------------------------------------------------
# Lot sizes / option security ids from Dhan scrip master (cached daily)
# --------------------------------------------------------------------------------------
def load_master(day: dt.date) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Dhan scrip master, cached daily as two small files:
    eq_<day>.csv     NSE cash-market equities (symbol -> securityId)
    optstk_<day>.csv NSE stock options (lot size, option securityId)"""
    eq_path, opt_path = OUT / f"eq_{day}.csv", OUT / f"optstk_{day}.csv"
    if eq_path.exists() and opt_path.exists():
        return pd.read_csv(eq_path), pd.read_csv(opt_path)
    try:
        print("Downloading Dhan scrip master (once a day, ~30 MB)...")
        m = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)
        nse = m[m["SEM_EXM_EXCH_ID"] == "NSE"]
        eq = nse[(nse["SEM_INSTRUMENT_NAME"] == "EQUITY") & (nse["SEM_SERIES"].astype(str).str.strip() == "EQ")]
        eq = eq[["SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL"]].rename(
            columns={"SEM_SMST_SECURITY_ID": "SID", "SEM_TRADING_SYMBOL": "SYMBOL"})
        eq["SYMBOL"] = eq["SYMBOL"].astype(str).str.strip().str.upper()
        opt = nse[nse["SEM_INSTRUMENT_NAME"] == "OPTSTK"]
        opt = opt[["SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL", "SEM_LOT_UNITS", "SEM_EXPIRY_DATE",
                   "SEM_STRIKE_PRICE", "SEM_OPTION_TYPE"]].copy()
        opt["UNDERLYING"] = opt["SEM_TRADING_SYMBOL"].astype(str).str.split("-").str[0].str.upper()
        opt["EXPIRY"] = opt["SEM_EXPIRY_DATE"].astype(str).str[:10]
        OUT.mkdir(exist_ok=True)
        eq.to_csv(eq_path, index=False)
        opt.to_csv(opt_path, index=False)
        return eq, opt
    except Exception as e:  # noqa: BLE001
        print(f"  (scrip master not available: {e}; using built-in WATCHLIST ids and LOT_FALLBACK)")
        return None, None


INDEX_CSV = {  # official NSE constituent lists (niftyindices.com, with an NSE archive mirror)
    "nifty50": "ind_nifty50list.csv",
    "niftynext50": "ind_niftynext50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
}


def fetch_index_symbols(name: str) -> list[str]:
    fname = INDEX_CSV[name]
    urls = [f"https://www.niftyindices.com/IndexConstituent/{fname}",
            f"https://nsearchives.nseindia.com/content/indices/{fname}"]
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "text/csv,*/*"}
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, headers=hdr, timeout=20)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            col = next(c for c in df.columns if c.strip().lower() == "symbol")
            syms = df[col].astype(str).str.strip().str.upper().tolist()
            if len(syms) >= 40:
                return syms
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"could not download {name} list ({last_err})")


def _latest_saved_list(part: str) -> tuple[list[str], str] | None:
    """Most recent successfully downloaded list for this index, e.g. scanner_out/index_nifty50_2026-10-14.json."""
    files = sorted(OUT.glob(f"index_{part}_*.json"))
    if not files:
        return None
    f = files[-1]
    return json.loads(f.read_text()), f.stem.rsplit("_", 1)[-1]


def build_universe(spec: str, day: dt.date, eq: pd.DataFrame | None, opt: pd.DataFrame | None) -> dict:
    """spec: comma list of nifty50 | niftynext50 | nifty100 | nifty200 | fno | watchlist.
    Returns {symbol: securityId}, keeping only stocks that have stock options (F&O).

    Where the list comes from (printed at start-up):
      1. today's download from NSE  -> saved as scanner_out/index_<name>_<date>.json
      2. if NSE can't be reached: the most recent saved download (e.g. yesterday's)
      3. only if there has never been a successful download: the built-in WATCHLIST
    """
    sid_of = dict(zip(eq["SYMBOL"], eq["SID"].astype(int))) if eq is not None else {}
    fno = set(opt["UNDERLYING"]) if opt is not None else None
    uni: dict[str, int] = {}
    OUT.mkdir(exist_ok=True)
    for part in [p.strip().lower() for p in spec.split(",") if p.strip()]:
        if part == "watchlist":
            uni.update(WATCHLIST)
            print(f"  watchlist: {len(WATCHLIST)} stocks from the built-in list (you asked for it)")
            continue
        if part == "fno":
            if fno is None or not sid_of:
                print("  fno: Dhan scrip master unavailable -> using built-in WATCHLIST")
                uni.update(WATCHLIST)
                continue
            syms, source = sorted(fno), "Dhan scrip master (today)"
        elif part in INDEX_CSV:
            saved_today = OUT / f"index_{part}_{day}.json"
            if saved_today.exists():
                syms, source = json.loads(saved_today.read_text()), f"NSE list downloaded today ({saved_today.name})"
            else:
                try:
                    syms = fetch_index_symbols(part)
                    saved_today.write_text(json.dumps(syms))
                    source = f"NSE list downloaded now -> {saved_today.name}"
                except Exception as e:  # noqa: BLE001
                    prev = _latest_saved_list(part)
                    if prev:
                        syms, when = prev
                        source = f"NSE download failed ({e}); using saved list from {when}"
                    else:
                        print(f"  {part}: NSE download failed ({e}) and no saved list yet -> built-in WATCHLIST")
                        uni.update(WATCHLIST)
                        continue
        else:
            sys.exit(f"Unknown universe '{part}'. Use nifty50, niftynext50, nifty100, nifty200, fno or watchlist.")
        no_id, no_fno = [], []
        for s in syms:
            sid = sid_of.get(s) or WATCHLIST.get(s)  # WATCHLIST only as an id backup if the scrip master failed
            if not sid:
                no_id.append(s)
                continue
            if fno is not None and s not in fno:
                no_fno.append(s)
                continue
            uni[s] = int(sid)
        print(f"  {part}: {len(syms)} symbols from {source}")
        if no_id:
            print(f"    no Dhan id found for: {', '.join(no_id)}")
        if no_fno:
            print(f"    skipped (no stock options): {', '.join(no_fno)}")
    if not uni:
        print("  nothing resolved -> built-in WATCHLIST")
        uni = dict(WATCHLIST)
    (OUT / f"universe_{spec.replace(',', '+')}_{day}.json").write_text(json.dumps(uni, indent=1))
    print(f"  scanning {len(uni)} F&O stocks: {', '.join(sorted(uni)[:12])}{' ...' if len(uni) > 12 else ''}")
    return uni


def lot_and_id(master, sym, expiry, strike, opt_type):
    if master is not None:
        hit = master[(master["UNDERLYING"] == sym) & (master["EXPIRY"] == expiry)
                     & (master["SEM_OPTION_TYPE"] == opt_type)
                     & ((master["SEM_STRIKE_PRICE"] - strike).abs() < 0.01)]
        if len(hit):
            return int(hit.iloc[0]["SEM_LOT_UNITS"]), int(hit.iloc[0]["SEM_SMST_SECURITY_ID"])
        any_lot = master[master["UNDERLYING"] == sym]
        if len(any_lot):
            return int(any_lot.iloc[0]["SEM_LOT_UNITS"]), None
    return LOT_FALLBACK.get(sym), None


# --------------------------------------------------------------------------------------
# Pre-market preparation (once per day, cached)
# --------------------------------------------------------------------------------------
def prepare(dh: Dhan, day: dt.date, symbols: dict) -> dict:
    path = OUT / f"prep_{day}.json"
    old = json.loads(path.read_text()) if path.exists() else {"stocks": {}, "expiry": None}
    todo = {s: i for s, i in symbols.items() if s not in old["stocks"]}
    if not todo and old.get("expiry"):
        return old
    print(f"Preparing daily levels for {len(todo)} stocks (about {max(1, len(todo) // 40)} min)...")
    stocks = dict(old["stocks"])
    symbols_all, symbols = symbols, todo
    prev_from = dt.datetime.combine(day - dt.timedelta(days=16), dt.time(9, 15), IST)
    prev_to = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(15, 30), IST)
    for sym, sid in symbols.items():
        try:
            d = dh.daily(sid)
            d = d[d.index.date < day]
            if len(d) < CONFIG["atr_period"] + 2:
                print(f"  {sym}: not enough daily data, skipped")
                continue
            box = d.tail(CONFIG["box_days"])
            i5 = dh.intraday(sid, prev_from, prev_to)
            i5 = i5[i5.index.date < day]
            slot_avg, cum_avg, warm = {}, {}, []
            if len(i5):
                i5 = i5.copy()
                i5["slot"] = i5.index.strftime("%H:%M")
                i5["d"] = i5.index.date
                i5["cum"] = i5.groupby("d")["volume"].cumsum()
                slot_avg = i5.groupby("slot")["volume"].mean().round(0).to_dict()
                cum_avg = i5.groupby("slot")["cum"].mean().round(0).to_dict()
                warm = i5["close"].tail(60).round(2).tolist()
            stocks[sym] = {
                "sid": sid,
                "atr": round(atr(d, CONFIG["atr_period"]), 2),
                "box_high": float(box["high"].max()), "box_low": float(box["low"].min()),
                # level ladders: highest high / lowest low of the last 1..N days (nearest one is the trigger)
                "res_levels": sorted({float(d["high"].tail(n).max()) for n in range(1, CONFIG["box_days"] + 1)}),
                "sup_levels": sorted({float(d["low"].tail(n).min()) for n in range(1, CONFIG["box_days"] + 1)},
                                     reverse=True),
                "prev_high": float(d["high"].iloc[-1]), "prev_low": float(d["low"].iloc[-1]),
                "prev_close": float(d["close"].iloc[-1]),
                "sma20": round(float(d["close"].tail(20).mean()), 2),
                "sma50": round(float(d["close"].tail(50).mean()), 2) if len(d) >= 50 else None,
                "chg_3m_pct": round((d["close"].iloc[-1] / d["close"].iloc[max(0, len(d) - 63)] - 1) * 100, 2),
                "avg_vol20": round(float(d["volume"].tail(20).mean()), 0),
                "slot_avg_vol": slot_avg, "cum_avg_vol": cum_avg, "warm_closes": warm,
            }
        except DhanAuthError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: prep failed ({e})")
    expiry = old.get("expiry")
    for sym in ("RELIANCE", "HDFCBANK", next(iter(symbols_all))):
        if expiry:
            break
        if sym in symbols_all:
            try:
                exps = [e for e in dh.expiries(symbols_all[sym]) if e >= str(day)]
                if exps:
                    expiry = sorted(exps)[0]
                    break
            except DhanAuthError:
                raise
            except Exception:  # noqa: BLE001
                pass
    prep = {"date": str(day), "expiry": expiry, "stocks": stocks}
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(prep))
    print(f"  done: {len(stocks)} stocks, nearest expiry {expiry}")
    return prep


# --------------------------------------------------------------------------------------
# Evaluation of one stock / one side
# --------------------------------------------------------------------------------------
def nifty_bias(dh: Dhan, quotes: dict, day: dt.date, now: dt.datetime) -> dict:
    q = (quotes.get("IDX_I") or {}).get(str(NIFTY_ID), {})
    ltp = q.get("last_price")
    o = (q.get("ohlc") or {}).get("open")
    pc = (q.get("ohlc") or {}).get("close")
    e9 = e21 = None
    try:
        n5 = dh.intraday(NIFTY_ID, dt.datetime.combine(day - dt.timedelta(days=4), dt.time(9, 15), IST), now,
                         seg="IDX_I", inst="INDEX")
        if len(n5) > 21:
            e9, e21 = float(ema(n5["close"], 9).iloc[-1]), float(ema(n5["close"], 21).iloc[-1])
    except DhanAuthError:
        raise
    except Exception:  # noqa: BLE001
        pass
    bias = "FLAT"
    if ltp and o:
        if ltp > o and (e9 is None or e9 > e21):
            bias = "UP"
        elif ltp < o and (e9 is None or e9 < e21):
            bias = "DOWN"
    return {"ltp": r2(ltp), "open": r2(o), "prev_close": r2(pc),
            "chg_pct": r2((ltp / pc - 1) * 100) if ltp and pc else None,
            "ema9_5m": r2(e9), "ema21_5m": r2(e21), "bias": bias}


def pick_option(chain: dict, side: str, ltp: float, entry: float):
    """Return ATM strike data and the nearest OI wall in the trade direction + the opposite floor."""
    oc = chain.get("oc") or {}
    rows = []
    for k, v in oc.items():
        try:
            strike = float(k)
        except ValueError:
            continue
        ce, pe = v.get("ce") or {}, v.get("pe") or {}
        rows.append((strike, ce, pe))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    leg = "ce" if side == "CE" else "pe"
    atm = min(rows, key=lambda x: abs(x[0] - ltp))
    opt = atm[1] if leg == "ce" else atm[2]
    if side == "CE":
        above = [(s, ce.get("oi", 0) or 0) for s, ce, _ in rows if s > entry]
        mx = max([oi for _, oi in above], default=0)
        wall = next((s for s, oi in above if mx and oi >= CONFIG["wall_frac"] * mx), None)
        below = [(s, pe.get("oi", 0) or 0) for s, _, pe in rows if s <= entry]
        floor = max(below, key=lambda x: x[1])[0] if below else None
    else:
        below = [(s, pe.get("oi", 0) or 0) for s, _, pe in reversed(rows) if s < entry]
        mx = max([oi for _, oi in below], default=0)
        wall = next((s for s, oi in below if mx and oi >= CONFIG["wall_frac"] * mx), None)
        above = [(s, ce.get("oi", 0) or 0) for s, ce, _ in rows if s >= entry]
        floor = max(above, key=lambda x: x[1])[0] if above else None
    return {"strike": atm[0], "opt": opt, "wall": wall, "opposite_wall": floor}


def nearest_level(p: dict, side: str, price: float) -> float:
    """Nearest resistance above (CE) / support below (PE) the price from the 1..5-day ladder.
    If price is already beyond all of them, return the outermost one (box edge)."""
    if side == "CE":
        lv = p.get("res_levels") or [p["box_high"]]
        return next((x for x in lv if x > price), lv[-1])
    lv = p.get("sup_levels") or [p["box_low"]]
    return next((x for x in lv if x < price), lv[-1])


def evaluate(dh, master, prep, sym, side, q, nifty, now, day, want_chain=True):
    p = prep["stocks"][sym]
    ohlc = q.get("ohlc") or {}
    ltp, o = float(q.get("last_price") or 0), float(ohlc.get("open") or 0)
    a = p["atr"]
    sign = 1 if side == "CE" else -1
    level = nearest_level(p, side, ltp)
    res = {"time": now.strftime("%H:%M:%S"), "symbol": sym, "side": side, "ltp": r2(ltp),
           "day_open": r2(o), "day_high": r2(ohlc.get("high")), "day_low": r2(ohlc.get("low")),
           "trigger_level": r2(level), "atr": r2(a)}
    fails, passes = [], []

    # 1. room left
    used = abs(ltp - o) / a if a and o else 9
    res["atr_used_pct"] = r2(used * 100)
    (passes if used <= CONFIG["max_atr_used"] else fails).append(f"ATR used {used*100:.0f}%")

    # 2. intraday candles (completed only)
    frm = dt.datetime.combine(day, dt.time(9, 15), IST)
    c = dh.intraday(p["sid"], frm, now)
    c = c[c.index.date == day]
    last_open_allowed = pd.Timestamp(now).tz_convert(IST) - pd.Timedelta(minutes=5)
    c = c[c.index <= last_open_allowed]  # keep only candles that have fully closed
    if len(c) < 4:
        res.update({"verdict": f"SKIP_{side}", "fails": ["fewer than 4 completed 5-min candles"], "passes": passes})
        return res
    last, prev3 = c.iloc[-1], c.iloc[-4:-1]
    # the level being broken = nearest ladder level beyond the previous candle's close
    level = nearest_level(p, side, float(c["close"].iloc[-2]))
    res["trigger_level"] = r2(level)
    closes = pd.concat([pd.Series(p.get("warm_closes") or [], dtype=float), c["close"].reset_index(drop=True)],
                       ignore_index=True)
    vw = vwap(c)
    e9, e21 = float(ema(closes, 9).iloc[-1]), float(ema(closes, 21).iloc[-1])
    rs, mh = rsi(closes), macd_hist(closes)
    slot = (c.index[-1]).strftime("%H:%M")
    cum_now, cum_avg = float(c["volume"].sum()), (p.get("cum_avg_vol") or {}).get(slot)
    rvol = cum_now / cum_avg if cum_avg else None
    vol_ratio = float(last["volume"] / prev3["volume"].mean()) if prev3["volume"].mean() else None
    res.update({"last_candle_time": c.index[-1].strftime("%H:%M"), "last_close": r2(last["close"]),
                "last_volume": int(last["volume"]), "vol_vs_prev3": r2(vol_ratio), "rel_volume_day": r2(rvol),
                "vwap": r2(vw), "ema9": r2(e9), "ema21": r2(e21), "rsi14_5m": r2(rs), "macd_hist_5m": r2(mh)})

    # 3. trigger: 5-min CLOSE beyond the box edge, fresh (this or previous candle crossed)
    beyond = sign * (last["close"] - level) > 0
    crossed_recent = any(sign * (x - level) <= 0 for x in c["close"].iloc[-3:-1])
    triggered = beyond and crossed_recent
    chase_pct = sign * (ltp - level) / level * 100
    dist_pct = -chase_pct  # % still to go to reach the level (positive = not reached)
    res["dist_to_trigger_pct"] = r2(dist_pct)
    res["triggered"] = bool(triggered)
    if triggered:
        passes.append(f"5-min close {last['close']:.2f} beyond {level:.2f}")
    elif beyond:
        fails.append("broke out more than 2 candles ago (not fresh)")
    else:
        fails.append(f"not triggered ({dist_pct:.2f}% to go)")
    # warn when the level was already crossed earlier today and price fell back (failed breakout)
    day_ext = c["high"].max() if side == "CE" else c["low"].min()
    if not beyond and sign * (day_ext - level) > 0:
        res["level_rejected_today"] = True
        fails.append(f"level already broken today (day {'high' if side == 'CE' else 'low'} "
                     f"{day_ext:.2f}) and price fell back - failed breakout")
    (passes if chase_pct <= CONFIG["max_chase_pct"] else fails).append(f"chase {max(chase_pct,0):.2f}%")
    vol_ok = vol_ratio is not None and vol_ratio > 1.0
    (passes if vol_ok else fails).append(f"breakout volume {vol_ratio or 0:.2f}x prev 3")
    t = now.strftime("%H:%M")
    time_ok = CONFIG["entry_start"] <= t <= CONFIG["entry_cutoff"]
    (passes if time_ok else fails).append(f"time {t}")
    vdist = sign * (ltp - vw) / vw * 100
    vwap_ok = 0 <= vdist <= CONFIG["max_vwap_dist_pct"]
    (passes if vwap_ok else fails).append(f"{vdist:+.2f}% from VWAP")
    nifty_ok = (side == "CE" and nifty["bias"] != "DOWN") or (side == "PE" and nifty["bias"] != "UP")
    (passes if nifty_ok else fails).append(f"Nifty {nifty['bias']}")

    # 4. trade plan on the underlying
    entry = max(ltp, level) if side == "CE" else min(ltp, level)
    risk = CONFIG["stop_atr_mult"] * a
    stop = entry - sign * risk
    plan = {"entry": r2(entry), "stop": r2(stop), "risk_per_share": r2(risk)}

    # 5. option chain: walls, strike, premium, lot
    opt_info = None
    if want_chain and prep.get("expiry"):
        try:
            opt_info = pick_option(dh.option_chain(p["sid"], prep["expiry"]), side, ltp, entry)
        except DhanAuthError:
            raise
        except Exception as e:  # noqa: BLE001
            fails.append(f"option chain error {e}")
    runway_ok = liquid_ok = False
    if opt_info:
        wall = opt_info["wall"]
        runway_r = (sign * (wall - entry) / risk) if wall else 99.0
        runway_ok = runway_r >= CONFIG["min_runway_r"]
        (passes if runway_ok else fails).append(f"runway {runway_r:.1f}R to wall {wall}")
        cap = (wall - sign * 0.1 * a) if wall else None
        t1, t2 = entry + sign * CONFIG["t1_r"] * risk, entry + sign * CONFIG["t2_r"] * risk
        if cap is not None:
            t1 = min(t1, cap) if side == "CE" else max(t1, cap)
            t2 = min(t2, cap) if side == "CE" else max(t2, cap)
        opt = opt_info["opt"] or {}
        prem = float(opt.get("last_price") or 0)
        bid, ask = float(opt.get("top_bid_price") or 0), float(opt.get("top_ask_price") or 0)
        spread = (ask - bid) / prem * 100 if prem and ask and bid else None
        liquid_ok = prem > 0 and (opt.get("oi") or 0) > 0 and (spread is None or spread <= CONFIG["max_option_spread_pct"])
        (passes if liquid_ok else fails).append(f"option liquidity (spread {spread if spread is None else round(spread,1)}%)")
        delta = abs(float(((opt.get("greeks") or {}).get("delta")) or 0.5)) or 0.5
        lot, opt_sid = lot_and_id(master, sym, prep["expiry"], opt_info["strike"], side)
        loss_lot = delta * risk * lot if lot else None
        plan.update({"t1": r2(t1), "t2": r2(t2), "next_wall": wall, "opposite_wall": opt_info["opposite_wall"],
                     "runway_r": r2(runway_r)})
        res["option"] = {
            "contract": f"{sym} {prep['expiry']} {opt_info['strike']:g} {side}",
            "security_id": opt_sid, "expiry": prep["expiry"], "strike": opt_info["strike"], "type": side,
            "premium": r2(prem), "bid": r2(bid), "ask": r2(ask), "iv": r2(opt.get("implied_volatility")),
            "oi": opt.get("oi"), "delta_used": r2(delta), "lot_size": lot,
            "cost_per_lot": r2(prem * lot) if lot else None,
            "premium_stop": r2(max(prem - delta * risk, 0.05)),
            "premium_t1": r2(prem + delta * abs(t1 - entry)), "premium_t2": r2(prem + delta * abs(t2 - entry)),
            "loss_per_lot_at_stop": r2(loss_lot),
            "loss_pct_of_capital": r2(loss_lot / CONFIG["capital"] * 100) if loss_lot else None,
        }
    res["plan"] = plan

    # 6. score (7 automatic factors; catalyst/news = null, judge it yourself / ask Claude)
    trend = (ltp > p["sma20"]) if side == "CE" else (ltp < p["sma20"])
    intra = ((ltp > vw and e9 > e21) if side == "CE" else (ltp < vw and e9 < e21))
    mom = ((55 <= rs <= 75 or mh > 0) if side == "CE" else (25 <= rs <= 45 or mh < 0))
    volf = bool(vol_ok and (rvol is None or rvol >= 0.8))
    factors = {"daily_trend": bool(trend), "intraday_trend": bool(intra), "momentum": bool(mom),
               "volume": volf, "breakout_close": bool(triggered), "oi_runway": bool(runway_ok),
               "catalyst": None, "nifty_aligned": bool(nifty_ok)}
    auto = sum(1 for v in factors.values() if v)
    res["factors"] = factors
    res["score"] = f"{auto}/7 auto (+ catalyst unknown)"

    # 7. verdict
    hard = [used <= CONFIG["max_atr_used"], triggered, chase_pct <= CONFIG["max_chase_pct"], vol_ok, time_ok,
            vwap_ok, runway_ok, liquid_ok, nifty_ok]
    if all(hard) and auto >= CONFIG["min_auto_score"]:
        verdict = f"BUY_{side}"
    elif (not triggered and 0 < dist_pct <= CONFIG["watch_dist_pct"] and used <= CONFIG["max_atr_used"]
          and time_ok and nifty_ok):
        verdict = f"WATCH_{side}"
    else:
        verdict = f"SKIP_{side}"
    over = res.get("option", {}).get("loss_pct_of_capital")
    if over and over > CONFIG["max_risk_pct"]:
        res["risk_warning"] = f"one lot risks {over}% of capital (max {CONFIG['max_risk_pct']}%)"
        if CONFIG["block_over_risk"] and verdict.startswith("BUY"):
            verdict = f"SKIP_{side}"
            fails.append("over max risk per trade")
    if verdict.startswith("WATCH"):
        res["watch_note"] = (f"Trigger: 5-min close {'above' if side=='CE' else 'below'} {level:.2f} "
                             f"with volume > avg of prior 3 candles, before {CONFIG['entry_cutoff']}")
    if verdict.startswith("BUY"):
        res["valid_range"] = [r2(level), r2(level * (1 + sign * CONFIG["max_chase_pct"] / 100))]
        res["rules"] = "Exit by 15:15. Move stop to entry after T1. Skip if price is already beyond valid_range."
    res["verdict"], res["fails"], res["passes"] = verdict, fails, passes
    return res


# --------------------------------------------------------------------------------------
# One scan
# --------------------------------------------------------------------------------------
def scan(dh, master, prep, day, exclude, fired, universe=None):
    now = dt.datetime.now(IST)
    syms = [s for s in (universe or prep["stocks"]) if s in prep["stocks"] and s not in exclude]
    quotes = dh.ohlc({"NSE_EQ": [prep["stocks"][s]["sid"] for s in syms], "IDX_I": [NIFTY_ID]})
    eq = quotes.get("NSE_EQ") or {}
    nifty = nifty_bias(dh, quotes, day, now)

    # cheap pre-filter on quotes: near a box edge and not already stretched
    pre = []
    for s in syms:
        q = eq.get(str(prep["stocks"][s]["sid"]))
        if not q or not q.get("last_price"):
            continue
        p, ltp = prep["stocks"][s], float(q["last_price"])
        o = float((q.get("ohlc") or {}).get("open") or ltp)
        used = abs(ltp - o) / p["atr"] if p["atr"] else 9
        for side, sign in (("CE", 1), ("PE", -1)):
            # nearest level ahead, or the one just broken (allow up to max_chase beyond it)
            ahead = nearest_level(p, side, ltp)
            behind = nearest_level(p, side, ltp * (1 - sign * CONFIG["max_chase_pct"] / 100))
            level = ahead if abs(ahead - ltp) <= abs(behind - ltp) else behind
            if side == "CE" and nifty["bias"] == "DOWN" or side == "PE" and nifty["bias"] == "UP":
                continue
            to_go = sign * (level - ltp) / ltp * 100  # % still to go (negative = beyond)
            if -CONFIG["max_chase_pct"] <= to_go <= CONFIG["watch_dist_pct"] and used <= CONFIG["max_atr_used"] + 0.1:
                pre.append((abs(to_go), s, side, q))
    pre.sort()
    pre = pre[: CONFIG["max_candidates_per_scan"]]

    results = []
    for _, s, side, q in pre:
        try:
            results.append(evaluate(dh, master, prep, s, side, q, nifty, now, day))
        except DhanAuthError:
            raise
        except Exception as e:  # noqa: BLE001
            results.append({"time": now.strftime("%H:%M:%S"), "symbol": s, "side": side,
                            "verdict": f"SKIP_{side}", "fails": [f"error: {e}"]})

    order = {"BUY": 0, "WATCH": 1, "SKIP": 2}
    results.sort(key=lambda r: order.get(r["verdict"].split("_")[0], 3))
    snapshot = {"scan_time": now.strftime("%Y-%m-%d %H:%M:%S"), "nifty": nifty,
                "expiry": prep.get("expiry"), "candidates_checked": len(results), "results": results}

    OUT.mkdir(exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(snapshot, indent=2))
    with open(OUT / f"scan_{day}.jsonl", "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"nifty_bias": nifty["bias"], **r}) + "\n")

    print_scan(snapshot)
    for r in results:
        key = f"{r['symbol']}_{r['side']}"
        if r["verdict"].startswith("BUY") and key not in fired:
            fired.add(key)
            signal = {"type": "SIGNAL", "scan_time": snapshot["scan_time"], "nifty": nifty, **r}
            with open(OUT / f"signals_{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(signal) + "\n")
            print("\n=========== SIGNAL FOR CLAUDE (copy from here) ===========")
            print(json.dumps(signal, indent=2))
            print("=========== END SIGNAL ===========\n")
            notify(r)
    return snapshot


def print_scan(snap):
    n = snap["nifty"]
    print(f"\n[{snap['scan_time']}] Nifty {n['ltp']} ({n['chg_pct']}%) bias {n['bias']} | "
          f"expiry {snap['expiry']} | candidates {snap['candidates_checked']}")
    if not snap["results"]:
        print("  no stock near a box edge with room left")
        return
    print(f"  {'VERDICT':10} {'SYMBOL':11} {'LTP':>9} {'LEVEL':>9} {'TOGO%':>6} {'ATR%':>5} {'VOLx':>5} {'SCORE':>5}  WHY NOT")
    for r in snap["results"]:
        fails = "; ".join(r.get("fails", [])[:3])
        print(f"  {r['verdict']:10} {r['symbol']:11} {r.get('ltp') or '':>9} {r.get('trigger_level') or '':>9} "
              f"{r.get('dist_to_trigger_pct') if r.get('dist_to_trigger_pct') is not None else '':>6} "
              f"{r.get('atr_used_pct') or '':>5} {r.get('vol_vs_prev3') or '':>5} "
              f"{(r.get('score') or '')[:3]:>5}  {fails}")


def notify(r):
    """Beep + optional Telegram (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."""
    print("\a", end="")
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    o, p = r.get("option", {}), r.get("plan", {})
    text = (f"{r['verdict']} {r['symbol']} @ {r['ltp']} (trigger {r['trigger_level']})\n"
            f"{o.get('contract')} ~Rs{o.get('premium')} lot {o.get('lot_size')}\n"
            f"Stop {p.get('stop')} / prem {o.get('premium_stop')} | T1 {p.get('t1')} / {o.get('premium_t1')} | "
            f"T2 {p.get('t2')} / {o.get('premium_t2')}\nLoss/lot {o.get('loss_per_lot_at_stop')} "
            f"{r.get('risk_warning', '')}")
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": chat, "text": text},
                      timeout=10)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def credentials():
    cid, tok = os.getenv("DHAN_CLIENT_ID"), os.getenv("DHAN_ACCESS_TOKEN")
    cfg = Path(__file__).resolve().parent / "config.json"
    if (not cid or not tok) and cfg.exists():
        c = json.loads(cfg.read_text())
        cid, tok = cid or c.get("client_id"), tok or c.get("access_token")
    if not cid or not tok:
        sys.exit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN (or config.json). See the top of this file.")
    return cid, tok


def hm(s):
    h, m = map(int, s.split(":"))
    return dt.time(h, m)


def main():
    ap = argparse.ArgumentParser(description="Intraday box-breakout scanner (Dhan API, read-only)")
    ap.add_argument("--once", action="store_true", help="run one scan now and exit")
    ap.add_argument("--exclude", default="", help="comma-separated symbols to skip (results today, F&O ban)")
    ap.add_argument("--universe", default="nifty50",
                    help="which stocks: nifty50 (default), niftynext50, nifty100, nifty200, fno (all F&O stocks), "
                         "watchlist (built-in list); combine with commas, e.g. nifty50,watchlist")
    args = ap.parse_args()
    exclude = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}

    dh = Dhan(*credentials())
    day = dt.datetime.now(IST).date()
    OUT.mkdir(exist_ok=True)
    try:
        eq, master = load_master(day)
        universe = build_universe(args.universe, day, eq, master)
        prep = prepare(dh, day, universe)
        fired: set[str] = set()
        sig_file = OUT / f"signals_{day}.jsonl"
        if sig_file.exists():  # don't repeat signals after a restart
            for line in sig_file.read_text().splitlines():
                j = json.loads(line)
                fired.add(f"{j['symbol']}_{j['side']}")

        if args.once:
            t = dt.datetime.now(IST).strftime("%H:%M")
            if not (CONFIG["entry_start"] <= t <= CONFIG["entry_cutoff"]):
                print(f"Note: {t} is outside the entry window {CONFIG['entry_start']}-{CONFIG['entry_cutoff']}, "
                      "so every result will be SKIP (time). This run only tests data and levels.")
            scan(dh, master, prep, day, exclude, fired, universe)
            return

        start, end = hm(CONFIG["run_start"]), hm(CONFIG["run_end"])
        print(f"Scanner running {CONFIG['run_start']}-{CONFIG['run_end']} IST, every 5 min. Ctrl+C to stop.")
        while True:
            now = dt.datetime.now(IST)
            if now.time() > end:
                print("Past run_end - stopping. Signals today:", sorted(fired) or "none")
                return
            # wake 20 s after the next 5-minute candle closes
            nxt = (now.replace(second=0, microsecond=0)
                   + dt.timedelta(minutes=5 - now.minute % 5)) + dt.timedelta(seconds=20)
            if now.time() < start:
                nxt = max(nxt, dt.datetime.combine(day, start, IST))
            time.sleep(max(1, (nxt - dt.datetime.now(IST)).total_seconds()))
            if dt.datetime.now(IST).time() < start:
                continue
            try:
                scan(dh, master, prep, day, exclude, fired, universe)
            except DhanAuthError:
                raise
            except requests.RequestException as e:
                print(f"  network error, will retry next candle: {e}")
    except DhanAuthError as e:
        sys.exit(f"Dhan rejected the token ({e}). Generate a new access token and restart.")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
