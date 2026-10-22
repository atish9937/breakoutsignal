#!/usr/bin/env python3
"""
breakout_scanner.py - Intraday "box breakout" scanner for NSE F&O stocks (Dhan API v2)
========================================================================================

Finds stocks that break out of their 5-day box with volume, still have room to run, and have
a clear runway to the next option-OI wall. Same rules as the "Intraday Breakout Checklist"
(JSW Steel, 15 Oct 2026).

By default it only reads data and prints/saves recommendations; optional live order
submission is disabled unless explicitly enabled (see "LIVE ORDER EXECUTION" below).

WHAT THE SCRIPT DECIDES, WHAT AI VALIDATES
    The script still makes the actual BUY_CE/BUY_PE recommendation itself, exactly as it always
    did: everything mechanical and gettable from the Dhan API (candles, ATR, box levels,
    VWAP/RSI/MACD, the fresh-breakout definition, option-chain walls/premiums/lot sizes, stop/
    target math) plus a quality score built from trend/momentum/VWAP/Nifty alignment. Clear the
    mechanical gates AND the quality-score bar (min_recommend_score) and the script calls it
    BUY_CE/BUY_PE - a real recommendation, not a neutral "here's some data" dump.

    What the script can never do is check things outside its own numbers: today's news/catalyst
    (it has no news access), whether several of today's recommendations are really the same bet
    in disguise (3 bank stocks all riding one Nifty move), and the broader market regime. So
    every BUY recommendation carries an "ai_validation" block - a checklist for you or an AI
    (e.g. Claude) to VALIDATE that recommendation against exactly those things, and sign off as
    VALIDATED / VETOED / SIZE-DOWN. AI reviews the call the script already made; it does not
    invent the call from a blank slate.

SETUP (one time)
    pip install requests pandas
    Set your Dhan credentials (Dhan web -> My Profile -> DhanHQ Trading APIs):
      Windows (PowerShell):  $env:DHAN_CLIENT_ID="1100xxxxxx"; $env:DHAN_ACCESS_TOKEN="eyJ..."
      Mac/Linux:             export DHAN_CLIENT_ID=1100xxxxxx; export DHAN_ACCESS_TOKEN=eyJ...
    (or copy .env.example to .env next to this file and set DHAN_CLIENT_ID and
    DHAN_ACCESS_TOKEN there; keep .env private with chmod 600)
    To renew Dhan Web-generated active tokens automatically when they are within 13 hours
    of expiry, set AUTO_RENEW_DHAN_TOKEN=true. The scanner reuses a token that is not near
    expiry, avoiding token rotation on every run. Renewal updates .env; expired tokens cannot
    be renewed. Keep the credentials in .env (not just exported variables) for persistence.
    API-key OAuth tokens are not supported by this renewal flow.
    To receive Telegram alerts for new BUY recommendations, create a bot with @BotFather,
    get your chat ID, then set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment
    or add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env; see .env.example.
    For a channel, add the bot as an administrator with permission to post;
    use @public_channel_name for a public channel or the -100... chat ID for a private channel.
    Telegram is optional; the scanner still beeps when it is not configured.
    To run breakout and trend-pullback together on one shared Dhan client and API rate limiter,
    use strategy_runner.py (Windows: run_all.bat; Unix: run_all_unix.sh). Do not start the
    standalone scanner loops at the same time as the combined runner.
    Dhan Web tokens expire after 24 h. With automatic renewal enabled, the scanner requests a
    replacement when it starts with 13 hours or less remaining and saves it into .env. Start the
    scanner before the token expires; once-daily runs can miss the window if they happen at or
    after expiry. If the token expires, enter a fresh Dhan Web token in .env. For a manual
    check/renewal, stop the scanner and run run.bat --renew-token-only before expiry.
    LIVE ORDER EXECUTION (optional, off unless explicitly enabled)
    Dhan order placement requires a static IP whitelisted in Dhan. Set
    ENABLE_LIVE_ORDERS=true in .env AND pass --enable-live-orders to the standalone
    breakout_scanner.py command to submit one INTRADAY LIMIT buy at the selected option's
    current ask for each new BUY signal. strategy_runner.py is recommendations-only.
    The selected contract must have an exact scrip-master security ID and lot size.
    Limit orders may remain unfilled. This only submits entries; it does not manage exits.
    Live orders do not wait for the AI-validation checklist. Each symbol/side is attempted at
    most once per day; rejected or uncertain submissions are not automatically retried.

UNIX / CRON
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp .env.example .env && chmod 600 .env
    Edit .env with Dhan credentials. Set AUTO_RENEW_DHAN_TOKEN=true to renew the active
    Dhan Web token only when within 13 hours of expiry. strategy_runner.py never places
    live orders. Live orders are available only in the standalone breakout scanner when
    ENABLE_LIVE_ORDERS=true is set and --enable-live-orders is passed.
    Run run_all_unix.sh --once to test all strategies without submitting orders.
    chmod +x run_unix.sh run_all_unix.sh
    The combined runner has one shared 5-minute loop. Add one daily cron entry
    (replace the old breakout-only scanner entry; do not run both):
      CRON_TZ=Asia/Kolkata
      0 9 * * 1-5 /var/www/breakoutsignal/run_all_unix.sh >> /var/www/breakoutsignal/scanner_out/cron.log 2>&1
    This starts both strategies at 09:00 IST; they scan sequentially using one Dhan client
    every five minutes from 09:20 through 12:35 IST (entry cutoff 12:30). The combined runner
    continues tracking saved BUY recommendations through 15:10 IST and sends the daily
    HTML performance report to TELEGRAM_CHAT_ID. It saves the SQLite ledger and HTML report
    under scanner_out/. Strategy checks run against the same candle-close time and shared
    API limiter.
    CRON_TZ sets the schedule timezone on cron implementations that support it, so the server
    timezone does not need to be changed. The script uses flock to prevent overlapping runs.
    Create scanner_out before adding the cron entry so the log redirection can open its file.
    To extend breakout entries to 15:00 IST while keeping trend-pullback at its normal
    12:30 entry cutoff, add --ignore-entry-cutoff to the combined runner cron command.
    The combined runner records both strategies' BUY recommendations in a local SQLite
    ledger, checks subsequent completed 1-minute underlying candles, and sends a colored
    HTML performance report to Telegram by 15:10 IST. During the session, a Dhan WebSocket
    recorder stores sampled option/underlying LTP and best bid/ask; after report delivery,
    Dhan's historical chart API archives post-signal 1-minute OHLC for the option and
    underlying. The HTML file and ledger are saved in scanner_out/. Install requirements.txt
    so the WebSocket recorder is available. Use strategy_runner.py --report-only to retry
    a failed Telegram upload. Breakout BUY recommendations stop at 15:00. WebSocket data
    starts only once this version is deployed and running; historical OHLC does not provide
    historical bid/ask or prove order fills. This is paper tracking, not actual fills or
    option P&L; the runner does not place live orders. After the intraday report, it also
    saves a separate swing-option candidate JSON and sends a paper-only Telegram summary
    after 15:35 IST, allowing the daily market candles to complete.
    Swing candidates use daily trend-pullback or 20-session-breakout checks, require an
    option expiry at least 7 calendar days away, compare ATM and slightly ITM quotes, and
    expire if the underlying trigger is not confirmed on the next trading session. They
    are not live signals. Compare exit rules with swing_option_backtest.py before relying
    on them; historical option candles lack bid/ask spreads and verified fills.
    Dhan tokens expire after 24 hours. To keep the token active across weekends and scanner
    downtime, add this all-days maintenance schedule (keep AUTO_RENEW_DHAN_TOKEN=true):
      0 6,18 * * * /var/www/breakoutsignal/run_unix.sh --renew-token-only >> /var/www/breakoutsignal/scanner_out/token-renewal.log 2>&1
    The 12-hour cadence and 13-hour renewal window leave a one-hour safety margin; these runs are
    outside scanner hours. If a renewal is missed until after expiry, generate a fresh Dhan Web
    access token and replace it in .env.
    Dhan order APIs require a static IP whitelisted in Dhan.

RUN
    python strategy_runner.py                   # breakout + trend pullback, shared client
    python strategy_runner.py --once            # one cycle of both strategies
    python breakout_scanner.py                  # breakout strategy only, standalone
    python breakout_scanner.py --once           # one scan now (for testing)
    python strategy_runner.py --ignore-entry-cutoff  # breakout BUY cutoff 15:00; daily report by 15:10 IST
    python strategy_runner.py --report-only          # retry/send today's saved HTML report
    python breakout_scanner.py --exclude TECHM,WIPRO   # skip stocks (results today / F&O ban)
    python breakout_scanner.py --universe nifty100      # which stocks to scan (default nifty50)

UNIVERSE (pulled fresh every day, cached in scanner_out/)
    nifty50, niftynext50, nifty100, nifty200, nifty500, niftymidcap50/100/150,
    niftysmallcap50/100/250, niftybank - official NSE constituent lists
    fno        - every stock that has stock options on NSE (from the Dhan scrip master, ~200)
    watchlist  - the built-in WATCHLIST below
    Combine with commas, e.g. nifty50,niftynext50,niftymidcap150,niftysmallcap250,niftybank.
    Stocks without stock options are dropped automatically.
    Symbols are mapped to Dhan security ids from the Dhan scrip master.
    Each successful download is saved as scanner_out/index_<name>_<date>.json. If NSE can't be
    reached, the most recent saved list is used. The built-in WATCHLIST is used only if no list
    has ever been downloaded. The start-up lines always say which source was used.

OUTPUT (folder ./scanner_out/)
    prep_YYYY-MM-DD.json      daily levels (box, ATR, prev day) - built once per day
    scan_YYYY-MM-DD.jsonl     every evaluation, one JSON per line
    latest.json               the most recent scan (market + all candidates)
    signals_YYYY-MM-DD.jsonl  only BUY_CE / BUY_PE recommendations
    orders_YYYY-MM-DD.jsonl   live order attempts and outcomes (when live orders are enabled)
    On screen: a table each scan, a "RECOMMENDATION FOR AI VALIDATION" JSON block for every new
    BUY, plus one "AI VALIDATION BATCH" block per scan with all of that scan's recommendations
    together (so the correlation/overlap check has something to compare). Paste either block
    into Claude (or your own AI layer) to validate it before you act - catalyst/news especially,
    since the script cannot check that itself.

THE CONCLUSIVE FIELD: "verdict"
    BUY_CE / BUY_PE    the script's own recommendation - mechanical gates + quality score both
                       passed. Still needs AI/human VALIDATION (see "ai_validation") before you
                       act on it - catalyst/news and cross-recommendation overlap aren't checked.
    WATCH_CE / WATCH_PE  near its trigger but not triggered yet - watch the level
    SKIP_CE / SKIP_PE    failed a mechanical gate or the quality-score bar ("fails" lists why)
    plus "risk_warning" when one lot's loss at the stop is above your max risk %.

Research tool, not SEBI-registered investment advice. Verify every level on your broker.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv, set_key

# --------------------------------------------------------------------------------------
# SETTINGS - change these to suit you
# --------------------------------------------------------------------------------------
CONFIG = {
    "capital": 72000,            # your trading capital in Rs
    "max_risk_pct": 2.0,         # max loss per trade as % of capital - reported via risk_warning
    "block_over_risk": False,    # True = a mechanical safety cap: downgrade BUY to SKIP if
                                  # one lot risks more than max_risk_pct. False (default) leaves
                                  # sizing to you/AI - the recommendation still shows with risk_warning set.
    "atr_period": 14,            # daily ATR length
    "box_days": 5,               # box = high/low of the last N completed days
    "max_atr_used": 0.60,        # mechanical: skip if |price - open| > 60% of ATR (move already done)
    "watch_dist_pct": 0.60,      # list as WATCH when within 0.6% of the box edge
    "max_chase_pct": 0.50,       # mechanical: don't recommend it if already >0.5% beyond the edge
    "max_vwap_dist_pct": 1.0,    # price must be within 1% of VWAP to count as "vwap_aligned"
    "stop_atr_mult": 0.35,       # stop distance = 0.35 x ATR (JSW: 0.35 x 30 = ~Rs 11)
    "min_runway_r": 2.0,         # mechanical: next OI wall must be >= 2R away, or there's no room to trade
    "t1_r": 1.5,                 # target 1 = 1.5R
    "t2_r": 2.5,                 # target 2 = 2.5R (capped just before the wall)
    "wall_frac": 0.5,            # a strike counts as a "wall" if its OI >= 50% of the biggest
    "min_recommend_score": 4,    # of 5 quality factors (catalyst/news is not counted - AI
                                  # validates that separately; see ai_validation on every BUY)
    "max_option_spread_pct": 3.0,  # mechanical: option must be liquid enough to actually fill
    "entry_start": "09:30",
    "entry_cutoff": "12:30",
    "run_start": "09:20",
    "run_end": "12:35",
    "extended_entry_cutoff": "15:00",
    "extended_run_end": "15:05",
    "max_candidates_per_scan": 15,  # option-chain calls are slow (1 per 3 s)
}

MIN_DISPLAY_QUALITY = 3

# The 5 quality factors (daily_trend, intraday_trend, momentum, vwap_aligned, nifty_aligned)
# drive the script's own BUY recommendation via min_recommend_score above - same role scoring
# always had. "catalyst" is the one factor the script can never fill in (no news access), so
# it's never counted in the script's score; AI/you must research it as part of validating any
# BUY the script recommends (see the "ai_validation" block on every BUY_CE/BUY_PE result).
QUALITY_FACTORS_NOTE = (
    "daily_trend, intraday_trend, momentum, vwap_aligned and nifty_aligned feed the script's own "
    "min_recommend_score bar. catalyst is always null - AI/you must check news before acting."
)

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
ENV_FILE = Path(__file__).resolve().parent / ".env"
TOKEN_RENEWAL_WINDOW_SECONDS = 13 * 60 * 60
DHAN_DATA_MIN_REQUEST_GAP_SECONDS = 0.21


# --------------------------------------------------------------------------------------
# Dhan REST client
# --------------------------------------------------------------------------------------
class DhanAuthError(RuntimeError):
    pass


class Dhan:
    """Minimal Dhan v2 client with simple rate limiting for read APIs."""

    def __init__(self, client_id: str, access_token: str):
        self.client_id = str(client_id)
        self.s = requests.Session()
        self.s.headers.update({
            "access-token": access_token,
            "client-id": str(client_id),
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._last: dict[str, float] = {}
        self._last_request: float | None = None
        self._request_lock = threading.Lock()

    def _post(self, path: str, body: dict, min_gap: float = 0.25,
              timeout: float | tuple[float, float] = 30) -> dict:
        for attempt in range(4):
            try:
                with self._request_lock:
                    now = time.monotonic()
                    rate_limit_key = (
                        "/optionchain"
                        if path in ("/optionchain", "/optionchain/expirylist")
                        else path
                    )
                    previous_path_request = self._last.get(rate_limit_key)
                    wait = max(
                        DHAN_DATA_MIN_REQUEST_GAP_SECONDS - (now - self._last_request)
                        if self._last_request is not None else 0.0,
                        min_gap - (now - previous_path_request)
                        if previous_path_request is not None else 0.0,
                        0.0,
                    )
                    if wait > 0:
                        time.sleep(wait)
                    started = time.monotonic()
                    self._last_request = started
                    self._last[rate_limit_key] = started
                    r = self.s.post(API + path, json=body, timeout=timeout)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
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

    def daily_range(self, sid, frm: dt.date, to: dt.date, seg="NSE_EQ", inst="EQUITY") -> pd.DataFrame:
        body = {"securityId": str(sid), "exchangeSegment": seg, "instrument": inst,
                "expiryCode": 0, "oi": False,
                "fromDate": str(frm), "toDate": str(to)}
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

    def rolling_option(self, body: dict) -> dict:
        return self._post("/charts/rollingoption", body, timeout=(10, 90))

    def place_order(self, body: dict) -> dict:
        """Submit a single order without automatic retries to avoid duplicate live orders."""
        response = self.s.post(API + "/orders", json=body, timeout=(10, 30))
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("Dhan order API returned invalid JSON") from exc
        if not isinstance(result, dict) or not result.get("orderId"):
            raise RuntimeError("Dhan order API response did not include an orderId")
        return result


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


INDEX_CSV = {  # official NSE constituent lists (NSE archive, with Nifty Indices mirror)
    "nifty50": "ind_nifty50list.csv",
    "niftynext50": "ind_niftynext50list.csv",
    "nifty100": "ind_nifty100list.csv",
    "nifty200": "ind_nifty200list.csv",
    "nifty500": "ind_nifty500list.csv",
    "niftymidcap50": "ind_niftymidcap50list.csv",
    "niftymidcap100": "ind_niftymidcap100list.csv",
    "niftymidcap150": "ind_niftymidcap150list.csv",
    "niftysmallcap50": "ind_niftysmallcap50list.csv",
    "niftysmallcap100": "ind_niftysmallcap100list.csv",
    "niftysmallcap250": "ind_niftysmallcap250list.csv",
    "niftybank": "ind_niftybanklist.csv",
}
INDEX_MIN_SYMBOLS = {
    "nifty50": 40,
    "niftynext50": 40,
    "nifty100": 80,
    "nifty200": 160,
    "nifty500": 400,
    "niftymidcap50": 40,
    "niftymidcap100": 80,
    "niftymidcap150": 120,
    "niftysmallcap50": 40,
    "niftysmallcap100": 80,
    "niftysmallcap250": 200,
    "niftybank": 10,
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
            if len(syms) >= INDEX_MIN_SYMBOLS[name]:
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
    """spec: comma list of supported NSE indices, fno, or watchlist.
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
            supported = ", ".join((*INDEX_CSV, "fno", "watchlist"))
            sys.exit(f"Unknown universe '{part}'. Choose from {supported}.")
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
    todo = {
        s: i for s, i in symbols.items()
        if s not in old["stocks"]
        or "ema20" not in old["stocks"][s]
        or "ema50" not in old["stocks"][s]
    }
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
                "ema20": round(float(ema(d["close"], 20).iloc[-1]), 2),
                "ema50": round(float(ema(d["close"], 50).iloc[-1]), 2),
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


def evaluate(dh, master, prep, sym, side, q, nifty, now, day, want_chain=True,
             ignore_entry_cutoff=False):
    p = prep["stocks"][sym]
    ohlc = q.get("ohlc") or {}
    ltp, o = float(q.get("last_price") or 0), float(ohlc.get("open") or 0)
    a = p["atr"]
    sign = 1 if side == "CE" else -1
    level = nearest_level(p, side, ltp)
    res = {"time": now.strftime("%H:%M:%S"), "symbol": sym, "side": side, "ltp": r2(ltp),
           "day_open": r2(o), "day_high": r2(ohlc.get("high")), "day_low": r2(ohlc.get("low")),
           "trigger_level": r2(level), "atr": r2(a), "mechanical_checks": []}
    fails, passes = [], []

    def record_check(name, current, criterion, passed):
        res["mechanical_checks"].append({
            "gate": name, "current": current, "criterion": criterion, "passed": bool(passed),
        })
        return bool(passed)

    # 1. room left
    used = abs(ltp - o) / a if a and o else 9
    res["atr_used_pct"] = r2(used * 100)
    atr_ok = record_check("ATR room", f"{used * 100:.2f}%", f"<= {CONFIG['max_atr_used'] * 100:.0f}%",
                          used <= CONFIG["max_atr_used"])
    (passes if atr_ok else fails).append(f"ATR used {used*100:.0f}%")

    # 2. intraday candles (completed only)
    frm = dt.datetime.combine(day, dt.time(9, 15), IST)
    c = dh.intraday(p["sid"], frm, now)
    c = c[c.index.date == day]
    last_open_allowed = pd.Timestamp(now).tz_convert(IST) - pd.Timedelta(minutes=5)
    c = c[c.index <= last_open_allowed]  # keep only candles that have fully closed
    if len(c) < 4:
        res["mechanical_checks"].append({
            "gate": "5-minute candle data", "current": f"{len(c)} completed candles",
            "criterion": "at least 4", "passed": False,
        })
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
    # Everything in this block is MECHANICAL (a formula on Dhan data) and gates the verdict -
    # it defines what "a fresh box breakout" even means, it isn't a judgment call.
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
    trigger_ok = record_check(
        "Fresh breakout",
        f"last close {last['close']:.2f}; trigger {level:.2f}; recent cross={crossed_recent}",
        "completed 5-min close beyond trigger, crossed within last 2 candles",
        triggered,
    )
    # warn when the level was already crossed earlier today and price fell back (failed breakout)
    day_ext = c["high"].max() if side == "CE" else c["low"].min()
    if not beyond and sign * (day_ext - level) > 0:
        res["level_rejected_today"] = True
        fails.append(f"level already broken today (day {'high' if side == 'CE' else 'low'} "
                     f"{day_ext:.2f}) and price fell back - failed breakout")
    chase_ok = record_check("Chase limit", f"{max(chase_pct, 0):.2f}%",
                            f"<= {CONFIG['max_chase_pct']:.2f}%", chase_pct <= CONFIG["max_chase_pct"])
    (passes if chase_ok else fails).append(f"chase {max(chase_pct,0):.2f}%")
    volume_current = f"{vol_ratio:.2f}x" if vol_ratio is not None else "unavailable"
    vol_ok = record_check("Breakout volume", volume_current, "> 1.00x previous 3-candle average",
                          vol_ratio is not None and vol_ratio > 1.0)
    (passes if vol_ok else fails).append(f"breakout volume {vol_ratio or 0:.2f}x prev 3")
    t = now.strftime("%H:%M")
    time_ok = is_entry_time(now, ignore_entry_cutoff)
    cutoff = CONFIG["extended_entry_cutoff"] if ignore_entry_cutoff else CONFIG["entry_cutoff"]
    time_ok = record_check("Entry time", t, f"{CONFIG['entry_start']}–{cutoff} IST", time_ok)
    time_note = (f"time {t} (extended cutoff {CONFIG['extended_entry_cutoff']})"
                 if ignore_entry_cutoff else f"time {t}")
    (passes if time_ok else fails).append(time_note)

    # INFORMATIONAL from here - reported for AI/you to weigh, never gates the verdict.
    vdist = sign * (ltp - vw) / vw * 100
    vwap_ok = 0 <= vdist <= CONFIG["max_vwap_dist_pct"]
    nifty_ok = (side == "CE" and nifty["bias"] != "DOWN") or (side == "PE" and nifty["bias"] != "UP")
    res["vwap_note"] = f"{vdist:+.2f}% from VWAP ({'within' if vwap_ok else 'outside'} {CONFIG['max_vwap_dist_pct']}% band)"
    res["nifty_note"] = f"Nifty bias {nifty['bias']}, trade is {'aligned' if nifty_ok else 'against it'}"

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
    if opt_info is None and not any("option chain error" in f for f in fails):
        fails.append("no option chain data (no expiry or chain unavailable) - "
                     "runway and liquidity can't be checked")
    if opt_info:
        wall = opt_info["wall"]
        runway_r = (sign * (wall - entry) / risk) if wall is not None and risk > 0 else None
        runway_ok = runway_r is not None and runway_r >= CONFIG["min_runway_r"]
        record_check("OI-wall runway", f"{runway_r:.2f}R to {wall}" if runway_r is not None else "no wall found",
                     f">= {CONFIG['min_runway_r']:.2f}R to a detected wall", runway_ok)
        runway_note = (f"runway {runway_r:.1f}R to wall {wall}" if runway_r is not None
                       else "no option OI wall found")
        (passes if runway_ok else fails).append(runway_note)
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
        spread_current = f"{spread:.2f}%" if spread is not None else "not available"
        liquidity_current = (f"premium {prem:.2f}; OI {opt.get('oi') or 0}; spread {spread_current}")
        record_check(
            "Option liquidity", liquidity_current,
            f"premium > 0; OI > 0; spread <= {CONFIG['max_option_spread_pct']:.2f}% when bid/ask available",
            liquid_ok,
        )
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
    else:
        record_check("OI-wall runway", "option-chain data unavailable", f">= {CONFIG['min_runway_r']:.2f}R", False)
        record_check("Option liquidity", "option-chain data unavailable",
                     f"premium > 0; OI > 0; spread <= {CONFIG['max_option_spread_pct']:.2f}% when bid/ask available",
                     False)
    res["plan"] = plan

    # 6. quality/conviction factors. These feed the script's own BUY recommendation below (same
    # role they always had) - they are also handed to AI as "factors"/"quality_score" so AI can
    # re-weigh them qualitatively during validation, rather than treating the count as gospel.
    # "catalyst" is always null: the script has no news access, so it is never part of the
    # script's own score - AI must research it as part of validating the recommendation.
    trend = (ltp > p["sma20"]) if side == "CE" else (ltp < p["sma20"])
    intra = ((ltp > vw and e9 > e21) if side == "CE" else (ltp < vw and e9 < e21))
    mom = ((55 <= rs <= 75 or mh > 0) if side == "CE" else (25 <= rs <= 45 or mh < 0))
    factors = {"daily_trend": bool(trend), "intraday_trend": bool(intra), "momentum": bool(mom),
               "vwap_aligned": bool(vwap_ok), "nifty_aligned": bool(nifty_ok), "catalyst": None}
    quality_true = sum(1 for v in factors.values() if v is True)
    quality_total = sum(1 for v in factors.values() if v is not None)
    res["factors"] = factors
    res["quality_score"] = f"{quality_true}/{quality_total} (catalyst not counted - unknown to the script)"

    # 7. verdict. MECHANICAL gates (fresh trigger, not chased, breakout volume, ATR budget, entry
    # window, OI-wall runway, liquid option) must ALL pass - that's the checklist's own definition
    # of a valid box breakout, not a judgment call. On top of that, the script makes its own
    # BUY_CE/BUY_PE recommendation exactly as it always did: mechanical gates passed AND the
    # quality score clears min_recommend_score. AI's role (see "ai_validation" below) is to
    # VALIDATE that already-made recommendation against catalyst/news, cross-candidate
    # correlation and regime - things the script can't check - not to originate the call itself.
    mechanical_gates = {
        "room left (ATR used)": atr_ok,
        "fresh trigger": trigger_ok,
        "not already chased": chase_ok,
        "breakout volume": vol_ok,
        "within entry window": time_ok,
        "runway to next OI wall": runway_ok,
        "option tradeable/liquid": liquid_ok,
    }
    mechanical_ok = all(mechanical_gates.values())
    if mechanical_ok and quality_true >= CONFIG["min_recommend_score"]:
        verdict = f"BUY_{side}"
    elif (not triggered and 0 < dist_pct <= CONFIG["watch_dist_pct"] and used <= CONFIG["max_atr_used"]
          and time_ok):
        verdict = f"WATCH_{side}"
    else:
        verdict = f"SKIP_{side}"
        res["failed_gates"] = [name for name, ok in mechanical_gates.items() if not ok]
        if mechanical_ok:
            fails.append(f"quality score {quality_true}/{quality_total} below "
                         f"min_recommend_score {CONFIG['min_recommend_score']}")
    over = res.get("option", {}).get("loss_pct_of_capital")
    if over and over > CONFIG["max_risk_pct"]:
        res["risk_warning"] = f"one lot risks {over}% of capital (max {CONFIG['max_risk_pct']}%)"
        if CONFIG["block_over_risk"] and verdict.startswith("BUY"):
            verdict = f"SKIP_{side}"
            fails.append("over max risk per trade (block_over_risk is on)")
    if verdict.startswith("WATCH"):
        cutoff_note = (f"before {CONFIG['extended_entry_cutoff']} (extended cutoff)" if ignore_entry_cutoff
                       else f"before {CONFIG['entry_cutoff']}")
        res["watch_note"] = (f"Trigger: 5-min close {'above' if side=='CE' else 'below'} {level:.2f} "
                             f"with volume > avg of prior 3 candles, {cutoff_note}")
    if verdict.startswith("BUY"):
        res["valid_range"] = [r2(level), r2(level * (1 + sign * CONFIG["max_chase_pct"] / 100))]
        res["rules"] = "Exit by 15:15. Move stop to entry after T1. Skip if price is already beyond valid_range."
        res["ai_validation"] = {
            "status": "RECOMMENDED_BY_SCRIPT_NEEDS_AI_VALIDATION",
            "why": (f"The script's own checklist recommends {verdict}: all mechanical gates "
                    f"passed and quality score {quality_true}/{quality_total} cleared "
                    f"min_recommend_score {CONFIG['min_recommend_score']}. AI validates this "
                    "recommendation against what the script can't check - it does not re-decide "
                    "from a blank slate."),
            "checklist_for_ai": [
                "Catalyst/news: search for results, filings, rating changes or sector news on "
                f"{sym} today. Contradicting news (e.g. bad news on a BUY_CE) should veto the "
                "recommendation; supportive or neutral news lets it stand.",
                "Sanity-check the quality factors qualitatively (see 'factors'/'quality_score') "
                "- the script's count is a heuristic, not proof; flag it if something looks off.",
                "Check this recommendation against the other entries in the same scan "
                "('scan_recommendations' in the batch block) for sector/index overlap - several "
                "correlated recommendations are not independent bets and may call for smaller size.",
                "Consider the broader regime (Nifty bias/chg_pct, how many BUYs vs SKIPs this "
                "scan found) - breakout systems do worse in a choppy/range-bound tape.",
                "If risk_warning is set, decide whether to size down rather than skip outright.",
                "Give a final sign-off: VALIDATED / VETOED / SIZE-DOWN, with reasoning.",
            ],
        }
    res["verdict"], res["fails"], res["passes"] = verdict, fails, passes
    return res


# --------------------------------------------------------------------------------------
# One scan
# --------------------------------------------------------------------------------------
def scan(dh, master, prep, day, exclude, fired, universe=None, ignore_entry_cutoff=False,
         live_orders_enabled=False, now=None):
    now = now or dt.datetime.now(IST)
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
            # Nifty alignment only feeds the quality score in evaluate(), it isn't a hard gate -
            # don't pre-filter on it here either, or a counter-trend setup never even gets scored.
            to_go = sign * (level - ltp) / ltp * 100  # % still to go (negative = beyond)
            if -CONFIG["max_chase_pct"] <= to_go <= CONFIG["watch_dist_pct"] and used <= CONFIG["max_atr_used"] + 0.1:
                pre.append((abs(to_go), s, side, q))
    pre.sort()
    pre = pre[: CONFIG["max_candidates_per_scan"]]

    results = []
    for _, s, side, q in pre:
        try:
            results.append(evaluate(dh, master, prep, s, side, q, nifty, now, day,
                                   ignore_entry_cutoff=ignore_entry_cutoff))
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
    new_signals = []
    for r in results:
        key = f"{r['symbol']}_{r['side']}"
        if r["verdict"].startswith("BUY") and key not in fired:
            fired.add(key)
            new_signals.append(r)
            order_state = {"status": "DISABLED"}
            if live_orders_enabled:
                order_state = {"status": "ARMED"}
            signal = {"type": "SIGNAL", "scan_time": snapshot["scan_time"], "nifty": nifty,
                      "execution": order_state, **r}
            with open(OUT / f"signals_{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(signal) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print("\n=========== RECOMMENDATION FOR AI VALIDATION (copy from here) ===========")
            print(json.dumps(signal, indent=2))
            print("=========== END RECOMMENDATION - script says BUY, AI must validate it ===========\n")
            if live_orders_enabled:
                order_state = submit_live_order(dh, r, day)
            notify(r, order_state)
    if new_signals:
        # One batch per scan, every new recommendation together, so the cross-recommendation
        # correlation and regime checks in "ai_validation" actually have something to compare.
        batch = {
            "type": "AI_VALIDATION_BATCH", "scan_time": snapshot["scan_time"], "nifty": nifty,
            "expiry": snapshot["expiry"],
            "scan_recommendations": [{"symbol": r["symbol"], "side": r["side"], "ltp": r.get("ltp"),
                                       "quality_score": r.get("quality_score")}
                                      for r in new_signals],
            "recommendations": new_signals,
        }
        print("\n=========== AI VALIDATION BATCH (all new recommendations this scan, copy from here) ===========")
        print(json.dumps(batch, indent=2))
        print("=========== END AI VALIDATION BATCH ===========\n")
    return snapshot


def print_scan(snap):
    n = snap["nifty"]
    print(f"\n[{snap['scan_time']}] Nifty {n['ltp']} ({n['chg_pct']}%) bias {n['bias']} | "
          f"expiry {snap['expiry']} | candidates {snap['candidates_checked']}")
    if not snap["results"]:
        print("  no stock near a box edge with room left")
        return
    visible_results = []
    hidden_count = 0
    for result in snap["results"]:
        factor_values = result.get("factors")
        if factor_values is not None:
            quality_count = sum(value is True for value in factor_values.values())
            if quality_count < MIN_DISPLAY_QUALITY:
                hidden_count += 1
                continue
        visible_results.append(result)

    if not visible_results:
        print(f"  no rows with QUAL >= {MIN_DISPLAY_QUALITY}/5 "
              f"({hidden_count} row(s) hidden below the display threshold)")
        return

    print(f"  Showing rows with QUAL >= {MIN_DISPLAY_QUALITY}/5")
    print(f"  {'VERDICT':8} {'SYMBOL':11} {'LTP':>9} {'LEVEL':>9} {'TOGO%':>6} {'ATR%':>5} {'VOLx':>5} {'QUAL':>5}  WHY NOT")
    for r in visible_results:
        fails = "; ".join(r.get("fails", [])[:3])
        print(f"  {r['verdict']:8} {r['symbol']:11} {r.get('ltp') or '':>9} {r.get('trigger_level') or '':>9} "
              f"{r.get('dist_to_trigger_pct') if r.get('dist_to_trigger_pct') is not None else '':>6} "
              f"{r.get('atr_used_pct') or '':>5} {r.get('vol_vs_prev3') or '':>5} "
              f"{(r.get('quality_score') or '')[:5]:>5}  {fails}")
        for check in r.get("mechanical_checks", []):
            status = "PASS" if check["passed"] else "FAIL"
            print(f"      {status}: {check['gate']} | current: {check['current']} | required: {check['criterion']}")
    if hidden_count:
        print(f"  ({hidden_count} row(s) hidden below {MIN_DISPLAY_QUALITY}/5 quality)")


def notify(r, order_state=None):
    """Beep + optional Telegram (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID).
    This is the script's own BUY recommendation - still needs AI/human validation (ai_validation)
    against catalyst/news and cross-recommendation overlap before you act on it."""
    print("\a", end="")
    load_env()
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    o, p = r.get("option", {}), r.get("plan", {})
    text = (f"[BREAKOUT INTRADAY OPTIONS SIGNAL]\n"
            f"{r['verdict']} {r['symbol']} @ {r['ltp']} (trigger {r['trigger_level']}) - "
            f"validate catalyst/news before acting\n"
            f"{o.get('contract')} ~Rs{o.get('premium')} lot {o.get('lot_size')}\n"
            f"Stop {p.get('stop')} / prem {o.get('premium_stop')} | T1 {p.get('t1')} / {o.get('premium_t1')} | "
            f"T2 {p.get('t2')} / {o.get('premium_t2')}\nLoss/lot {o.get('loss_per_lot_at_stop')} "
            f"{r.get('risk_warning', '')}")
    if order_state:
        text += f"\nOrder: {order_state['status']}"
        if order_state.get("order_id"):
            text += f" (Dhan ID {order_state['order_id']}, {order_state.get('order_status', 'status unknown')})"
        if order_state.get("reason"):
            text += f" - {order_state['reason']}"
    send_telegram_message(tok, chat, text)


def send_telegram_message(token, chat_id, text):
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Telegram notification failed ({type(exc).__name__}); check bot credentials and connectivity.",
              file=sys.stderr)
        return

    try:
        result = response.json()
    except ValueError:
        print("Telegram notification failed: API returned invalid JSON.", file=sys.stderr)
        return
    if not isinstance(result, dict):
        print("Telegram notification failed: API returned an unexpected response.", file=sys.stderr)
        return
    if not result.get("ok"):
        description = result.get("description", "unknown Telegram API error")
        print(f"Telegram notification failed: {description}", file=sys.stderr)


def build_order_payload(r, client_id, day):
    side = r.get("side")
    option = r.get("option") or {}
    if r.get("verdict") != f"BUY_{side}" or side not in ("CE", "PE"):
        raise ValueError("only BUY_CE and BUY_PE signals can be submitted")
    if r.get("risk_warning"):
        raise ValueError(f"signal exceeds the configured one-lot risk limit: {r['risk_warning']}")
    security_id = option.get("security_id")
    lot_size = option.get("lot_size")
    ask = option.get("ask")
    if not security_id or not str(security_id).isdigit():
        raise ValueError("exact option security ID is unavailable from the Dhan scrip master")
    if not isinstance(lot_size, int) or lot_size <= 0:
        raise ValueError("valid option lot size is unavailable")
    if option.get("type") != side:
        raise ValueError("selected option type does not match the recommendation")
    if not isinstance(ask, (int, float)) or not math.isfinite(ask) or ask <= 0:
        raise ValueError("a valid current option ask price is unavailable")
    if ask * lot_size > CONFIG["capital"]:
        raise ValueError("one-lot premium cost exceeds CONFIG['capital']")

    signal_key = f"{day}:{r.get('symbol')}:{side}"
    correlation_id = "bs" + hashlib.sha256(signal_key.encode("utf-8")).hexdigest()[:16]
    return {
        "dhanClientId": str(client_id),
        "correlationId": correlation_id,
        "transactionType": "BUY",
        "exchangeSegment": "NSE_FNO",
        "productType": "INTRADAY",
        "orderType": "LIMIT",
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": lot_size,
        "disclosedQuantity": 0,
        "price": float(ask),
        "triggerPrice": 0,
        "afterMarketOrder": False,
        "amoTime": "",
    }


def append_order_event(day, event):
    path = OUT / f"orders_{day}.jsonl"
    OUT.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()
        os.fsync(f.fileno())


def submit_live_order(dh, r, day):
    """Place one lot once; ambiguous API errors are logged, never retried automatically."""
    try:
        payload = build_order_payload(r, dh.client_id, day)
    except ValueError as exc:
        state = {"status": "BLOCKED", "reason": str(exc)}
        append_order_event(day, {"time": dt.datetime.now(IST).isoformat(), "symbol": r["symbol"],
                                 "side": r["side"], **state})
        print(f"  order blocked for {r['symbol']} {r['side']}: {exc}", file=sys.stderr)
        return state

    correlation_id = payload["correlationId"]
    attempt = {
        "time": dt.datetime.now(IST).isoformat(),
        "symbol": r["symbol"],
        "side": r["side"],
        "correlation_id": correlation_id,
        "security_id": payload["securityId"],
        "quantity": payload["quantity"],
        "limit_price": payload["price"],
        "status": "SUBMITTING",
    }
    append_order_event(day, attempt)
    try:
        response = dh.place_order(payload)
    except (requests.RequestException, RuntimeError) as exc:
        state = {"status": "UNKNOWN", "reason": type(exc).__name__, "correlation_id": correlation_id}
        print(f"  order submission outcome unknown for {r['symbol']} {r['side']} "
              f"({type(exc).__name__}); check Dhan order book before taking action.",
              file=sys.stderr)
    else:
        order_status = str(response.get("orderStatus") or "UNKNOWN")
        state = {
            "status": "REJECTED" if order_status == "REJECTED" else "SUBMITTED",
            "order_id": str(response["orderId"]),
            "order_status": order_status,
            "correlation_id": correlation_id,
        }
        if state["status"] == "REJECTED":
            state["reason"] = "Dhan rejected the order"
    append_order_event(day, {"time": dt.datetime.now(IST).isoformat(), "symbol": r["symbol"],
                             "side": r["side"], **state})
    return state


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def load_env():
    """Load optional .env values without overriding explicitly exported environment variables."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def credentials():
    load_env()
    cid, tok = os.getenv("DHAN_CLIENT_ID"), os.getenv("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        sys.exit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env or the process environment.")
    if (os.getenv("AUTO_RENEW_DHAN_TOKEN", "").strip().lower() in {"1", "true", "yes", "on"}
            and token_expires_within(tok, TOKEN_RENEWAL_WINDOW_SECONDS)):
        try:
            tok = renew_dhan_access_token(cid, tok)
        except RuntimeError as exc:
            sys.exit(str(exc))
        print("Dhan access token renewed and saved to .env.")
    return cid, tok


def token_expires_within(access_token, seconds):
    """Read JWT exp for renewal scheduling; this does not validate the token's signature."""
    parts = access_token.split(".")
    if len(parts) != 3:
        raise RuntimeError("Cannot determine Dhan token expiry; provide a valid JWT access token")
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Cannot read Dhan token expiry from its JWT payload") from exc
    expiry = claims.get("exp") if isinstance(claims, dict) else None
    if not isinstance(expiry, (int, float)):
        raise RuntimeError("Dhan token JWT does not contain a numeric expiry; cannot schedule renewal safely")
    return expiry - time.time() <= seconds


def renew_dhan_access_token(client_id, access_token):
    """Renew an active Dhan Web token once and persist the replacement into .env."""
    if not ENV_FILE.is_file():
        raise RuntimeError("AUTO_RENEW_DHAN_TOKEN requires credentials to be stored in the local .env file")

    env_values = dotenv_values(ENV_FILE)
    if env_values.get("DHAN_ACCESS_TOKEN") != access_token:
        raise RuntimeError(
            "AUTO_RENEW_DHAN_TOKEN requires DHAN_ACCESS_TOKEN in .env to match the active token; "
            "remove any overriding DHAN_ACCESS_TOKEN from the process environment"
        )
    if os.name == "posix":
        os.chmod(ENV_FILE, 0o600)

    try:
        response = requests.get(
            f"{API}/RenewToken",
            headers={"access-token": access_token, "dhanClientId": str(client_id)},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        status_note = f" (HTTP {status})" if status is not None else ""
        raise RuntimeError(
            f"Dhan token renewal failed{status_note} ({type(exc).__name__}); "
            "ensure it is an unexpired Dhan Web-generated token"
        ) from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("Dhan token renewal returned invalid JSON") from exc
    if isinstance(result, dict):
        token_data = result.get("data") if isinstance(result.get("data"), dict) else result
        new_token = (
            token_data.get("accessToken")
            or token_data.get("access_token")
            or token_data.get("token")
        )
        if not isinstance(new_token, str):
            fields = ", ".join(sorted(str(key) for key in result))
            nested = result.get("data")
            if isinstance(nested, dict):
                fields += f"; data fields: {', '.join(sorted(str(key) for key in nested))}"
            raise RuntimeError(
                f"Dhan token renewal returned HTTP 200 without an access token "
                f"(response fields: {fields or 'none'}); the existing .env token was not changed"
            )
    else:
        raise RuntimeError(
            f"Dhan token renewal returned unexpected response type {type(result).__name__}; "
            "the existing .env token was not changed"
        )
    new_token = new_token.strip()
    if not new_token or new_token == access_token:
        raise RuntimeError("Dhan token renewal returned an empty or unchanged access token")

    success, key, _ = set_key(
        str(ENV_FILE),
        "DHAN_ACCESS_TOKEN",
        new_token,
        quote_mode="never",
    )
    if not success or key != "DHAN_ACCESS_TOKEN":
        raise RuntimeError("Could not persist the renewed token to .env")
    os.environ["DHAN_ACCESS_TOKEN"] = new_token
    return new_token


def hm(s):
    h, m = map(int, s.split(":"))
    return dt.time(h, m)


def is_entry_time(now: dt.datetime, ignore_entry_cutoff: bool = False) -> bool:
    current = now.strftime("%H:%M")
    cutoff = (CONFIG["extended_entry_cutoff"] if ignore_entry_cutoff
              else CONFIG["entry_cutoff"])
    return CONFIG["entry_start"] <= current <= cutoff


def main():
    ap = argparse.ArgumentParser(description="Intraday box-breakout scanner (Dhan API)")
    ap.add_argument("--once", action="store_true", help="run one scan now and exit")
    ap.add_argument("--ignore-entry-cutoff", action="store_true",
                    help="extend entry cutoff to 15:00 IST and keep scheduled scans running until 15:05")
    ap.add_argument("--exclude", default="", help="comma-separated symbols to skip (results today, F&O ban)")
    ap.add_argument("--universe", default="nifty50",
                    help="stocks from supported NSE indices (nifty50 default), fno (all F&O stocks), or watchlist; "
                         "combine with commas, e.g. nifty50,niftymidcap150,niftybank")
    ap.add_argument("--enable-live-orders", action="store_true",
                    help="allow live Dhan orders only when ENABLE_LIVE_ORDERS=true in .env")
    ap.add_argument("--renew-token-only", action="store_true",
                    help="renew the active Dhan Web token if near expiry, save it to .env, and exit")
    args = ap.parse_args()
    exclude = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}

    load_env()
    if args.renew_token_only:
        if os.getenv("AUTO_RENEW_DHAN_TOKEN", "").strip().lower() not in {"1", "true", "yes", "on"}:
            sys.exit("--renew-token-only requires AUTO_RENEW_DHAN_TOKEN=true in .env")
        current_token = os.getenv("DHAN_ACCESS_TOKEN")
        credentials()
        if os.getenv("DHAN_ACCESS_TOKEN") == current_token:
            print("Dhan access token is valid; renewal is not due yet.")
        return

    live_orders_enabled = (
        args.enable_live_orders
        and os.getenv("ENABLE_LIVE_ORDERS", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    if args.enable_live_orders and not live_orders_enabled:
        print("Live order submission remains disabled: set ENABLE_LIVE_ORDERS=true in .env "
              "as well as passing --enable-live-orders.")

    dh = Dhan(*credentials())
    day = dt.datetime.now(IST).date()
    OUT.mkdir(exist_ok=True)
    try:
        eq, master = load_master(day)
        universe = build_universe(args.universe, day, eq, master)
        prep = prepare(dh, day, universe)
        fired: set[str] = set()
        sig_file = OUT / f"signals_{day}.jsonl"
        if sig_file.exists():  # don't repeat recommendations after a restart
            for line in sig_file.read_text().splitlines():
                j = json.loads(line)
                fired.add(f"{j['symbol']}_{j['side']}")

        if args.once:
            now = dt.datetime.now(IST)
            t = now.strftime("%H:%M")
            entry_cutoff = (CONFIG["extended_entry_cutoff"] if args.ignore_entry_cutoff
                            else CONFIG["entry_cutoff"])
            if not is_entry_time(now, args.ignore_entry_cutoff):
                print(f"Note: {t} is outside the entry window {CONFIG['entry_start']}-{entry_cutoff}, "
                      "so every result will be SKIP (time). This run only tests data and levels.")
            elif args.ignore_entry_cutoff and t > CONFIG["entry_cutoff"]:
                print(f"Note: {t} is after the usual {CONFIG['entry_cutoff']} cutoff; "
                      f"the extended cutoff {entry_cutoff} is active for this run.")
            scan(dh, master, prep, day, exclude, fired, universe,
                 ignore_entry_cutoff=args.ignore_entry_cutoff,
                 live_orders_enabled=live_orders_enabled)
            return

        start = hm(CONFIG["run_start"])
        end_text = CONFIG["extended_run_end"] if args.ignore_entry_cutoff else CONFIG["run_end"]
        end = hm(end_text)
        if args.ignore_entry_cutoff:
            print(f"Scanner running {CONFIG['run_start']}-{end_text} IST, every 5 min; "
                  f"entry cutoff extended to {CONFIG['extended_entry_cutoff']}. Ctrl+C to stop.")
        else:
            print(f"Scanner running {CONFIG['run_start']}-{end_text} IST, every 5 min. Ctrl+C to stop.")
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
            after_sleep = dt.datetime.now(IST)
            if after_sleep.time() < start:
                continue
            scheduled_end = dt.datetime.combine(day, end, IST)
            if after_sleep > scheduled_end:
                print("Past run_end - stopping. Signals today:", sorted(fired) or "none")
                return
            try:
                scan(dh, master, prep, day, exclude, fired, universe,
                     ignore_entry_cutoff=args.ignore_entry_cutoff,
                     live_orders_enabled=live_orders_enabled)
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
