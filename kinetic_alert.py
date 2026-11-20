"""Penny-stock movers (+20%) -> Kinetic Gaussian turns green on 30m / 1h -> Telegram alert.

Run:   python kinetic_alert.py            (loop)
       python kinetic_alert.py --once     (one cycle, then exit)
       python kinetic_alert.py --check SYM   (print Gaussian state/flips to compare with TradingView)
       python kinetic_alert.py --get-chat-id (find your Telegram chat id)
       python kinetic_alert.py --test-telegram
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).parent
ET = ZoneInfo("America/New_York")
INTERVALS = {"30m": pd.Timedelta(minutes=30), "60m": pd.Timedelta(minutes=60)}
LABEL = {"30m": "30분", "60m": "1시간"}


# ---------------------------------------------------------------- config
def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env(HERE / ".env")


def env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


TG_TOKEN = env("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = env("TELEGRAM_CHAT_ID", "")
MIN_CHANGE = env("MIN_CHANGE_PCT", 20.0, float)   # day % change to enter the watchlist
MAX_PRICE = env("MAX_PRICE", 5.0, float)          # "penny" = below this price
MIN_VOLUME = env("MIN_VOLUME", 500_000, int)      # day volume filter (kills illiquid junk)
INCLUDE_OTC = env("INCLUDE_OTC", 0, int)
MAX_WATCH = env("MAX_WATCH", 60, int)
POLL = env("POLL_SECONDS", 60, int)
USE_FORMING_BAR = env("USE_FORMING_BAR", 1, int)  # 1 = alert intrabar (live), 0 = only on bar close
FLIP_LOOKBACK = env("FLIP_LOOKBACK_BARS", 2, int)  # also catch flips within the last N bars
PREPOST = env("PREPOST", 1, int)                  # include pre/after-hours bars
MOVER_ALERTS = env("MOVER_ALERTS", 1, int)        # alert when a penny stock first hits +MIN_CHANGE
R2_ALERTS = env("R2_ALERTS", 1, int)              # alert when a watch-list stock trades above the 3-minute Pivot R2
R2_MIN_RANGE_PCT = env("R2_MIN_RANGE_PCT", 0.5, float)  # skip if the previous 3m bar's (high-low)/close is below this %
R2_MAX_AGE_SEC = env("R2_MAX_AGE_SEC", 240, int)  # the last trade must be this recent (no alert on a stale price)
CUSTOM_ALERTS = env("CUSTOM_ALERTS", 1, int)      # tickers you type in Telegram -> alert when price goes above the 3m "5H"
CUSTOM_COOLDOWN_SEC = env("CUSTOM_COOLDOWN_SEC", 300, int)   # min seconds between two alerts for the same ticker
CUSTOM_MAX = env("CUSTOM_MAX", 30, int)           # max tickers you can watch at once
CUSTOM_MSG_MAX_AGE_SEC = env("CUSTOM_MSG_MAX_AGE_SEC", 10800, int)  # ignore chat messages older than this (3 h)
PICK_ALERTS = env("PICK_ALERTS", 0, int)          # recommend stocks up >= PICK_MIN_CHANGE whose 3m and 10m Pivots are similar
PICK_MIN_CHANGE = env("PICK_MIN_CHANGE", 16.0, float)   # day % change for a recommendation candidate
PICK_SIM_PCT = env("PICK_SIM_PCT", 0.5, float)    # "similar" = P and R1 of the 3m / 10m Pivots differ by at most this % of the price
PICK_MAX = env("PICK_MAX", 40, int)               # max recommended stocks tracked per day
PICK_COOLDOWN_SEC = env("PICK_COOLDOWN_SEC", 600, int)  # min seconds between two entry alerts for the same stock
PICK_MAX_ENTRIES = env("PICK_MAX_ENTRIES", 3, int)      # max entry alerts per stock per day (R1 is crossed 10-80 times a day)
MIN_AVG_VOL = env("MIN_AVG_VOL", 200_000, int)    # pre/post-market universe: 3-month avg volume floor
MIN_EXT_VOLUME = env("MIN_EXT_VOLUME", 50_000, int)  # pre-market volume floor (only if Yahoo reports it)
MIN_BARS = 60
PERIOD = "58d"  # Yahoo caps 30m data at 60 days
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"

# Kinetic Stocks Gaussian settings (same as the Pine defaults)
G_LENGTH, G_POLES, G_SMOOTH, G_FLATTEN = 260, 2, 26, 10
ST_FACTOR, ST_ATR = 0.15, 21

STATE_FILE = HERE / "state.json"


def log(msg):
    print(f"[{datetime.now(ET):%m-%d %H:%M:%S ET}] {msg}", flush=True)


# ---------------------------------------------------------------- indicator (port of the Pine code)
def gaussian_alpha(length, order):
    f = 2.0 * np.pi / length
    b = (1.0 - np.cos(f)) / (1.414 ** (2.0 / order) - 1.0)
    return -b + np.sqrt(b * b + 2.0 * b)


def gaussian_filter(x, poles, a):
    # Pine starts the recursion from 0; seeding with the first close is the converged equivalent
    # (the filter has unity DC gain) and avoids a huge startup transient on short histories.
    n, om = len(x), 1.0 - a
    y = np.empty(n)
    for i in range(n):
        h = [y[i - k] if i >= k else x[0] for k in (1, 2, 3, 4)]
        if poles == 1:
            y[i] = a * x[i] + om * h[0]
        elif poles == 2:
            y[i] = a**2 * x[i] + 2 * om * h[0] - om**2 * h[1]
        elif poles == 3:
            y[i] = a**3 * x[i] + 3 * om * h[0] - 3 * om**2 * h[1] + om**3 * h[2]
        else:
            y[i] = a**4 * x[i] + 4 * om * h[0] - 6 * om**2 * h[1] + 4 * om**3 * h[2] - om**4 * h[3]
    return y


def linreg(y, length, offset):
    n = len(y)
    out = np.full(n, np.nan)
    xs = np.arange(length, dtype=float)
    xm = xs.mean()
    sxx = ((xs - xm) ** 2).sum()
    for i in range(length - 1, n):
        w = y[i - length + 1 : i + 1]
        wm = w.mean()
        slope = ((xs - xm) * (w - wm)).sum() / sxx
        out[i] = (wm - slope * xm) + slope * (length - 1 - offset)
    return out


def atr_rma(h, l, c, length):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = np.full(n, np.nan)
    if n >= length:
        atr[length - 1] = tr[:length].mean()
        for i in range(length, n):
            atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr


def supertrend(src, factor, atr):
    n = len(src)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    st = np.full(n, np.nan)
    d = np.ones(n, dtype=int)
    for i in range(n):
        lo = src[i] - factor * atr[i]
        up = src[i] + factor * atr[i]
        p_lo = 0.0 if i == 0 or np.isnan(lower[i - 1]) else lower[i - 1]
        p_up = 0.0 if i == 0 or np.isnan(upper[i - 1]) else upper[i - 1]
        s1 = src[i - 1] if i > 0 else np.nan
        # comparisons with NaN are False, exactly like Pine's na handling
        lo = lo if (lo > p_lo or s1 < p_lo) else p_lo
        up = up if (up < p_up or s1 > p_up) else p_up
        lower[i], upper[i] = lo, up
        if i == 0 or np.isnan(atr[i - 1]):
            d[i] = 1
        elif st[i - 1] == p_up:
            d[i] = -1 if src[i] > up else 1
        else:
            d[i] = 1 if src[i] < lo else -1
        st[i] = lo if d[i] == -1 else up
    return st


def kinetic_gaussian(df):
    """Return (line, trend[+1 green / -1 red], valid) for one OHLC frame."""
    c = df["Close"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    line = linreg(gaussian_filter(c, G_POLES, gaussian_alpha(G_LENGTH, G_POLES)), G_SMOOTH, G_FLATTEN)
    st = supertrend(line, ST_FACTOR, atr_rma(h, l, c, ST_ATR))
    trend = np.where(line > st, 1, -1)
    valid = ~np.isnan(line) & ~np.isnan(st)
    return line, trend, valid


def analyze(df, interval):
    """Drop the forming bar if configured, then report current trend and recent green flips."""
    if len(df) and not USE_FORMING_BAR:
        if datetime.now(ET) < df.index[-1].tz_convert(ET) + INTERVALS[interval]:
            df = df.iloc[:-1]
    if len(df) < MIN_BARS:
        return None
    _, trend, valid = kinetic_gaussian(df)
    last = len(df) - 1
    flips = []
    for j in range(max(1, last - FLIP_LOOKBACK + 1), last + 1):
        if valid[j] and valid[j - 1] and trend[j] == 1 and trend[j - 1] == -1:
            flips.append((df.index[j], last - j))
    return {"trend": int(trend[last]) if valid[last] else 0, "flips": flips, "close": float(df["Close"].iloc[-1]), "bars": len(df)}


# ---------------------------------------------------------------- data
def screen_movers(min_change=None):
    """Penny stocks up >= MIN_CHANGE% today (Yahoo screener)."""
    from yfinance import EquityQuery

    conds = [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("gte", ["percentchange", MIN_CHANGE if min_change is None else min_change]),
        EquityQuery("lt", ["intradayprice", MAX_PRICE]),
        EquityQuery("gte", ["dayvolume", MIN_VOLUME]),
    ]
    if not INCLUDE_OTC:
        conds.append(EquityQuery("is-in", ["exchange", "NMS", "NCM", "NGM", "NYQ", "ASE", "BTS"]))
    res = yf.screen(EquityQuery("and", conds), sortField="percentchange", sortAsc=False, size=100)
    out = []
    for q in res.get("quotes", []):
        sym = q.get("symbol", "")
        if not sym or "-" in sym or "=" in sym or q.get("quoteType", "EQUITY") != "EQUITY":
            continue
        if len(sym) == 5 and sym[-1] in "WRU":  # Nasdaq warrant / right / unit suffix
            continue
        out.append({
            "sym": sym,
            "name": q.get("shortName") or q.get("longName") or "",
            "price": q.get("regularMarketPrice"),
            "chg": q.get("regularMarketChangePercent"),
            "vol": q.get("regularMarketVolume"),
        })
    return out


def market_phase():
    """pre 04:00-09:30 ET, regular 09:30-16:00, post 16:00-20:00 (override with FORCE_PHASE for testing)."""
    forced = os.environ.get("FORCE_PHASE")
    if forced in ("pre", "regular", "post"):
        return forced
    t = datetime.now(ET)
    m = t.hour * 60 + t.minute
    return "pre" if m < 570 else ("regular" if m < 960 else "post")


_universe = {"t": 0.0, "syms": []}


def load_universe():
    """Penny-ish US stocks with enough average volume; refreshed every 30 minutes."""
    if _universe["syms"] and time.time() - _universe["t"] < 1800:
        return _universe["syms"]
    from yfinance import EquityQuery

    conds = [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("lt", ["intradayprice", MAX_PRICE * 2]),  # a $2.5 stock can be a $5 stock by +100%
        EquityQuery("gte", ["avgdailyvol3m", MIN_AVG_VOL]),
    ]
    if not INCLUDE_OTC:
        conds.append(EquityQuery("is-in", ["exchange", "NMS", "NCM", "NGM", "NYQ", "ASE", "BTS"]))
    q, syms = EquityQuery("and", conds), []
    for off in range(0, 2000, 250):
        res = yf.screen(q, offset=off, size=250, sortField="avgdailyvol3m", sortAsc=False)
        rows = res.get("quotes", [])
        syms += [r["symbol"] for r in rows if r.get("quoteType", "EQUITY") == "EQUITY"]
        if len(rows) < 250:
            break
    _universe.update(t=time.time(), syms=syms)
    log(f"universe refreshed: {len(syms)} symbols")
    return syms


_yf_data = None


def batch_quotes(symbols):
    global _yf_data
    from yfinance.data import YfData

    if _yf_data is None:
        _yf_data = YfData()  # handles Yahoo cookie + crumb
    out = []
    for i in range(0, len(symbols), 150):
        r = _yf_data.get(QUOTE_URL, params={"symbols": ",".join(symbols[i : i + 150]), "formatted": "false"})
        out += r.json().get("quoteResponse", {}).get("result", [])
    return out


def extended_movers(phase, min_change=None):
    """Pre-market / after-hours movers: % change vs the previous regular close."""
    floor = MIN_CHANGE if min_change is None else min_change
    out = []
    for q in batch_quotes(load_universe()):
        sym = q.get("symbol", "")
        if not sym or "-" in sym or "=" in sym or (len(sym) == 5 and sym[-1] in "WRU"):
            continue
        reg = q.get("regularMarketChangePercent") or 0.0
        if phase == "pre":
            price, chg, vol = q.get("preMarketPrice"), q.get("preMarketChangePercent"), q.get("preMarketVolume")
            if vol is not None and vol < MIN_EXT_VOLUME:
                continue
        else:
            price, post, vol = q.get("postMarketPrice"), q.get("postMarketChangePercent"), q.get("regularMarketVolume")
            chg = ((1 + reg / 100) * (1 + post / 100) - 1) * 100 if post is not None else None
            if vol is not None and vol < MIN_VOLUME:
                continue
        if price is None or chg is None or chg < floor or price >= MAX_PRICE:
            continue
        out.append({"sym": sym, "name": q.get("shortName") or q.get("longName") or "", "price": price, "chg": chg, "vol": vol})
    return out


def find_movers(min_change=None):
    """Regular hours: screener. Pre-market: extended quotes. After-hours: both, merged by symbol."""
    phase = market_phase()
    found = {}
    if phase in ("regular", "post"):
        for m in screen_movers(min_change):
            found[m["sym"]] = m
    if phase in ("pre", "post"):
        try:
            for m in extended_movers(phase, min_change):
                if m["sym"] not in found or (m["chg"] or 0) > (found[m["sym"]]["chg"] or 0):
                    found[m["sym"]] = m
        except Exception as e:
            log(f"extended movers failed: {e!r}")
    return sorted(found.values(), key=lambda m: -(m["chg"] or 0))


def fetch_bars(symbols, interval):
    out = {}
    for k in range(0, len(symbols), 40):
        chunk = symbols[k : k + 40]
        raw = yf.download(chunk, period=PERIOD, interval=interval, prepost=bool(PREPOST), group_by="ticker",
                          auto_adjust=False, threads=False, progress=False)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            present = set(raw.columns.get_level_values(0))
            for s in chunk:
                if s in present:
                    d = raw[s].dropna(subset=["Close"])
                    if len(d):
                        out[s] = d
        else:
            out[chunk[0]] = raw.dropna(subset=["Close"])
    return out


# ---------------------------------------------------------------- telegram
def send(text):
    if not TG_TOKEN or not TG_CHAT:
        print("\n--- (Telegram not configured; console only) ---\n" + text + "\n")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", timeout=15, data={
            "chat_id": TG_CHAT, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"})
        if not r.ok:
            log(f"Telegram error {r.status_code}: {r.text[:200]}")
    except requests.RequestException as e:
        log(f"Telegram request failed: {type(e).__name__}")  # no URL: it contains the bot token


def fmt_vol(v):
    if not v:
        return "-"
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"


def link(sym):
    return f'<a href="https://www.tradingview.com/symbols/{sym}/">차트</a>'


# ---------------------------------------------------------------- state
def load_state():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    carry = {}
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if s.get("date") == today:
                return s
            # a new day resets the movers watch-list, but your own tickers and the Telegram read position carry over
            carry = {k: s[k] for k in ("custom", "tg_offset") if k in s}
        except Exception:
            pass
    return {"date": today, "watch": {}, "sent": [], **carry}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


def session_open():
    t = datetime.now(ET)
    return t.weekday() < 5 and 4 <= t.hour < 20


def seconds_to_open():
    """Seconds until today's 04:00 ET pre-market open on a weekday, or None if the open is not today."""
    t = datetime.now(ET)
    if t.weekday() >= 5 or t.hour >= 4:
        return None
    return ((4 - t.hour) * 60 - t.minute) * 60 - t.second


# ---------------------------------------------------------------- 3-minute Pivot R2
def bars_3m(rows):
    return bars_nm(rows, 3)


def bars_nm(rows, n):
    """1-minute rows [(ts, o, h, l, c, v)] (ascending, aware ET) -> n-minute bars aligned to the clock (04:00, 04:03, ...).
    A minute without trades has no row, so a bar is built only from minutes that exist (nothing is invented)."""
    out = []
    for ts, o, h, l, c, v in rows:
        key = ts.replace(minute=ts.minute - ts.minute % n, second=0, microsecond=0)
        if out and out[-1]["t"] == key:
            b = out[-1]
            b["h"], b["l"], b["c"], b["v"] = max(b["h"], h), min(b["l"], l), c, b["v"] + v
        else:
            out.append({"t": key, "o": o, "h": h, "l": l, "c": c, "v": v})
    return out


def pivot_levels(h, l, c):
    """Classic (Standard) pivot points from one bar's high / low / close."""
    p = (h + l + c) / 3.0
    return {"P": p, "R1": 2 * p - l, "R2": p + (h - l), "S1": 2 * p - h, "S2": p - (h - l)}


def r2_state(rows, now):
    """Latest price vs. R2 of the PREVIOUS (completed) 3-minute bar.  None = cannot be judged."""
    if not rows:
        return None
    last = rows[-1]
    if (now - last[0]).total_seconds() > R2_MAX_AGE_SEC:
        return None                                   # no recent trade: never alert on a stale price
    b3 = bars_3m(rows)
    if len(b3) < 2 or b3[-2]["t"].date() != b3[-1]["t"].date():
        return None
    prev, cur = b3[-2], b3[-1]
    if prev["c"] <= 0:
        return None
    range_pct = (prev["h"] - prev["l"]) / prev["c"] * 100.0
    lv = pivot_levels(prev["h"], prev["l"], prev["c"])
    price = last[4]
    # a previous bar with (almost) no range puts R2 right on its high: any uptick would "break" it, so skip
    return {"price": price, "price_ts": last[0], "levels": lv, "prev": prev, "cur_t": cur["t"], "range_pct": range_pct,
            "above": range_pct >= R2_MIN_RANGE_PCT and price > lv["R2"]}


def fmt_px(x):
    return f"{x:.4f}" if x < 1 else f"{x:.2f}"


def fetch_1m_rows(symbols):
    out = {}
    for k in range(0, len(symbols), 40):
        chunk = symbols[k : k + 40]
        raw = yf.download(chunk, period="1d", interval="1m", prepost=bool(PREPOST), group_by="ticker",
                          auto_adjust=False, threads=False, progress=False)
        if raw is None or raw.empty:
            continue
        frames = {}
        if isinstance(raw.columns, pd.MultiIndex):
            present = set(raw.columns.get_level_values(0))
            frames = {s: raw[s] for s in chunk if s in present}
        else:
            frames = {chunk[0]: raw}
        for s, d in frames.items():
            d = d.dropna(subset=["Close"])
            if len(d):
                out[s] = [(ts.tz_convert(ET).to_pydatetime(), float(o), float(h), float(l), float(c), float(v) if v == v else 0.0)
                          for ts, o, h, l, c, v in zip(d.index, d["Open"], d["High"], d["Low"], d["Close"], d["Volume"])]
    return out


def r2_message(sym, info, st):
    lv, prev = st["levels"], st["prev"]
    chg = info.get("chg")
    head = f"🔥 <b>{sym}</b> 3분봉 Pivot R2 돌파" + (f" (당일 +{chg:.1f}%)" if chg is not None else "")
    return (f"{head}\n"
            f"현재가 <b>${fmt_px(st['price'])}</b> > R2 <b>${fmt_px(lv['R2'])}</b>  (마지막 체결 {st['price_ts']:%H:%M:%S} ET)\n"
            f"기준: 직전 3분봉 {prev['t']:%H:%M} · H {fmt_px(prev['h'])} L {fmt_px(prev['l'])} C {fmt_px(prev['c'])}\n"
            f"P {fmt_px(lv['P'])} | R1 {fmt_px(lv['R1'])} | R2 {fmt_px(lv['R2'])} | S1 {fmt_px(lv['S1'])}\n"
            f"{info.get('name', '')}\n{link(sym)}")


def check_r2(state):
    """Watch-list stocks (already up >= MIN_CHANGE): alert once per day when the last price is above R2."""
    watch = state["watch"]
    sent = state.setdefault("r2_sent", [])
    syms = [s for s in watch if s not in sent]
    if not syms:
        return
    now = datetime.now(ET)
    rows = fetch_1m_rows(syms)
    n_above = 0
    for s in syms:
        st = r2_state(rows.get(s), now)
        if st and st["above"]:
            n_above += 1
            sent.append(s)                      # recorded first: a send problem must not cause a repeat storm
            send(r2_message(s, watch[s], st))
    log(f"r2 check: {len(syms)} symbols, 1m data for {len(rows)}, above R2: {n_above}")


# ---------------------------------------------------------------- recommendations: 3m Pivot ~ 10m Pivot on a +16% stock, entry on R1
def pivot_pair(rows, now):
    """Latest price vs. the Pivots of the previous completed 3-minute AND 10-minute bars.  None = cannot be judged.
    similar = P and R1 of the two timeframes lie within PICK_SIM_PCT % of the price;  above_r1 = price > the 3m R1."""
    if not rows:
        return None
    last = rows[-1]
    if (now - last[0]).total_seconds() > R2_MAX_AGE_SEC:
        return None                                   # stale price: never judge it
    b3, b10 = bars_nm(rows, 3), bars_nm(rows, 10)
    if len(b3) < 2 or len(b10) < 2:
        return None
    if b3[-2]["t"].date() != b3[-1]["t"].date() or b10[-2]["t"].date() != b10[-1]["t"].date():
        return None
    p3, p10 = b3[-2], b10[-2]
    price = last[4]
    if price <= 0 or p3["c"] <= 0 or p10["c"] <= 0:
        return None
    range3 = (p3["h"] - p3["l"]) / p3["c"] * 100.0
    lv3, lv10 = pivot_levels(p3["h"], p3["l"], p3["c"]), pivot_levels(p10["h"], p10["l"], p10["c"])
    diff = max(abs(lv3["P"] - lv10["P"]), abs(lv3["R1"] - lv10["R1"])) / price * 100.0
    # a 3m bar with (almost) no range puts P, R1 and its high on the same spot: any uptick would "break" R1, so no verdict
    ok = range3 >= R2_MIN_RANGE_PCT
    return {"price": price, "price_ts": last[0], "lv3": lv3, "lv10": lv10, "prev3": p3, "prev10": p10,
            "diff_pct": diff, "similar": ok and diff <= PICK_SIM_PCT, "above_r1": ok and price > lv3["R1"]}


def _pivot_lines(st):
    a, b = st["lv3"], st["lv10"]
    return (f"3분 Pivot  P {fmt_px(a['P'])} | R1 {fmt_px(a['R1'])} | R2 {fmt_px(a['R2'])}  (기준 {st['prev3']['t']:%H:%M})\n"
            f"10분 Pivot P {fmt_px(b['P'])} | R1 {fmt_px(b['R1'])} | R2 {fmt_px(b['R2'])}  (기준 {st['prev10']['t']:%H:%M})\n"
            f"두 Pivot 차이 {st['diff_pct']:.2f}% (기준 ≤ {PICK_SIM_PCT:.1f}%)")


def pick_message(sym, info, st):
    chg = f"+{info['chg']:.1f}%" if info.get("chg") is not None else "-"
    state = "이미 R1 위 (R1 아래로 내려갔다가 다시 돌파하면 진입 알림)" if st["above_r1"] else f"R1 ${fmt_px(st['lv3']['R1'])} 돌파 시 진입 알림"
    return (f"🎯 <b>{sym}</b> 추천 · 당일 {chg} · ${fmt_px(st['price'])}\n"
            f"{_pivot_lines(st)}\n"
            f"진입: {state}\n{info.get('name', '')}\n{link(sym)}")


def entry_message(sym, info, st):
    chg = f"+{info['chg']:.1f}%" if info.get("chg") is not None else "-"
    return (f"✅ <b>{sym}</b> 진입 · 3분봉 R1 돌파 (당일 {chg})\n"
            f"현재가 <b>${fmt_px(st['price'])}</b> > R1 <b>${fmt_px(st['lv3']['R1'])}</b>  (마지막 체결 {st['price_ts']:%H:%M:%S} ET)\n"
            f"{_pivot_lines(st)}\n{link(sym)}")


def check_picks(state, movers_all):
    """Candidates = penny movers up >= PICK_MIN_CHANGE.  A candidate whose 3m and 10m Pivots are similar is recommended once a day;
    after that, an alert fires each time the price crosses from below to above the 3m R1 (the entry)."""
    picks = state.setdefault("picks", {})
    now = datetime.now(ET)
    chg_now = {m["sym"]: m for m in movers_all}
    cands = [m for m in movers_all if m["sym"] not in picks and (m.get("chg") or 0) >= PICK_MIN_CHANGE]
    syms = [m["sym"] for m in cands] + list(picks)
    if not syms:
        return
    rows = fetch_1m_rows(syms)
    n_new = n_entry = 0
    added = set()
    for m in cands:
        if len(picks) >= PICK_MAX:
            break
        st = pivot_pair(rows.get(m["sym"]), now)
        if st and st["similar"]:
            picks[m["sym"]] = {"name": m.get("name", ""), "chg": m.get("chg"), "above": st["above_r1"], "last_entry": None,
                               "rec_at": now.isoformat()}
            added.add(m["sym"])
            n_new += 1
            send(pick_message(m["sym"], picks[m["sym"]], st))
    for s, info in list(picks.items()):
        if s in chg_now:
            info["chg"] = chg_now[s].get("chg")
        if s in added:                          # just recommended this cycle: its state is already recorded
            continue
        st = pivot_pair(rows.get(s), now)
        if not st:
            continue
        was, info["above"] = info.get("above"), st["above_r1"]
        if st["above_r1"] and was is False:
            last = info.get("last_entry")
            if last and (now - datetime.fromisoformat(last)).total_seconds() < PICK_COOLDOWN_SEC:
                continue
            if info.get("entries", 0) >= PICK_MAX_ENTRIES:
                continue
            info["last_entry"] = now.isoformat()
            info["entries"] = info.get("entries", 0) + 1
            n_entry += 1
            send(entry_message(s, info, st))
    log(f"picks: candidates {len(cands)}, tracked {len(picks)}, new {n_new}, entry alerts {n_entry}")


# ---------------------------------------------------------------- your own tickers (typed in Telegram) -> 3m "5H" breakout
TICKER_RE = re.compile(r"^\$?([A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,2})?)$")
REMOVE_WORDS = {"삭제", "제거", "remove", "del", "delete", "rm"}
LIST_WORDS = {"목록", "리스트", "list", "상태"}
HELP_WORDS = {"도움말", "help", "?"}
IGNORE_WORDS = {"hi", "hello", "hey", "ok", "okay", "yes", "no", "go", "thx", "thanks", "test", "start", "stop", "시작", "중지"}
HELP_TEXT = ("사용법\n• 티커를 보내면 감시 시작 (여러 개는 띄어쓰기): GYGY ABCD\n• 삭제 GYGY\n• 목록\n"
             "3분봉 '5H'(직전 완성된 3분봉 5개의 최고 고가) 위로 올라오면 알려 드립니다.")


def parse_command(text):
    """Telegram text -> ('add'|'remove'|'list'|'help', [tickers]) or None when it is not a command."""
    t = (text or "").strip()
    if not t or len(t) > 80:
        return None
    parts = [p for p in re.split(r"[\s,;]+", t) if p]
    if not parts:
        return None
    head = parts[0].lower()
    if head in HELP_WORDS and len(parts) == 1:
        return ("help", [])
    if head in LIST_WORDS and len(parts) == 1:
        return ("list", [])
    if head in REMOVE_WORDS:
        syms = [m.group(1).upper() for p in parts[1:] if (m := TICKER_RE.match(p))]
        return ("remove", syms) if syms and len(syms) == len(parts) - 1 else None
    if len(parts) > 15 or any(p.lower() in IGNORE_WORDS for p in parts):
        return None
    syms = []
    for p in parts:
        m = TICKER_RE.match(p)
        if not m:
            return None
        syms.append(m.group(1).upper())
    return ("add", syms)


def five_high_state(rows, now, n=5):
    """Last price vs. the 3-minute 'nH': the highest HIGH of the previous n COMPLETED 3-minute bars.
    The bar that is still forming is excluded, otherwise the price could never be above its own high."""
    if not rows:
        return None
    last = rows[-1]
    out = {"price": last[4], "price_ts": last[0], "stale": (now - last[0]).total_seconds() > R2_MAX_AGE_SEC,
           "level": None, "above": None, "span": None}
    b3 = bars_3m(rows)
    recent = b3[-(n + 1):]
    if len(recent) < n + 1 or any(b["t"].date() != recent[-1]["t"].date() for b in recent):
        return out                                    # not enough completed bars yet today
    done = recent[:-1]
    out["level"] = max(b["h"] for b in done)
    out["above"] = last[4] > out["level"]
    out["span"] = (done[0]["t"], done[-1]["t"])
    return out


def symbol_exists(sym):
    try:
        return len(yf.Ticker(sym).history(period="5d")) > 0
    except Exception:
        return False


def tg_get_updates(offset):
    params = {"timeout": 0, "allowed_updates": json.dumps(["message"])}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", params=params, timeout=15)
    except requests.RequestException as e:
        log(f"telegram getUpdates failed: {type(e).__name__}")     # no URL: it contains the bot token
        return None
    if r.status_code == 409:
        log("telegram getUpdates 409 (a webhook is set?) - ticker commands are off")
        return None
    if not r.ok:
        log(f"telegram getUpdates HTTP {r.status_code}")
        return None
    return r.json().get("result", [])


def add_custom(state, sym):
    custom = state.setdefault("custom", {})
    if sym in custom:
        send(f"ℹ️ <b>{sym}</b> 이미 감시 중입니다.")
        return
    if len(custom) >= CUSTOM_MAX:
        send(f"⚠️ 감시 종목이 {CUSTOM_MAX}개라 더 추가할 수 없습니다. '삭제 티커'로 정리해 주세요.")
        return
    now = datetime.now(ET)
    rows = fetch_1m_rows([sym]).get(sym)
    if not rows:
        if not symbol_exists(sym):
            send(f"❌ <b>{sym}</b>: 종목을 찾을 수 없습니다. 티커를 확인해 주세요.")
            return
        custom[sym] = {"added": now.isoformat(), "above": None, "last_alert": None}
        send(f"✅ <b>{sym}</b> 감시 시작 (오늘 거래 데이터가 아직 없어서 거래가 생기면 판정합니다)")
        return
    st = five_high_state(rows, now)
    custom[sym] = {"added": now.isoformat(), "above": st["above"] if st and not st["stale"] else None, "last_alert": None}
    if st["level"] is None:
        send(f"✅ <b>{sym}</b> 감시 시작 · 현재가 ${fmt_px(st['price'])} (완성된 3분봉이 5개 미만이라 5H 계산 대기)")
        return
    pos = "5H <b>위</b>" if st["above"] else "5H 아래"
    note = "  ※ 마지막 체결이 오래됨" if st["stale"] else ""
    extra = "\n이미 위에 있어서, 아래로 내려갔다가 다시 올라올 때 알려 드립니다." if st["above"] else ""
    send(f"✅ <b>{sym}</b> 감시 시작 · 현재가 ${fmt_px(st['price'])} · 3분봉 5H ${fmt_px(st['level'])} → {pos}{note}{extra}")


def handle_command(state, cmd):
    kind, syms = cmd
    custom = state.setdefault("custom", {})
    if kind == "help":
        send(HELP_TEXT)
    elif kind == "list":
        if not custom:
            send("감시 중인 종목이 없습니다. 티커를 보내 주세요. (예: GYGY)")
            return
        rows = fetch_1m_rows(list(custom))
        now = datetime.now(ET)
        lines = []
        for s in custom:
            st = five_high_state(rows.get(s), now)
            if not st or st["level"] is None:
                lines.append(f"• {s}: 5H 계산 대기")
            else:
                lines.append(f"• {s}: ${fmt_px(st['price'])} / 5H ${fmt_px(st['level'])} → {'위' if st['above'] else '아래'}")
        send("📋 감시 목록\n" + "\n".join(lines))
    elif kind == "remove":
        gone = [s for s in syms if custom.pop(s, None) is not None]
        miss = [s for s in syms if s not in gone]
        send((f"🗑 삭제: {', '.join(gone)}" if gone else "") + (("\n" if gone else "") + f"목록에 없음: {', '.join(miss)}" if miss else ""))
    elif kind == "add":
        for s in syms:
            add_custom(state, s)


def process_telegram(state):
    """Read new messages from YOUR chat only and run add / remove / list commands."""
    if not TG_TOKEN or not TG_CHAT:
        return
    updates = tg_get_updates(state.get("tg_offset") or None)
    if not updates:
        return
    now_ts = time.time()
    for u in updates:
        state["tg_offset"] = u["update_id"] + 1
        m = u.get("message") or {}
        if str((m.get("chat") or {}).get("id")) != str(TG_CHAT):
            continue                                  # anyone else who finds the bot is ignored
        if (m.get("from") or {}).get("is_bot") or now_ts - m.get("date", 0) > CUSTOM_MSG_MAX_AGE_SEC:
            continue
        cmd = parse_command(m.get("text"))
        if cmd:
            try:
                handle_command(state, cmd)
            except Exception as e:                    # one bad command must not stop the others
                log(f"command failed: {e!r}")


def check_custom(state):
    """Alert when a ticker you added goes from BELOW to ABOVE its 3-minute 5H."""
    custom = state.get("custom") or {}
    if not custom:
        return
    now = datetime.now(ET)
    rows = fetch_1m_rows(list(custom))
    n_alert = 0
    for sym, ent in custom.items():
        st = five_high_state(rows.get(sym), now)
        if not st or st["level"] is None or st["stale"]:
            continue                                  # cannot judge: keep the previous state
        if ent.get("above") is False and st["above"]:
            last = ent.get("last_alert")
            if not last or (now - datetime.fromisoformat(last)).total_seconds() >= CUSTOM_COOLDOWN_SEC:
                a, b = st["span"]
                send(f"🔼 <b>{sym}</b> 3분봉 5H 돌파\n현재가 <b>${fmt_px(st['price'])}</b> > 5H <b>${fmt_px(st['level'])}</b>  "
                     f"(마지막 체결 {st['price_ts']:%H:%M:%S} ET)\n기준: 직전 완성된 3분봉 5개({a:%H:%M}~{b:%H:%M})의 최고 고가\n{link(sym)}")
                ent["last_alert"] = now.isoformat()
                n_alert += 1
        ent["above"] = st["above"]
    log(f"custom check: {len(custom)} tickers, 1m data for {len(rows)}, alerts: {n_alert}")


# ---------------------------------------------------------------- one cycle
def cycle(state):
    try:
        movers_all = find_movers(min(MIN_CHANGE, PICK_MIN_CHANGE) if PICK_ALERTS else None)
    except Exception as e:  # keep running on transient Yahoo errors
        log(f"screener failed: {e!r}")
        movers_all = []
    movers = [m for m in movers_all if (m.get("chg") or 0) >= MIN_CHANGE]

    watch = state["watch"]
    fresh = []
    for m in movers:
        if m["sym"] in watch:
            watch[m["sym"]].update({k: m[k] for k in ("price", "chg", "vol")})
        elif len(watch) < MAX_WATCH:
            watch[m["sym"]] = {**m, "first_seen": datetime.now(ET).isoformat()}
            fresh.append(m)
    if fresh and MOVER_ALERTS:
        if len(fresh) <= 3:
            for m in fresh:
                send(f"🚀 <b>{m['sym']}</b> +{m['chg']:.1f}% · ${m['price']:.2f} · 거래량 {fmt_vol(m['vol'])}\n{m['name']}\n{link(m['sym'])}")
        else:
            rows = "\n".join(f"• <b>{m['sym']}</b> +{m['chg']:.1f}% ${m['price']:.2f} ({fmt_vol(m['vol'])})" for m in fresh)
            send(f"🚀 페니 +{MIN_CHANGE:.0f}% 신규 {len(fresh)}종목\n{rows}")
    log(f"movers={len(movers)} new={len(fresh)} watchlist={len(watch)}")
    if CUSTOM_ALERTS:
        try:
            process_telegram(state)
            check_custom(state)
        except Exception as e:                  # never let this stop the main alerts
            log(f"custom watch failed: {e!r}")
    if R2_ALERTS and watch:
        try:
            check_r2(state)
        except Exception as e:                  # never let the R2 check stop the main alerts
            log(f"r2 check failed: {e!r}")
    if PICK_ALERTS:
        try:
            check_picks(state, movers_all)
        except Exception as e:                  # never let the recommendations stop the main alerts
            log(f"picks check failed: {e!r}")
    if not watch:
        return

    syms = list(watch)
    res = {s: {} for s in syms}
    for iv in INTERVALS:
        bars = fetch_bars(syms, iv)
        for s, df in bars.items():
            try:
                res[s][iv] = analyze(df, iv)
            except Exception as e:
                log(f"{s} {iv} analyze failed: {e!r}")

    sent = set(state["sent"])
    for s in syms:
        for iv, r in res[s].items():
            if not r:
                continue
            for ts, ago in r["flips"]:
                key = f"{s}|{iv}|{ts.isoformat()}"
                if key in sent:
                    continue
                sent.add(key)
                info = watch[s]
                trends = {i: (res[s].get(i) or {}).get("trend", 0) for i in INTERVALS}
                both = all(t == 1 for t in trends.values())
                mark = lambda t: "🟢" if t == 1 else ("🔴" if t == -1 else "⚪")
                when = "지금" if ago == 0 else f"{ago}봉 전"
                chg = f"+{info['chg']:.1f}%" if info.get("chg") is not None else "-"
                send(
                    f"🟢 <b>{s}</b> Kinetic Gaussian GREEN — <b>{LABEL[iv]}봉</b> ({when})\n"
                    f"${r['close']:.2f} ({chg}) · 거래량 {fmt_vol(info.get('vol'))}\n"
                    f"30분 {mark(trends['30m'])} | 1시간 {mark(trends['60m'])}"
                    + ("\n⭐ 30분·1시간 동시 green" if both else "")
                    + f"\n{link(s)}"
                )
    state["sent"] = sorted(sent)


# ---------------------------------------------------------------- commands
def cmd_check(sym):
    global USE_FORMING_BAR, FLIP_LOOKBACK
    USE_FORMING_BAR, FLIP_LOOKBACK = 1, 10**6
    for iv in INTERVALS:
        df = fetch_bars([sym], iv).get(sym)
        if df is None or len(df) < MIN_BARS:
            print(f"{sym} {iv}: not enough bars ({0 if df is None else len(df)})")
            continue
        _, trend, valid = kinetic_gaussian(df)
        print(f"\n{sym} {iv}: {len(df)} bars, now {'GREEN' if trend[-1] == 1 else 'RED'}")
        idx = [j for j in range(1, len(df)) if valid[j] and valid[j - 1] and trend[j] != trend[j - 1]]
        for j in idx[-8:]:
            print(f"  {df.index[j].tz_convert(ET):%m-%d %H:%M ET}  -> {'GREEN' if trend[j] == 1 else 'RED'}  close {df['Close'].iloc[j]:.4f}")


def cmd_get_chat_id():
    if not TG_TOKEN:
        sys.exit("Set TELEGRAM_BOT_TOKEN in .env first, send any message to your bot, then rerun.")
    r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", timeout=15).json()
    seen = {u["message"]["chat"]["id"]: u["message"]["chat"].get("first_name", "") for u in r.get("result", []) if "message" in u}
    print(seen or "No messages yet - send your bot a message first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--always", action="store_true", help="run even outside 04:00-20:00 ET")
    ap.add_argument("--max-minutes", type=float, default=0, help="exit gracefully after N minutes (CI jobs)")
    ap.add_argument("--exit-when-closed", action="store_true", help="exit instead of sleeping when the market is closed (CI)")
    ap.add_argument("--check", metavar="SYM")
    ap.add_argument("--get-chat-id", action="store_true")
    ap.add_argument("--test-telegram", action="store_true")
    a = ap.parse_args()

    if a.check:
        return cmd_check(a.check.upper())
    if a.get_chat_id:
        return cmd_get_chat_id()
    if a.test_telegram:
        return send("✅ Kinetic penny alert: Telegram 연결 테스트")

    log(f"start: +{MIN_CHANGE:.0f}% / <${MAX_PRICE} / vol>={fmt_vol(MIN_VOLUME)} / forming_bar={USE_FORMING_BAR} / poll={POLL}s")
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    while deadline is None or time.time() < deadline:
        if a.always or a.once or session_open():
            state = load_state()
            try:
                cycle(state)
            except Exception as e:  # never let one bad cycle kill a long-running job
                log(f"cycle failed: {e!r}")
            finally:
                save_state(state)
            if a.once:
                return
            time.sleep(POLL)
        else:
            wait = seconds_to_open()
            # waiting up to 150 min lets a job that GitHub starts early be ready when the pre-market opens
            # (the open is 08:00 UTC in summer time, 09:00 UTC in winter time)
            if a.exit_when_closed and (wait is None or wait > 150 * 60):
                return log("market closed - exiting")
            # sleep in 5-minute steps, but wake exactly at the open
            nap = 300 if wait is None or wait > 300 else max(1, wait)
            log(f"market closed (04:00-20:00 ET, Mon-Fri) - sleeping {nap}s")
            time.sleep(nap)
    log("max runtime reached - exiting for handoff")


if __name__ == "__main__":
    main()
