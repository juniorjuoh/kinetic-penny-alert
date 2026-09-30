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
def screen_movers():
    """Penny stocks up >= MIN_CHANGE% today (Yahoo screener)."""
    from yfinance import EquityQuery

    conds = [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("gte", ["percentchange", MIN_CHANGE]),
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


def extended_movers(phase):
    """Pre-market / after-hours movers: % change vs the previous regular close."""
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
        if price is None or chg is None or chg < MIN_CHANGE or price >= MAX_PRICE:
            continue
        out.append({"sym": sym, "name": q.get("shortName") or q.get("longName") or "", "price": price, "chg": chg, "vol": vol})
    return out


def find_movers():
    """Regular hours: screener. Pre-market: extended quotes. After-hours: both, merged by symbol."""
    phase = market_phase()
    found = {}
    if phase in ("regular", "post"):
        for m in screen_movers():
            found[m["sym"]] = m
    if phase in ("pre", "post"):
        try:
            for m in extended_movers(phase):
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
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if s.get("date") == today:
                return s
        except Exception:
            pass
    return {"date": today, "watch": {}, "sent": []}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


def session_open():
    t = datetime.now(ET)
    return t.weekday() < 5 and 4 <= t.hour < 20


def minutes_to_open():
    """Minutes until today's 04:00 ET open on a weekday, or None if the open is not today."""
    t = datetime.now(ET)
    if t.weekday() >= 5 or t.hour >= 4:
        return None
    return (4 - t.hour) * 60 - t.minute


# ---------------------------------------------------------------- one cycle
def cycle(state):
    try:
        movers = find_movers()
    except Exception as e:  # keep running on transient Yahoo errors
        log(f"screener failed: {e!r}")
        movers = []

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
            wait = minutes_to_open()
            if a.exit_when_closed and (wait is None or wait > 90):
                return log("market closed - exiting")
            log("market closed (04:00-20:00 ET, Mon-Fri) - sleeping 5 min")
            time.sleep(300)
    log("max runtime reached - exiting for handoff")


if __name__ == "__main__":
    main()
