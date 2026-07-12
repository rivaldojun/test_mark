#!/usr/bin/env python3
"""
Shared strategy engine — used by BOTH the live bot (deriv_ws_scalper.py) and
the backtester (backtest_xau.py). One code path = what you backtest is what
trades live.

CONTRACT
    Every strategy receives `data`: a dict {granularity_seconds: DataFrame}
    containing ONLY CLOSED candles (columns: open/high/low/close, UTC
    DatetimeIndex, ascending). `iloc[-1]` is therefore the last closed bar.
    It returns a Signal (direction + absolute SL/TP price levels) or None.

    Strategies are stateful (they remember which setups already fired) —
    instantiate one object per run/session.

SESSION TIMES (UTC) — ⚠ assumption to review twice a year
    London open 07:00 UTC, New York open 13:30 UTC. These are the SUMMER
    (BST/EDT) values. In winter (GMT/EST) they shift to 08:00 / 14:30 UTC.
    Adjust LONDON_OPEN_UTC / NY_OPEN_UTC below when DST changes.

VWAP CAVEAT
    Deriv's feed provides no volume, so "VWAP" here is an equal-weighted
    cumulative mean of typical price (hlc3) since the session anchor — i.e.
    a session TWAP used as VWAP proxy. This is a documented approximation.
"""

from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Session configuration (UTC, summer time — see module docstring) ──────────
LONDON_OPEN_UTC = dtime(7, 0)
NY_OPEN_UTC     = dtime(13, 30)
LONDON_WINDOW   = (dtime(7, 0),  dtime(10, 30))   # entries allowed
NY_WINDOW       = (dtime(13, 30), dtime(17, 0))

M1, M5, H1 = 60, 300, 3600

#: granularities the Deriv candle feed actually serves
VALID_GRANS = [60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400]

_TF_ALIASES = {
    "m1": 60, "m2": 120, "m3": 180, "m5": 300, "m10": 600, "m15": 900,
    "m30": 1800, "h1": 3600, "h2": 7200, "h4": 14400, "h8": 28800, "d1": 86400,
}


def parse_tf(value) -> int:
    """'M5' / '5m' / '300' → 300 (seconds, snapped to a valid granularity)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _TF_ALIASES:
        return _TF_ALIASES[s]
    if s.endswith("m") and s[:-1].isdigit():
        return snap_tf(int(s[:-1]) * 60)
    if s.endswith("h") and s[:-1].isdigit():
        return snap_tf(int(s[:-1]) * 3600)
    if s.isdigit():
        return snap_tf(int(s))
    raise ValueError(f"Timeframe invalide: {value!r} (ex: M1, M5, M15, H1, 300, 5m)")


def snap_tf(seconds: int) -> int:
    """Nearest granularity Deriv can serve."""
    return min(VALID_GRANS, key=lambda g: abs(g - seconds))


@dataclass
class Signal:
    direction: str      # 'long' | 'short'
    sl_price: float     # absolute stop-loss level
    tp_price: Optional[float] = None   # absolute TP level; None = pas de TP fixe (géré par trailing)
    reason: str = ""
    # Trailing par paliers: à chaque trail_step de gain latent MAX, on verrouille
    # trail_lock de profit. None = pas de trailing (SL/TP fixes + break-even).
    # trail_kind: "pnl" → step/lock en $ de P&L du compte (dépend du notionnel) ;
    #             "dist" → step/lock en DISTANCE de prix absolue (indépendant de
    #                      la taille ; c'est ainsi que les modes R et ATR sont
    #                      transmis, déjà convertis en prix par la stratégie).
    trail_step: Optional[float] = None
    trail_lock: Optional[float] = None
    trail_kind: str = "pnl"


# ──────────────────────────────────────────────────────────────────────────────
#  Indicators (causal — value at row i uses only rows ≤ i)
# ──────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed moving average (Wilder) — the MA Bill Williams uses."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def alligator_lines(df: pd.DataFrame):
    """Williams Alligator: SMMA of median price, shifted FORWARD — so the
    value at bar i comes from bars ≤ i−shift. Strictly causal.
    Returns (jaw, teeth, lips) series aligned on df.index."""
    med = (df["high"] + df["low"]) / 2.0
    jaw   = smma(med, 13).shift(8)
    teeth = smma(med, 8).shift(5)
    lips  = smma(med, 5).shift(3)
    return jaw, teeth, lips


def ensure_cols(df: pd.DataFrame, needs: Dict[str, tuple]) -> pd.DataFrame:
    """Adds indicator columns if absent. The backtester precomputes them once
    on the full history (causal indicators ⇒ no lookahead); the live bot lets
    this compute them fresh on each small window."""
    for col, (kind, period) in needs.items():
        if col not in df.columns:
            if kind == "ema":
                df[col] = ema(df["close"], period)
            elif kind == "rsi":
                df[col] = rsi(df["close"], period)
            elif kind == "atr":
                df[col] = atr(df, period)
    return df


def confirmed_pivots(df: pd.DataFrame, k: int = 2):
    """Swing highs/lows with k bars on each side. A pivot at index i is only
    CONFIRMED once k bars have closed after it — since `df` contains closed
    bars up to 'now', any pivot at i ≤ len-1-k is confirmed, later ones are
    not returned. Returns (high_idx, low_idx) as integer positions."""
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    highs, lows = [], []
    for i in range(k, n - k):
        wh = h[i - k:i + k + 1]
        wl = l[i - k:i + k + 1]
        if h[i] == wh.max() and (wh.argmax() == k):
            highs.append(i)
        if l[i] == wl.min() and (wl.argmin() == k):
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


def session_anchor(ts: pd.Timestamp) -> pd.Timestamp:
    """Latest session anchor (00:00, London open, NY open) at or before ts."""
    day = ts.normalize()
    anchors = [
        day,
        day + pd.Timedelta(hours=LONDON_OPEN_UTC.hour, minutes=LONDON_OPEN_UTC.minute),
        day + pd.Timedelta(hours=NY_OPEN_UTC.hour, minutes=NY_OPEN_UTC.minute),
    ]
    return max(a for a in anchors if a <= ts)


def anchored_vwap(df: pd.DataFrame) -> pd.Series:
    """Equal-weighted cumulative mean of hlc3 since the current session anchor
    (volume proxy — see module docstring). Causal."""
    if "vwap" in df.columns:
        return df["vwap"]
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    anchors = df.index.to_series().apply(session_anchor)
    grp = hlc3.groupby(anchors.values)
    return grp.cumsum() / grp.cumcount().add(1)


def in_window(ts: pd.Timestamp) -> Optional[str]:
    t = ts.time()
    if LONDON_WINDOW[0] <= t < LONDON_WINDOW[1]:
        return "london"
    if NY_WINDOW[0] <= t < NY_WINDOW[1]:
        return "ny"
    return None


def session_of(ts: pd.Timestamp) -> str:
    """'london' / 'ny' / 'off' — the session bucket of a timestamp."""
    return in_window(ts) or "off"


def parse_sessions(value) -> Optional[tuple]:
    """'london' / 'ny,off' / ('london',) → validated tuple, None = no filter."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        parts = [str(s).strip().lower() for s in value]
    else:
        parts = [s.strip().lower() for s in str(value).split(",") if s.strip()]
    valid = {"london", "ny", "off"}
    bad = [s for s in parts if s not in valid]
    if bad:
        raise ValueError(f"Sessions invalides: {bad} (choix: london, ny, off)")
    return tuple(parts) or None


# ──────────────────────────────────────────────────────────────────────────────
#  Base class
# ──────────────────────────────────────────────────────────────────────────────

class Strategy:
    name: str = "base"
    #: default working (entry) timeframe — override with the tf= constructor arg
    DEFAULT_TF: int = M5
    #: reward/risk multiple used for TP when the strategy is R-based
    rr: float = 2.0

    def __init__(self, rr: Optional[float] = None, tf: Optional[int] = None,
                 sessions: Optional[tuple] = None):
        if rr:
            self.rr = rr
        self.tf = snap_tf(tf) if tf else self.DEFAULT_TF
        #: {granularity_seconds: min bars needed} — rebuilt for the chosen tf
        self.granularities = self._grans(self.tf)
        #: how often the live bot should poll (scales with tf)
        self.poll_seconds = max(10, min(60, self.tf // 4))
        #: restrict entries to these session buckets (None = strategy default).
        #: e.g. ("london",) / ("off",) / ("london","ny")
        self.sessions = parse_sessions(sessions)
        self._done = set()   # setup keys that already fired (dedupe)

    def session_ok(self, ts: pd.Timestamp) -> bool:
        """Engine-level entry gate for the user's --sessions filter."""
        return self.sessions is None or session_of(ts) in self.sessions

    def _grans(self, tf: int) -> Dict[int, int]:
        """Timeframes needed as a function of the working tf. Override in
        strategies that use higher structure/bias timeframes."""
        return {tf: 300}

    def active(self, ts: pd.Timestamp) -> bool:
        """Fast pre-check: can this strategy possibly fire at `ts`? The
        backtester skips the (expensive) data slicing + signal() call when
        False. Must be side-effect-free and conservative (never False when
        signal() could fire)."""
        return True

    def signal(self, data: Dict[int, pd.DataFrame], now: pd.Timestamp) -> Optional[Signal]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
#  1) Legacy EMA(8/21) × RSI filter (the original bot strategy, M5)
# ──────────────────────────────────────────────────────────────────────────────

class EmaRsiStrategy(Strategy):
    name = "ema-rsi"
    DEFAULT_TF = M5

    SL_ATR, TP_ATR = 1.5, 2.5

    def signal(self, data, now):
        df = data[self.tf]
        if len(df) < 40:
            return None
        df = ensure_cols(df, {
            "ema_fast": ("ema", 8), "ema_slow": ("ema", 21),
            "rsi": ("rsi", 14), "atr": ("atr", 14),
        })
        cur, prev = df.iloc[-1], df.iloc[-2]
        a = float(cur["atr"])
        if a <= 0 or np.isnan(a):
            return None
        crossed_up   = cur["ema_fast"] > cur["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]
        crossed_down = cur["ema_fast"] < cur["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]
        r = float(cur["rsi"])
        entry = float(cur["close"])
        key = ("x", df.index[-1])
        if key in self._done:
            return None
        if crossed_up and 40 < r < 65:
            self._done.add(key)
            return Signal("long", entry - self.SL_ATR * a, entry + self.TP_ATR * a, "EMA cross up + RSI")
        if crossed_down and 35 < r < 60:
            self._done.add(key)
            return Signal("short", entry + self.SL_ATR * a, entry - self.TP_ATR * a, "EMA cross down + RSI")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  2) Liquidity Sweep + Market Structure Shift  (mechanical ICT variant)
# ──────────────────────────────────────────────────────────────────────────────

class SweepMssStrategy(Strategy):
    """
    Mechanical translation of the discretionary playbook:
      bias    : H1 close vs EMA(50) → longs only above, shorts only below
      sweep   : an M5 bar wicks BELOW a confirmed M5 swing low but CLOSES back
                above it (liquidity grab) — within the last SWEEP_LOOKBACK bars
      shift   : an M1 close breaks the most recent confirmed M1 swing high
                that formed during/just before the sweep (structure shift)
      entry   : on that M1 breaking close (we skip the FVG/OB refinement —
                that part of ICT is not mechanically well-defined)
      stop    : sweep wick low − 0.1×ATR(M5)
      target  : rr × R (default 2R)
      session : London/NY killzones only (standard ICT practice — and the
                off-session trades were the worst losers in backtests)
    """
    name = "sweep-mss"
    DEFAULT_TF = M1              # entry timeframe; structure=5×tf, bias=60×tf
    SWEEP_LOOKBACK = 12          # structure-tf bars in which the sweep must have happened
    MSS_TIMEOUT_M1 = 30          # entry-tf bars allowed between sweep and structure shift

    def _grans(self, tf):
        self.struct_tf = snap_tf(5 * tf)
        self.bias_tf   = snap_tf(max(H1, 60 * tf))
        return {self.bias_tf: 120, self.struct_tf: 200, tf: 240}

    def active(self, ts):
        return in_window(ts) is not None

    def signal(self, data, now):
        h1, m5, m1 = data[self.bias_tf], data[self.struct_tf], data[self.tf]
        if len(h1) < 60 or len(m5) < 40 or len(m1) < 40:
            return None
        if in_window(m1.index[-1]) is None:
            return None
        h1 = ensure_cols(h1, {"ema50": ("ema", 50)})
        m5 = ensure_cols(m5, {"atr": ("atr", 14)})
        bias = "long" if float(h1["close"].iloc[-1]) > float(h1["ema50"].iloc[-1]) else "short"

        atr5 = float(m5["atr"].iloc[-1])
        if atr5 <= 0 or np.isnan(atr5):
            return None

        piv_h, piv_l = confirmed_pivots(m5, k=2)
        lows, highs = m5["low"].values, m5["high"].values
        closes = m5["close"].values
        n5 = len(m5)

        sweep = None   # (m5_pos, swept_level, wick_extreme)
        for b in range(max(0, n5 - self.SWEEP_LOOKBACK), n5):
            if bias == "long":
                prior = piv_l[piv_l < b - 2]
                if len(prior) == 0:
                    continue
                level = lows[prior[-1]]
                if lows[b] < level and closes[b] > level:
                    sweep = (b, level, lows[b])
            else:
                prior = piv_h[piv_h < b - 2]
                if len(prior) == 0:
                    continue
                level = highs[prior[-1]]
                if highs[b] > level and closes[b] < level:
                    sweep = (b, level, highs[b])
        if sweep is None:
            return None

        b, level, wick = sweep
        key = (bias, m5.index[b])
        if key in self._done:
            return None
        sweep_end = m5.index[b] + pd.Timedelta(seconds=self.struct_tf)

        # M1 bars strictly after the sweep bar closed, bounded by timeout
        after = m1[m1.index >= sweep_end]
        if len(after) == 0 or len(after) > self.MSS_TIMEOUT_M1:
            return None

        # structure level on M1: last confirmed swing high (long) formed in the
        # 30 minutes up to the sweep close
        ctx_span = pd.Timedelta(seconds=30 * self.tf)   # ≙ 30 min when tf=M1
        ctx = m1[(m1.index >= sweep_end - ctx_span) & (m1.index < sweep_end)]
        if len(ctx) < 6:
            return None
        c_h, c_l = confirmed_pivots(ctx, k=2)
        closes_after = after["close"].values
        if bias == "long":
            if len(c_h) == 0:
                return None
            structure = float(ctx["high"].values[c_h[-1]])
            # fire only on the FIRST M1 close breaking the structure level
            if not (closes_after[-1] > structure and not (closes_after[:-1] > structure).any()):
                return None
            entry = float(closes_after[-1])
            sl = wick - 0.1 * atr5
            if entry <= sl:
                return None
            self._done.add(key)
            return Signal("long", sl, entry + self.rr * (entry - sl), f"sweep@{level:.2f}+MSS")
        else:
            if len(c_l) == 0:
                return None
            structure = float(ctx["low"].values[c_l[-1]])
            if not (closes_after[-1] < structure and not (closes_after[:-1] < structure).any()):
                return None
            entry = float(closes_after[-1])
            sl = wick + 0.1 * atr5
            if entry >= sl:
                return None
            self._done.add(key)
            return Signal("short", sl, entry - self.rr * (sl - entry), f"sweep@{level:.2f}+MSS")


# ──────────────────────────────────────────────────────────────────────────────
#  3) Session VWAP Reclaim
# ──────────────────────────────────────────────────────────────────────────────

class VwapReclaimStrategy(Strategy):
    """
      deviation : price stretched ≥ DEV_ATR×ATR(M1) below the session VWAP
                  at some point in the last DEV_LOOKBACK bars
      reclaim   : a close crosses back above the VWAP
      retest    : within RETEST_TIMEOUT bars, a bar dips to VWAP (±RETEST_ATR
                  ×ATR) and closes back above → entry
      stop      : VWAP − SL_ATR×ATR    target: rr×R
      sessions  : London open & NY open windows only (that's where the edge is
                  claimed to live; also keeps the TWAP-proxy honest)
    """
    name = "vwap-reclaim"
    DEFAULT_TF = M1
    DEV_ATR, SL_ATR, RETEST_ATR = 1.0, 1.0, 0.25
    DEV_LOOKBACK, RETEST_TIMEOUT = 60, 10

    def _grans(self, tf):
        return {tf: 480}

    def __init__(self, rr=None, tf=None, sessions=None):
        super().__init__(rr, tf, sessions)
        self._pending = None   # (dir, cross_ts, vwap_at_cross)

    def active(self, ts):
        return in_window(ts) is not None

    def signal(self, data, now):
        df = data[self.tf]
        if len(df) < self.DEV_LOOKBACK + 20:
            return None
        if in_window(df.index[-1]) is None:
            self._pending = None
            return None

        df = ensure_cols(df, {"atr": ("atr", 14)})
        vwap = anchored_vwap(df)
        a = float(df["atr"].iloc[-1])
        if a <= 0 or np.isnan(a):
            return None

        c, v = df["close"], vwap
        crossed_up   = c.iloc[-2] <= v.iloc[-2] and c.iloc[-1] > v.iloc[-1]
        crossed_down = c.iloc[-2] >= v.iloc[-2] and c.iloc[-1] < v.iloc[-1]

        look = slice(-self.DEV_LOOKBACK - 1, -1)
        max_dev_below = float((v.iloc[look] - df["low"].iloc[look]).max())
        max_dev_above = float((df["high"].iloc[look] - v.iloc[look]).max())

        if crossed_up and max_dev_below >= self.DEV_ATR * a:
            self._pending = ("long", df.index[-1], float(v.iloc[-1]))
            return None
        if crossed_down and max_dev_above >= self.DEV_ATR * a:
            self._pending = ("short", df.index[-1], float(v.iloc[-1]))
            return None

        if self._pending is None:
            return None
        pdir, pts, _ = self._pending
        bars_since = int((df.index[-1] - pts).total_seconds() // self.tf)
        if bars_since > self.RETEST_TIMEOUT:
            self._pending = None
            return None

        vn = float(v.iloc[-1])
        key = (pdir, pts)
        if key in self._done:
            return None
        if pdir == "long":
            if float(df["low"].iloc[-1]) <= vn + self.RETEST_ATR * a and float(c.iloc[-1]) > vn:
                entry = float(c.iloc[-1])
                sl = vn - self.SL_ATR * a
                if entry <= sl:
                    return None
                self._done.add(key)
                self._pending = None
                return Signal("long", sl, entry + self.rr * (entry - sl), "VWAP reclaim")
        else:
            if float(df["high"].iloc[-1]) >= vn - self.RETEST_ATR * a and float(c.iloc[-1]) < vn:
                entry = float(c.iloc[-1])
                sl = vn + self.SL_ATR * a
                if entry >= sl:
                    return None
                self._done.add(key)
                self._pending = None
                return Signal("short", sl, entry - self.rr * (sl - entry), "VWAP reclaim")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  4) Opening Range Breakout (ORB) with retest
# ──────────────────────────────────────────────────────────────────────────────

class OrbStrategy(Strategy):
    """
      range    : first OR_BARS M1 bars of the London / NY session
      filter   : H1 trend (close vs EMA50) must agree with breakout direction;
                 volatility rising (ATR(M1) now > ATR ATR_CMP_BARS bars ago)
      breakout : M1 close beyond the range
      retest   : within RETEST_TIMEOUT bars, price touches the broken edge
                 (±RETEST_ATR×ATR) and closes back in the breakout direction
      stop     : the opposite side of the range      target : rr×R
      limit    : one trade per session per direction
    """
    name = "orb"
    DEFAULT_TF = M1               # OR = first OR_BARS bars of tf after the open
    OR_BARS, RETEST_TIMEOUT, ATR_CMP_BARS = 5, 15, 20
    RETEST_ATR = 0.1
    SESSIONS = ("london", "ny")   # which session opens to trade

    def _grans(self, tf):
        self.bias_tf = snap_tf(max(H1, 12 * tf))
        return {tf: 480, self.bias_tf: 120}

    def __init__(self, rr=None, tf=None, sessions=None):
        super().__init__(rr, tf, sessions)
        # ORB's own session-open logic follows the user's filter ("off" has no
        # session open, so it can't apply here and is dropped)
        if self.sessions:
            narrowed = tuple(s for s in self.sessions if s in ("london", "ny"))
            if narrowed:
                self.SESSIONS = narrowed
        self._breakout = {}   # session_key -> (dir, breakout_ts, or_high, or_low)

    def _session_open(self, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        day = ts.normalize()
        opens = {"london": LONDON_OPEN_UTC, "ny": NY_OPEN_UTC}
        for name in self.SESSIONS:
            t0 = opens[name]
            o = day + pd.Timedelta(hours=t0.hour, minutes=t0.minute)
            if o <= ts < o + pd.Timedelta(hours=3):
                return o
        return None

    def active(self, ts):
        return self._session_open(ts) is not None

    def signal(self, data, now):
        m1, h1 = data[self.tf], data[self.bias_tf]
        if len(m1) < 60 or len(h1) < 60:
            return None
        ts = m1.index[-1]
        s_open = self._session_open(ts)
        if s_open is None:
            return None

        orb = m1[(m1.index >= s_open)]
        if len(orb) < self.OR_BARS + 1:
            return None
        rng = orb.iloc[:self.OR_BARS]
        or_high, or_low = float(rng["high"].max()), float(rng["low"].min())
        skey = s_open

        h1 = ensure_cols(h1, {"ema50": ("ema", 50)})
        m1 = ensure_cols(m1, {"atr": ("atr", 14)})
        trend = "long" if float(h1["close"].iloc[-1]) > float(h1["ema50"].iloc[-1]) else "short"
        a = float(m1["atr"].iloc[-1])
        if a <= 0 or np.isnan(a):
            return None
        atr_rising = float(m1["atr"].iloc[-1]) > float(m1["atr"].iloc[-1 - self.ATR_CMP_BARS])

        c = float(m1["close"].iloc[-1])

        # phase 1: detect breakout close (in trend direction, vol rising)
        bo = self._breakout.get(skey)
        if bo is None:
            if trend == "long" and c > or_high and atr_rising:
                self._breakout[skey] = ("long", ts, or_high, or_low)
            elif trend == "short" and c < or_low and atr_rising:
                self._breakout[skey] = ("short", ts, or_high, or_low)
            return None

        # phase 2: retest of the broken edge
        bdir, bts, bhigh, blow = bo
        key = (skey, bdir)
        if key in self._done:
            return None
        bars_since = int((ts - bts).total_seconds() // self.tf)
        if bars_since > self.RETEST_TIMEOUT:
            del self._breakout[skey]
            return None

        if bdir == "long":
            if float(m1["low"].iloc[-1]) <= bhigh + self.RETEST_ATR * a and c > bhigh:
                sl = blow
                if c <= sl:
                    return None
                self._done.add(key)
                return Signal("long", sl, c + self.rr * (c - sl), f"ORB {s_open.time()} retest")
        else:
            if float(m1["high"].iloc[-1]) >= blow - self.RETEST_ATR * a and c < blow:
                sl = bhigh
                if c >= sl:
                    return None
                self._done.add(key)
                return Signal("short", sl, c - self.rr * (sl - c), f"ORB {s_open.time()} retest")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  5) Williams Alligator + fractal breakout
# ──────────────────────────────────────────────────────────────────────────────

class AlligatorStrategy(Strategy):
    """
    Bill Williams' classic system, mechanized on M5:
      lines   : SMMA of median price (H+L)/2 — jaw 13/8, teeth 8/5, lips 5/3
                (period / forward shift — shifted values are past data ⇒ causal)
      awake   : lines fanned in order (lips > teeth > jaw for long) AND the
                jaw–lips spread is wider than SPREAD_BARS bars ago (the
                alligator is "opening its mouth", not sleeping)
      trigger : first close breaking the last confirmed up-fractal (k=2)
                sitting above the teeth — Williams' fractal entry
      stop    : last confirmed opposite fractal
      target  : rr × R (default 2R)
    """
    name = "alligator"
    DEFAULT_TF = M5
    SPREAD_BARS = 5

    def signal(self, data, now):
        df = data[self.tf]
        if len(df) < 80:
            return None
        if not {"jaw", "teeth", "lips"} <= set(df.columns):
            df = df.copy()
            df["jaw"], df["teeth"], df["lips"] = alligator_lines(df)

        jaw, teeth, lips = (float(df[c].iloc[-1]) for c in ("jaw", "teeth", "lips"))
        if any(np.isnan(x) for x in (jaw, teeth, lips)):
            return None

        spread_now  = abs(lips - jaw)
        spread_then = abs(float(df["lips"].iloc[-1 - self.SPREAD_BARS])
                          - float(df["jaw"].iloc[-1 - self.SPREAD_BARS]))
        if np.isnan(spread_then) or spread_now <= spread_then:
            return None   # alligator sleeping or closing

        piv_h, piv_l = confirmed_pivots(df, k=2)
        c_now, c_prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])

        if lips > teeth > jaw and len(piv_h) and len(piv_l):
            f_high = float(df["high"].values[piv_h[-1]])
            key = ("L", df.index[piv_h[-1]])
            # first close through an up-fractal that sits above the teeth
            if f_high > teeth and c_now > f_high >= c_prev and key not in self._done:
                sl = float(df["low"].values[piv_l[-1]])
                if c_now > sl:
                    self._done.add(key)
                    return Signal("long", sl, c_now + self.rr * (c_now - sl), "alligator+fractal")

        if lips < teeth < jaw and len(piv_h) and len(piv_l):
            f_low = float(df["low"].values[piv_l[-1]])
            key = ("S", df.index[piv_l[-1]])
            if f_low < teeth and c_now < f_low <= c_prev and key not in self._done:
                sl = float(df["high"].values[piv_h[-1]])
                if c_now < sl:
                    self._done.add(key)
                    return Signal("short", sl, c_now - self.rr * (sl - c_now), "alligator+fractal")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  6) Croco — Alligator optimisé + règle minée sur les données
# ──────────────────────────────────────────────────────────────────────────────

class CrocoStrategy(AlligatorStrategy):
    """
    Williams Alligator avec la configuration optimale issue du sweep TRAIN
    (2026-03-25 → 2026-06-15, frxXAUUSD) + une règle minée dans les trades:

      Config : tf M5, rr 3.0, SPREAD_BARS 5, entrées hors-session par défaut
               (TRAIN: 167 trades, +$912, PF 1.11 — vs PF ~0.6 en M1 et
               pertes systématiques à rr 1.5/2)
      R1     : aucune entrée entre 00:00 et 04:59 UTC. Sur TRAIN ce bloc
               horaire adjacent perdait de façon cohérente (58 trades,
               winrate 10%, −$1413) — début de séance asiatique sans
               direction sur l'or. Avec R1: 109 trades, +$2326, winrate 24%.

    Règles envisagées et REJETÉES (voir analyze_patterns.py):
      - mouth ≤ 1 ATR: positif seul, mais dégrade combiné à R1
      - jeudi exclu: chiffres flatteurs mais aucun mécanisme plausible → overfit
      - alignement H1, body ratio, anti-streak: aucun signal dans les données

    ⚠ Les seuils viennent du TRAIN. La validité réelle se juge UNIQUEMENT sur
      la période TEST (2026-06-15 → 2026-07-06) et en démo live.
    """
    name = "croco"
    DEFAULT_TF = M5
    rr = 3.0
    SPREAD_BARS = 5
    DEAD_HOURS = range(0, 5)   # UTC — aucune entrée dans ce bloc

    def __init__(self, rr=None, tf=None, sessions=None):
        super().__init__(rr, tf, sessions)
        if self.sessions is None:
            self.sessions = ("off",)   # config optimale du sweep

    def signal(self, data, now):
        sig = super().signal(data, now)
        if sig is None:
            return None
        if data[self.tf].index[-1].hour in self.DEAD_HOURS:
            return None   # R1 (le setup est consommé, comme dans le mining)
        return sig


# ──────────────────────────────────────────────────────────────────────────────
#  7) Alligator v2 — retournement à travers la gueule, filtré par la tendance H1
# ──────────────────────────────────────────────────────────────────────────────

class AlligatorV2Strategy(Strategy):
    """Bill Williams inversé — on trade la CASSURE de la gueule, pas sa tendance.

    ... (votre documentation) ...
    """

    name = "alligator-v2"
    DEFAULT_TF = M5
    ABOVE_LOOKBACK = 10
    SL_MODE = "body"
    SL_BUFFER_ATR = 0.5
    TRAIL_MODE = "pnl"
    TRAIL_STEP = 1.0
    TRAIL_LOCK = 1.0

    def _grans(self, tf):
        self.bias_tf = snap_tf(max(H1, 12 * tf))
        return {tf: 120, self.bias_tf: 120}

    def __init__(self, rr=None, tf=None, sessions=None):
        super().__init__(rr, tf, sessions)
        self._pending = None
        self._last_bar = None

    def plot_current_state(self, data, nb_bougies=100):
        """Génère une seule image du marché à l'état actuel avec vos lignes

        d'Alligator pour comparaison avec TradingView.
        """
        df = data[self.tf]
        if len(df) < 60:
            print("Pas assez de données pour tracer le graphique.")
            return

        # On s'assure que les colonnes Alligator existent
        if not {"jaw", "teeth", "lips"} <= set(df.columns):
            df = df.copy()
            df["jaw"], df["teeth"], df["lips"] = alligator_lines(df)

        # Extraction des N dernières bougies pour le visuel
        df_plot = df.tail(nb_bougies)

        plt.figure(figsize=(14, 7))

        # Rendu des bougies (Open, High, Low, Close)
        for idx, row in df_plot.iterrows():
            color = "green" if row["close"] >= row["open"] else "red"
            # Mèches
            plt.plot(
                [idx, idx], [row["low"], row["high"]], color=color, linewidth=1
            )
            # Corps
            plt.plot(
                [idx, idx],
                [row["open"], row["close"]],
                color=color,
                linewidth=4,
                solid_capstyle="butt",
            )

        # Tracé de VOS lignes Alligator calculées par le bot
        plt.plot(
            df_plot.index,
            df_plot["jaw"],
            label="Jaw (Bleu)",
            color="blue",
            linewidth=1.5,
        )
        plt.plot(
            df_plot.index,
            df_plot["teeth"],
            label="Teeth (Rouge)",
            color="red",
            linewidth=1.5,
        )
        plt.plot(
            df_plot.index,
            df_plot["lips"],
            label="Lips (Vert)",
            color="green",
            linewidth=1.5,
        )

        plt.title(
            f"Vérification Alligator - TF: {self.tf} (Dernières {nb_bougies} bougies)",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()

        # Affiche l'image instantanément
        plt.show()

    def signal(self, data, now):
        df = data[self.tf]
        if len(df) < 60:
            return None
        if not {"jaw", "teeth", "lips"} <= set(df.columns):
            df = df.copy()
            df["jaw"], df["teeth"], df["lips"] = alligator_lines(df)

        self.plot_current_state(data)
        # ---------------------------------------------------------------------
        # NOTE : Pour voir le graphique en direct au moment où le bot tourne,
        # vous pouvez appeler la méthode ici (attention, bloquant à chaque bougie) :
        # self.plot_current_state(data)
        # ---------------------------------------------------------------------

        ts = df.index[-1]
        if ts == self._last_bar:
            return None
        self._last_bar = ts

        h1 = data[self.bias_tf]
        if len(h1) < 60:
            return None
        h1 = ensure_cols(h1, {"ema50": ("ema", 50)})
        h1_bull = float(h1["close"].iloc[-1]) > float(h1["ema50"].iloc[-1])

        df = ensure_cols(df, {"atr": ("atr", 14)})
        o, c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
        jaw = float(df["jaw"].iloc[-1])
        teeth = float(df["teeth"].iloc[-1])
        lips = float(df["lips"].iloc[-1])
        atr = float(df["atr"].iloc[-1])
        if any(np.isnan(x) for x in (jaw, teeth, lips)):
            self._pending = None
            return None
        is_green, is_red = c > o, c < o

        # ── 1) résolution d'une confirmation en attente ──────────────────────
        if self._pending is not None:
            p = self._pending
            good = is_green if p["dir"] == "long" else is_red
            if good:
                self._pending = None
                return self._fire(p, c, atr, h1_bull)
            if not p["tolerated"]:
                p["tolerated"] = True
                return None
            self._pending = None

        # ── 2) détection d'un déclencheur sur la bougie clôturée ─────────────
        n = len(df)
        w = slice(max(0, n - 1 - self.ABOVE_LOOKBACK), n - 1)
        lo_w, hi_w = df["low"].values[w], df["high"].values[w]
        jw, th, lp = (
            df["jaw"].values[w],
            df["teeth"].values[w],
            df["lips"].values[w],
        )

        if lips > teeth > jaw and o > jaw > c:
            if np.any((lo_w > lp) & (lp > th) & (th > jw)):
                self._pending = {
                    "dir": "short",
                    "trig_ts": ts,
                    "sl_price": max(o, c),
                    "tolerated": False,
                }
        elif jaw > teeth > lips and o < jaw < c:
            if np.any((hi_w < lp) & (lp < th) & (th < jw)):
                self._pending = {
                    "dir": "long",
                    "trig_ts": ts,
                    "sl_price": min(o, c),
                    "tolerated": False,
                }
        return None

    def _fire(self, p, close, atr, h1_bull):
        key = (p["dir"], p["trig_ts"])
        if key in self._done:
            return None

        sl = p["sl_price"]
        if (p["dir"] == "long" and close <= sl) or (
            p["dir"] == "short" and close >= sl
        ):
            return None
        self._done.add(key)
        reason = "alligator-v2 " + ("buy" if p["dir"] == "long" else "sell")

        mode = self.TRAIL_MODE
        if mode == "pnl":
            return Signal(
                p["dir"],
                sl,
                None,
                reason,
                trail_kind="pnl",
                trail_step=self.TRAIL_STEP,
                trail_lock=self.TRAIL_LOCK,
            )
        if mode == "r":
            unit = abs(close - sl)
        elif mode == "atr":
            unit = atr
        else:
            raise ValueError(f"TRAIL_MODE inconnu: {mode!r}")
        if not unit or np.isnan(unit) or unit <= 0:
            return None
        return Signal(
            p["dir"],
            sl,
            None,
            reason,
            trail_kind="dist",
            trail_step=self.TRAIL_STEP * unit,
            trail_lock=self.TRAIL_LOCK * unit,
        )
    


# ──────────────────────────────────────────────────────────────────────────────
#  Registry
# ──────────────────────────────────────────────────────────────────────────────

REGISTRY = {
    cls.name: cls
    for cls in (EmaRsiStrategy, SweepMssStrategy, VwapReclaimStrategy, OrbStrategy,
                AlligatorStrategy, CrocoStrategy, AlligatorV2Strategy)
}


def make_strategy(name: str, rr: Optional[float] = None, tf=None, sessions=None) -> Strategy:
    """tf: entry timeframe — seconds or 'M1'/'M5'/'M15'/'H1'/'5m'... (None = strategy default).
    Higher structure/bias timeframes scale with it and snap to valid Deriv granularities.
    sessions: restrict entries to 'london' / 'ny' / 'off' (comma-separated), None = default.
    Note: vwap-reclaim & sweep-mss already require london/ny internally, so
    sessions='off' yields zero trades for them — same for orb (needs a session open)."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name](rr=rr, tf=parse_tf(tf), sessions=parse_sessions(sessions))
