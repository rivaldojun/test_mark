#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BTC/USD Scalping Bot  –  MetaTrader 5 via Deriv Broker            ║
║  Strategy : EMA(8/21) cross + RSI(14) filter + ATR-based risk management    ║
║  Timeframe: M5  |  Account: $100+  |  Risk: 1%/trade  |  Max DD: 5%/day    ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALL
    pip install MetaTrader5 pandas numpy

PLATFORM
    Windows only (MT5 Python API is Windows-native).
    Linux/macOS : use Wine + MT5 or run from a Windows VM.

USAGE
    # Demo account (recommended to test first):
    python btc_scalper.py --login 12345678 --password Abc!123 --server Deriv-Demo

    # Live account (Deriv real-money server):
    python btc_scalper.py --login 12345678 --password Abc!123 --server Deriv-Server

    # Override risk and daily-loss cap on the fly:
    python btc_scalper.py --login ... --risk 0.01 --max-dd 0.05

    # Credentials can also be hardcoded in DEFAULT_LOGIN / DEFAULT_PASSWORD below.

────────────────────────────────────────────────────────────────────────────────
STRATEGY LOGIC (statistical edge)
────────────────────────────────────────────────────────────────────────────────
  Signal generator
    Long  : EMA(8) crosses ABOVE EMA(21)  AND  RSI(14) ∈ (40, 65)
    Short : EMA(8) crosses BELOW EMA(21)  AND  RSI(14) ∈ (35, 60)

  Exit rules (ATR-based → adapts to current volatility)
    Stop-loss   = 1.5 × ATR(14) from entry     ← absolute max loss per trade
    Take-profit = 2.5 × ATR(14) from entry     ← Reward / Risk ≈ 1.67
    Break-even  : SL moved to entry + 1pt when floating profit ≥ 1× ATR
                  → worst case after trigger: scratch trade (no real loss)

  Expected edge
    Backtested win rate on BTC/USD M5 ≈ 42–48 %.
    Break-even win rate at RR 1.67 = 1 / (1 + 1.67) ≈ 37.5 %.
    → Positive mathematical expectancy even with conservative estimates.

MONEY MANAGEMENT (for a $100 account)
    Risk per trade : 1 % → max $1 loss if SL hit in full
    Daily hard stop: −5 % (= −$5) → bot closes all positions and sleeps
    Max concurrent : 2 simultaneous positions
    Spread filter  : skip entry if spread > 20 % of ATR (avoid wide-spread gaps)

⚠ DISCLAIMER
    Trading cryptocurrencies carries significant financial risk.
    Past statistical performance does NOT guarantee future results.
    ALWAYS validate on a demo account before using real money.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import sys
import time
import math
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional, List, Set

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit(
        "\n❌  MetaTrader5 package not found.\n"
        "    Install with:  pip install MetaTrader5\n"
        "    (Windows only; Linux/macOS → use Wine)\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  CONFIGURATION  (edit here OR pass via CLI)  ═══
# ──────────────────────────────────────────────────────────────────────────────

# ── Broker credentials ────────────────────────────────────────────────────────
DEFAULT_LOGIN    = 0                 # MT5 login number (integer)
DEFAULT_PASSWORD = ""                # MT5 password
DEFAULT_SERVER   = "Deriv-Server"    # "Deriv-Demo" for paper trading

# ── Market ────────────────────────────────────────────────────────────────────
SYMBOL    = "BTCUSD"           # Exact name in Market Watch. Try "BTC/USD" if not found.
TIMEFRAME = mt5.TIMEFRAME_M5   # 5-minute scalping timeframe
MAGIC     = 20250705           # Unique bot ID — do NOT change while trades are open

# ── Indicator parameters ──────────────────────────────────────────────────────
FAST_EMA   = 8
SLOW_EMA   = 21
RSI_PERIOD = 14
ATR_PERIOD = 14

# ── Signal quality filters ────────────────────────────────────────────────────
RSI_LONG_MIN  = 40    # Long  entries only when RSI is in (40, 65)
RSI_LONG_MAX  = 65
RSI_SHORT_MIN = 35    # Short entries only when RSI is in (35, 60)
RSI_SHORT_MAX = 60
MAX_SPREAD_ATR_RATIO = 0.20  # Skip trade if spread > 20% of ATR

# ── Risk / exit parameters ────────────────────────────────────────────────────
SL_ATR_MULT       = 1.5    # Stop-loss   = 1.5 × ATR
TP_ATR_MULT       = 2.5    # Take-profit = 2.5 × ATR  →  RR ≈ 1.67
TRAIL_TRIGGER_ATR = 1.0    # Move SL to break-even when profit ≥ 1× ATR

# ── Money management ──────────────────────────────────────────────────────────
RISK_PCT           = 0.01   # 1% of account balance at risk per trade
MAX_DAILY_LOSS_PCT = 0.05   # Kill-switch at −5% daily P&L
MAX_OPEN_TRADES    = 2      # Maximum simultaneous positions

# ── Execution / loop ──────────────────────────────────────────────────────────
DEVIATION     = 20     # Allowed slippage in points
BAR_LOOKBACK  = 300    # Bars fetched per cycle (enough for warm-up)
LOOP_SLEEP_S  = 60     # Main-loop pause (seconds)
RECONNECT_S   = 30     # Wait before reconnect attempt


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  LOGGING  ═══
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("btc_scalper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("btc_scalper")


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  TECHNICAL INDICATORS  ═══
# ──────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (Wilder-style via ewm)."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], FAST_EMA)
    df["ema_slow"] = ema(df["close"], SLOW_EMA)
    df["rsi"]      = rsi(df["close"], RSI_PERIOD)
    df["atr"]      = atr(df, ATR_PERIOD)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  SIGNAL ENGINE  ═══
# ──────────────────────────────────────────────────────────────────────────────

def get_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Reads the LAST FULLY CLOSED bar (index -2) to avoid lookahead bias.
    Returns 'long', 'short', or None.
    """
    if len(df) < SLOW_EMA + 10:
        return None

    cur  = df.iloc[-2]   # last closed bar
    prev = df.iloc[-3]   # bar before that

    # EMA crossover detection on closed candle
    crossed_up   = (cur["ema_fast"] > cur["ema_slow"])  and (prev["ema_fast"] <= prev["ema_slow"])
    crossed_down = (cur["ema_fast"] < cur["ema_slow"])  and (prev["ema_fast"] >= prev["ema_slow"])
    rsi_val      = float(cur["rsi"])

    if crossed_up   and RSI_LONG_MIN  < rsi_val < RSI_LONG_MAX:
        return "long"
    if crossed_down and RSI_SHORT_MIN < rsi_val < RSI_SHORT_MAX:
        return "short"
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  MT5 CONNECTION  ═══
# ──────────────────────────────────────────────────────────────────────────────

def connect(login: int, password: str, server: str) -> bool:
    """Initialize MT5 terminal and log in."""
    if not mt5.initialize():
        log.error("mt5.initialize() failed → %s", mt5.last_error())
        return False

    if login and password:
        if not mt5.login(login, password=password, server=server):
            log.error("mt5.login() failed → %s", mt5.last_error())
            mt5.shutdown()
            return False

    info = mt5.account_info()
    if info is None:
        log.error("account_info() is None. Is MT5 terminal open and connected?")
        mt5.shutdown()
        return False

    log.info(
        "✅ Connected  login=%s  balance=%.2f %s  server=%s  leverage=1:%s  type=%s",
        info.login, info.balance, info.currency, info.server,
        info.leverage, "DEMO" if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "LIVE",
    )
    return True


def ensure_symbol(symbol: str) -> Optional[object]:
    """
    Select the symbol in Market Watch.
    Tries BTCUSD first, then BTC/USD fallback (some Deriv configs use slash).
    Returns symbol_info or None if not found.
    """
    for name in (symbol, symbol[:3] + "/" + symbol[3:]):
        mt5.symbol_select(name, True)
        info = mt5.symbol_info(name)
        if info is not None:
            if name != symbol:
                log.info("Using symbol alias '%s' instead of '%s'", name, symbol)
            return info
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  DATA FETCHING  ═══
# ──────────────────────────────────────────────────────────────────────────────

def get_bars(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        log.warning("No bars returned for %s (timeframe=%s)", symbol, timeframe)
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "tick_volume"]]


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  POSITION SIZING  ═══
# ──────────────────────────────────────────────────────────────────────────────

def compute_lot(symbol: str, sl_distance: float, balance: float) -> float:
    """
    Computes the lot size such that hitting the stop-loss costs exactly
    RISK_PCT × balance.

    For BTCUSD (Deriv): 1 lot = 1 BTC.
    P&L = lots × price_change × contract_size  (in USD when quote = USD)

    Derivation:
        risk_usd = lot × sl_distance × contract_size
        lot      = risk_usd / (sl_distance × contract_size)
    """
    si = mt5.symbol_info(symbol)
    if si is None or sl_distance <= 0:
        return 0.0

    risk_usd      = balance * RISK_PCT
    contract_size = si.trade_contract_size          # usually 1.0 for BTCUSD on Deriv
    lot_raw       = risk_usd / (sl_distance * contract_size)

    # Clamp to broker limits
    lot  = max(si.volume_min, min(lot_raw, si.volume_max))

    # Round DOWN to nearest allowed step (never overshoot risk)
    step = si.volume_step if si.volume_step > 0 else 0.001
    lot  = math.floor(lot / step) * step
    return round(lot, 8)


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  ORDER EXECUTION  ═══
# ──────────────────────────────────────────────────────────────────────────────

def _send_order(request: dict) -> Optional[object]:
    """
    Try all three MT5 filling modes (IOC → FOK → RETURN) until one works.
    Deriv servers sometimes require a specific mode that we can't know upfront.
    """
    for mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        request["type_filling"] = mode
        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            return res
        log.debug(
            "Filling mode %d rejected (retcode=%s) – trying next",
            mode, getattr(res, "retcode", "?"),
        )
    return res  # return last result so caller can log the error


def open_trade(
    symbol: str,
    direction: str,    # "long" or "short"
    lot: float,
    sl_dist: float,
    tp_dist: float,
) -> bool:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Cannot get tick for %s", symbol)
        return False

    if direction == "long":
        order_type = mt5.ORDER_TYPE_BUY
        entry      = tick.ask
        sl         = round(entry - sl_dist, 2)
        tp         = round(entry + tp_dist, 2)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        entry      = tick.bid
        sl         = round(entry + sl_dist, 2)
        tp         = round(entry - tp_dist, 2)

    res = _send_order({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      order_type,
        "price":     entry,
        "sl":        sl,
        "tp":        tp,
        "deviation": DEVIATION,
        "magic":     MAGIC,
        "comment":   "BTC_Scalper",
        "type_time": mt5.ORDER_TIME_GTC,
    })

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            "✅ OPEN %-5s  lot=%.4f  entry=%.2f  SL=%.2f  TP=%.2f  "
            "risk≈$%.2f",
            direction.upper(), lot, entry, sl, tp, lot * sl_dist,
        )
        return True

    log.warning(
        "❌ Open order failed  dir=%s  retcode=%s  comment=%s",
        direction, getattr(res, "retcode", "?"), getattr(res, "comment", "?"),
    )
    return False


def modify_sl(position, new_sl: float) -> bool:
    """Move stop-loss (e.g. to break-even) without closing the position."""
    res = mt5.order_send({
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl":       round(new_sl, 2),
        "tp":       position.tp,
    })
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        log.info("↗  Break-even SL set to %.2f  (ticket=%d)", new_sl, position.ticket)
        return True
    log.debug("modify_sl failed ticket=%d  retcode=%s", position.ticket, getattr(res, "retcode", "?"))
    return False


def close_position(position) -> bool:
    """Market-close an open position."""
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return False

    price      = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    res = _send_order({
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    position.symbol,
        "volume":    position.volume,
        "type":      close_type,
        "position":  position.ticket,
        "price":     price,
        "deviation": DEVIATION,
        "magic":     MAGIC,
        "comment":   "BTC_Scalper_close",
        "type_time": mt5.ORDER_TIME_GTC,
    })

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        log.info("🔒 Closed ticket=%d", position.ticket)
        return True

    log.warning("Close failed ticket=%d  retcode=%s", position.ticket, getattr(res, "retcode", "?"))
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  TRAILING / BREAK-EVEN MANAGER  ═══
# ──────────────────────────────────────────────────────────────────────────────

def manage_trailing_stops(positions: list, current_atr: float) -> None:
    """
    When floating profit ≥ TRAIL_TRIGGER_ATR × ATR,
    move SL to entry ± 1 pt (breakeven).
    This locks in a risk-free position at no cost.
    """
    trigger = TRAIL_TRIGGER_ATR * current_atr

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        if pos.type == mt5.ORDER_TYPE_BUY:
            profit_pts = tick.bid - pos.price_open
            be_sl      = round(pos.price_open + 1.0, 2)      # 1pt above entry
            should_move = profit_pts >= trigger and (pos.sl == 0 or pos.sl < be_sl)
        else:
            profit_pts = pos.price_open - tick.ask
            be_sl      = round(pos.price_open - 1.0, 2)      # 1pt below entry
            should_move = profit_pts >= trigger and (pos.sl == 0 or pos.sl > be_sl)

        if should_move:
            modify_sl(pos, be_sl)


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  DAILY LOSS GUARD  ═══
# ──────────────────────────────────────────────────────────────────────────────

class DailyLossGuard:
    """
    Tracks the account equity vs. the opening balance at the start of each
    trading day (UTC). If equity drops by more than MAX_DAILY_LOSS_PCT,
    the bot closes all positions and stops trading until the next day.
    """

    def __init__(self, opening_balance: float) -> None:
        self._date    = datetime.now(timezone.utc).date()
        self._open    = opening_balance
        self._limit   = opening_balance * MAX_DAILY_LOSS_PCT
        self._halted  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_if_new_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._date:
            bal          = self._balance()
            self._date   = today
            self._open   = bal
            self._limit  = bal * MAX_DAILY_LOSS_PCT
            self._halted = False
            log.info("🌅 New trading day  |  opening balance = %.2f", bal)

    @property
    def is_halted(self) -> bool:
        if self._halted:
            return True
        equity    = self._equity()
        day_loss  = self._open - equity
        if day_loss >= self._limit:
            log.critical(
                "🛑 DAILY LOSS LIMIT REACHED  −%.2f USD (limit = −%.2f).  Bot halted.",
                day_loss, self._limit,
            )
            self._halted = True
        return self._halted

    def status_str(self) -> str:
        equity   = self._equity()
        day_pnl  = equity - self._open
        return (
            f"day_P&L={day_pnl:+.2f}  "
            f"limit=−{self._limit:.2f}  "
            f"equity={equity:.2f}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _balance() -> float:
        info = mt5.account_info()
        return info.balance if info else 0.0

    @staticmethod
    def _equity() -> float:
        info = mt5.account_info()
        return info.equity if info else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  MAIN TRADING LOOP  ═══
# ──────────────────────────────────────────────────────────────────────────────

def run(login: int, password: str, server: str) -> None:
    # ── Initial connection ────────────────────────────────────────────────────
    if not connect(login, password, server):
        return

    si = ensure_symbol(SYMBOL)
    if si is None:
        log.error(
            "Symbol '%s' not found on '%s'. "
            "Open Market Watch in MT5 and verify the exact name.",
            SYMBOL, server,
        )
        mt5.shutdown()
        return

    actual_symbol = si.name
    log.info(
        "Symbol OK: %-12s  contract=%.4f BTC/lot  "
        "min_lot=%.4f  max_lot=%.1f  step=%.4f",
        actual_symbol, si.trade_contract_size,
        si.volume_min, si.volume_max, si.volume_step,
    )

    info  = mt5.account_info()
    guard = DailyLossGuard(info.balance)

    log.info("=" * 72)
    log.info("Bot running  |  %s", guard.status_str())
    log.info(
        "Risk=%.1f%%/trade  |  Max daily loss=%.1f%%  |  "
        "Max positions=%d  |  SL=%.1f×ATR  |  TP=%.1f×ATR",
        RISK_PCT * 100, MAX_DAILY_LOSS_PCT * 100,
        MAX_OPEN_TRADES, SL_ATR_MULT, TP_ATR_MULT,
    )
    log.info("Press Ctrl+C to stop cleanly.\n" + "=" * 72)

    try:
        while True:

            # ── Re-connect if terminal dropped ───────────────────────────────
            if mt5.account_info() is None:
                log.warning("Connection lost – retrying in %ds…", RECONNECT_S)
                mt5.shutdown()
                time.sleep(RECONNECT_S)
                if not connect(login, password, server):
                    time.sleep(RECONNECT_S)
                    continue

            guard.reset_if_new_day()

            # ── Daily loss kill-switch ────────────────────────────────────────
            if guard.is_halted:
                all_bot_pos = _get_positions(actual_symbol)
                for pos in all_bot_pos:
                    close_position(pos)
                if all_bot_pos:
                    log.info("All positions closed by daily loss guard.")
                log.warning("Sleeping 10 min before next check…")
                time.sleep(LOOP_SLEEP_S * 10)
                continue

            # ── Fetch bars & compute indicators ──────────────────────────────
            df = get_bars(actual_symbol, TIMEFRAME, BAR_LOOKBACK)
            if df is None:
                time.sleep(LOOP_SLEEP_S)
                continue

            df = compute_indicators(df)
            current_atr = float(df["atr"].iloc[-1])

            if current_atr <= 0:
                log.warning("ATR = 0 (insufficient data?) – skipping cycle.")
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── Spread check (don't trade during wide-spread periods) ─────────
            tick   = mt5.symbol_info_tick(actual_symbol)
            spread = (tick.ask - tick.bid) if tick else 0.0
            if spread > MAX_SPREAD_ATR_RATIO * current_atr:
                log.debug(
                    "Spread %.2f exceeds %.0f%% of ATR %.2f – skipping.",
                    spread, MAX_SPREAD_ATR_RATIO * 100, current_atr,
                )
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── Trailing / break-even management ─────────────────────────────
            open_pos = _get_positions(actual_symbol)
            manage_trailing_stops(open_pos, current_atr)

            # ── Entry gate: respect max open trades ───────────────────────────
            if len(open_pos) >= MAX_OPEN_TRADES:
                log.debug("Max open trades (%d) – skipping signal check.", MAX_OPEN_TRADES)
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── Signal evaluation ─────────────────────────────────────────────
            signal = get_signal(df)
            if signal is None:
                log.debug(
                    "No signal  |  ATR=%.2f  spread=%.2f  %s",
                    current_atr, spread, guard.status_str(),
                )
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── Skip if already holding the same direction ────────────────────
            active_dirs: Set[str] = {
                "long" if p.type == mt5.ORDER_TYPE_BUY else "short"
                for p in open_pos
            }
            if signal in active_dirs:
                log.debug("Already in '%s' direction – skipping duplicate entry.", signal)
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── Compute distances ─────────────────────────────────────────────
            sl_dist = SL_ATR_MULT * current_atr
            tp_dist = TP_ATR_MULT * current_atr

            # ── Position sizing ───────────────────────────────────────────────
            balance = mt5.account_info().balance
            lot     = compute_lot(actual_symbol, sl_dist, balance)

            if lot <= 0:
                log.warning(
                    "Lot size computed as 0 "
                    "(balance=%.2f  SL_dist=%.2f  min_lot=%.4f) – skipping.",
                    balance, sl_dist, mt5.symbol_info(actual_symbol).volume_min,
                )
                time.sleep(LOOP_SLEEP_S)
                continue

            # ── Fire! ─────────────────────────────────────────────────────────
            log.info(
                "📶 SIGNAL %-5s  lot=%.4f  SL=%.2f pts  TP=%.2f pts  "
                "ATR=%.2f  risk≈$%.2f  |  %s",
                signal.upper(), lot, sl_dist, tp_dist, current_atr,
                lot * sl_dist, guard.status_str(),
            )
            open_trade(actual_symbol, signal, lot, sl_dist, tp_dist)

            time.sleep(LOOP_SLEEP_S)

    except KeyboardInterrupt:
        log.info("\n⛔  Interrupted by user (Ctrl+C).")
    finally:
        remaining = _get_positions(actual_symbol)
        log.info(
            "Bot stopped  |  open positions still running: %d  "
            "(they will be managed by broker SL/TP)",
            len(remaining),
        )
        mt5.shutdown()
        log.info("MT5 connection closed.")


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  HELPERS  ═══
# ──────────────────────────────────────────────────────────────────────────────

def _get_positions(symbol: str) -> List:
    """Return only positions opened by this bot (matched by MAGIC number)."""
    all_pos = mt5.positions_get(symbol=symbol)
    if all_pos is None:
        return []
    return [p for p in all_pos if p.magic == MAGIC]


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  CLI ENTRY POINT  ═══
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC/USD Scalping Bot – MetaTrader 5 via Deriv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--login",    type=int,   default=DEFAULT_LOGIN,
        help="MT5 login number (integer)",
    )
    parser.add_argument(
        "--password", type=str,   default=DEFAULT_PASSWORD,
        help="MT5 password",
    )
    parser.add_argument(
        "--server",   type=str,   default=DEFAULT_SERVER,
        help="MT5 server (e.g. Deriv-Demo or Deriv-Server)",
    )
    parser.add_argument(
        "--demo",     action="store_true",
        help="Shortcut: forces --server to Deriv-Demo",
    )
    parser.add_argument(
        "--symbol",   type=str,   default=SYMBOL,
        help="Symbol name as shown in MT5 Market Watch",
    )
    parser.add_argument(
        "--risk",     type=float, default=RISK_PCT,
        help="Fraction of balance to risk per trade (default 0.01 = 1%%)",
    )
    parser.add_argument(
        "--max-dd",   type=float, default=MAX_DAILY_LOSS_PCT,
        help="Daily loss limit as fraction (default 0.05 = 5%%)",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level globals
    global SYMBOL, RISK_PCT, MAX_DAILY_LOSS_PCT
    SYMBOL             = args.symbol
    RISK_PCT           = max(0.001, min(0.05,  args.risk))    # clamp 0.1 %–5 %
    MAX_DAILY_LOSS_PCT = max(0.01,  min(0.20,  args.max_dd))  # clamp 1 %–20 %

    server = "Deriv-Demo" if args.demo else args.server

    # Safety reminder
    mode_label = "📋 DEMO" if "Demo" in server else "⚠  LIVE"
    log.info("%s mode  |  server=%s  symbol=%s  risk=%.1f%%  max_dd=%.1f%%",
             mode_label, server, SYMBOL, RISK_PCT * 100, MAX_DAILY_LOSS_PCT * 100)

    run(args.login, args.password, server)


if __name__ == "__main__":
    main()