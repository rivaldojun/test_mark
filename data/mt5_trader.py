#!/usr/bin/env python3
"""
mt5_trader.py — exécute les stratégies de strategies.py sur MetaTrader 5
(Deriv MT5 Standard / CFDs), en réutilisant EXACTEMENT le même code de signal
que le backtester (backtest_xau.py) et le bot Multipliers (deriv_ws_scalper.py).

DIFFÉRENCES vs le bot Multipliers
    • Sizing en LOTS (risque % → lot via tick_value), pas stake × multiplicateur.
    • SL/TP en NIVEAUX DE PRIX, posés côté broker (protègent même bot déconnecté).
    • Le trailing peut VERROUILLER DU PROFIT (SL au-delà de l'entrée) — impossible
      sur les Multipliers Deriv. Le comportement colle donc enfin au backtest.
    • Coûts réels: commission MT5 (ex. XAUUSD 2.4$/100k = 0.0024%) + SPREAD + SWAP
      (non modélisés par le backtest → à revalider en démo).

CONTRAT (identique à strategies.py)
    On fournit à la stratégie data = {granularité_s: DataFrame} de bougies
    CLÔTURÉES (open/high/low/close, index UTC croissant). iloc[-1] = dernière
    bougie close. Le Signal renvoie des niveaux de PRIX absolus (SL/TP) et,
    éventuellement, une spec de trailing (trail_kind/step/lock).

⚠ WINDOWS UNIQUEMENT — le package MetaTrader5 ne tourne pas sur Mac/Linux.
   Terminal MT5 installé, connecté au compte, « Algo Trading » activé, laissé
   OUVERT. Voir GUIDE_VPS_MT5.md.

USAGE
    pip install MetaTrader5 pandas numpy
    python mt5_trader.py --symbol XAUUSD --strategy alligator-v2 --tf M1 \
        --login 12345678 --password "…" --server Deriv-Demo
    # (ou sans --login si le terminal est déjà connecté au bon compte)
"""

import sys
import math
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("Installe le package :  pip install MetaTrader5   (Windows uniquement)")

import strategies as st

# ── Money management (mêmes valeurs que les autres bots) ──────────────────────
RISK_PCT           = 0.01     # 1% du solde risqué par trade
MAX_DAILY_LOSS_PCT = 0.05     # coupe-circuit à −5% sur la journée
MAX_OPEN_TRADES    = 2        # positions simultanées max
MAGIC              = 20260708 # identifie NOS positions (n'y touche pas à la main)
DEVIATION          = 20       # slippage max autorisé (points) sur l'ordre marché
STALE_FACTOR       = 5        # bougie plus vieille que N×granularité ⇒ marché fermé
RECONNECT_S        = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mt5_trader")

# Granularité (secondes, comme dans strategies.py) → constante timeframe MT5
_MT5_TF = {
    60: mt5.TIMEFRAME_M1, 120: mt5.TIMEFRAME_M2, 180: mt5.TIMEFRAME_M3,
    300: mt5.TIMEFRAME_M5, 600: mt5.TIMEFRAME_M10, 900: mt5.TIMEFRAME_M15,
    1800: mt5.TIMEFRAME_M30, 3600: mt5.TIMEFRAME_H1, 7200: mt5.TIMEFRAME_H2,
    14400: mt5.TIMEFRAME_H4, 28800: mt5.TIMEFRAME_H8, 86400: mt5.TIMEFRAME_D1,
}


# ──────────────────────────────────────────────────────────────────────────────
#  CONNEXION / SYMBOLE
# ──────────────────────────────────────────────────────────────────────────────

def connect(login: Optional[int], password: Optional[str], server: Optional[str],
            terminal_path: Optional[str]):
    kwargs = {"path": terminal_path} if terminal_path else {}
    if not mt5.initialize(**kwargs):
        sys.exit(f"mt5.initialize() a échoué : {mt5.last_error()}  "
                 "(terminal MT5 installé et lancé ?)")
    if login:
        if not mt5.login(int(login), password=password, server=server):
            err = mt5.last_error()
            mt5.shutdown()
            sys.exit(f"mt5.login() a échoué : {err}  (login/mot de passe/serveur corrects ?)")
    acc = mt5.account_info()
    if acc is None:
        mt5.shutdown()
        sys.exit("account_info() est None — le terminal est-il connecté à un compte ?")
    return acc


def symbol_meta(symbol: str):
    si = mt5.symbol_info(symbol)
    if si is None:
        return None
    if not si.visible:
        mt5.symbol_select(symbol, True)
        si = mt5.symbol_info(symbol)
    return si


# ──────────────────────────────────────────────────────────────────────────────
#  DONNÉES (bougies CLÔTURÉES uniquement — contrat strategies.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_candles(symbol: str, gran: int, count: int, offset_h: float) -> Optional[pd.DataFrame]:
    tf = _MT5_TF.get(gran)
    if tf is None:
        log.error("Granularité %ss non supportée par MT5.", gran)
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count + 1)
    if rates is None or len(rates) < 2:
        return None
    df = pd.DataFrame(rates).iloc[:-1]          # on jette la bougie en cours de formation
    if df.empty:
        return None
    # 'time' MT5 = heure SERVEUR ; on la ramène en UTC via l'offset (Deriv=0)
    df["time"] = pd.to_datetime(df["time"] - offset_h * 3600, unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close"]].astype(float)


def get_strategy_data(symbol: str, strat, offset_h: float) -> Optional[Dict[int, pd.DataFrame]]:
    data: Dict[int, pd.DataFrame] = {}
    for gran, count in strat.granularities.items():
        df = get_candles(symbol, gran, max(count, 60), offset_h)
        if df is None or len(df) < 30:
            return None
        data[gran] = df
    return data


# ──────────────────────────────────────────────────────────────────────────────
#  SIZING (en lots, piloté par le risque)
# ──────────────────────────────────────────────────────────────────────────────

def price_per_dollar_inv(si, volume: float) -> float:
    """$ de P&L par 1.0 de mouvement de prix, pour `volume` lots."""
    if si.trade_tick_size <= 0:
        return 0.0
    return si.trade_tick_value / si.trade_tick_size * volume


def compute_lot(si, entry: float, sl_price: float, risk_usd: float) -> float:
    """Lot tel que toucher le SL coûte ≈ risk_usd. Perte/lot = SL_dist / tick_size
    × tick_value. Arrondi au pas de volume, borné [min, max]."""
    sl_dist = abs(entry - sl_price)
    if sl_dist <= 0 or si.trade_tick_size <= 0 or si.trade_tick_value <= 0:
        return 0.0
    loss_per_lot = sl_dist / si.trade_tick_size * si.trade_tick_value
    if loss_per_lot <= 0:
        return 0.0
    step = si.volume_step or 0.01
    lot = math.floor((risk_usd / loss_per_lot) / step) * step
    lot = max(si.volume_min, min(lot, si.volume_max))
    return round(lot, 8)


def margin_ok(symbol: str, order_type: int, lot: float, price: float) -> bool:
    req = mt5.order_calc_margin(order_type, symbol, lot, price)
    if req is None:
        return True                                   # non calculable → on laisse le broker trancher
    acc = mt5.account_info()
    return acc is not None and req <= acc.margin_free


# ──────────────────────────────────────────────────────────────────────────────
#  EXÉCUTION
# ──────────────────────────────────────────────────────────────────────────────

def _filling(si) -> int:
    fm = si.filling_mode                              # bitmask: FOK=1, IOC=2
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def open_trade(si, direction: str, lot: float, sl_price: float,
               tp_price: Optional[float], reason: str):
    tick = mt5.symbol_info_tick(si.name)
    if tick is None:
        return None
    if direction == "long":
        otype, price = mt5.ORDER_TYPE_BUY, tick.ask
    else:
        otype, price = mt5.ORDER_TYPE_SELL, tick.bid

    d = si.digits
    sl = round(sl_price, d)
    tp = round(tp_price, d) if tp_price is not None else 0.0

    min_dist = si.trade_stops_level * si.point        # distance mini SL/TP imposée par le broker
    if min_dist > 0 and abs(price - sl) < min_dist:
        log.warning("SL trop proche du prix (min %.*f) — trade ignoré.", d, min_dist)
        return None
    if not margin_ok(si.name, otype, lot, price):
        log.warning("Marge libre insuffisante pour %.2f lot — trade ignoré.", lot)
        return None

    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": si.name, "volume": lot,
        "type": otype, "price": price, "sl": sl, "tp": tp,
        "deviation": DEVIATION, "magic": MAGIC, "comment": reason[:31],
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": _filling(si),
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning("order_send échec : retcode=%s comment=%s",
                    getattr(res, "retcode", None), getattr(res, "comment", None))
        return None
    log.info("✅ OPEN %-5s  lot=%.2f  @%.*f  SL=%.*f  TP=%s  ticket=%s",
             direction.upper(), lot, d, res.price, d, sl,
             (f"{tp:.{d}f}" if tp else "trailing"), res.order)
    return res


def _modify_sl(si, ticket: int, sl: float, tp: float) -> bool:
    req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": si.name,
           "position": ticket, "sl": round(sl, si.digits), "tp": tp or 0.0}
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        log.debug("modif SL échec ticket=%s retcode=%s", ticket, getattr(res, "retcode", None))
        return False
    log.info("↗  Trailing : SL déplacé à %.*f (ticket=%s)", si.digits, sl, ticket)
    return True


def _close_position(si, p) -> bool:
    tick = mt5.symbol_info_tick(si.name)
    if tick is None:
        return False
    if p.type == mt5.ORDER_TYPE_BUY:
        otype, price = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        otype, price = mt5.ORDER_TYPE_BUY, tick.ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": si.name, "volume": p.volume,
           "type": otype, "position": p.ticket, "price": price, "deviation": DEVIATION,
           "magic": MAGIC, "type_time": mt5.ORDER_TIME_GTC, "type_filling": _filling(si)}
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log.info("🔒 Clôture ticket=%s  %s", p.ticket, "OK" if ok else f"échec {getattr(res, 'retcode', None)}")
    return ok


def our_positions(symbol: str):
    return [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == MAGIC]


def close_all(si):
    for p in our_positions(si.name):
        _close_position(si, p)


# ──────────────────────────────────────────────────────────────────────────────
#  TRAILING (identique au moteur de backtest, mais SL en prix côté broker)
# ──────────────────────────────────────────────────────────────────────────────

def manage_positions(si, pos_state: dict) -> None:
    """Pour chaque position portant une spec de trailing : verrouille `lock` par
    palier de `step` de gain latent MAX (cliquet, ne se relâche jamais). Sur MT5
    on PEUT poser le SL au-delà de l'entrée → vrai verrouillage de profit."""
    positions = our_positions(si.name)
    live = {p.ticket for p in positions}
    for t in list(pos_state):                    # oublie les positions déjà fermées
        if t not in live:
            pos_state.pop(t, None)

    for p in positions:
        stt = pos_state.get(p.ticket)
        if stt is None or stt.get("trail") is None:
            continue
        kind, step, lock = stt["trail"]
        entry = stt["entry"]
        is_long = p.type == mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(si.name)
        if tick is None:
            continue
        cur = tick.bid if is_long else tick.ask   # prix de sortie du bon côté

        if kind == "pnl":
            stt["peak_profit"] = max(stt.get("peak_profit", 0.0), p.profit)
            ppu = price_per_dollar_inv(si, p.volume)
            locked_usd = lock * math.floor(stt["peak_profit"] / step) if stt["peak_profit"] > 0 else 0.0
            locked_dist = (locked_usd / ppu) if (locked_usd > 0 and ppu > 0) else 0.0
        else:                                     # "dist" : step/lock en distance de prix (modes r/atr)
            stt["peak"] = max(stt.get("peak", entry), cur) if is_long else min(stt.get("peak", entry), cur)
            exc = (stt["peak"] - entry) if is_long else (entry - stt["peak"])
            locked_dist = lock * math.floor(exc / step) if exc > 0 else 0.0

        if locked_dist <= 0:
            continue
        new_sl = round(entry + locked_dist if is_long else entry - locked_dist, si.digits)
        better = (new_sl > p.sl) if is_long else (new_sl < p.sl or p.sl == 0.0)
        if better:
            _modify_sl(si, p.ticket, new_sl, p.tp)


# ──────────────────────────────────────────────────────────────────────────────
#  COUPE-CIRCUIT JOURNALIER (repris de deriv_ws_scalper.py)
# ──────────────────────────────────────────────────────────────────────────────

class DailyLossGuard:
    def __init__(self, opening_balance: float):
        self._date = datetime.now(timezone.utc).date()
        self._open = opening_balance
        self._limit = opening_balance * MAX_DAILY_LOSS_PCT
        self._halted = False

    def reset_if_new_day(self, current_balance: float):
        today = datetime.now(timezone.utc).date()
        if today != self._date:
            self._date, self._open = today, current_balance
            self._limit, self._halted = current_balance * MAX_DAILY_LOSS_PCT, False
            log.info("🌅 Nouvelle journée  |  solde d'ouverture = %.2f", current_balance)

    def is_halted(self, current_equity: float) -> bool:
        if self._halted:
            return True
        if self._open - current_equity >= self._limit:
            log.critical("🛑 LIMITE DE PERTE JOURNALIÈRE atteinte (−%.2f).  Bot en pause.",
                         self._open - current_equity)
            self._halted = True
        return self._halted

    def status_str(self, equity: float) -> str:
        return f"day_P&L={equity - self._open:+.2f}  limit=−{self._limit:.2f}  equity={equity:.2f}"


# ──────────────────────────────────────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    acc = connect(args.login, args.password, args.server, args.terminal_path)
    demo = getattr(acc, "trade_mode", 0) == mt5.ACCOUNT_TRADE_MODE_DEMO
    log.info("Connecté  |  compte=%s (%s)  serveur=%s  solde=%.2f %s  levier=1:%s",
             acc.login, "DÉMO" if demo else "RÉEL ⚠", acc.server, acc.balance,
             acc.currency, acc.leverage)

    strat = st.make_strategy(args.strategy, rr=args.rr, tf=args.tf, sessions=args.sessions)
    si = symbol_meta(args.symbol)
    if si is None:
        mt5.shutdown()
        sys.exit(f"Symbole {args.symbol!r} introuvable sur ce compte "
                 "(vérifie l'orthographe / le Market Watch du terminal).")
    log.info("Stratégie=%s  timeframes=%s  poll=%ss  |  %s: contrat=%s  lot∈[%s..%s] pas=%s  digits=%s",
             strat.name, sorted(strat.granularities), strat.poll_seconds, si.name,
             si.trade_contract_size, si.volume_min, si.volume_max, si.volume_step, si.digits)
    log.info("Risk=%.1f%%/trade  |  Max daily loss=%.1f%%  |  Max positions=%d  |  Ctrl+C pour stopper",
             RISK_PCT * 100, MAX_DAILY_LOSS_PCT * 100, MAX_OPEN_TRADES)

    guard = DailyLossGuard(acc.balance)
    pos_state: dict = {}
    primary_gran = min(strat.granularities)

    while True:
        try:
            acc = mt5.account_info()
            if acc is None:
                log.warning("account_info None — tentative de reconnexion…")
                mt5.shutdown()
                time.sleep(RECONNECT_S)
                connect(args.login, args.password, args.server, args.terminal_path)
                continue
            equity, balance = acc.equity, acc.balance
            guard.reset_if_new_day(balance)

            manage_positions(si, pos_state)

            if guard.is_halted(equity):
                close_all(si)
                time.sleep(600)
                continue

            data = get_strategy_data(si.name, strat, args.server_utc_offset)
            if data is None:
                time.sleep(strat.poll_seconds)
                continue
            last_ts = data[primary_gran].index[-1]

            age = (datetime.now(timezone.utc) - last_ts.to_pydatetime()).total_seconds()
            if age > STALE_FACTOR * primary_gran:
                log.debug("Marché fermé / flux périmé (dernière bougie %.0fs).", age)
                time.sleep(max(strat.poll_seconds, 120))
                continue

            positions = our_positions(si.name)
            if len(positions) >= MAX_OPEN_TRADES:
                time.sleep(strat.poll_seconds)
                continue
            if not (strat.active(last_ts) and strat.session_ok(last_ts)):
                time.sleep(strat.poll_seconds)
                continue

            sig = strat.signal(data, last_ts)
            if sig is None:
                log.debug("Pas de signal  |  %s", guard.status_str(equity))
                time.sleep(strat.poll_seconds)
                continue

            active_dirs = {("long" if p.type == mt5.ORDER_TYPE_BUY else "short") for p in positions}
            if sig.direction in active_dirs:
                time.sleep(strat.poll_seconds)
                continue

            tick = mt5.symbol_info_tick(si.name)
            entry = tick.ask if sig.direction == "long" else tick.bid
            sl_dist = abs(entry - sig.sl_price)
            trailing = sig.trail_step is not None
            tp_dist = abs(sig.tp_price - entry) if sig.tp_price is not None else 0.0
            if sl_dist <= 0 or (not trailing and tp_dist <= 0):
                log.debug("Distances SL/TP dégénérées — skip.")
                time.sleep(strat.poll_seconds)
                continue

            lot = compute_lot(si, entry, sig.sl_price, balance * RISK_PCT)
            if lot <= 0:
                log.warning("Lot calculé = 0 (SL %.5f trop large / méta manquante) — skip.", sl_dist)
                time.sleep(strat.poll_seconds)
                continue

            log.info("📶 SIGNAL %-5s [%s]  lot=%.2f  entry~%.*f  SL=%.*f  risque=$%.2f  |  %s",
                     sig.direction.upper(), sig.reason, lot, si.digits, entry, si.digits,
                     sig.sl_price, balance * RISK_PCT, guard.status_str(equity))

            res = open_trade(si, sig.direction, lot, sig.sl_price, sig.tp_price, sig.reason)
            if res is not None:
                pos_state[res.order] = {
                    "dir": sig.direction, "entry": res.price,
                    "trail": (sig.trail_kind, sig.trail_step, sig.trail_lock) if trailing else None,
                    "peak": res.price, "peak_profit": 0.0,
                }
            time.sleep(strat.poll_seconds)

        except KeyboardInterrupt:
            log.info("\n⛔  Arrêt demandé (Ctrl+C).")
            break
        except Exception as exc:                     # robustesse: on ne meurt pas sur une erreur transitoire
            log.exception("Erreur dans la boucle : %s", exc)
            time.sleep(RECONNECT_S)

    mt5.shutdown()


def main():
    global RISK_PCT, MAX_DAILY_LOSS_PCT
    p = argparse.ArgumentParser(description="Bot MT5 multi-stratégies (réutilise strategies.py)")
    p.add_argument("--symbol", default="XAUUSD", help="symbole MT5 (ex: XAUUSD, BTCUSD)")
    p.add_argument("--strategy", default="alligator-v2", choices=list(st.REGISTRY))
    p.add_argument("--tf", default=None, help="timeframe d'entrée: M1, M5, H1... (défaut: celui de la stratégie)")
    p.add_argument("--sessions", default=None, help="london, ny, off ou 'london,ny' (défaut: comportement de la stratégie)")
    p.add_argument("--rr", type=float, default=None, help="override du R:R (ignoré si la stratégie fait du trailing)")
    p.add_argument("--risk", type=float, default=RISK_PCT, help="fraction du solde risquée par trade (défaut 0.01)")
    p.add_argument("--max-dd", type=float, default=MAX_DAILY_LOSS_PCT, help="perte journalière max (défaut 0.05)")
    p.add_argument("--login", type=int, default=None, help="n° de compte MT5 (sinon: compte déjà connecté dans le terminal)")
    p.add_argument("--password", default=None, help="mot de passe MT5 (avec --login)")
    p.add_argument("--server", default=None, help="serveur MT5, ex: Deriv-Demo (avec --login)")
    p.add_argument("--terminal-path", default=None, help="chemin de terminal64.exe si plusieurs terminaux installés")
    p.add_argument("--server-utc-offset", type=float, default=0.0,
                   help="décalage (heures) de l'heure SERVEUR MT5 vs UTC. Deriv=0. "
                        "Si tes sessions london/ny semblent décalées, ajuste ici.")
    args = p.parse_args()

    RISK_PCT = max(0.001, min(0.05, args.risk))
    MAX_DAILY_LOSS_PCT = max(0.01, min(0.20, args.max_dd))
    run(args)


if __name__ == "__main__":
    main()
