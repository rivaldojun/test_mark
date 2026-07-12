#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Backtester XAU/USD (ou tout symbole Deriv) — moteur M1 multi-timeframe    ║
╚══════════════════════════════════════════════════════════════════════════════╝

PRINCIPE DE FIABILITÉ
    • Les stratégies exécutées ici sont LE MÊME CODE que le bot live
      (module strategies.py importé tel quel) — zéro divergence backtest/live.
    • Données: bougies M1 réelles téléchargées depuis le feed public Deriv
      (le même feed que le bot trade), mises en cache localement en CSV.
    • Anti-lookahead: signaux calculés uniquement sur bougies CLÔTURÉES;
      entrée à l'OPEN de la bougie suivante; les timeframes supérieurs (M5/H1)
      sont resamplés et un bar n'est visible qu'après sa clôture.
    • Exécution conservatrice: si SL et TP sont touchés dans la même bougie
      M1, le SL est compté en premier (pire cas).
    • Coûts: commission Deriv multiplier (% du notionnel, mesurée ~0.018%
      sur frxXAUUSD) + slippage paramétrable sur l'entrée.
    • Sizing identique au bot live: risque % du solde, multiplicateur choisi
      dynamiquement (SL$ ≤ 95% du stake), stake plafonné (500$ / balance÷2).

USAGE
    # 1) Télécharger l'historique M1 (une fois; reprend/complète le cache):
    python3 backtest_xau.py download --days 120

    # 2) Backtester une stratégie:
    python3 backtest_xau.py run --strategy orb --from 2026-04-01 --to 2026-07-01

    # 3) Comparer les stratégies sur la même période:
    python3 backtest_xau.py compare --from 2026-04-01 --to 2026-07-01

    Options communes: --balance 10000 --risk 0.01 --rr 2.0 --slippage 0.05
                      --commission-pct 0.018 --symbol frxXAUUSD
    Sorties: métriques console + trades_<strategy>.csv + equity_<strategy>.csv
"""

import os
import sys
import json
import time
import asyncio
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import strategies as st

try:
    import websockets
except ImportError:
    sys.exit("pip3 install websockets pandas numpy")

PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
DATA_DIR = Path(__file__).parent / "data"

# Sizing identique au bot live (voir deriv_ws_scalper.py)
MULTIPLIERS = [50, 100, 150, 250, 500]   # frxXAUUSD (vue publique)
BROKER_MAX_STAKE = 500.0
MAX_OPEN_TRADES = 2                      # le live réserve balance/2 par slot
BE_TRIGGER_R = 0.67                      # passage break-even à 0.67×R de gain
                                         # (≈ l'ancien 1×ATR pour un SL à 1.5×ATR),
                                         # défini en R pour rester cohérent quelle
                                         # que soit la stratégie — comme le live


# ──────────────────────────────────────────────────────────────────────────────
#  TÉLÉCHARGEMENT / CACHE
# ──────────────────────────────────────────────────────────────────────────────

def cache_path(symbol: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{symbol}_M1.csv"


async def _download(symbol: str, start_epoch: int, end_epoch: int) -> pd.DataFrame:
    """Pagine le feed public par fenêtres de 1000 bougies max."""
    rows: List[dict] = []
    rid = 0
    async with websockets.connect(PUBLIC_WS, open_timeout=20) as ws:
        async def req(payload):
            nonlocal rid
            rid += 1
            payload["req_id"] = rid
            await ws.send(json.dumps(payload))
            while True:
                resp = json.loads(await ws.recv())
                if resp.get("req_id") == rid:
                    return resp

        cursor = start_epoch
        window = 1000 * 60   # 1000 bougies M1
        n_req = 0
        while cursor < end_epoch:
            w_end = min(cursor + window, end_epoch)
            r = await req({
                "ticks_history": symbol, "style": "candles", "granularity": 60,
                "start": cursor, "end": str(w_end), "count": 1000,
            })
            n_req += 1
            if "error" in r:
                print(f"  ! {r['error']['message']} @ {cursor}")
            else:
                rows.extend(r.get("candles", []))
            cursor = w_end
            if n_req % 20 == 0:
                got = pd.to_datetime(rows[-1]["epoch"], unit="s") if rows else "-"
                print(f"  … {n_req} requêtes, {len(rows)} bougies (jusqu'à {got})")
            await asyncio.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch")
    return df[["epoch", "open", "high", "low", "close"]]


def download(symbol: str, days: int) -> None:
    path = cache_path(symbol)
    now = int(time.time())
    start = now - days * 86400

    existing = None
    if path.exists():
        existing = pd.read_csv(path)
        if len(existing):
            last = int(existing["epoch"].max())
            print(f"Cache existant: {len(existing)} bougies (jusqu'à "
                  f"{pd.to_datetime(last, unit='s')}). Reprise depuis là.")
            start = max(start, last + 60)

    if start >= now - 60:
        print("Cache déjà à jour.")
        return

    print(f"Téléchargement {symbol} M1: "
          f"{pd.to_datetime(start, unit='s')} → {pd.to_datetime(now, unit='s')}")
    df = asyncio.run(_download(symbol, start, now))
    print(f"Reçu {len(df)} bougies.")
    if existing is not None and len(existing):
        df = pd.concat([existing, df]).drop_duplicates("epoch").sort_values("epoch")
    df.to_csv(path, index=False)
    print(f"Cache: {path}  ({len(df)} bougies au total)")


def ensure_data(symbol: str, d_from: Optional[str], d_to: Optional[str]) -> None:
    """Complète automatiquement le cache si la période demandée le déborde
    (vers le passé ou le futur). Marge d'1 jour pour ignorer les trous de
    week-end. NB: Deriv ne sert que ~6 mois d'historique M1 — au-delà, la
    requête revient simplement vide."""
    path = cache_path(symbol)
    now = int(time.time())
    want_start = int(pd.Timestamp(d_from, tz="UTC").timestamp()) if d_from else None
    want_end   = min(now, int(pd.Timestamp(d_to, tz="UTC").timestamp())) if d_to else None

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["epoch"])
    margin = 86400

    fetch_ranges = []
    if len(existing) == 0:
        fetch_ranges.append((want_start or now - 120 * 86400, want_end or now))
    else:
        first, last = int(existing["epoch"].min()), int(existing["epoch"].max())
        if want_start is not None and want_start < first - margin:
            fetch_ranges.append((want_start, first - 60))
        if want_end is not None and want_end > last + margin:
            fetch_ranges.append((last + 60, want_end))

    for start, end in fetch_ranges:
        print(f"Cache incomplet → téléchargement {symbol} M1: "
              f"{pd.to_datetime(start, unit='s')} → {pd.to_datetime(end, unit='s')}")
        df_new = asyncio.run(_download(symbol, start, end))
        print(f"  reçu {len(df_new)} bougies.")
        if len(df_new) == 0 and start < now - 200 * 86400:
            print("  (période probablement au-delà de l'historique servi par Deriv)")
        if len(df_new):
            existing = pd.concat([existing, df_new]).drop_duplicates("epoch").sort_values("epoch")
            existing.to_csv(path, index=False)


def load_m1(symbol: str, d_from: Optional[str], d_to: Optional[str]) -> pd.DataFrame:
    ensure_data(symbol, d_from, d_to)
    path = cache_path(symbol)
    if not path.exists():
        sys.exit(f"Pas de cache {path}. Lance d'abord:  python3 backtest_xau.py download --days 120")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("time").sort_index()[["open", "high", "low", "close"]].astype(float)
    if d_from:
        df = df[df.index >= pd.Timestamp(d_from, tz="UTC")]
    if d_to:
        df = df[df.index < pd.Timestamp(d_to, tz="UTC")]
    if len(df) < 1000:
        sys.exit(f"Seulement {len(df)} bougies dans la période — élargis --from/--to ou re-télécharge.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  RESAMPLING MULTI-TIMEFRAME (sans lookahead)
# ──────────────────────────────────────────────────────────────────────────────

def resample(m1: pd.DataFrame, gran: int) -> pd.DataFrame:
    if gran == 60:
        return m1
    rule = f"{gran}s"
    out = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return out


def precompute(df: pd.DataFrame, gran: int) -> pd.DataFrame:
    """Ajoute une fois les colonnes d'indicateurs (causales) que les stratégies
    utiliseront via ensure_cols() — grosse accélération, zéro lookahead."""
    df = df.copy()
    df["atr"]      = st.atr(df, 14)
    df["ema_fast"] = st.ema(df["close"], 8)
    df["ema_slow"] = st.ema(df["close"], 21)
    df["ema50"]    = st.ema(df["close"], 50)
    df["rsi"]      = st.rsi(df["close"], 14)
    df["jaw"], df["teeth"], df["lips"] = st.alligator_lines(df)
    if gran == 60:
        hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
        anchors = df.index.to_series().apply(st.session_anchor)
        grp = hlc3.groupby(anchors.values)
        df["vwap"] = grp.cumsum() / grp.cumcount().add(1)
    return df


def prepare_frames(m1: pd.DataFrame, granularities) -> tuple:
    """Frames par granularité + epochs de clôture — factorisé pour que le mode
    grid ne repaye pas le resample/precompute à chaque combinaison."""
    frames, closes_at = {}, {}
    for gran in granularities:
        f = precompute(resample(m1, gran), gran)
        frames[gran] = f
        closes_at[gran] = (f.index.view("int64") // 10**9 + gran).astype(np.int64)
    return frames, closes_at


# ──────────────────────────────────────────────────────────────────────────────
#  SIZING (répliqué du bot live)
# ──────────────────────────────────────────────────────────────────────────────

def size_trade(balance: float, entry: float, sl_dist: float, risk_pct: float):
    """→ (stake, multiplier, sl_amount, notional) ou None si intraitable."""
    usable = [m for m in MULTIPLIERS if m * sl_dist / entry <= 0.95]
    if not usable:
        return None
    mult = max(usable)
    risk_usd = balance * risk_pct
    raw_stake = risk_usd * entry / (mult * sl_dist)
    max_stake = min(BROKER_MAX_STAKE, balance / MAX_OPEN_TRADES)
    stake = max(1.0, min(raw_stake, max_stake))
    notional = stake * mult
    sl_amount = notional * sl_dist / entry
    if sl_amount > 0.95 * stake:      # garde-fou miroir du live
        return None
    return stake, mult, sl_amount, notional


# ──────────────────────────────────────────────────────────────────────────────
#  MOTEUR DE SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(strategy_name: str, m1: pd.DataFrame, balance0: float,
                 risk_pct: float, rr: Optional[float], slippage: float,
                 commission_pct: float, quiet: bool = False,
                 overrides: Optional[dict] = None,
                 prepared: Optional[tuple] = None,
                 tf=None, sessions=None) -> dict:
    strat = st.make_strategy(strategy_name, rr=rr, tf=tf, sessions=sessions)
    if overrides:
        for k, v in overrides.items():
            setattr(strat, k, v)

    # frames par granularité, indicateurs précalculés
    if prepared is not None:
        frames, closes_at = prepared
    else:
        frames, closes_at = prepare_frames(m1, strat.granularities)

    m1_epochs = (m1.index.view("int64") // 10**9).astype(np.int64)
    n = len(m1)
    o, h, l, c = (m1[k].values for k in ("open", "high", "low", "close"))

    # Petit warmup fixe: le gate par-timeframe dans la boucle (j < min(need,60))
    # attend de toute façon ~60 bougies clôturées de CHAQUE granularité avant
    # d'autoriser un signal. (L'ancienne formule exigeait ~20 jours et
    # tronquait silencieusement le début de chaque backtest.)
    warmup = min(600, max(0, n - 2))

    balance = balance0
    equity_curve = []
    trades: List[dict] = []
    pos = None   # position ouverte

    for i in range(warmup, n - 1):
        bar_close_epoch = m1_epochs[i] + 60

        # ── gestion de la position ouverte sur la bougie i+1 (après entrée) ──
        if pos is not None:
            hi, lo = h[i], l[i]
            be_trigger = BE_TRIGGER_R * pos["sl_dist"]
            allow_be = strat.use_break_even
            exit_price, why = None, None
            # fills gap-aware: si la bougie OUVRE déjà au-delà du niveau
            # (gap week-end/news), le fill réel est l'open, pas le niveau.
            if pos["dir"] == "long":
                if allow_be and not pos["be"] and hi - pos["entry"] >= be_trigger:
                    pos["sl"] = pos["entry"]          # break-even (scratch)
                    pos["be"] = True
                if lo <= pos["sl"]:
                    exit_price = min(o[i], pos["sl"])
                    why = "BE" if pos["be"] else "SL"
                elif hi >= pos["tp"]:
                    exit_price, why = max(o[i], pos["tp"]), "TP"
            else:
                if allow_be and not pos["be"] and pos["entry"] - lo >= be_trigger:
                    pos["sl"] = pos["entry"]
                    pos["be"] = True
                if hi >= pos["sl"]:
                    exit_price = max(o[i], pos["sl"])
                    why = "BE" if pos["be"] else "SL"
                elif lo <= pos["tp"]:
                    exit_price, why = min(o[i], pos["tp"]), "TP"

            if exit_price is not None:
                sgn = 1 if pos["dir"] == "long" else -1
                gross = pos["notional"] * sgn * (exit_price - pos["entry"]) / pos["entry"]
                net = gross - pos["commission"]
                balance += net
                trades.append({
                    "time_in": pos["t_in"], "time_out": m1.index[i],
                    "dir": pos["dir"], "entry": pos["entry"], "exit": exit_price,
                    "sl0": pos["sl0"], "tp": pos["tp"], "why": why,
                    "stake": pos["stake"], "mult": pos["mult"],
                    "gross": round(gross, 2), "commission": round(pos["commission"], 2),
                    "net": round(net, 2), "balance": round(balance, 2),
                    "reason": pos["reason"], "session": st.in_window(pos["t_in"]) or "off",
                })
                pos = None
            else:
                equity_curve.append((m1.index[i], balance))
                continue    # une seule position à la fois

        # ── fast-gate: hors fenêtre de session ou filtre --sessions ──────────
        ts_i = m1.index[i]
        if not (strat.active(ts_i) and strat.session_ok(ts_i)):
            equity_curve.append((ts_i, balance))
            continue

        # ── signal sur bougies clôturées jusqu'à i inclus ─────────────────────
        data = {}
        ok = True
        for gran, f in frames.items():
            j = int(np.searchsorted(closes_at[gran], bar_close_epoch, side="right"))
            need = strat.granularities[gran]
            if j < min(need, 60):
                ok = False
                break
            data[gran] = f.iloc[max(0, j - need):j]
        if not ok:
            equity_curve.append((m1.index[i], balance))
            continue

        sig = strat.signal(data, m1.index[i])
        if sig is None:
            equity_curve.append((m1.index[i], balance))
            continue

        # ── entrée à l'open de la bougie suivante + slippage ─────────────────
        raw_entry = o[i + 1]
        entry = raw_entry + slippage if sig.direction == "long" else raw_entry - slippage
        sl_dist = abs(entry - sig.sl_price)
        tp_dist = abs(sig.tp_price - entry)
        if sl_dist <= 0 or tp_dist <= 0:
            continue

        sized = size_trade(balance, entry, sl_dist, risk_pct)
        if sized is None:
            continue
        stake, mult, sl_amount, notional = sized
        commission = notional * commission_pct / 100.0

        pos = {
            "dir": sig.direction, "entry": entry,
            "sl": sig.sl_price, "sl0": sig.sl_price, "tp": sig.tp_price,
            "sl_dist": sl_dist,
            "stake": stake, "mult": mult, "notional": notional,
            "commission": commission,
            "be": False, "t_in": m1.index[i + 1], "reason": sig.reason,
        }
        equity_curve.append((m1.index[i], balance))

    # position restante ignorée (non clôturée en fin de période)

    return summarize(strategy_name, trades, equity_curve, balance0, quiet)


# ──────────────────────────────────────────────────────────────────────────────
#  MÉTRIQUES
# ──────────────────────────────────────────────────────────────────────────────

def summarize(name: str, trades: List[dict], curve, balance0: float, quiet=False) -> dict:
    if not trades:
        if not quiet:
            print(f"\n### {name}: AUCUN trade sur la période.")
        return {"strategy": name, "trades": 0, "winrate_%": 0.0, "profit_factor": 0.0,
                "net_$": 0.0, "return_%": 0.0, "avg_net_$": 0.0, "max_dd_%": 0.0,
                "commission_$": 0.0, "TP": 0, "SL": 0, "BE": 0}

    t = pd.DataFrame(trades)
    wins = t[t["net"] > 0]
    losses = t[t["net"] <= 0]
    gp, gl = wins["net"].sum(), -losses["net"].sum()
    eq = pd.Series([b for _, b in curve], index=[ts for ts, _ in curve])
    peak = eq.cummax()
    dd = (eq - peak)
    dd_pct = (dd / peak * 100).min()

    res = {
        "strategy": name,
        "trades": len(t),
        "winrate_%": round(100 * len(wins) / len(t), 1),
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "net_$": round(t["net"].sum(), 2),
        "return_%": round(100 * t["net"].sum() / balance0, 2),
        "avg_net_$": round(t["net"].mean(), 2),
        "max_dd_%": round(dd_pct, 2),
        "commission_$": round(t["commission"].sum(), 2),
        "TP": int((t["why"] == "TP").sum()),
        "SL": int((t["why"] == "SL").sum()),
        "BE": int((t["why"] == "BE").sum()),
    }

    if not quiet:
        print(f"\n{'='*64}\n###  {name}\n{'='*64}")
        for k, v in res.items():
            if k != "strategy":
                print(f"  {k:<16} {v}")
        print("\n  Par session:")
        for s, grp in t.groupby("session"):
            print(f"    {s:<8} {len(grp):>4} trades   net ${grp['net'].sum():>9.2f}   "
                  f"winrate {100*len(grp[grp['net']>0])/len(grp):.0f}%")
        tpath = Path(__file__).parent / f"trades_{name}.csv"
        epath = Path(__file__).parent / f"equity_{name}.csv"
        t.to_csv(tpath, index=False)
        eq.to_csv(epath, header=["balance"])
        print(f"\n  → {tpath.name}, {epath.name}")

    return res


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

_WORKER: dict = {}


def _grid_init(symbol, d_from, d_to, tf):
    """Chaque worker charge les données et précalcule les frames UNE fois."""
    m1 = load_m1(symbol, d_from, d_to)
    _WORKER["m1"] = m1
    probe = st.make_strategy("orb", tf=tf)
    _WORKER["prepared"] = prepare_frames(m1, probe.granularities)


def _grid_combo(params):
    rr, bars, sessions, balance, risk, slippage, commission, tf = params
    res = run_backtest(
        "orb", _WORKER["m1"], balance, risk, rr, slippage, commission,
        quiet=True, overrides={"OR_BARS": bars, "SESSIONS": sessions},
        prepared=_WORKER["prepared"], tf=tf,
    )
    return res


def run_grid(m1: pd.DataFrame, args) -> None:
    """Balayage systématique des paramètres d'ORB: R:R × taille du range ×
    session, parallélisé sur --jobs process (chaque worker précalcule les
    frames une fois, puis enchaîne ses combinaisons)."""
    rr_list   = [float(x) for x in args.grid_rr.split(",")]
    bars_list = [int(x) for x in args.grid_bars.split(",")]
    sess_list = [s.strip() for s in args.grid_sessions.split(",")]
    sess_map = {"london": ("london",), "ny": ("ny",), "both": ("london", "ny")}

    combos = [(r, b, s) for r in rr_list for b in bars_list for s in sess_list]
    jobs = max(1, args.jobs)
    print(f"Grille ORB: {len(rr_list)} R:R × {len(bars_list)} ranges × "
          f"{len(sess_list)} sessions = {len(combos)} backtests  ({jobs} process)\n", flush=True)

    tasks = [(r, b, sess_map[s], args.balance, args.risk, args.slippage, args.commission_pct, args.tf)
             for r, b, s in combos]

    rows = []
    if jobs == 1:
        _grid_init(args.symbol, args.d_from, args.d_to, args.tf)
        results = map(_grid_combo, tasks)
    else:
        ctx = mp.get_context("spawn")   # fork est fragile sur macOS
        pool = ctx.Pool(jobs, initializer=_grid_init,
                        initargs=(args.symbol, args.d_from, args.d_to, args.tf))
        results = pool.imap(_grid_combo, tasks)

    for k, ((rr, bars, sess), res) in enumerate(zip(combos, results), 1):
        res.update({"rr": rr, "or_bars": bars, "session": sess})
        rows.append(res)
        print(f"  [{k:>2}/{len(combos)}] rr={rr:<4} range={bars:>2}m session={sess:<6} "
              f"→ {res['trades']:>3} trades  net ${res['net_$']:>8.2f}  PF {res['profit_factor']}",
              flush=True)

    df = pd.DataFrame(rows)[
        ["rr", "or_bars", "session", "trades", "winrate_%", "profit_factor",
         "net_$", "return_%", "max_dd_%", "commission_$", "TP", "SL", "BE"]
    ].sort_values("net_$", ascending=False)

    out = Path(__file__).parent / "grid_orb.csv"
    df.to_csv(out, index=False)
    print(f"\n{'='*72}\n###  GRILLE ORB — classée par résultat net\n{'='*72}")
    print(df.to_string(index=False))
    print(f"\n→ {out.name}")
    print("\n⚠ Attention à l'overfitting: la meilleure combinaison d'une grille est")
    print("  par construction optimiste. Valide-la sur une AUTRE période (--from/--to)")
    print("  avant d'y croire (walk-forward).")


def main():
    p = argparse.ArgumentParser(description="Backtester Deriv M1 multi-stratégies")
    p.add_argument("mode", choices=["download", "run", "compare", "grid"])
    p.add_argument("--symbol", default="frxXAUUSD")
    p.add_argument("--days", type=int, default=120, help="download: profondeur d'historique")
    p.add_argument("--strategy", default="orb", choices=list(st.REGISTRY))
    p.add_argument("--from", dest="d_from", default=None, help="ex: 2026-04-01")
    p.add_argument("--to", dest="d_to", default=None)
    p.add_argument("--balance", type=float, default=10000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--rr", type=float, default=None, help="override du R:R des stratégies")
    p.add_argument("--tf", default=None,
                   help="timeframe d'entrée: M1, M5, M15, M30, H1... (défaut: celui de la stratégie). "
                        "Les TF de structure/biais s'échelonnent automatiquement.")
    p.add_argument("--sessions", default=None,
                   help="restreint les entrées: london, ny, off, ou combinaison 'london,ny' "
                        "(défaut: comportement propre à la stratégie)")
    p.add_argument("--or-bars", type=int, default=None,
                   help="orb uniquement: taille du range d'ouverture en bougies de --tf (défaut 5)")
    p.add_argument("--set", action="append", default=[], metavar="ATTR=VAL",
                   help="run: override d'attribut de stratégie, répétable "
                        "(ex: --set EXIT_MODE=atr --set RR_X=3 --set H1_FILTER=True)")
    p.add_argument("--multipliers", default=None,
                   help="liste des multiplicateurs du symbole (ex: '50,100,200,300,500' pour R_75; "
                        "défaut: ceux de frxXAUUSD)")
    p.add_argument("--slippage", type=float, default=0.05, help="en unités de prix (USD sur XAU)")
    p.add_argument("--commission-pct", type=float, default=0.0024,
                   help="commission Deriv en %% du notionnel (mesurée ~0.018 sur XAU)")
    # grid mode (ORB)
    p.add_argument("--grid-rr", default="1.5,2,2.5,3,4", help="grid: liste de R:R")
    p.add_argument("--grid-bars", default="5,10,15,30", help="grid: tailles du range (minutes)")
    p.add_argument("--grid-sessions", default="london,ny,both", help="grid: sessions")
    p.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)),
                   help="grid: process en parallèle")
    args = p.parse_args()

    if args.mode == "download":
        download(args.symbol, args.days)
        return

    m1 = load_m1(args.symbol, args.d_from, args.d_to)
    print(f"Données: {args.symbol} M1  {m1.index[0]} → {m1.index[-1]}  ({len(m1)} bougies)")

    if args.multipliers:
        global MULTIPLIERS
        MULTIPLIERS = sorted(float(x) for x in args.multipliers.split(","))
        print(f"Multiplicateurs: {MULTIPLIERS}")

    if args.mode == "run":
        overrides = {}
        if args.or_bars and args.strategy == "orb":
            overrides["OR_BARS"] = args.or_bars
        import ast
        for item in args.set:
            key, _, val = item.partition("=")
            try:
                overrides[key.strip()] = ast.literal_eval(val.strip())
            except (ValueError, SyntaxError):
                overrides[key.strip()] = val.strip()   # chaîne brute (ex: atr)
        run_backtest(args.strategy, m1, args.balance, args.risk, args.rr,
                     args.slippage, args.commission_pct, tf=args.tf,
                     sessions=args.sessions, overrides=overrides or None)
    elif args.mode == "grid":
        run_grid(m1, args)
    else:   # compare
        rows = []
        for name in st.REGISTRY:
            rows.append(run_backtest(name, m1, args.balance, args.risk, args.rr,
                                     args.slippage, args.commission_pct,
                                     tf=args.tf, sessions=args.sessions))
        print(f"\n{'='*64}\n###  COMPARAISON\n{'='*64}")
        print(pd.DataFrame(rows).set_index("strategy").to_string())


if __name__ == "__main__":
    main()
