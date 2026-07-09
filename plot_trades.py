#!/usr/bin/env python3
"""
Visualisation des trades d'un backtest — un PNG par position.

Pour chaque trade du CSV: chandelier du timeframe choisi, 20 bougies avant
l'entrée et 20 après, avec lignes de l'Alligator (jaw/teeth/lips), niveau
d'entrée, SL initial, TP (si présent), marqueurs d'entrée/sortie et résultat.

USAGE
    python3 plot_trades.py --trades data/trades_alligator-v2.csv --symbol cryBTCUSD --tf M5 --last 30
    python3 plot_trades.py --trades trades_orb.csv --tf M1 --sample 20
    # --from/--to pour filtrer une période; --outdir pour changer la sortie
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Contournement d'un bug matplotlib+macOS: le scan de polices via
# system_profiler crashe (KeyError '_items') pendant l'import. On fait échouer
# cet appel précis en OSError — que matplotlib attrape en retournant [] — puis
# on restaure subprocess. Les polices par défaut suffisent pour nos plots.
import subprocess as _sp
_orig_co = _sp.check_output
def _no_sysprofiler(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)) and cmd and "system_profiler" in str(cmd[0]):
        raise OSError("macOS font scan disabled (workaround)")
    return _orig_co(cmd, *a, **k)
_sp.check_output = _no_sysprofiler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_sp.check_output = _orig_co

import strategies as st
import backtest_xau as bt

BARS_BEFORE = 20
BARS_AFTER = 20


def draw_candles(ax, df: pd.DataFrame, tf_sec: int):
    w = pd.Timedelta(seconds=tf_sec * 0.7)
    for ts, row in df.iterrows():
        color = "#26a69a" if row.close >= row.open else "#ef5350"
        ax.plot([ts, ts], [row.low, row.high], color=color, linewidth=0.8, zorder=2)
        body_lo, body_hi = sorted((row.open, row.close))
        if body_hi == body_lo:
            body_hi = body_lo + (row.high - row.low) * 0.001 + 1e-9
        ax.bar(ts, body_hi - body_lo, bottom=body_lo, width=w,
               color=color, edgecolor=color, zorder=3)


def plot_trade(i, trade, tfdf, tf_sec, outdir):
    t_in = trade["time_in"]
    # position de la bougie d'entrée dans le frame tf
    pos = tfdf.index.searchsorted(t_in) - 1
    lo = max(0, pos - BARS_BEFORE)
    hi = min(len(tfdf), pos + BARS_AFTER + 1)
    win = tfdf.iloc[lo:hi]
    if len(win) < 10:
        return False

    fig, ax = plt.subplots(figsize=(14, 7))
    draw_candles(ax, win, tf_sec)

    # lignes alligator
    for col, color, label in (("jaw", "#1f77b4", "jaw(13,8)"),
                              ("teeth", "#d62728", "teeth(8,5)"),
                              ("lips", "#2ca02c", "lips(5,3)")):
        ax.plot(win.index, win[col], color=color, linewidth=1.2, label=label, zorder=4)

    # niveaux du trade
    ax.axhline(trade["entry"], color="black", linewidth=1, linestyle="--", label=f"entry {trade['entry']:.2f}")
    if not pd.isna(trade.get("sl0")):
        ax.axhline(trade["sl0"], color="#ef5350", linewidth=1, linestyle=":", label=f"SL {trade['sl0']:.2f}")
    if not pd.isna(trade.get("tp")):
        ax.axhline(trade["tp"], color="#26a69a", linewidth=1, linestyle=":", label=f"TP {trade['tp']:.2f}")

    # marqueurs entrée/sortie
    is_long = trade["dir"] == "long"
    ax.scatter([t_in], [trade["entry"]], marker="^" if is_long else "v",
               s=180, color="#1b5e20" if is_long else "#b71c1c", zorder=6,
               edgecolors="white", linewidths=1.2, label=f"entrée {trade['dir']}")
    t_out = trade["time_out"]
    if pd.notna(t_out) and win.index[0] <= t_out <= win.index[-1] + pd.Timedelta(seconds=tf_sec):
        ax.scatter([t_out], [trade["exit"]], marker="x", s=140, color="black",
                   zorder=6, label=f"sortie {trade['exit']:.2f} ({trade.get('why','?')})")

    net = trade.get("net", float("nan"))
    res = "WIN" if net > 0 else "LOSS" if net < 0 else "FLAT"
    ax.set_title(
        f"#{i}  {trade['dir'].upper()}  {t_in}  →  net ${net:+.2f} [{res}]   "
        f"{trade.get('reason','')}  (session {trade.get('session','?')})",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    fname = outdir / f"{i:04d}_{t_in.strftime('%Y%m%d_%H%M')}_{trade['dir']}_{res}.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument("--symbol", default="frxXAUUSD")
    p.add_argument("--tf", default="M5")
    p.add_argument("--from", dest="d_from", default=None)
    p.add_argument("--to", dest="d_to", default=None)
    p.add_argument("--last", type=int, default=0, help="ne tracer que les N derniers trades")
    p.add_argument("--sample", type=int, default=0, help="échantillon aléatoire de N trades (seed fixe)")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    tf_sec = st.parse_tf(args.tf)
    trades = pd.read_csv(args.trades, parse_dates=["time_in", "time_out"])
    if args.d_from:
        trades = trades[trades["time_in"] >= pd.Timestamp(args.d_from, tz="UTC")]
    if args.d_to:
        trades = trades[trades["time_in"] < pd.Timestamp(args.d_to, tz="UTC")]

    m1 = bt.load_m1(args.symbol, None, None)
    # ne garder que les trades couverts par les données
    cov_lo = m1.index[0] + pd.Timedelta(seconds=tf_sec * (BARS_BEFORE + 30))
    cov_hi = m1.index[-1] - pd.Timedelta(seconds=tf_sec * BARS_AFTER)
    n0 = len(trades)
    trades = trades[(trades["time_in"] >= cov_lo) & (trades["time_in"] <= cov_hi)]
    if len(trades) < n0:
        print(f"⚠ {n0 - len(trades)} trades hors couverture du cache M1 "
              f"({m1.index[0]} → {m1.index[-1]}) — ignorés.")

    if args.sample and len(trades) > args.sample:
        trades = trades.sample(args.sample, random_state=42).sort_values("time_in")
    elif args.last and len(trades) > args.last:
        trades = trades.tail(args.last)

    if len(trades) == 0:
        print("Aucun trade à tracer.")
        return

    tfdf = bt.resample(m1, tf_sec)
    tfdf["jaw"], tfdf["teeth"], tfdf["lips"] = st.alligator_lines(tfdf)

    name = Path(args.trades).stem.replace("trades_", "")
    outdir = Path(args.outdir) if args.outdir else Path(__file__).parent / "plots" / name
    outdir.mkdir(parents=True, exist_ok=True)

    done = 0
    for i, (_, tr) in enumerate(trades.iterrows(), 1):
        if plot_trade(i, tr, tfdf, tf_sec, outdir):
            done += 1
    print(f"{done} plots → {outdir}/")


if __name__ == "__main__":
    main()
