#!/usr/bin/env python3
"""
Fouille de patterns sur les trades d'une stratégie (recherche, pas production).

Enrichit chaque trade de trades_<strategy>.csv avec ses caractéristiques au
moment de l'entrée (calculées SANS lookahead: uniquement les bougies clôturées
avant l'entrée), puis affiche winrate / net / n par bucket. Les patterns avec
un petit échantillon (n < --min-n) sont marqués comme non fiables.

⚠ Tout pattern trouvé ici est par construction ajusté à la période analysée.
   Ne JAMAIS en déduire une règle sans la valider sur une période disjointe.

USAGE
    python3 analyze_patterns.py --trades trades_alligator.csv --tf M5
"""

import argparse

import numpy as np
import pandas as pd

import strategies as st
import backtest_xau as bt


def build_features(trades: pd.DataFrame, symbol: str, tf: int) -> pd.DataFrame:
    m1 = bt.load_m1(symbol, None, None)
    ftf = bt.precompute(bt.resample(m1, tf), tf)          # colonnes causales
    h1  = bt.precompute(bt.resample(m1, 3600), 3600)

    t = trades.copy()
    t["time_in"] = pd.to_datetime(t["time_in"], utc=True)
    t["win"] = t["net"] > 0

    feats = []
    atr_hist = ftf["atr"].dropna()
    for _, row in t.iterrows():
        ts = row["time_in"]
        # dernière bougie tf CLÔTURÉE avant l'entrée
        i = ftf.index.searchsorted(ts) - 1
        j = h1.index.searchsorted(ts) - 1
        if i < 60 or j < 60:
            feats.append({})
            continue
        bar = ftf.iloc[i]
        atr = bar["atr"]
        # percentile de l'ATR vs les 500 dernières bougies (régime de vol)
        window = ftf["atr"].iloc[max(0, i - 500):i]
        atr_pctl = float((window < atr).mean() * 100) if len(window) else np.nan
        jaw, lips = bar["jaw"], bar["lips"]
        mouth_atr = abs(lips - jaw) / atr if atr > 0 else np.nan
        h1bar = h1.iloc[j]
        h1_trend = "up" if h1bar["close"] > h1bar["ema50"] else "down"
        body = abs(bar["close"] - bar["open"])
        rng = bar["high"] - bar["low"]
        feats.append({
            "hour": ts.hour,
            "dow": ts.dayofweek,           # 0=lundi
            "atr_pctl": atr_pctl,
            "mouth_atr": mouth_atr,
            "h1_trend": h1_trend,
            "with_h1": (row["dir"] == "long") == (h1_trend == "up"),
            "body_ratio": body / rng if rng > 0 else np.nan,
        })

    f = pd.DataFrame(feats, index=t.index)
    t = pd.concat([t, f], axis=1)
    # résultat du trade précédent (streaks)
    t["prev_win"] = t["win"].shift(1)
    return t


def bucket_report(t: pd.DataFrame, col: str, buckets, min_n: int) -> None:
    if buckets is not None:
        grp = t.groupby(pd.cut(t[col], buckets), observed=True)
    else:
        grp = t.groupby(col)
    print(f"\n─── {col} ───")
    for key, g in grp:
        if len(g) == 0:
            continue
        flag = "" if len(g) >= min_n else "  ⚠ n trop petit"
        print(f"  {str(key):<14} n={len(g):>4}  winrate={100*g['win'].mean():5.1f}%  "
              f"net=${g['net'].sum():>8.2f}  avg=${g['net'].mean():>7.2f}{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="trades_alligator.csv")
    p.add_argument("--symbol", default="frxXAUUSD")
    p.add_argument("--tf", default="M5", help="tf de la stratégie qui a produit les trades")
    p.add_argument("--min-n", type=int, default=25)
    args = p.parse_args()

    trades = pd.read_csv(args.trades)
    print(f"{len(trades)} trades depuis {args.trades}")
    t = build_features(trades, args.symbol, st.parse_tf(args.tf))
    t.to_csv("trades_enriched.csv", index=False)

    print(f"\nBaseline: winrate={100*t['win'].mean():.1f}%  net=${t['net'].sum():.2f}  "
          f"avg=${t['net'].mean():.2f}  (n={len(t)})")

    bucket_report(t, "hour", None, args.min_n)
    bucket_report(t, "dow", None, args.min_n)
    bucket_report(t, "dir", None, args.min_n)
    bucket_report(t, "atr_pctl", [0, 25, 50, 75, 100], args.min_n)
    bucket_report(t, "mouth_atr", [0, 0.25, 0.5, 1.0, 10], args.min_n)
    bucket_report(t, "with_h1", None, args.min_n)
    bucket_report(t, "body_ratio", [0, 0.33, 0.66, 1.0], args.min_n)
    bucket_report(t, "prev_win", None, args.min_n)

    print("\n→ trades_enriched.csv (pour analyses manuelles)")


if __name__ == "__main__":
    main()
