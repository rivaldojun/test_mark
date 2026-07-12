#!/usr/bin/env python3
"""
Optimisation de la stratégie `scratch` — TRAIN uniquement.

Balaye les leviers STRUCTURELS (sorties ATR vs $, ratio TP/SL, filtre de
spread en ATR, alignement H1, coupure vendredi) sur la période d'entraînement.
La meilleure config doit ensuite être validée UNE fois sur la période TEST
(jamais utilisée ici) — c'est backtest_xau.py run qui s'en charge.

USAGE
    python3 sweep_scratch.py --from 2026-01-15 --to 2026-05-31 --jobs 4
"""

import argparse
import multiprocessing as mp

import pandas as pd

import strategies as st
import backtest_xau as bt

# grille: le premier combo = config d'origine de l'utilisateur (baseline)
COMBOS = [
    # (exit_mode, sl_atrx, rr_x, spread_mode, h1_filter, friday_cutoff)
    ("usd", None, None, "usd", False, None),          # baseline utilisateur
]
for sl_atrx in (1.0, 1.5):
    for rr_x in (1.0, 1.5, 2.0, 3.0):
        for spread_mode in ("usd", "atr"):
            for h1 in (False, True):
                COMBOS.append(("atr", sl_atrx, rr_x, spread_mode, h1, None))

_G = {}


def _init(symbol, d_from, d_to):
    m1 = bt.load_m1(symbol, d_from, d_to)
    _G["m1"] = m1
    probe = st.make_strategy("scratch")
    _G["prepared"] = bt.prepare_frames(m1, probe.granularities)


def _combo(params):
    (exit_mode, sl_atrx, rr_x, spread_mode, h1, friday), balance, risk, slip, comm = params
    overrides = {"EXIT_MODE": exit_mode, "SPREAD_MODE": spread_mode,
                 "H1_FILTER": h1, "FRIDAY_CUTOFF": friday}
    if sl_atrx:
        overrides["SL_ATRX"] = sl_atrx
    if rr_x:
        overrides["RR_X"] = rr_x
    return bt.run_backtest("scratch", _G["m1"], balance, risk, None, slip, comm,
                           quiet=True, overrides=overrides, prepared=_G["prepared"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="frxXAUUSD")
    p.add_argument("--from", dest="d_from", required=True)
    p.add_argument("--to", dest="d_to", required=True)
    p.add_argument("--balance", type=float, default=10000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--commission-pct", type=float, default=0.018)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    bt.load_m1(args.symbol, args.d_from, args.d_to)   # complète le cache AVANT les workers

    tasks = [(c, args.balance, args.risk, args.slippage, args.commission_pct) for c in COMBOS]
    print(f"Sweep scratch: {len(COMBOS)} combos ({args.jobs} process)", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init,
                  initargs=(args.symbol, args.d_from, args.d_to)) as pool:
        rows = []
        for combo, res in zip(COMBOS, pool.imap(_combo, tasks)):
            exit_mode, sl_atrx, rr_x, spread_mode, h1, friday = combo
            res.update({"exit": exit_mode, "sl_atrx": sl_atrx, "rr_x": rr_x,
                        "spread": spread_mode, "h1": h1})
            rows.append(res)
            print(f"  [{len(rows):>2}/{len(COMBOS)}] exit={exit_mode:<4} slx={str(sl_atrx):<5} "
                  f"rr={str(rr_x):<4} spread={spread_mode:<4} h1={int(h1)}  "
                  f"→ {res['trades']:>4} trades  net ${res['net_$']:>9.2f}  "
                  f"PF {res['profit_factor']}", flush=True)

    df = pd.DataFrame(rows)[
        ["exit", "sl_atrx", "rr_x", "spread", "h1", "trades", "winrate_%",
         "profit_factor", "net_$", "return_%", "max_dd_%", "commission_$", "TP", "SL"]
    ].sort_values("net_$", ascending=False)
    df.to_csv("sweep_scratch.csv", index=False)
    print("\n###  SWEEP SCRATCH — classement TRAIN")
    print(df.to_string(index=False))
    print("\n→ sweep_scratch.csv")
    print("\n⚠ Valider le gagnant UNE SEULE fois sur la période TEST avant d'y croire.")


if __name__ == "__main__":
    main()
