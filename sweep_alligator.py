#!/usr/bin/env python3
"""
Sweep des paramètres Williams Alligator (recherche, pas production).

Balaye tf × sessions × rr × SPREAD_BARS sur une période d'ENTRAÎNEMENT
uniquement — la validation finale se fait sur une période disjointe.

USAGE
    python3 sweep_alligator.py --from 2026-03-25 --to 2026-06-15 --jobs 6
"""

import argparse
import multiprocessing as mp

import pandas as pd

import strategies as st
import backtest_xau as bt

TFS      = ["M1", "M5", "M15"]    # restreignable via --tfs
SESSIONS = [None, "off"]          # None = toutes les heures
RRS      = [1.5, 2.0, 3.0]
SPREADS  = [3, 5, 8]              # SPREAD_BARS: vitesse d'ouverture exigée

_G = {}


def _init(symbol, d_from, d_to):
    m1 = bt.load_m1(symbol, d_from, d_to)
    _G["m1"] = m1
    _G["frames"] = {}
    for tf_name in TFS:
        tf = st.parse_tf(tf_name)
        probe = st.make_strategy("alligator", tf=tf)
        _G["frames"][tf_name] = bt.prepare_frames(m1, probe.granularities)


def _combo(params):
    tf_name, sessions, rr, spread, balance, risk, slippage, commission = params
    res = bt.run_backtest(
        "alligator", _G["m1"], balance, risk, rr, slippage, commission,
        quiet=True, overrides={"SPREAD_BARS": spread},
        prepared=_G["frames"][tf_name], tf=st.parse_tf(tf_name), sessions=sessions,
    )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="frxXAUUSD")
    p.add_argument("--from", dest="d_from", required=True)
    p.add_argument("--to", dest="d_to", required=True)
    p.add_argument("--balance", type=float, default=10000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--commission-pct", type=float, default=0.018)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--tfs", default=None, help="ex: 'M15' pour ne balayer qu'un tf")
    p.add_argument("--out", default="sweep_alligator.csv")
    args = p.parse_args()

    global TFS
    if args.tfs:
        TFS = [t.strip() for t in args.tfs.split(",")]

    # s'assure que le cache couvre la période AVANT de spawner les workers
    bt.load_m1(args.symbol, args.d_from, args.d_to)

    combos = [(tf, s, rr, sp) for tf in TFS for s in SESSIONS for rr in RRS for sp in SPREADS]
    tasks = [(tf, s, rr, sp, args.balance, args.risk, args.slippage, args.commission_pct)
             for tf, s, rr, sp in combos]
    print(f"Sweep alligator: {len(combos)} combos ({args.jobs} process)", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init,
                  initargs=(args.symbol, args.d_from, args.d_to)) as pool:
        rows = []
        for (tf, s, rr, sp), res in zip(combos, pool.imap(_combo, tasks)):
            res.update({"tf": tf, "sessions": s or "all", "rr": rr, "spread_bars": sp})
            rows.append(res)
            print(f"  [{len(rows):>2}/{len(combos)}] tf={tf:<4} sess={s or 'all':<4} "
                  f"rr={rr:<4} spread={sp}  → {res['trades']:>4} trades  "
                  f"net ${res['net_$']:>8.2f}  PF {res['profit_factor']}", flush=True)

    df = pd.DataFrame(rows)[
        ["tf", "sessions", "rr", "spread_bars", "trades", "winrate_%",
         "profit_factor", "net_$", "return_%", "max_dd_%", "commission_$", "TP", "SL", "BE"]
    ].sort_values("net_$", ascending=False)
    df.to_csv(args.out, index=False)
    print("\n###  SWEEP ALLIGATOR — top 15 (période d'entraînement)")
    print(df.head(15).to_string(index=False))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
