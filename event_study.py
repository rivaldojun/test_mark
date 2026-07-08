#!/usr/bin/env python3
"""
Harnais de criblage statistique de signaux (event study) — recherche.

Pour chaque condition candidate (calculée SANS lookahead sur bougies M5
clôturées), mesure sur toutes ses occurrences la course TP-vs-SL symétrique
(±k×ATR, premier touché, résolution M1, SL prioritaire si même bougie M1) :

    n événements, événements/jour, winrate, expectancy brute (R),
    expectancy NETTE de commission (R), par direction long/short.

L'expectancy nette en R utilise: commission_R = c% × prix / distance_stop.
C'est le filtre n°1: un signal doit battre ~0.05-0.10R de frais par trade.

⚠ Multiple testing: avec ~30 conditions × 2 directions, plusieurs "edges"
   apparaîtront par pur hasard. Ce harnais CRIBLE sur TRAIN; tout candidat
   retenu doit être confirmé sur TEST (période disjointe) puis en démo.

USAGE
    python3 event_study.py --from 2026-03-25 --to 2026-06-15            # TRAIN
    python3 event_study.py --from 2026-06-15 --to 2026-07-06 --only rsi2_low
"""

import argparse

import numpy as np
import pandas as pd

import strategies as st
import backtest_xau as bt

M5 = 300


# ──────────────────────────────────────────────────────────────────────────────
#  FEATURES M5 (toutes causales: valeur en i n'utilise que les bougies ≤ i)
# ──────────────────────────────────────────────────────────────────────────────

def build_m5(m1: pd.DataFrame) -> pd.DataFrame:
    df = bt.resample(m1, M5)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    df["atr"] = st.atr(df, 14)
    df["rsi2"] = st.rsi(c, 2)
    df["rsi14"] = st.rsi(c, 14)
    df["ema8"] = st.ema(c, 8)
    df["ema21"] = st.ema(c, 21)
    df["ema50"] = st.ema(c, 50)

    ret1 = c.pct_change()
    df["ret_z"] = (ret1 - ret1.rolling(96).mean()) / ret1.rolling(96).std()

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_z"] = (c - ma20) / sd20

    rng = h - l
    df["range_pctl"] = rng.rolling(96).rank(pct=True)
    df["nr7"] = rng == rng.rolling(7).min()
    df["inside"] = (h <= h.shift()) & (l >= l.shift())

    body = (c - o).abs()
    df["body_ratio"] = body / rng.replace(0, np.nan)
    df["up_wick"] = (h - np.maximum(c, o)) / rng.replace(0, np.nan)
    df["dn_wick"] = (np.minimum(c, o) - l) / rng.replace(0, np.nan)

    up = (c > o).astype(int)
    grp = (up != up.shift()).cumsum()
    df["streak"] = up.groupby(grp).cumcount() + 1
    df["streak"] *= np.where(up == 1, 1, -1)

    df["dist_ema50_atr"] = (c - df["ema50"]) / df["atr"]
    df["big_bar"] = rng / df["atr"]

    # VWAP de session (proxy TWAP hlc3 — pas de volume sur le feed)
    hlc3 = (h + l + c) / 3.0
    anchors = df.index.to_series().apply(st.session_anchor)
    g = hlc3.groupby(anchors.values)
    df["vwap"] = g.cumsum() / g.cumcount().add(1)
    df["dist_vwap_atr"] = (c - df["vwap"]) / df["atr"]

    # niveaux ronds de l'or ($25): distance du close au multiple le plus proche
    df["round_dist"] = (c % 25).where(lambda s: s <= 12.5, 25 - (c % 25)) / df["atr"]

    df["hour"] = df.index.hour
    df["day_open"] = o.groupby(df.index.normalize()).transform("first")

    # CLV: position du close dans le range (proxy OHLC de pression fin de bougie)
    df["clv"] = (c - l) / rng.replace(0, np.nan)

    # largeur Bollinger relative + percentile (squeeze)
    bbw = (4 * sd20) / ma20
    df["bbw_pctl"] = bbw.rolling(200).rank(pct=True)

    # décomposition M1: nb de clôtures M1 haussières dans chaque bougie M5
    up_m1 = (m1["close"] > m1["close"].shift()).astype(float)
    df["m1_up"] = up_m1.groupby(m1.index.floor("5min")).sum().reindex(df.index)

    # donchian 20 (bornes de la bougie précédente, causal)
    df["dc_hi20"] = h.shift().rolling(20).max()
    df["dc_lo20"] = l.shift().rolling(20).min()
    df["atr_med"] = df["atr"].rolling(288).median()

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  CONDITIONS CANDIDATES  →  {nom: (Series bool, direction)}
#  direction: "long", "short", ou "both" (long, et miroir short testé à part)
# ──────────────────────────────────────────────────────────────────────────────

def conditions(df: pd.DataFrame) -> dict:
    c, h, l = df["close"], df["high"], df["low"]
    cond = {}

    # mean reversion
    cond["rsi2_low"] = (df["rsi2"] < 5, "long")
    cond["rsi2_high"] = (df["rsi2"] > 95, "short")
    cond["bb_ext_low"] = (df["bb_z"] < -2.2, "long")
    cond["bb_ext_high"] = (df["bb_z"] > 2.2, "short")
    cond["ret_z_down"] = (df["ret_z"] < -2.5, "long")
    cond["ret_z_up"] = (df["ret_z"] > 2.5, "short")
    cond["streak_dn4"] = (df["streak"] <= -4, "long")
    cond["streak_up4"] = (df["streak"] >= 4, "short")
    cond["vwap_ext_low"] = (df["dist_vwap_atr"] < -2.0, "long")
    cond["vwap_ext_high"] = (df["dist_vwap_atr"] > 2.0, "short")
    cond["big_red_bar"] = ((df["big_bar"] > 2.0) & (c < df["open"]), "long")
    cond["big_green_bar"] = ((df["big_bar"] > 2.0) & (c > df["open"]), "short")
    cond["dn_wick_extreme"] = ((df["dn_wick"] > 0.6) & (df["big_bar"] > 1.2), "long")
    cond["up_wick_extreme"] = ((df["up_wick"] > 0.6) & (df["big_bar"] > 1.2), "short")

    # momentum / breakout
    cond["hh12_break"] = ((c > h.shift().rolling(12).max()) & (df["range_pctl"] > 0.5), "long")
    cond["ll12_break"] = ((c < l.shift().rolling(12).min()) & (df["range_pctl"] > 0.5), "short")
    cond["nr7_break_up"] = (df["nr7"].shift() & (c > h.shift()), "long")
    cond["nr7_break_dn"] = (df["nr7"].shift() & (c < l.shift()), "short")
    cond["inside_break_up"] = (df["inside"].shift() & (c > h.shift(2)), "long")
    cond["inside_break_dn"] = (df["inside"].shift() & (c < l.shift(2)), "short")
    cond["mom3_up"] = ((c > c.shift(3)) & (df["ema8"] > df["ema21"]) & (df["body_ratio"] > 0.6) & (c > df["open"]), "long")
    cond["mom3_dn"] = ((c < c.shift(3)) & (df["ema8"] < df["ema21"]) & (df["body_ratio"] > 0.6) & (c < df["open"]), "short")

    # trend-pullback
    cond["pb_ema21_up"] = ((df["ema8"] > df["ema21"]) & (df["ema21"] > df["ema50"]) & (l <= df["ema21"]) & (c > df["ema21"]), "long")
    cond["pb_ema21_dn"] = ((df["ema8"] < df["ema21"]) & (df["ema21"] < df["ema50"]) & (h >= df["ema21"]) & (c < df["ema21"]), "short")

    # temps / sessions
    cond["h_london_1st"] = ((df["hour"] == 7) & (df.index.minute < 30), "both")
    cond["h_ny_1st"] = ((df["hour"] == 13) & (df.index.minute >= 30), "both")
    cond["asia_fade_up"] = ((df["hour"].isin([1, 2, 3])) & (df["dist_vwap_atr"] > 1.2), "short")
    cond["asia_fade_dn"] = ((df["hour"].isin([1, 2, 3])) & (df["dist_vwap_atr"] < -1.2), "long")

    # niveaux
    cond["round_bounce"] = ((df["round_dist"] < 0.15) & (df["rsi2"] < 10), "long")
    cond["round_reject"] = ((df["round_dist"] < 0.15) & (df["rsi2"] > 90), "short")
    cond["above_day_open"] = ((c > df["day_open"]) & (c.shift() <= df["day_open"].shift()), "long")
    cond["below_day_open"] = ((c < df["day_open"]) & (c.shift() >= df["day_open"].shift()), "short")

    # ── idées de l'agent (non couvertes par la 1re bibliothèque) ─────────────
    o = df["open"]
    body_signed = c - o

    # 34. persistance du chemin M1 intrabar (info invisible en OHLC M5 pur)
    cond["m1_path_up"] = ((df["m1_up"] >= 4) & (c > o), "long")
    cond["m1_path_dn"] = ((df["m1_up"] <= 1) & (c < o), "short")

    # 35. CLV persistant: deux clôtures de suite à l'extrême du range
    cond["clv_hi2"] = ((df["clv"] >= 0.9) & (df["clv"].shift() >= 0.9) & (df["big_bar"] >= 0.8), "long")
    cond["clv_lo2"] = ((df["clv"] <= 0.1) & (df["clv"].shift() <= 0.1) & (df["big_bar"] >= 0.8), "short")

    # 36. asymétrie baissière de l'or: continuation des grosses rouges (+ contrôle long)
    cond["down_asym_cont"] = (body_signed <= -2.0 * df["atr"], "short")
    cond["up_asym_ctrl"] = (body_signed >= 2.0 * df["atr"], "long")

    # 10. squeeze BB multi-heures + release directionnel
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    cond["squeeze_rel_up"] = ((df["bbw_pctl"] <= 0.15) & (c > ma20 + 2 * sd20), "long")
    cond["squeeze_rel_dn"] = ((df["bbw_pctl"] <= 0.15) & (c < ma20 - 2 * sd20), "short")

    # 17. ignition d'ouverture de session (direction = sens de la 1re bougie)
    first_bar = ((df["hour"] == 7) & (df.index.minute == 0)) | ((df["hour"] == 13) & (df.index.minute == 30))
    cond["sess_ignite_up"] = (first_bar & (body_signed >= 0.8 * df["atr"]), "long")
    cond["sess_ignite_dn"] = (first_bar & (body_signed <= -0.8 * df["atr"]), "short")

    # 7. donchian20 + régime de vol haute (version agent de mon hh12 mort)
    cond["dc20_hivol_up"] = ((c > df["dc_hi20"]) & (df["atr"] > df["atr_med"]) & df["hour"].between(7, 17), "long")
    cond["dc20_hivol_dn"] = ((c < df["dc_lo20"]) & (df["atr"] > df["atr_med"]) & df["hour"].between(7, 17), "short")

    # 14. marubozu: corps plein ≥85% + range ≥1.5 ATR → continuation
    cond["marubozu_up"] = ((df["body_ratio"] >= 0.85) & (df["big_bar"] >= 1.5) & (c > o), "long")
    cond["marubozu_dn"] = ((df["body_ratio"] >= 0.85) & (df["big_bar"] >= 1.5) & (c < o), "short")

    return cond


# ──────────────────────────────────────────────────────────────────────────────
#  PREMIER-PASSAGE TP/SL sur M1 (vectorisé par fenêtres glissantes)
# ──────────────────────────────────────────────────────────────────────────────

def first_passage(m1: pd.DataFrame, event_times, atrs, direction: str,
                  k_sl: float, k_tp: float, horizon_m1: int, commission_pct: float,
                  time_exit_m1: int = 0):
    """→ dict de stats. event_times = heure d'OUVERTURE de la bougie M5 signal;
    l'entrée se fait à l'open de la première M1 qui suit sa CLÔTURE (t+5min) —
    aucune information de la bougie signal n'est utilisée après coup."""
    o = m1["open"].values
    h = m1["high"].values
    l = m1["low"].values
    c = m1["close"].values
    idx = m1.index

    # fenêtres forward de highs/lows
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(m1)
    Hw = sliding_window_view(h, horizon_m1)   # [n-H+1, H]
    Lw = sliding_window_view(l, horizon_m1)

    # première M1 dont l'open est ≥ clôture de la bougie M5 signal
    close_times = event_times + pd.Timedelta(seconds=M5)
    pos = idx.searchsorted(close_times)
    valid = pos < (n - horizon_m1 - 1)
    pos, atrs = pos[valid], atrs[valid]
    if len(pos) == 0:
        return None
    entries = o[pos]                          # vrai prix d'entrée: open M1 suivant

    sgn = 1.0 if direction == "long" else -1.0

    if time_exit_m1:
        # sortie au close après N bougies M1, R mesuré vs un stop virtuel k_sl×ATR
        # (le stop sert au sizing/commission; on suppose qu'il n'est pas touché —
        #  approximation optimiste, à confirmer en backtest complet)
        exit_c = c[np.minimum(pos + time_exit_m1 - 1, n - 1)]
        r = sgn * (exit_c - entries) / (k_sl * atrs)
        comm_r = (commission_pct / 100.0) * entries / (k_sl * atrs)
        r_net = r - comm_r
        win = r > 0
        return {
            "n": len(pos),
            "winrate_%": round(100 * win.mean(), 1),
            "timeout_%": 100.0,
            "exp_R": round(float(r.mean()), 4),
            "exp_R_net": round(float(r_net.mean()), 4),
            "comm_R": round(float(comm_r.mean()), 4),
        }

    sl = entries - sgn * k_sl * atrs
    tp = entries + sgn * k_tp * atrs

    Hev, Lev = Hw[pos], Lw[pos]
    if direction == "long":
        tp_hit = Hev >= tp[:, None]
        sl_hit = Lev <= sl[:, None]
    else:
        tp_hit = Lev <= tp[:, None]
        sl_hit = Hev >= sl[:, None]

    t_tp = np.where(tp_hit.any(1), tp_hit.argmax(1), horizon_m1 + 1)
    t_sl = np.where(sl_hit.any(1), sl_hit.argmax(1), horizon_m1 + 1)

    win = t_tp < t_sl                       # SL prioritaire si égalité (conservateur)
    loss = t_sl <= t_tp
    timeout = (~win) & (~loss)
    # timeout: mark-to-market à l'horizon, en R
    exit_c = c[np.minimum(pos + horizon_m1 - 1, n - 1)]
    mtm_r = sgn * (exit_c - entries) / (k_sl * atrs)

    r = np.where(win, k_tp / k_sl, np.where(loss, -1.0, np.clip(mtm_r, -1.0, k_tp / k_sl)))
    comm_r = (commission_pct / 100.0) * entries / (k_sl * atrs)
    r_net = r - comm_r

    return {
        "n": len(pos),
        "winrate_%": round(100 * win.mean(), 1),
        "timeout_%": round(100 * timeout.mean(), 1),
        "exp_R": round(float(r.mean()), 4),
        "exp_R_net": round(float(r_net.mean()), 4),
        "comm_R": round(float(comm_r.mean()), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  CRIBLAGE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="frxXAUUSD")
    p.add_argument("--from", dest="d_from", required=True)
    p.add_argument("--to", dest="d_to", required=True)
    p.add_argument("--k-sl", type=float, default=1.0, help="stop en ×ATR(M5)")
    p.add_argument("--k-tp", type=float, default=1.0, help="target en ×ATR(M5)")
    p.add_argument("--horizon", type=int, default=24, help="bougies M5 max avant timeout")
    p.add_argument("--commission-pct", type=float, default=0.018)
    p.add_argument("--only", default=None, help="ne tester qu'une condition (nom exact)")
    p.add_argument("--min-n", type=int, default=80)
    p.add_argument("--time-exit", type=int, default=0,
                   help="sortie au close après N bougies M5 (0 = course TP/SL); "
                        "le stop --k-sl reste la base du sizing/commission")
    args = p.parse_args()

    m1 = bt.load_m1(args.symbol, args.d_from, args.d_to)
    days = max(1, np.busday_count(m1.index[0].date(), m1.index[-1].date()))
    print(f"{args.symbol} M1 {m1.index[0]} → {m1.index[-1]}  ({len(m1)} bougies, ~{days}j ouvrés)")
    print(f"course symétrique ±{args.k_sl}/{args.k_tp}×ATR(M5), horizon {args.horizon}×M5, "
          f"commission {args.commission_pct}%\n")

    m5 = build_m5(m1)
    cond = conditions(m5)
    if args.only:
        cond = {args.only: cond[args.only]}

    rows = []
    for name, (mask, direc) in cond.items():
        mask = mask.fillna(False)
        ev = m5.index[mask]
        atrs_all = m5["atr"][mask].values
        ok = ~np.isnan(atrs_all) & (atrs_all > 0)
        ev, atrs_all = ev[ok], atrs_all[ok]

        dirs = ["long", "short"] if direc == "both" else [direc]
        for d in dirs:
            s = first_passage(m1, ev, atrs_all, d,
                              args.k_sl, args.k_tp, args.horizon * 5, args.commission_pct,
                              time_exit_m1=args.time_exit * 5)
            if s is None:
                continue
            s.update({"signal": name, "dir": d, "per_day": round(s["n"] / days, 2)})
            rows.append(s)

    df = pd.DataFrame(rows)[
        ["signal", "dir", "n", "per_day", "winrate_%", "timeout_%", "exp_R", "comm_R", "exp_R_net"]
    ].sort_values("exp_R_net", ascending=False)

    small = df["n"] < args.min_n
    print(df.to_string(index=False))
    print(f"\n⚠ {small.sum()} lignes avec n < {args.min_n} (peu fiables)")
    df.to_csv("event_study.csv", index=False)
    print("→ event_study.csv")


if __name__ == "__main__":
    main()
