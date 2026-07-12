import websocket
import json
import time
import pandas as pd
import numpy as np
import mplfinance as mpf

from datetime import datetime, timezone, timedelta

PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
SYMBOL = "frxXAUUSD"
GRANULARITY = 900
DAYS = 5

# --- PARAMÈTRES DE LA STRATÉGIE ---
MIN_SPREAD = 0.50 
LOT_SIZE = 0.01
CONTRACT_SIZE = 100  # 1 lot standard = 100 onces sur Deriv

# Paramètres de gestion des risques fixes (en Dollars de PnL pour 0.01 lot)
TARGET_PROFIT_USD = 4.00
MAX_LOSS_USD = 4.00

# ==========================================================
# DOWNLOAD DERIV
# ==========================================================
def get_candles(symbol, granularity, days):
    end = int(time.time())
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    ws = websocket.create_connection(PUBLIC_WS)
    ws.send(json.dumps({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "start": start,
        "end": end,
        "style": "candles",
        "granularity": granularity
    }))

    candles = None
    while True:
        msg = json.loads(ws.recv())
        if "candles" in msg:
            candles = msg["candles"]
            break
        if "error" in msg:
            raise Exception(msg["error"])

    ws.close()
    return candles

# ==========================================================
# DATAFRAME
# ==========================================================
def candles_to_df(candles):
    rows = []
    for c in candles:
        rows.append({
            "time": datetime.fromtimestamp(c["epoch"], timezone.utc),
            "Open": float(c["open"]),
            "High": float(c["high"]),
            "Low": float(c["low"]),
            "Close": float(c["close"])
        })

    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df

# ==========================================================
# TRADINGVIEW SMMA
# ==========================================================
def smma(series, length):
    out = pd.Series(np.nan, index=series.index)
    out.iloc[length-1] = series.iloc[:length].mean()

    for i in range(length, len(series)):
        out.iloc[i] = (out.iloc[i-1] * (length-1) + series.iloc[i]) / length
    return out

# ==========================================================
# ALLIGATOR
# ==========================================================
def add_alligator(df):
    hl2 = (df["High"] + df["Low"]) / 2
    df["jaw"] = smma(hl2, 13)
    df["teeth"] = smma(hl2, 8)
    df["lips"] = smma(hl2, 5)
    return df

# ==========================================================
# EVENTS DETECTION WITH FIXED TP/SL
# ==========================================================
def detect_events(df):
    bullish_entries = []
    bullish_exits = []
    bearish_entries = []
    bearish_exits = []
    
    trades_history = []

    current_state = "NONE" 
    entry_price = 0
    trade_start_time = None

    # Niveaux de prix pour le TP et le SL
    take_profit_price = 0
    stop_loss_price = 0

    i = 4
    while i < len(df):
        jaw = df["jaw"].iloc[i]
        teeth = df["teeth"].iloc[i]
        lips = df["lips"].iloc[i]

        if np.isnan(jaw):
            i += 1
            continue

        volume_factor = LOT_SIZE * CONTRACT_SIZE # Vaut 1 pour 0.01 lot

        # Conversion des dollars en distance de prix sur l'or (1$ USD = 1$ sur l'or)
        price_dist_tp = TARGET_PROFIT_USD / volume_factor
        price_dist_sl = MAX_LOSS_USD / volume_factor

        # --------------------------------------------------
        # GESTION DES EXITS FIXES (TP / SL)
        # --------------------------------------------------
        if current_state == "BUY":
            # 1. Est-ce que la mèche haute a touché le Take Profit ?
            if df["High"].iloc[i] >= take_profit_price:
                # Si l'ouverture a directement gapé au-dessus du TP, on prend l'Open, sinon le TP précis
                exit_p = max(df["Open"].iloc[i], take_profit_price)
                pnl = (exit_p - entry_price) * volume_factor
                trades_history.append({"type": "BUY", "entry_time": trade_start_time, "exit_time": df.index[i], "pnl": pnl, "reason": "Take Profit (+3$)"})
                bullish_exits.append({"time": df.index[i], "price": exit_p})
                current_state = "NONE"
                i += 1
                continue

            # 2. Est-ce que la mèche basse a touché le Stop Loss ?
            elif df["Low"].iloc[i] <= stop_loss_price:
                exit_p = min(df["Open"].iloc[i], stop_loss_price)
                pnl = (exit_p - entry_price) * volume_factor
                trades_history.append({"type": "BUY", "entry_time": trade_start_time, "exit_time": df.index[i], "pnl": pnl, "reason": "Stop Loss (-4$)"})
                bullish_exits.append({"time": df.index[i], "price": exit_p})
                current_state = "NONE"
                i += 1
                continue

        elif current_state == "SELL":
            # 1. Est-ce que la mèche basse a touché le Take Profit (vers le bas) ?
            if df["Low"].iloc[i] <= take_profit_price:
                exit_p = min(df["Open"].iloc[i], take_profit_price)
                pnl = (entry_price - exit_p) * volume_factor
                trades_history.append({"type": "SELL", "entry_time": trade_start_time, "exit_time": df.index[i], "pnl": pnl, "reason": "Take Profit (+3$)"})
                bearish_exits.append({"time": df.index[i], "price": exit_p})
                current_state = "NONE"
                i += 1
                continue

            # 2. Est-ce que la mèche haute a touché le Stop Loss (vers le haut) ?
            elif df["High"].iloc[i] >= stop_loss_price:
                exit_p = max(df["Open"].iloc[i], stop_loss_price)
                pnl = (entry_price - exit_p) * volume_factor
                trades_history.append({"type": "SELL", "entry_time": trade_start_time, "exit_time": df.index[i], "pnl": pnl, "reason": "Stop Loss (-4$)"})
                bearish_exits.append({"time": df.index[i], "price": exit_p})
                current_state = "NONE"
                i += 1
                continue

        # --------------------------------------------------
        # BULLISH SETUP DETECTION (ENTRIES)
        # --------------------------------------------------
        if current_state == "NONE" and (jaw > teeth > lips):
            was_below_lips = (df["Low"].iloc[i-3:i] < df["lips"].iloc[i-3:i]).any()
            green_cross = df["Close"].iloc[i] > df["Open"].iloc[i]
            break_up = df["Close"].iloc[i] > jaw
            current_spread = jaw - lips

            if was_below_lips and green_cross and break_up and (current_spread >= MIN_SPREAD):

                # Condition 1 : Suivante verte
                if i + 1 < len(df) and (df["Close"].iloc[i+1] > df["Open"].iloc[i+1]):
                    entry_price = df["Close"].iloc[i+1]
                    trade_start_time = df.index[i+1]
                    bullish_entries.append({"time": trade_start_time, "price": entry_price})
                    
                    # Calcul des objectifs fixes pour le BUY
                    take_profit_price = entry_price + price_dist_tp
                    stop_loss_price = entry_price - price_dist_sl
                    
                    current_state = "BUY"
                    i += 2  
                    continue

                # Condition 2 : Suivante rouge puis verte avec corps au-dessus
                if i + 2 < len(df):
                    next_red = df["Close"].iloc[i+1] < df["Open"].iloc[i+1]
                    after_next_green = df["Close"].iloc[i+2] > df["Open"].iloc[i+2]
                    upper_body_above = df["Close"].iloc[i+2] > df["jaw"].iloc[i+2]

                    if next_red and after_next_green and upper_body_above:
                        entry_price = df["Close"].iloc[i+2]
                        trade_start_time = df.index[i+2]
                        bullish_entries.append({"time": trade_start_time, "price": entry_price})
                        
                        # Calcul des objectifs fixes pour le BUY
                        take_profit_price = entry_price + price_dist_tp
                        stop_loss_price = entry_price - price_dist_sl
                        
                        current_state = "BUY"
                        i += 3
                        continue

        # --------------------------------------------------
        # BEARISH SETUP DETECTION (ENTRIES)
        # --------------------------------------------------
        if current_state == "NONE" and (lips > teeth > jaw):
            was_above_lips = (df["High"].iloc[i-3:i] > df["lips"].iloc[i-3:i]).any()
            red_cross = df["Close"].iloc[i] < df["Open"].iloc[i]
            break_down = df["Close"].iloc[i] < jaw
            current_spread = lips - jaw

            if was_above_lips and red_cross and break_down and (current_spread >= MIN_SPREAD):

                # Condition 1 : Suivante rouge
                if i + 1 < len(df) and (df["Close"].iloc[i+1] < df["Open"].iloc[i+1]):
                    entry_price = df["Close"].iloc[i+1]
                    trade_start_time = df.index[i+1]
                    bearish_entries.append({"time": trade_start_time, "price": entry_price})
                    
                    # Calcul des objectifs fixes pour le SELL (Inversé)
                    take_profit_price = entry_price - price_dist_tp
                    stop_loss_price = entry_price + price_dist_sl
                    
                    current_state = "SELL"
                    i += 2
                    continue

                # Condition 2 : Suivante verte puis rouge avec corps en dessous
                if i + 2 < len(df):
                    next_green = df["Close"].iloc[i+1] > df["Open"].iloc[i+1]
                    after_next_red = df["Close"].iloc[i+2] < df["Open"].iloc[i+2]
                    upper_body_below = df["Open"].iloc[i+2] < df["jaw"].iloc[i+2]

                    if next_green and after_next_red and upper_body_below:
                        entry_price = df["Close"].iloc[i+2]
                        trade_start_time = df.index[i+2]
                        bearish_entries.append({"time": trade_start_time, "price": entry_price})
                        
                        # Calcul des objectifs fixes pour le SELL (Inversé)
                        take_profit_price = entry_price - price_dist_tp
                        stop_loss_price = entry_price + price_dist_sl
                        
                        current_state = "SELL"
                        i += 3
                        continue

        i += 1

    return bullish_entries, bullish_exits, bearish_entries, bearish_exits, trades_history

# ==========================================================
# PLOT & PRINT REPORT
# ==========================================================
def plot(df):
    df_plot = df.tail(600)
    bull_entries, bull_exits, bear_entries, bear_exits, trades_history = detect_events(df)

    print("\n" + "="*50)
    print(f" RAPPORT DE PERFORMANCE FIXE (TP: +{TARGET_PROFIT_USD}$, SL: -{MAX_LOSS_USD}$)")
    print("="*50)
    
    total_pnl = 0
    winning_trades = 0
    
    for idx, t in enumerate(trades_history, 1):
        print(f"Trade #{idx} | {t['type']} | {t['entry_time'].strftime('%d/%m %H:%M')} -> {t['exit_time'].strftime('%d/%m %H:%M')}")
        print(f"        Résultat: {t['pnl']:.2f} USD ({t['reason']})")
        total_pnl += t['pnl']
        if t['pnl'] > 0:
            winning_trades += 1
            
    print("-"*50)
    total_trades = len(trades_history)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Nombre total de trades : {total_trades}")
    print(f"Taux de réussite (Win Rate) : {win_rate:.2f}%")
    print(f"GAIN / PERTE NET GLOBALE : {total_pnl:.2f} USD")
    print("="*50 + "\n")

    # --- PLOT GRAPHIC ---
    plots = [
        mpf.make_addplot(df_plot["jaw"], color="blue"),
        mpf.make_addplot(df_plot["teeth"], color="red"),
        mpf.make_addplot(df_plot["lips"], color="green")
    ]

    def add_marker(event_list, marker_shape, marker_color):
        if event_list:
            marker_series = pd.Series(np.nan, index=df_plot.index)
            for e in event_list:
                if e["time"] in marker_series.index:
                    marker_series.loc[e["time"]] = e["price"]
            plots.append(mpf.make_addplot(marker_series, type="scatter", marker=marker_shape, markersize=90, color=marker_color))

    add_marker(bull_entries, "^", "green")
    add_marker(bear_entries, "v", "black")
    add_marker(bull_exits, "o", "orange")
    add_marker(bear_exits, "o", "red")

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        addplot=plots,
        figsize=(16, 8),
        title=f"XAUUSD M15 - Fixed TP/SL (TP:+{TARGET_PROFIT_USD}$ / SL:-{MAX_LOSS_USD}$)"
    )

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("Téléchargement XAUUSD M15...")
    try:
        candles = get_candles(SYMBOL, GRANULARITY, DAYS)
        df = candles_to_df(candles)
        df = add_alligator(df)
        plot(df)
    except Exception as e:
        print(f"Erreur rencontrée : {e}")