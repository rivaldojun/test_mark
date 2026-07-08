#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Scalping Bot multi-stratégies  –  Deriv Options Trading API (2026)      ║
║  Symbols  : frxXAUUSD (défaut), cryBTCUSD, ...  |  Risk: 1%/trade           ║
║  Strategies (--strategy): ema-rsi · sweep-mss · vwap-reclaim · orb          ║
║  Le code des stratégies est PARTAGÉ avec backtest_xau.py (strategies.py).  ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW THE NEW DERIV API WORKS (verified against the live servers, July 2026)
    Deriv migrated from the legacy v3 WebSocket (numeric app_id + short tokens)
    to a new "Options Trading" API:

      1. REST  GET  https://api.derivws.com/trading/v1/options/accounts
                    Headers: Authorization: Bearer pat_...   Deriv-App-ID: ...
                    → lists your DOT... trading accounts (demo/real)
      2. REST  POST .../accounts/{account_id}/otp
                    → returns a one-time WebSocket URL bound to that account
      3. WS    connect to that URL → then the classic JSON message protocol
                    (proposal / buy / sell / contract_update / balance / ...)
                    with `underlying_symbol` instead of the old `symbol` field.

    Personal Access Tokens (pat_...) generated at home.deriv.com/dashboard
    only work with THIS API — they are always rejected as "invalid" by the
    legacy v3 endpoint. That is not an error in this bot; it's by design.

INSTALL
    pip3 install websockets pandas numpy

GET YOUR CREDENTIALS
    1. PAT token : home.deriv.com/dashboard → API token → Create token
                   (scope "Trade" required; 90-day max expiry)
    2. App ID    : home.deriv.com/dashboard → Applications → register an app.
                   Passed as the Deriv-App-ID header. NOTE: an OAuth "Client
                   ID" (UUID) is NOT accepted here — the server answers
                   "Invalid application". Create/use a PAT-type app instead.

USAGE
    # Demo account (default), stratégie ORB sur l'or:
    python3 deriv_ws_scalper.py --token pat_... --app-id YOUR_APP_ID --strategy orb

    # Autres stratégies:
    python3 deriv_ws_scalper.py --token ... --app-id ... --strategy vwap-reclaim
    python3 deriv_ws_scalper.py --token ... --app-id ... --strategy sweep-mss
    python3 deriv_ws_scalper.py --token ... --app-id ... --strategy ema-rsi --symbol cryBTCUSD

    # Real-money account (deliberate opt-in):
    python3 deriv_ws_scalper.py --token pat_... --app-id YOUR_APP_ID --live

    # Override risk / daily-loss cap / RR / leverage:
    ... --risk 0.01 --max-dd 0.05 --rr 2.5 --multiplier 100

STRATEGIES (détail complet + hypothèses de session dans strategies.py)
    ema-rsi       EMA(8/21) cross + filtre RSI(14) (défaut M5, bot d'origine)
    sweep-mss     Balayage de liquidité + cassure de structure (ICT, défaut M1)
    vwap-reclaim  Reclaim du VWAP de session, fenêtres Londres/NY (défaut M1)
    orb           Opening Range Breakout + retest, Londres/NY (défaut M1)
    alligator     Williams Alligator + cassure de fractale (défaut M5)
    --tf M1|M5|M15|H1… change le timeframe d'entrée; les TF de structure/biais
    s'échelonnent automatiquement (mêmes valeurs dans le backtester).
    ⚠ Les heures de session sont en UTC heure d'ÉTÉ (Londres 07:00, NY 13:30)
      — à ajuster dans strategies.py aux changements d'heure.

MONEY MANAGEMENT
    Risk per trade : 1 % of balance → sized via stake × multiplier
    Daily hard stop: −5 % → bot closes all positions and sleeps until next day
    Max concurrent : 2 simultaneous positions

    Deriv Multiplier contracts take stop_loss/take_profit as DOLLAR amounts
    (auto-close when contract P&L reaches them), not price levels like MT5.
    The bot converts the strategy's SL/TP price levels into the equivalent
    dollar amounts for the computed stake/multiplier — identical realized risk.

    Le multiplicateur est choisi PAR TRADE: le plus haut disponible tel que
    SL$ ≤ 95% du stake (Deriv interdit de perdre plus que le stake).

⚠ DISCLAIMER
    Trading carries significant financial risk. Past statistical performance
    does NOT guarantee future results. ALWAYS validate on a demo account and
    on the backtester (backtest_xau.py) before using real money.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import sys
import json
import math
import asyncio
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

try:
    import websockets
except ImportError:
    sys.exit(
        "\n❌  'websockets' package not found.\n"
        "    Install with:  pip3 install websockets pandas numpy\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  CONFIGURATION  (edit here OR pass via CLI)  ═══
# ──────────────────────────────────────────────────────────────────────────────

# ── Broker credentials ────────────────────────────────────────────────────────
DEFAULT_TOKEN  = ""   # PAT from home.deriv.com/dashboard (scope: Trade)
DEFAULT_APP_ID = ""   # App ID from home.deriv.com/dashboard → Applications

REST_BASE = "https://api.derivws.com"
ACCOUNTS_PATH = "/trading/v1/options/accounts"

# ── Market ────────────────────────────────────────────────────────────────────
SYMBOL = "frxXAUUSD"    # default: gold. --symbol cryBTCUSD for bitcoin, etc.

# ── Strategy ──────────────────────────────────────────────────────────────────
# Signal logic lives in strategies.py (shared with the backtester — what you
# backtest is exactly what trades live). Pick with --strategy, timeframe with
# --tf (M1/M5/M15/H1…; default: each strategy's own).
DEFAULT_STRATEGY = "orb"

# ── Risk / exit parameters ────────────────────────────────────────────────────
BE_TRIGGER_R = 0.67        # Tighten SL to break-even at 0.67×R favorable move
                           # (same constant in backtest_xau.py — keep in sync)
STALE_DATA_FACTOR = 5      # candles older than N×granularity ⇒ market closed

# ── Money management ──────────────────────────────────────────────────────────
RISK_PCT           = 0.01   # 1% of account balance at risk per trade
MAX_DAILY_LOSS_PCT = 0.05   # Kill-switch at −5% daily P&L
MAX_OPEN_TRADES    = 2      # Maximum simultaneous positions

# ── Execution / loop ──────────────────────────────────────────────────────────
RECONNECT_S   = 30     # Wait before reconnect attempt
REQUEST_TIMEOUT_S = 15
APP_PING_S    = 30     # Application-level {"ping": 1} keepalive


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  LOGGING  ═══
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("btc_scalper_deriv.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("btc_scalper_deriv")


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  STRATEGIES (shared with the backtester — see strategies.py)  ═══
# ──────────────────────────────────────────────────────────────────────────────

import strategies as st


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  REST HELPERS (auth + account discovery + OTP handshake)  ═══
# ──────────────────────────────────────────────────────────────────────────────

class DerivRestError(RuntimeError):
    pass


def _rest_call(method: str, path: str, token: str, app_id: str, body: Optional[dict] = None) -> dict:
    """Blocking REST call to api.derivws.com with PAT bearer auth."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Deriv-App-ID": app_id,
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()

    req = urllib.request.Request(REST_BASE + path, headers=headers, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise DerivRestError(f"{method} {path} → HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise DerivRestError(f"{method} {path} → network error: {e.reason}") from None


def list_accounts(token: str, app_id: str) -> List[dict]:
    """Returns [{'account_id': 'DOT...', 'balance': ..., 'currency': 'USD',
    'account_type': 'demo'|'real', 'status': 'active'}, ...]"""
    resp = _rest_call("GET", ACCOUNTS_PATH, token, app_id)
    return resp.get("data", [])


def get_ws_url(token: str, app_id: str, account_id: str) -> str:
    """One-time-password handshake: returns a WebSocket URL bound to the account."""
    resp = _rest_call("POST", f"{ACCOUNTS_PATH}/{account_id}/otp", token, app_id)
    url = resp.get("data", {}).get("url")
    if not url:
        raise DerivRestError(f"OTP response missing WebSocket URL: {resp}")
    return url


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  DERIV WEBSOCKET CLIENT  ═══
# ──────────────────────────────────────────────────────────────────────────────

class DerivWSClient:
    """
    Request/response wrapper around the Deriv Options Trading WebSocket.
    Every outgoing message gets a unique req_id; a background reader task
    dispatches incoming frames to the Future waiting on that req_id.
    Connection auth: OTP embedded in the URL (fetched via REST) — there is
    no `authorize` message on this API.
    """

    def __init__(self, token: str, app_id: str) -> None:
        self.token  = token
        self.app_id = app_id
        self.account: Dict[str, Any] = {}
        self.ws = None
        self._req_id  = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    async def connect(self, account: dict) -> None:
        """`account` is one entry from list_accounts()."""
        self.account = account
        loop = asyncio.get_event_loop()
        ws_url = await loop.run_in_executor(
            None, get_ws_url, self.token, self.app_id, account["account_id"]
        )
        self.ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=10)
        self._reader_task = asyncio.create_task(self._reader())
        self._ping_task   = asyncio.create_task(self._app_ping())

    async def _reader(self) -> None:
        try:
            async for raw in self.ws:
                data = json.loads(raw)
                rid  = data.get("req_id")
                fut  = self._pending.pop(rid, None) if rid is not None else None
                if fut and not fut.done():
                    fut.set_result(data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket connection closed"))
            self._pending.clear()

    async def _app_ping(self) -> None:
        """Application-level keepalive on top of protocol pings."""
        while True:
            await asyncio.sleep(APP_PING_S)
            try:
                await self.request({"ping": 1})
            except Exception:
                return

    async def request(self, payload: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
        self._req_id += 1
        rid = self._req_id
        payload = {**payload, "req_id": rid}

        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self.ws.send(json.dumps(payload))
        return await asyncio.wait_for(fut, timeout)

    async def close(self) -> None:
        for task in (self._reader_task, self._ping_task):
            if task:
                task.cancel()
        if self.ws:
            await self.ws.close()


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  DATA / ACCOUNT HELPERS  ═══
# ──────────────────────────────────────────────────────────────────────────────

async def get_candles(client: DerivWSClient, symbol: str, granularity: int, count: int) -> Optional[pd.DataFrame]:
    """Returns CLOSED candles only — the strategy contract (strategies.py)
    guarantees iloc[-1] is a finished bar, so the still-forming one is dropped."""
    resp = await client.request({
        "ticks_history": symbol,
        "end": "latest",
        "count": count + 1,
        "style": "candles",
        "granularity": granularity,
    })
    if "error" in resp:
        log.warning("ticks_history error: %s", resp["error"]["message"])
        return None
    candles = resp.get("candles", [])
    if not candles:
        return None
    now_epoch = datetime.now(timezone.utc).timestamp()
    if candles[-1]["epoch"] + granularity > now_epoch:
        candles = candles[:-1]
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close"]].astype(float)


async def get_strategy_data(client: DerivWSClient, symbol: str, strat: "st.Strategy") -> Optional[Dict[int, pd.DataFrame]]:
    """Fetches every timeframe the strategy needs. None if any is missing."""
    data: Dict[int, pd.DataFrame] = {}
    for gran, count in strat.granularities.items():
        df = await get_candles(client, symbol, gran, count)
        if df is None or len(df) < 30:
            return None
        data[gran] = df
    return data


async def get_balance(client: DerivWSClient) -> Optional[float]:
    resp = await client.request({"balance": 1})
    if "error" in resp:
        log.warning("balance error: %s", resp["error"]["message"])
        return None
    return float(resp["balance"]["balance"])


async def get_contracts_for(client: DerivWSClient, symbol: str) -> Optional[dict]:
    resp = await client.request({"contracts_for": symbol})
    if "error" in resp:
        log.error("contracts_for error: %s", resp["error"]["message"])
        return None
    return resp["contracts_for"]


def pick_multiplier(available: List[float], preferred: Optional[float]) -> float:
    """Multiplier values allowed by Deriv are a discrete list, not a range."""
    if preferred is None:
        return max(available)
    if preferred in available:
        return preferred
    return min(available, key=lambda x: abs(x - preferred))


async def get_open_contract(client: DerivWSClient, contract_id: int) -> Optional[dict]:
    resp = await client.request({"proposal_open_contract": 1, "contract_id": contract_id})
    if "error" in resp:
        log.warning("proposal_open_contract error: %s", resp["error"]["message"])
        return None
    return resp.get("proposal_open_contract")


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  POSITION SIZING  ═══
# ──────────────────────────────────────────────────────────────────────────────

def compute_stake(
    balance: float,
    entry_price: float,
    sl_dist: float,
    multiplier: float,
    min_stake: float,
    max_stake: float,
) -> float:
    """
    Computes the stake such that hitting the stop-loss costs exactly
    RISK_PCT × balance, given the contract's fixed multiplier (leverage).

    Multiplier contract P&L ≈ stake × multiplier × (price_change / entry_price).
    Solving for stake at price_change = sl_dist and P&L = risk_usd:
        stake = risk_usd × entry_price / (multiplier × sl_dist)
    """
    if sl_dist <= 0 or multiplier <= 0 or entry_price <= 0:
        return 0.0

    risk_usd = balance * RISK_PCT
    raw_stake = risk_usd * entry_price / (multiplier * sl_dist)

    stake = max(min_stake, min(raw_stake, max_stake))
    return round(stake, 2)


def dollar_amount_for_price_distance(
    stake: float, multiplier: float, price_dist: float, entry_price: float
) -> float:
    """Converts an ATR-based price distance into the equivalent dollar P&L
    amount for a given stake/multiplier — the unit Deriv's limit_order needs."""
    if entry_price <= 0:
        return 0.0
    return stake * multiplier * price_dist / entry_price


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  ORDER EXECUTION  ═══
# ──────────────────────────────────────────────────────────────────────────────

async def _get_proposal(
    client: DerivWSClient,
    contract_type: str,
    currency: str,
    symbol: str,
    stake: float,
    multiplier: float,
    sl_amount: float,
    tp_amount: float,
) -> dict:
    return await client.request({
        "proposal": 1,
        "amount": round(stake, 2),
        "basis": "stake",
        "contract_type": contract_type,
        "currency": currency,
        "underlying_symbol": symbol,
        "multiplier": multiplier,
        "limit_order": {"stop_loss": round(sl_amount, 2), "take_profit": round(tp_amount, 2)},
    })


async def open_trade(
    client: DerivWSClient,
    symbol: str,
    currency: str,
    direction: str,     # "long" or "short"
    stake: float,
    multiplier: float,
    sl_amount: float,
    tp_amount: float,
    sl_dist: float,     # SL distance in PRICE units (for break-even trigger)
) -> Optional[dict]:
    contract_type = "MULTUP" if direction == "long" else "MULTDOWN"

    # First proposal: fetch live validation bounds (min/max stake and SL/TP
    # dollar bounds change over time — we clamp to whatever is valid now).
    proposal = await _get_proposal(
        client, contract_type, currency, symbol, stake, multiplier, sl_amount, tp_amount
    )
    if "error" in proposal:
        log.warning("proposal rejected (%s) – skipping entry.", proposal["error"]["message"])
        return None

    p  = proposal["proposal"]
    vp = p.get("validation_params", {})
    cd = p.get("contract_details", {})

    def _clamp(value, lo, hi):
        return max(float(lo), min(value, float(hi)))

    c_stake = _clamp(stake, cd.get("minimum_stake", stake), cd.get("maximum_stake", stake))
    c_sl, c_tp = sl_amount, tp_amount
    if "stop_loss" in vp:
        c_sl = _clamp(sl_amount, vp["stop_loss"]["min"], vp["stop_loss"]["max"])
    if "take_profit" in vp:
        c_tp = _clamp(tp_amount, vp["take_profit"]["min"], vp["take_profit"]["max"])

    # If clamping changed anything, re-price: the buy references the proposal
    # id, which is bound to the ORIGINAL parameters.
    if (round(c_stake, 2), round(c_sl, 2), round(c_tp, 2)) != (round(stake, 2), round(sl_amount, 2), round(tp_amount, 2)):
        log.info(
            "Clamped to broker limits: stake %.2f→%.2f  SL %.2f→%.2f  TP %.2f→%.2f",
            stake, c_stake, sl_amount, c_sl, tp_amount, c_tp,
        )
        stake, sl_amount, tp_amount = c_stake, c_sl, c_tp
        proposal = await _get_proposal(
            client, contract_type, currency, symbol, stake, multiplier, sl_amount, tp_amount
        )
        if "error" in proposal:
            log.warning("re-proposal rejected (%s) – skipping entry.", proposal["error"]["message"])
            return None
        p = proposal["proposal"]

    res = await client.request({
        "buy": p["id"],
        "price": float(p["ask_price"]),
    })

    if "error" in res:
        log.warning("❌ Open order failed  dir=%s  reason=%s", direction, res["error"]["message"])
        return None

    buy = res["buy"]
    contract_id = buy["contract_id"]

    # Fetch entry_spot / commission once, needed for break-even management.
    oc = await get_open_contract(client, contract_id)
    entry_spot = float(oc["entry_spot"]) if oc and oc.get("entry_spot") is not None else None
    commission = float(oc.get("commission", 0.0)) if oc else 0.0

    log.info(
        "✅ OPEN %-5s  stake=$%.2f  x%s  SL=$%.2f  TP=$%.2f  entry=%.2f  contract_id=%s",
        direction.upper(), stake, multiplier, sl_amount, tp_amount, entry_spot or 0.0, contract_id,
    )

    return {
        "contract_id": contract_id,
        "direction": direction,
        "stake": stake,
        "multiplier": multiplier,
        "entry_spot": entry_spot,
        "commission": commission,
        "sl_amount": sl_amount,
        "tp_amount": tp_amount,
        "sl_dist": sl_dist,
        "be_triggered": False,
    }


async def update_stop_loss(client: DerivWSClient, contract_id: int, new_sl: float) -> bool:
    res = await client.request({
        "contract_update": 1,
        "contract_id": contract_id,
        "limit_order": {"stop_loss": round(new_sl, 2)},
    })
    if "error" in res:
        log.debug("contract_update failed contract_id=%s reason=%s", contract_id, res["error"]["message"])
        return False
    log.info("↗  Break-even SL set to $%.2f  (contract_id=%s)", new_sl, contract_id)
    return True


async def close_position(client: DerivWSClient, contract_id: int) -> bool:
    res = await client.request({"sell": contract_id, "price": 0})
    if "error" in res:
        log.warning("Close failed contract_id=%s reason=%s", contract_id, res["error"]["message"])
        return False
    log.info("🔒 Closed contract_id=%s  sold_for=%.2f", contract_id, float(res["sell"].get("sold_for", 0.0)))
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  TRAILING / BREAK-EVEN MANAGER  ═══
# ──────────────────────────────────────────────────────────────────────────────

async def manage_trailing_stops(client: DerivWSClient, positions: Dict[int, dict]) -> None:
    """
    When price has moved ≥ BE_TRIGGER_R × (the trade's own SL distance) in the
    trade's favor, tighten stop_loss down to "commission only" — the Deriv
    equivalent of a break-even stop: worst case after trigger is a scratch
    trade. Defined in R units so it behaves identically for every strategy —
    and identically in the backtester.
    """
    for contract_id, pos in list(positions.items()):
        if pos["be_triggered"] or pos["entry_spot"] is None:
            continue
        trigger_dist = BE_TRIGGER_R * pos["sl_dist"]

        oc = await get_open_contract(client, contract_id)
        if oc is None or oc.get("is_sold"):
            continue

        current_spot = float(oc["current_spot"])
        if pos["direction"] == "long":
            favorable_move = current_spot - pos["entry_spot"]
        else:
            favorable_move = pos["entry_spot"] - current_spot

        if favorable_move >= trigger_dist:
            be_sl = max(pos["commission"], 0.01)
            if await update_stop_loss(client, contract_id, be_sl):
                pos["be_triggered"] = True
                pos["sl_amount"] = be_sl


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  DAILY LOSS GUARD  ═══
# ──────────────────────────────────────────────────────────────────────────────

class DailyLossGuard:
    """
    Tracks account equity vs. the opening balance at the start of each
    trading day (UTC). If equity drops by more than MAX_DAILY_LOSS_PCT,
    the bot closes all positions and stops trading until the next day.
    """

    def __init__(self, opening_balance: float) -> None:
        self._date   = datetime.now(timezone.utc).date()
        self._open   = opening_balance
        self._limit  = opening_balance * MAX_DAILY_LOSS_PCT
        self._halted = False

    def reset_if_new_day(self, current_balance: float) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._date:
            self._date   = today
            self._open   = current_balance
            self._limit  = current_balance * MAX_DAILY_LOSS_PCT
            self._halted = False
            log.info("🌅 New trading day  |  opening balance = %.2f", current_balance)

    def is_halted(self, current_equity: float) -> bool:
        if self._halted:
            return True
        day_loss = self._open - current_equity
        if day_loss >= self._limit:
            log.critical(
                "🛑 DAILY LOSS LIMIT REACHED  −%.2f USD (limit = −%.2f).  Bot halted.",
                day_loss, self._limit,
            )
            self._halted = True
        return self._halted

    def status_str(self, current_equity: float) -> str:
        day_pnl = current_equity - self._open
        return f"day_P&L={day_pnl:+.2f}  limit=−{self._limit:.2f}  equity={current_equity:.2f}"


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  MAIN TRADING LOOP  ═══
# ──────────────────────────────────────────────────────────────────────────────

def select_account(accounts: List[dict], want_real: bool) -> Optional[dict]:
    wanted_type = "real" if want_real else "demo"
    candidates = [
        a for a in accounts
        if a.get("account_type") == wanted_type and a.get("status", "active") == "active"
    ]
    return candidates[0] if candidates else None


async def run(token: str, app_id: str, symbol: str, preferred_multiplier: Optional[float],
              allow_live: bool, strategy_name: str, rr: Optional[float],
              tf=None, sessions=None, or_bars: Optional[int] = None) -> None:
    strat = st.make_strategy(strategy_name, rr=rr, tf=tf, sessions=sessions)
    if or_bars is not None:
        if strategy_name != "orb":
            log.warning("--or-bars ignoré: ne s'applique qu'à la stratégie orb.")
        else:
            strat.OR_BARS = or_bars
    loop_sleep = strat.poll_seconds
    primary_gran = min(strat.granularities)   # smallest TF drives entry price & BE trigger
    log.info(
        "Strategy: %s  |  timeframes=%s  |  poll every %ds",
        strat.name, sorted(strat.granularities), loop_sleep,
    )

    # ── Account discovery (REST) ──────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    accounts = await loop.run_in_executor(None, list_accounts, token, app_id)
    if not accounts:
        log.error("No Options trading accounts found for this token/app. "
                  "Check the dashboard at home.deriv.com/dashboard.")
        return
    log.info(
        "Accounts found: %s",
        ", ".join(f"{a['account_id']} ({a.get('account_type','?')}, {a.get('balance','?')} {a.get('currency','')})"
                  for a in accounts),
    )

    account = select_account(accounts, want_real=allow_live)
    if account is None:
        log.error("No %s account available.", "real" if allow_live else "demo")
        return

    # ── Safety belt: never trade real money without explicit --live ───────────
    if account.get("account_type") == "real" and not allow_live:
        log.critical("🛑 Selected account is REAL but --live was not passed. Aborting.")
        return

    client = DerivWSClient(token, app_id)
    await client.connect(account)

    currency = account.get("currency", "USD")
    log.info(
        "✅ Connected  account=%s  type=%s  balance=%.2f %s",
        account["account_id"], account.get("account_type"),
        float(account.get("balance", 0)), currency,
    )

    cfor = await get_contracts_for(client, symbol)
    if cfor is None:
        await client.close()
        return

    mult_entries = [a for a in cfor["available"] if a["contract_type"] in ("MULTUP", "MULTDOWN")]
    if not mult_entries:
        log.error("Symbol '%s' has no MULTUP/MULTDOWN offering on this account.", symbol)
        await client.close()
        return

    available_multipliers = sorted(set(mult_entries[0]["multiplier_range"]))
    log.info(
        "Symbol OK: %-12s  available multipliers=%s  (picked per-trade to fit ATR)",
        symbol, available_multipliers,
    )

    balance = await get_balance(client)
    if balance is None:
        log.error("Could not fetch balance – aborting.")
        await client.close()
        return
    guard = DailyLossGuard(balance)
    positions: Dict[int, dict] = {}

    log.info("=" * 72)
    log.info("Bot running  |  %s", guard.status_str(balance))
    log.info(
        "Strategy=%s  |  Risk=%.1f%%/trade  |  Max daily loss=%.1f%%  |  Max positions=%d",
        strat.name, RISK_PCT * 100, MAX_DAILY_LOSS_PCT * 100, MAX_OPEN_TRADES,
    )
    log.info("Press Ctrl+C to stop cleanly.\n" + "=" * 72)

    try:
        while True:
            try:
                # ── Refresh open positions & drop any the broker already closed ──
                for contract_id in list(positions.keys()):
                    oc = await get_open_contract(client, contract_id)
                    if oc is None:
                        continue
                    if oc.get("is_sold"):
                        log.info(
                            "📤 Position closed by broker  contract_id=%s  profit=%.2f",
                            contract_id, float(oc.get("profit", 0.0)),
                        )
                        del positions[contract_id]

                balance = await get_balance(client)
                if balance is None:
                    await asyncio.sleep(RECONNECT_S)
                    continue

                open_profits = 0.0
                for contract_id in positions:
                    oc = await get_open_contract(client, contract_id)
                    if oc:
                        open_profits += float(oc.get("profit", 0.0))
                equity = balance + open_profits

                guard.reset_if_new_day(balance)

                # ── Daily loss kill-switch ────────────────────────────────────
                if guard.is_halted(equity):
                    for contract_id in list(positions.keys()):
                        if await close_position(client, contract_id):
                            del positions[contract_id]
                    log.warning("Sleeping 10 min before next check…")
                    await asyncio.sleep(600)
                    continue

                # ── Fetch all timeframes the strategy needs (closed bars only) ─
                data = await get_strategy_data(client, symbol, strat)
                if data is None:
                    await asyncio.sleep(loop_sleep)
                    continue

                primary = data[primary_gran]
                now_utc = datetime.now(timezone.utc)

                # Market closed / stale feed (gold closes nights & weekends)
                last_close_age = (now_utc - primary.index[-1].to_pydatetime()).total_seconds()
                if last_close_age > STALE_DATA_FACTOR * primary_gran:
                    log.debug("Market closed or stale feed (last bar %.0fs old) – waiting.", last_close_age)
                    await asyncio.sleep(max(loop_sleep, 120))
                    continue

                # ── Trailing / break-even management ──────────────────────────
                await manage_trailing_stops(client, positions)

                # ── Entry gate: respect max open trades ───────────────────────
                if len(positions) >= MAX_OPEN_TRADES:
                    log.debug("Max open trades (%d) – skipping signal check.", MAX_OPEN_TRADES)
                    await asyncio.sleep(loop_sleep)
                    continue

                # ── Session filter (--sessions) ────────────────────────────────
                if not strat.session_ok(primary.index[-1]):
                    log.debug("Outside allowed sessions (%s) – no entries.", strat.sessions)
                    await asyncio.sleep(loop_sleep)
                    continue

                # ── Signal evaluation (shared strategy code) ───────────────────
                sig = strat.signal(data, primary.index[-1])
                if sig is None:
                    log.debug("No signal  |  %s", guard.status_str(equity))
                    await asyncio.sleep(loop_sleep)
                    continue

                active_dirs = {p["direction"] for p in positions.values()}
                if sig.direction in active_dirs:
                    log.debug("Already in '%s' direction – skipping duplicate entry.", sig.direction)
                    await asyncio.sleep(loop_sleep)
                    continue

                entry_price = float(primary["close"].iloc[-1])
                sl_dist = abs(entry_price - sig.sl_price)
                tp_dist = abs(sig.tp_price - entry_price)
                if sl_dist <= 0 or tp_dist <= 0:
                    log.debug("Degenerate SL/TP distances – skipping.")
                    await asyncio.sleep(loop_sleep)
                    continue

                # Rough bounds to seed the sizing formula — open_trade() re-clamps
                # against the broker's live validation bounds before buying.
                # Stake can never exceed cash on hand; reserve headroom so up
                # to MAX_OPEN_TRADES can each still get a slot concurrently.
                min_stake = 1.0
                max_stake = min(500.0, balance / MAX_OPEN_TRADES)

                if max_stake < min_stake:
                    log.warning(
                        "Balance too low ($%.2f) to open a new position – skipping.",
                        balance,
                    )
                    await asyncio.sleep(loop_sleep)
                    continue

                # Deriv caps stop_loss at 100% of stake (you can't lose more than
                # you staked). SL$ = stake × mult × sl_dist/entry, so we need
                # mult × sl_dist/entry ≤ 1. Pick the highest multiplier that
                # satisfies it with a 5% margin — dynamic per trade.
                usable_multipliers = [
                    m for m in available_multipliers
                    if m * sl_dist / entry_price <= 0.95
                ]
                if not usable_multipliers:
                    log.warning(
                        "SL distance %.2f too wide for available multipliers (%s) – skipping entry.",
                        sl_dist, available_multipliers,
                    )
                    await asyncio.sleep(loop_sleep)
                    continue
                multiplier = pick_multiplier(usable_multipliers, preferred_multiplier)

                stake = compute_stake(balance, entry_price, sl_dist, multiplier, min_stake, max_stake)
                if stake <= 0:
                    log.warning("Stake computed as 0 – skipping.")
                    await asyncio.sleep(loop_sleep)
                    continue

                sl_amount = dollar_amount_for_price_distance(stake, multiplier, sl_dist, entry_price)
                tp_amount = dollar_amount_for_price_distance(stake, multiplier, tp_dist, entry_price)

                actual_risk_pct = (sl_amount / balance * 100) if balance > 0 else 0.0
                if actual_risk_pct < RISK_PCT * 100 * 0.9:
                    log.info(
                        "⚠ Stake capped by limits — actual risk ≈%.2f%% of balance (target %.1f%%).",
                        actual_risk_pct, RISK_PCT * 100,
                    )

                log.info(
                    "📶 SIGNAL %-5s  [%s]  stake=$%.2f  x%s  SL=$%.2f  TP=$%.2f  |  %s",
                    sig.direction.upper(), sig.reason, stake, multiplier,
                    sl_amount, tp_amount, guard.status_str(equity),
                )

                new_pos = await open_trade(
                    client, symbol, currency, sig.direction, stake, multiplier,
                    sl_amount, tp_amount, sl_dist,
                )
                if new_pos:
                    positions[new_pos["contract_id"]] = new_pos

                await asyncio.sleep(loop_sleep)

            except (ConnectionError, asyncio.TimeoutError, websockets.exceptions.ConnectionClosed, DerivRestError) as exc:
                log.warning("Connection issue (%s) – reconnecting in %ds…", exc, RECONNECT_S)
                await client.close()
                await asyncio.sleep(RECONNECT_S)
                try:
                    await client.connect(account)   # fresh OTP each time
                    log.info("✅ Reconnected  account=%s", account["account_id"])
                except Exception as reconnect_exc:
                    log.error("Reconnect failed: %s", reconnect_exc)
                    await asyncio.sleep(RECONNECT_S)

    except KeyboardInterrupt:
        log.info("\n⛔  Interrupted by user (Ctrl+C).")
    finally:
        log.info(
            "Bot stopped  |  open positions still running: %d  "
            "(they will be managed by Deriv's own stop_loss/take_profit)",
            len(positions),
        )
        await client.close()
        log.info("WebSocket connection closed.")


# ──────────────────────────────────────────────────────────────────────────────
#  ═══  CLI ENTRY POINT  ═══
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global RISK_PCT, MAX_DAILY_LOSS_PCT

    parser = argparse.ArgumentParser(
        description="Scalping Bot multi-stratégies – Deriv Options Trading API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token",  type=str, default=DEFAULT_TOKEN, required=not DEFAULT_TOKEN,
                         help="Deriv PAT (home.deriv.com/dashboard → API token, scope Trade)")
    parser.add_argument("--app-id", type=str, default=DEFAULT_APP_ID, required=not DEFAULT_APP_ID,
                         help="Deriv App ID (home.deriv.com/dashboard → Applications)")
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY, choices=list(st.REGISTRY),
                         help="Trading strategy (shared with backtest_xau.py)")
    parser.add_argument("--symbol", type=str, default=SYMBOL,
                         help="Deriv symbol code (frxXAUUSD=gold, cryBTCUSD=bitcoin)")
    parser.add_argument("--risk",   type=float, default=RISK_PCT,
                         help="Fraction of balance to risk per trade (default 0.01 = 1%%)")
    parser.add_argument("--max-dd", type=float, default=MAX_DAILY_LOSS_PCT,
                         help="Daily loss limit as fraction (default 0.05 = 5%%)")
    parser.add_argument("--rr", type=float, default=None,
                         help="Override the strategy's reward/risk multiple (default: per-strategy)")
    parser.add_argument("--tf", type=str, default=None,
                         help="Entry timeframe: M1, M5, M15, H1... (default: strategy's own). "
                              "Structure/bias timeframes scale automatically.")
    parser.add_argument("--sessions", type=str, default=None,
                         help="Restrict entries to sessions: london, ny, off, or 'london,ny' "
                              "(default: strategy's own behavior)")
    parser.add_argument("--or-bars", type=int, default=None,
                         help="ORB only: opening-range size in bars of --tf "
                              "(default 5; e.g. 10 = 10 min range on M1)")
    parser.add_argument("--multiplier", type=float, default=None,
                         help="Preferred leverage multiplier (auto-picks the closest one Deriv allows). "
                              "Default: highest multiplier that fits the trade's SL distance.")
    parser.add_argument("--live", action="store_true",
                         help="Trade the REAL-MONEY account. Without it, the demo account is used.")
    args = parser.parse_args()

    RISK_PCT           = max(0.001, min(0.05, args.risk))     # clamp 0.1%–5%
    MAX_DAILY_LOSS_PCT = max(0.01,  min(0.20, args.max_dd))   # clamp 1%–20%

    log.info(
        "Starting  |  strategy=%s  symbol=%s  mode=%s  risk=%.1f%%  max_dd=%.1f%%",
        args.strategy, args.symbol, "LIVE ⚠" if args.live else "DEMO",
        RISK_PCT * 100, MAX_DAILY_LOSS_PCT * 100,
    )

    try:
        asyncio.run(run(args.token, args.app_id, args.symbol, args.multiplier,
                        args.live, args.strategy, args.rr, args.tf, args.sessions,
                        args.or_bars))
    except (RuntimeError, DerivRestError, ValueError) as exc:
        sys.exit(f"\n❌  {exc}\n")


if __name__ == "__main__":
    main()
