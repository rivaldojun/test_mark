# Guide — Lancer `mt5_trader.py` sur un VPS Windows

Objectif : faire tourner le bot **24/7** sur un serveur Windows, connecté au
terminal **Deriv MT5 Standard** (démo d'abord). Le bot réutilise `strategies.py`
tel quel — c'est seulement l'exécution qui passe par MT5.

> ⚠️ **Windows obligatoire** : le package Python `MetaTrader5` ne fonctionne
> **pas** sur Mac/Linux. D'où le VPS Windows (ou une VM Windows).

---

## 1. Prendre un VPS Windows

- N'importe quel VPS **Windows Server 2019/2022** (ex. Contabo, Vultr, Kamatera,
  ou un « Forex VPS »). ~10–20 $/mois suffisent.
- Choisis une région proche des serveurs du broker (Europe pour Deriv) pour la latence.
- Connecte-toi en **Bureau à distance (RDP)** : `mstsc` sur Windows,
  *Microsoft Remote Desktop* sur Mac.

## 2. Préparer le VPS

Sur le VPS :

1. **Python 3.11 (64-bit)** : https://www.python.org/downloads/ → coche
   *« Add python.exe to PATH »*.
2. **Copie le projet** (au minimum `strategies.py` + `mt5_trader.py`) dans un
   dossier, ex. `C:\bot\`. (Copier-coller via RDP fonctionne.)
3. Ouvre *PowerShell* / *cmd* dans ce dossier et installe les dépendances :
   ```
   pip install MetaTrader5 pandas numpy
   ```

## 3. Installer et connecter le terminal Deriv MT5

1. Télécharge **Deriv MT5** : deriv.com → *Deriv MT5* → *Download* (Windows). Installe.
2. Récupère tes identifiants MT5 sur **home.deriv.com** → compte **MT5 CFDs Standard
   Demo** → **Details** :
   - **Login** (un numéro, ex. `12345678`)
   - **Mot de passe** (le *trading password* MT5 que tu as défini)
   - **Serveur** (ex. `Deriv-Demo`)
3. Ouvre le terminal MT5 → *Fichier → Se connecter à un compte* → saisis
   login / mot de passe / serveur. Vérifie que le solde démo (~10 000 $) s'affiche.
4. **Active l'Algo Trading** : dans la barre d'outils, le bouton **« Algo Trading »**
   doit être **vert/activé** (sinon `order_send` est refusé : *AutoTrading disabled*).
5. Ajoute **XAUUSD** au *Market Watch* (clic droit → *Symboles* → active-le) pour que
   le bot puisse lire ses cotations.
6. **Laisse le terminal OUVERT** (le package Python se connecte au terminal en cours).

## 4. Lancer le bot

Deux façons de s'authentifier :

**A. Le terminal est déjà connecté au bon compte** (le plus simple) — pas besoin de login :
```
python mt5_trader.py --symbol XAUUSD --strategy alligator-v2 --tf M1
```

**B. Laisser le bot se connecter** (utile si plusieurs comptes) :
```
python mt5_trader.py --symbol XAUUSD --strategy alligator-v2 --tf M1 ^
    --login 12345678 --password "TON_MDP_MT5" --server Deriv-Demo
```

Options utiles :
- `--risk 0.01` (1 % par trade, défaut) · `--max-dd 0.05` (coupe-circuit −5 %/jour)
- `--sessions london,ny,off` (mêmes filtres que le backtest)
- `--terminal-path "C:\Program Files\Deriv MT5\terminal64.exe"` si plusieurs terminaux
- `--server-utc-offset 0` — **à ajuster si les sessions semblent décalées** (voir §7)

Au 1er trade tu verras :
```
📶 SIGNAL SHORT [alligator-v2 sell]  lot=0.11  entry~3300.50  SL=3306.00  risque=$100.00
✅ OPEN SHORT  lot=0.11  @3300.40  SL=3306.00  TP=trailing  ticket=...
↗  Trailing : SL déplacé à 3298.10 (ticket=...)   ← le SL peut verrouiller du profit (impossible sur Multipliers)
```

## 5. Garder le bot en vie 24/7

- **Ne te DÉCONNECTE pas en « Fermer la session »** du RDP → ça tue les applis GUI.
  Utilise la croix de la fenêtre RDP (*disconnect*), qui **laisse la session tourner**.
- Le terminal MT5 **et** le script Python doivent rester ouverts. Lance le script
  dans une fenêtre cmd que tu laisses ouverte.
- Redémarrage propre après reboot : mets un raccourci du terminal MT5 dans
  *shell:startup*, et (optionnel) une tâche planifiée pour relancer le `.py`.

## 6. Surveiller / arrêter

- Les logs défilent dans la fenêtre cmd. Pour garder un historique, redirige :
  ```
  python mt5_trader.py --symbol XAUUSD --strategy alligator-v2 --tf M1 >> C:\bot\bot.log 2>&1
  ```
- **Arrêt propre** : `Ctrl+C` dans la fenêtre. Les positions déjà ouvertes restent
  protégées par leur SL côté broker.
- Le coupe-circuit ferme tout et met en pause si la perte du jour dépasse `--max-dd`.

## 7. Points de vigilance (⚠ lire avant de croire aux résultats)

- **Spread + swap** : MT5 facture le spread (bid/ask) et le swap overnight, que le
  **backtest ne modélise pas**. Sur du scalping, le spread peut redevenir décisif.
  → Compare les résultats démo aux résultats backtest sur quelques jours.
- **Commission réelle** : vérifie-la dans l'onglet *Trade/Historique* du terminal
  (colonne Commission). Pour XAUUSD Standard c'est ~2,4 $/100k = 0,0024 %.
- **Heure serveur** : la stratégie filtre les sessions en **UTC**. Si le serveur MT5
  n'est pas en UTC, les entrées london/ny seront décalées → règle `--server-utc-offset`
  (ex. serveur en GMT+2 → `--server-utc-offset 2`).
- **Symbole** : le nom exact peut varier (`XAUUSD`, parfois un suffixe). Prends celui
  affiché dans le *Market Watch*.

## 8. Passer au réel (plus tard, prudemment)

Une fois la démo validée sur **plusieurs semaines** :
1. Connecte le terminal au compte **MT5 Standard RÉEL**.
2. Relance avec `--login` du compte réel (ou terminal déjà connecté au réel).
3. Commence avec un `--risk` **plus petit** (ex. 0.005 = 0,5 %).

Le bot log `RÉEL ⚠` au démarrage quand le compte n'est pas un compte démo — vérifie
toujours cette ligne avant de laisser tourner.
