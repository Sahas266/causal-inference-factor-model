# Telegram notifications — setup

One-time setup for the model + portfolio update channel. The code side is
already wired: `causal_portfolio/execution/notify.py` sends automatically on
every live execution result, unresolved leg failure, and CPCM causal target write.

## 1. Create the bot

1. In Telegram, message **@BotFather** → `/newbot`.
2. Pick a display name (e.g. `CPCM Portfolio`) and a unique username ending
   in `bot` (e.g. `cpcm_portfolio_bot`).
3. BotFather replies with the **bot token** (`123456789:AA...`). Treat it
   like a private key — anyone with it can send as the bot.

## 2. Create the channel

1. Telegram → New Channel (private is fine), e.g. `CPCM Updates`.
2. Channel settings → Administrators → Add admin → your bot. It only needs
   the "Post messages" permission.

## 3. Get the chat id

Post any message in the channel, then:

```
curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
```

Find `"chat":{"id":-100xxxxxxxxxx,...}` in the response. Channel ids are
negative and start with `-100`.

## 4. Configure

Add to `.env` in the command's working directory or a parent (never commit):

```
TELEGRAM_BOT_TOKEN=123456789:AA...
TELEGRAM_CHAT_ID=-100xxxxxxxxxx
```

## 5. Verify

```bash
python -m causal_portfolio.execution.notify --test     # ping the channel
python -m causal_portfolio.execution.notify --state    # live testnet account summary
python -m causal_portfolio.execution.notify --pnl       # live unrealized PnL, once
python -m causal_portfolio.execution.notify --message "hello"
```

Formatted messages (state, PnL, execution results, CPCM causal targets) use bold
labels, monospace ids, and 🟢/🔴/⚠️/🛑 status emoji so a scroll through the
channel reads at a glance. `--message` stays plain text — arbitrary operator
text is never parsed as markup.

## 6. Recurring PnL updates (every 30 minutes)

Windows Task Scheduler runs `causal_portfolio/execution/run_pnl_notifier.cmd`
every 30 minutes under `CPCM_Portfolio_30m_HL_Testnet`. The wrapper is one-shot;
the scheduler owns recurrence. Each tick logs SQLite portfolio metrics, updates
the local control panel, then posts PnL and the next-rebalance countdown.
If Hyperliquid is unreachable, the tick records a health event, refreshes the
panel with the failure status, and exits non-zero instead of inventing PnL.

`--pnl-loop --interval-minutes 30` remains available for non-Windows hosts but
is not the deployed Windows path.

## What gets sent automatically

- **Execution results** (`execute_plan`): network, target id, orders, gross,
  leg-repair outcome, unresolved drifts, any error. Clean dry-runs are
  silent; live submissions and all failures notify.
- **Model updates** (`causal_daily.run_once`): target date, DAG v2 congestion
  innovation, registered holdout status, and BTC weight on every eligible target
  write. A pending or failed holdout produces no target notification. The
  one-time, wallet/reference-scoped RP-PCA retirement still uses the normal
  execution-result notification.
- **PnL updates**: the scheduled one-shot task posts every 30 minutes.

Unconfigured env = notifications silently disabled. Telegram errors are logged
and swallowed; the HTTP timeout caps notification delay at 15 seconds.
