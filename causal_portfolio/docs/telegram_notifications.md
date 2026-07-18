# Telegram notifications — setup

One-time setup for the model + portfolio update channel. The code side is
already wired: `causal_portfolio/execution/notify.py` sends automatically on
every live execution result, unresolved leg failure, and RP-PCA target write.

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

Add to repo-root `.env` (never commit):

```
TELEGRAM_BOT_TOKEN=123456789:AA...
TELEGRAM_CHAT_ID=-100xxxxxxxxxx
```

## 5. Verify

```bash
python -m causal_portfolio.execution.notify --test     # ping the channel
python -m causal_portfolio.execution.notify --state    # live testnet account summary
python -m causal_portfolio.execution.notify --message "hello"
```

## What gets sent automatically

- **Execution results** (`execute_plan`): network, target id, orders, gross,
  leg-repair outcome, unresolved drifts, any error. Clean dry-runs are
  silent; live submissions and all failures notify.
- **Model updates** (`rppca_daily.run_once`): target date + weight vector on
  every target write.

Unconfigured env = notifications silently disabled; a Telegram outage never
blocks or fails an execution run (send errors are logged and swallowed).
