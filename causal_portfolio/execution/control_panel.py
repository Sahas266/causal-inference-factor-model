"""Generate the local framework-free execution control panel."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from causal_portfolio.execution.types import AccountState

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CPCM Execution</title>
<style>
:root{color-scheme:dark;--bg:216 27% 6%;--surface:216 25% 11%;--text:216 48% 94%;--muted:215 18% 66%;--border:216 19% 23%;--accent:216 100% 71%;--positive:153 55% 60%;--negative:354 100% 71%;font:16px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;background:hsl(var(--bg));color:hsl(var(--text));-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{max-width:1100px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;gap:16px;align-items:end;border-bottom:1px solid hsl(var(--border));padding-bottom:16px}h1,h2{text-wrap:balance;margin:0 0 8px}.muted{color:hsl(var(--muted))}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:16px 0}.card{background:hsl(var(--surface));border:1px solid hsl(var(--border));border-radius:6px;padding:12px}.value{font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;background:hsl(var(--surface))}th,td{text-align:right;padding:12px 8px;border-bottom:1px solid hsl(var(--border));font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}th{color:hsl(var(--muted))}.positive{color:hsl(var(--positive))}.negative{color:hsl(var(--negative))}.status{padding:12px;border-left:3px solid hsl(var(--accent));background:hsl(var(--surface));overflow-wrap:anywhere}.skip{position:absolute;left:-9999px}.skip:focus{left:8px;top:8px;background:hsl(var(--surface));padding:12px;z-index:1}:focus-visible{outline:2px solid hsl(var(--accent));outline-offset:2px}@media(max-width:650px){body{padding:12px}header{display:block}}
</style>
</head>
<body>
<a class="skip" href="#main-content">Skip to main content</a>
<header><div><h1>CPCM Execution</h1><div id="identity" class="muted"></div></div><div id="countdown"></div></header>
<main id="main-content">
<section aria-labelledby="metrics-title"><h2 id="metrics-title">Portfolio</h2><div id="metrics" class="grid"></div></section>
<section aria-labelledby="weights-title"><h2 id="weights-title">Weights</h2><div class="table-wrap"><table><caption class="skip">Target weights compared with current portfolio weights</caption><thead><tr><th scope="col">Asset</th><th scope="col">Target weight</th><th scope="col">Actual weight</th><th scope="col">Mid</th><th scope="col">Position</th><th scope="col">Unrealized PnL</th></tr></thead><tbody id="assets"></tbody></table></div></section>
<section aria-labelledby="status-title"><h2 id="status-title">Latest execution</h2><div id="status" class="status" aria-live="polite"></div></section>
</main>
<script id="panel-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('panel-data').textContent);const $=id=>document.getElementById(id);const money=v=>v==null?'—':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(v);const pct=v=>v==null?'—':`${(v*100).toFixed(2)}%`;const metric=(label,value)=>{const box=document.createElement('div');box.className='card';const name=document.createElement('div');name.className='muted';name.textContent=label;const val=document.createElement('div');val.className='value';val.textContent=value;box.append(name,val);return box};const meta=data.meta||{},m=data.metrics||{};$('identity').textContent=`${meta.strategy||'unknown'} · target ${meta.target_id||'unknown'} · updated ${m.updated_at||'never'}`;[['Equity',money(m.equity_usd)],['Unrealized PnL',money(m.unrealized_pnl_usd)],['Free margin',money(m.free_margin_usd)],['Gross exposure',money(m.gross_exposure_usd)],['Net exposure',money(m.net_exposure_usd)]].forEach(x=>$('metrics').append(metric(...x)));for(const a of data.assets||[]){const tr=document.createElement('tr');[a.coin,pct(a.target_weight),pct(a.actual_weight),money(a.mid_price),money(a.position_usd),money(a.unrealized_pnl_usd)].forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i===5&&a.unrealized_pnl_usd!=null)td.className=a.unrealized_pnl_usd>=0?'positive':'negative';tr.append(td)});$('assets').append(tr)}const next=meta.expected_next_rebalance?new Date(meta.expected_next_rebalance):null;if(!next||Number.isNaN(next.valueOf())){$('countdown').textContent='Next rebalance: unknown'}else{const delta=next-Date.now();$('countdown').textContent=delta<=0?`Next rebalance: OVERDUE (${next.toISOString()})`:`Next rebalance: ${Math.floor(delta/3600000)}h ${Math.floor((delta%3600000)/60000)}m (${next.toISOString()})`}$('status').textContent=Object.keys(data.status||{}).length?JSON.stringify(data.status):'No execution recorded';
</script>
</body>
</html>
"""


def _default_output_dir() -> Path:
    configured = os.environ.get("CPCM_EXECUTION_CONTROL_PANEL_DIR")
    return Path(configured) if configured else Path.home() / ".cpcm-execution" / "control-panel"


def write_control_panel(
    state: AccountState,
    mids: dict[str, float],
    *,
    log_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Atomically replace index.html with the latest active-cycle snapshot."""
    from causal_portfolio.execution.trace import panel_data

    data = panel_data(log_dir)
    if not data["metrics"]:
        notionals = [position.notional_usd for position in state.positions.values()]
        data["metrics"] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "equity_usd": state.account_value_usd,
            "margin_used_usd": state.margin_used_usd,
            "free_margin_usd": state.free_margin_usd,
            "gross_exposure_usd": sum(abs(value) for value in notionals),
            "net_exposure_usd": sum(notionals),
        }
        data["assets"] = [
            {
                "coin": coin,
                "ticker": coin.lower(),
                "mid_price": mids.get(coin),
                "target_weight": None,
                "actual_weight": (
                    position.notional_usd / state.account_value_usd
                    if state.account_value_usd
                    else None
                ),
                "position_size": position.size,
                "position_usd": position.notional_usd,
                "unrealized_pnl_usd": (
                    (mids[coin] - position.entry_px) * position.size
                    if coin in mids
                    else None
                ),
            }
            for coin, position in sorted(state.positions.items())
        ]
    serialized = json.dumps(
        data, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).replace("<", "\\u003c")
    directory = output_dir or _default_output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "index.html"
    temporary = directory / "index.html.tmp"
    temporary.write_text(_TEMPLATE.replace("__DATA__", serialized), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
