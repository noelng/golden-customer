"""
run_pipeline.py — Orchestrator: ETL → Agent → Report → Dashboard → Browser

This is the single entry point for the demo.
"""

import json
import logging
import os
import sqlite3
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent
DB_PATH   = ROOT / "golden_customers.db"
LOG_PATH  = ROOT / "agent_decisions.jsonl"
DASH_PATH = ROOT / "dashboard.html"
DATA_DIR  = ROOT / "data"


# ---------------------------------------------------------------------------
# Dashboard generator
# ---------------------------------------------------------------------------
def generate_dashboard(stats: dict):
    """Build a self-contained HTML dashboard and write it to DASH_PATH."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    golden_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM golden_customers ORDER BY canonical_name"
    ).fetchall()]

    agent_log_rows = [dict(r) for r in conn.execute(
        "SELECT * FROM agent_decision_log ORDER BY created_at"
    ).fetchall()]

    conn.close()

    # Load JSONL log too (in case DB write lagged)
    jsonl_rows = []
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        jsonl_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    # Merge, prefer DB
    db_decision_ids = {r["decision_id"] for r in agent_log_rows}
    for row in jsonl_rows:
        if row.get("decision_id") not in db_decision_ids:
            agent_log_rows.append(row)

    # Stats for chart
    auto_email  = stats.get("auto_merge_email", 0)
    auto_name   = stats.get("auto_merge_name", 0)
    agent_merge = stats.get("agent_merge", 0)
    agent_split = stats.get("agent_split", 0)
    new_recs    = stats.get("new_records", 0)
    true_dups   = stats.get("true_duplicates", 0)
    total_raw   = stats.get("total_raw", 90)
    agent_mode  = stats.get("agent_mode", "mock")

    sources_json = json.dumps(stats.get("sources", {}))
    golden_json  = json.dumps(golden_rows, ensure_ascii=False, default=str)
    agent_json   = json.dumps(agent_log_rows, ensure_ascii=False, default=str)
    run_time     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mode_badge = (
        '<span style="background:#16a34a;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">LIVE · gpt-4o-mini</span>'
        if agent_mode == "live" else
        '<span style="background:#d97706;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">DEMO · Mock AI</span>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Golden Customer Record — Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system,"Segoe UI",system-ui,sans-serif; background:#f0f2f5; color:#1f2328; font-size:14px; line-height:1.6; }}
  header {{ background:linear-gradient(135deg,#1e2761 0%,#0a3d62 100%); color:#fff; padding:24px 32px; }}
  header h1 {{ font-size:22px; font-weight:700; letter-spacing:-.3px; }}
  header p  {{ font-size:13px; color:#a8c8e8; margin-top:4px; }}
  .badge-row {{ display:flex; gap:10px; margin-top:12px; align-items:center; flex-wrap:wrap; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; padding:20px 32px; }}
  .kpi {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:16px; text-align:center; }}
  .kpi .num {{ font-size:32px; font-weight:800; color:#1e2761; line-height:1; }}
  .kpi .lbl {{ font-size:11px; color:#57606a; margin-top:4px; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi .sub {{ font-size:11px; color:#3b82d4; margin-top:2px; }}
  .tabs {{ display:flex; gap:0; padding:0 32px; border-bottom:2px solid #e5e7eb; background:#fff; }}
  .tab  {{ padding:12px 22px; cursor:pointer; font-weight:600; font-size:13px; color:#57606a; border-bottom:3px solid transparent; transition:all .15s; }}
  .tab.active {{ color:#1e2761; border-bottom-color:#1e2761; }}
  .tab:hover  {{ color:#1e2761; }}
  .panel {{ display:none; padding:24px 32px; }}
  .panel.active {{ display:block; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px; overflow:hidden; border:1px solid #e5e7eb; font-size:13px; }}
  th {{ background:#1e2761; color:#fff; padding:10px 12px; text-align:left; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.4px; white-space:nowrap; }}
  td {{ padding:9px 12px; border-bottom:1px solid #f0f2f5; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#f7f8fa; }}
  .search-row {{ display:flex; gap:12px; margin-bottom:14px; align-items:center; }}
  .search-row input {{ flex:1; padding:9px 14px; border:1px solid #d0d7de; border-radius:8px; font-size:13px; outline:none; }}
  .search-row input:focus {{ border-color:#1e2761; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; }}
  .badge-merge {{ background:#dcfce7; color:#16a34a; }}
  .badge-split {{ background:#fee2e2; color:#dc2626; }}
  .badge-auto  {{ background:#dbeafe; color:#1d4ed8; }}
  .badge-agent {{ background:#fef3c7; color:#d97706; }}
  .conf-bar {{ height:8px; background:#e5e7eb; border-radius:4px; overflow:hidden; min-width:60px; }}
  .conf-fill {{ height:100%; border-radius:4px; background:linear-gradient(90deg,#3b82d4,#1e2761); }}
  .chart-wrap {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:20px; margin-bottom:20px; }}
  .chart-title {{ font-weight:700; font-size:14px; color:#1e2761; margin-bottom:16px; }}
  .bar-chart {{ display:flex; flex-direction:column; gap:10px; }}
  .bar-row {{ display:flex; align-items:center; gap:10px; }}
  .bar-label {{ width:180px; font-size:12px; color:#57606a; text-align:right; flex-shrink:0; }}
  .bar-track {{ flex:1; height:22px; background:#f0f2f5; border-radius:4px; overflow:hidden; }}
  .bar-fill  {{ height:100%; border-radius:4px; display:flex; align-items:center; padding-left:8px; font-size:11px; font-weight:700; color:#fff; transition:width .6s ease; }}
  .bar-count {{ width:30px; font-size:12px; font-weight:700; color:#1f2328; }}
  .src-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }}
  .src-card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:14px; }}
  .src-card .name {{ font-size:12px; color:#57606a; text-transform:uppercase; letter-spacing:.4px; }}
  .src-card .count {{ font-size:24px; font-weight:800; color:#1e2761; }}
  .rationale {{ font-size:12px; color:#57606a; font-style:italic; max-width:300px; }}
  footer {{ text-align:center; font-size:11px; color:#57606a; padding:20px 0 30px; border-top:1px solid #e5e7eb; margin-top:10px; }}
  @media(max-width:700px) {{ .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} .tabs {{ overflow-x:auto; }} .src-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<header>
  <h1>🏦 Golden Customer Record Dashboard</h1>
  <p>Master Data Management · KYC/AML Deduplication Pipeline · Run: {run_time}</p>
  <div class="badge-row">
    {mode_badge}
    <span style="font-size:12px;color:#a8c8e8;">Total raw records: {total_raw} across 3 source systems</span>
  </div>
</header>

<div class="kpi-grid" id="kpiGrid"></div>

<div class="tabs">
  <div class="tab active" onclick="showTab('golden')">🏅 Golden Records</div>
  <div class="tab" onclick="showTab('stats')">📊 Match Statistics</div>
  <div class="tab" onclick="showTab('agent')">🤖 AI Decisions</div>
</div>

<!-- ===== TAB: GOLDEN RECORDS ===== -->
<div id="panel-golden" class="panel active">
  <div class="search-row">
    <input id="searchInput" type="text" placeholder="Search by name, email, phone, product..." oninput="filterTable()">
    <span id="recordCount" style="color:#57606a;font-size:12px;white-space:nowrap;"></span>
  </div>
  <table id="goldenTable">
    <thead>
      <tr>
        <th>#</th>
        <th>Name</th>
        <th>DOB</th>
        <th>Email</th>
        <th>Phone</th>
        <th>Products</th>
        <th>Source Systems</th>
        <th>Confidence</th>
        <th>Resolved By</th>
      </tr>
    </thead>
    <tbody id="goldenBody"></tbody>
  </table>
</div>

<!-- ===== TAB: MATCH STATISTICS ===== -->
<div id="panel-stats" class="panel">
  <div class="src-grid" id="srcGrid"></div>
  <div class="chart-wrap">
    <div class="chart-title">Deduplication Pipeline — Record Flow</div>
    <div class="bar-chart" id="pipelineChart"></div>
  </div>
  <div class="chart-wrap">
    <div class="chart-title">Match Decision Breakdown</div>
    <div class="bar-chart" id="matchChart"></div>
  </div>
</div>

<!-- ===== TAB: AI AGENT DECISIONS ===== -->
<div id="panel-agent" class="panel">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Record A</th>
        <th>Record B</th>
        <th>Name Score</th>
        <th>Email Sim.</th>
        <th>DOB Match</th>
        <th>Agent Steps</th>
        <th>Decision</th>
        <th>Confidence</th>
        <th>Rationale</th>
      </tr>
    </thead>
    <tbody id="agentBody"></tbody>
  </table>
</div>

<footer>
  <div>Golden Customer Record Pipeline · IBM Bob Hackathon Demo</div>
  <div style="margin-top:4px;color:#8b949e;">Made with IBM Bob</div>
</footer>

<script>
// ---- Data injected by Python ----
const GOLDEN  = {golden_json};
const AGENTS  = {agent_json};
const STATS   = {{
  autoEmail: {auto_email},
  autoName:  {auto_name},
  agentMerge: {agent_merge},
  agentSplit: {agent_split},
  newRecs:   {new_recs},
  trueDups:  {true_dups},
  totalRaw:  {total_raw},
  sources:   {sources_json},
}};

// ---- KPI Cards ----
const kpis = [
  {{ num: GOLDEN.length,         lbl: "Golden Records",      sub: "unified identities",  color:"#1e2761" }},
  {{ num: STATS.totalRaw,        lbl: "Raw Records",         sub: "across 3 systems",    color:"#0a3d62" }},
  {{ num: STATS.trueDups,        lbl: "True Duplicates",     sub: "auto-removed",        color:"#dc2626" }},
  {{ num: STATS.autoEmail + STATS.autoName, lbl: "Auto-Merges", sub: "rules engine",     color:"#1d4ed8" }},
  {{ num: AGENTS.length,         lbl: "AI Decisions",        sub: "agent-resolved",      color:"#d97706" }},
  {{ num: STATS.agentMerge,      lbl: "AI Merges",           sub: "uncertain resolved",  color:"#16a34a" }},
  {{ num: STATS.agentSplit,      lbl: "AI Splits",           sub: "kept separate",       color:"#7c5cd8" }},
  {{ num: GOLDEN.filter(g => JSON.parse(g.products||'[]').length > 1).length, lbl: "Multi-Product", sub: "cross-system customers", color:"#0891b2" }},
];
const grid = document.getElementById('kpiGrid');
kpis.forEach(k => {{
  grid.innerHTML += `<div class="kpi"><div class="num" style="color:${{k.color}}">${{k.num}}</div><div class="lbl">${{k.lbl}}</div><div class="sub">${{k.sub}}</div></div>`;
}});

// ---- Source cards ----
const srcGrid = document.getElementById('srcGrid');
Object.entries(STATS.sources).forEach(([src, cnt]) => {{
  const label = src.replace(/_/g,' ').replace(/\\b\\w/g, c => c.toUpperCase());
  srcGrid.innerHTML += `<div class="src-card"><div class="name">${{label}}</div><div class="count">${{cnt}}</div><div style="font-size:11px;color:#57606a;">source records</div></div>`;
}});

// ---- Pipeline chart ----
const pipelineData = [
  {{ label:"Raw Records Ingested", val: STATS.totalRaw, color:"#0a3d62" }},
  {{ label:"True Duplicates Removed", val: STATS.trueDups, color:"#dc2626" }},
  {{ label:"Email Auto-Merges", val: STATS.autoEmail, color:"#1d4ed8" }},
  {{ label:"Name+DOB Auto-Merges", val: STATS.autoName, color:"#3b82d4" }},
  {{ label:"Sent to AI Agent", val: AGENTS.length, color:"#d97706" }},
  {{ label:"Final Golden Records", val: GOLDEN.length, color:"#16a34a" }},
];
const maxPipeline = Math.max(...pipelineData.map(d => d.val));
const pChart = document.getElementById('pipelineChart');
pipelineData.forEach(d => {{
  const pct = maxPipeline > 0 ? (d.val / maxPipeline * 100) : 0;
  pChart.innerHTML += `<div class="bar-row">
    <div class="bar-label">${{d.label}}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{d.color}}">${{d.val > 0 ? d.val : ''}}</div></div>
    <div class="bar-count">${{d.val}}</div>
  </div>`;
}});

// ---- Match chart ----
const matchData = [
  {{ label:"Exact Email Match", val: STATS.autoEmail, color:"#1d4ed8" }},
  {{ label:"Name + DOB Auto-Merge", val: STATS.autoName, color:"#0891b2" }},
  {{ label:"AI Agent → MERGE", val: STATS.agentMerge, color:"#16a34a" }},
  {{ label:"AI Agent → SPLIT (kept sep.)", val: STATS.agentSplit, color:"#7c5cd8" }},
  {{ label:"New Records (no match)", val: STATS.newRecs, color:"#57606a" }},
];
const maxMatch = Math.max(...matchData.map(d => d.val));
const mChart = document.getElementById('matchChart');
matchData.forEach(d => {{
  const pct = maxMatch > 0 ? (d.val / maxMatch * 100) : 0;
  mChart.innerHTML += `<div class="bar-row">
    <div class="bar-label">${{d.label}}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{d.color}}">${{d.val > 0 ? d.val : ''}}</div></div>
    <div class="bar-count">${{d.val}}</div>
  </div>`;
}});

// ---- Golden records table ----
let allRows = GOLDEN;
function renderGolden(rows) {{
  const tbody = document.getElementById('goldenBody');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {{
    const products  = JSON.parse(r.products || '[]');
    const sourceIds = JSON.parse(r.source_ids || '{{}}');
    const prodBadges = products.map(p => `<span class="badge badge-auto" style="margin:1px;">${{p.replace(/_/g,' ')}}</span>`).join('');
    const srcIds = Object.entries(sourceIds).map(([s,id]) => `<span style="font-size:10px;color:#57606a;">${{s.split('_').map(w=>w[0].toUpperCase()+w.slice(1)).join(' ')}}: ${{id}}</span>`).join('<br>');
    const conf = parseFloat(r.confidence) || 0;
    const decBadge = r.decision_source === 'agent'
      ? '<span class="badge badge-agent">🤖 AI Agent</span>'
      : '<span class="badge badge-auto">⚡ Auto</span>';
    tbody.innerHTML += `<tr>
      <td style="color:#57606a;font-size:11px;">${{i+1}}</td>
      <td><strong>${{r.canonical_name}}</strong></td>
      <td style="white-space:nowrap;">${{r.dob || '<span style="color:#d0d7de">—</span>'}}</td>
      <td style="font-size:12px;">${{r.email || '<span style="color:#d0d7de">—</span>'}}</td>
      <td style="font-size:12px;white-space:nowrap;">${{r.phone || '<span style="color:#d0d7de">—</span>'}}</td>
      <td>${{prodBadges}}</td>
      <td style="font-size:11px;">${{srcIds}}</td>
      <td>
        <div class="conf-bar"><div class="conf-fill" style="width:${{conf*100}}%"></div></div>
        <div style="font-size:10px;color:#57606a;margin-top:2px;">${{(conf*100).toFixed(0)}}%</div>
      </td>
      <td>${{decBadge}}</td>
    </tr>`;
  }});
  document.getElementById('recordCount').textContent = `${{rows.length}} of ${{GOLDEN.length}} records`;
}}
renderGolden(allRows);

function filterTable() {{
  const q = document.getElementById('searchInput').value.toLowerCase();
  if (!q) {{ renderGolden(allRows); return; }}
  const filtered = allRows.filter(r =>
    (r.canonical_name||'').toLowerCase().includes(q) ||
    (r.email||'').toLowerCase().includes(q) ||
    (r.phone||'').toLowerCase().includes(q) ||
    (r.products||'').toLowerCase().includes(q) ||
    (r.source_ids||'').toLowerCase().includes(q)
  );
  renderGolden(filtered);
}}

// ---- Agent decisions table ----
const agentBody = document.getElementById('agentBody');
if (AGENTS.length === 0) {{
  agentBody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:#57606a;padding:30px;">No agent decisions recorded in this run.</td></tr>';
}} else {{
  AGENTS.forEach((d, i) => {{
    const decBadge = d.decision === 'MERGE'
      ? '<span class="badge badge-merge">✓ MERGE</span>'
      : '<span class="badge badge-split">✗ SPLIT</span>';
    const dobIcon = d.dob_match === 1 ? '✓' : d.dob_match === 0 ? '✗' : '—';
    const conf = parseFloat(d.confidence) || 0;
    agentBody.innerHTML += `<tr>
      <td style="color:#57606a;font-size:11px;">${{i+1}}</td>
      <td><strong style="font-size:12px;">${{d.record_a_name||''}}</strong><br><span style="font-size:10px;color:#57606a;">${{d.record_a_source||''}}: ${{d.record_a_id||''}}</span></td>
      <td><strong style="font-size:12px;">${{d.record_b_name||''}}</strong><br><span style="font-size:10px;color:#57606a;">${{d.record_b_source||''}}: ${{d.record_b_id||''}}</span></td>
      <td>
        <div class="conf-bar"><div class="conf-fill" style="width:${{(d.name_score||0)}}%;background:#3b82d4"></div></div>
        <span style="font-size:10px;color:#57606a;">${{d.name_score||0}}%</span>
      </td>
      <td>
        <div class="conf-bar"><div class="conf-fill" style="width:${{((d.email_score||0)*100).toFixed(0)}}%;background:#0891b2"></div></div>
        <span style="font-size:10px;color:#57606a;">${{((d.email_score||0)*100).toFixed(0)}}%</span>
      </td>
      <td style="text-align:center;font-size:13px;">${{dobIcon}}</td>
      <td style="text-align:center;font-size:12px;">${{d.agent_steps||0}}</td>
      <td>${{decBadge}}</td>
      <td>
        <div class="conf-bar"><div class="conf-fill" style="width:${{conf*100}}%"></div></div>
        <span style="font-size:10px;color:#57606a;">${{(conf*100).toFixed(0)}}%</span>
      </td>
      <td class="rationale">${{d.rationale||''}}</td>
    </tr>`;
  }});
}}

// ---- Tabs ----
function showTab(name) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    DASH_PATH.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s", DASH_PATH)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*62)
    print("  GOLDEN CUSTOMER RECORD — PIPELINE STARTING")
    print("="*62)

    # Clean old DB and log for a fresh demo run
    if DB_PATH.exists():
        DB_PATH.unlink()
        logger.info("Removed old database for fresh run")
    if LOG_PATH.exists():
        LOG_PATH.unlink()
        logger.info("Removed old agent log for fresh run")

    # Step 1: Initialize agent
    logger.info("Step 1/4 — Initializing Agentic AI layer...")
    sys.path.insert(0, str(ROOT))
    from agent.decision_agent import DecisionAgent
    agent = DecisionAgent(log_path=str(LOG_PATH))
    print(f"  [OK] Agent mode: {agent.mode.upper()}")
    if agent.mode == "mock":
        print("  [i]  Set OPENAI_API_KEY env var to use live gpt-4o-mini")

    # Step 2: Run ETL
    logger.info("Step 2/4 - Running ETL pipeline...")
    from etl_pipeline import run_etl
    stats = run_etl(data_dir=DATA_DIR, db_path=DB_PATH, agent=agent)
    stats["agent_mode"] = agent.mode
    print(f"  [OK] ETL complete - {stats['golden_total']} golden records created")

    # Step 3: Generate dashboard
    logger.info("Step 3/4 - Generating dashboard...")
    generate_dashboard(stats)
    print(f"  [OK] Dashboard: {DASH_PATH}")

    # Step 4: Print report
    logger.info("Step 4/4 - Generating report...")
    from report import print_report
    print_report(stats)

    # Open dashboard in browser
    dashboard_url = DASH_PATH.resolve().as_uri()
    print(f"\n  Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)
    print("  Done! [OK]\n")


if __name__ == "__main__":
    main()
