#!/usr/bin/env python3
"""
monitor.py — Live training dashboard server.
Reads E:/pretrain_checkpoints/*/train_log.jsonl and serves a dashboard at http://localhost:6006

Run:  python monitor.py
Then open browser at http://localhost:6006
"""
import json, math, pathlib, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

CHECKPOINTS_DIR = pathlib.Path("E:/pretrain_checkpoints")
PORT = 6006

ARCH_COLORS = {
    "gqa":       {"line": "#3B82F6", "fill": "rgba(59,130,246,0.15)"},
    "gqe":       {"line": "#10B981", "fill": "rgba(16,185,129,0.15)"},
    "hyper_gqe": {"line": "#8B5CF6", "fill": "rgba(139,92,246,0.15)"},
}
SEEDS = [42, 123, 456]
TOTAL_STEPS = 15259   # 500M tokens / 32768 tokens per step


def read_log(arch: str, seed: int) -> list:
    path = CHECKPOINTS_DIR / f"{arch}_seed{seed}" / "train_log.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except:
                    pass
    return rows


def read_run_config(arch: str, seed: int) -> dict:
    path = CHECKPOINTS_DIR / f"{arch}_seed{seed}" / "run_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except:
            pass
    return {}


def aggregate_data():
    data = {}
    for arch in ["gqa", "gqe", "hyper_gqe"]:
        data[arch] = {}
        for seed in SEEDS:
            rows   = read_log(arch, seed)
            config = read_run_config(arch, seed)
            train_steps = [r for r in rows if "loss" in r and "eval" not in r]
            eval_steps  = [r for r in rows if "eval" in r]

            if train_steps:
                latest = train_steps[-1]
                step = latest.get("step", 0)
                pct  = round(step / TOTAL_STEPS * 100, 1)
                elapsed = (step * 5)   # rough estimate 5s/step
                remaining = max(0, (TOTAL_STEPS - step) * 5)
                tok_seen  = latest.get("tokens_seen", step * 32768)
                speed_vals = [r.get("tokens_seen", 0) for r in train_steps[-10:]]
                speed = 0
                if len(speed_vals) >= 2 and len(train_steps) >= 2:
                    dt = (train_steps[-1].get("step",1) - train_steps[-10].get("step",1)) * 5
                    ds = speed_vals[-1] - speed_vals[0]
                    speed = int(ds / max(dt, 1))
            else:
                step = pct = elapsed = remaining = tok_seen = speed = 0

            data[arch][str(seed)] = {
                "running":     len(train_steps) > 0,
                "step":        step,
                "total_steps": TOTAL_STEPS,
                "pct":         pct,
                "tok_seen":    tok_seen,
                "speed":       speed,
                "remaining_s": remaining,
                "train": {
                    "steps":  [r.get("step", 0) for r in train_steps],
                    "loss":   [round(r.get("loss", 0), 4) for r in train_steps],
                    "lr":     [r.get("lr", 0) for r in train_steps],
                    "tokens": [r.get("tokens_seen", 0) for r in train_steps],
                },
                "eval": {
                    "steps": [r.get("step", 0) for r in eval_steps],
                    "loss":  [round(r.get("eval", {}).get("val_loss", 0), 4) for r in eval_steps],
                    "ppl":   [round(r.get("eval", {}).get("perplexity", 0), 2) for r in eval_steps],
                },
                "config": config,
            }
    return data


DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HyperGQE Pre-training Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0e1a;--bg2:#0f1629;--bg3:#141e35;--card:#1a2540;--card2:#1e2c47;
  --border:#243154;--text:#e2e8f0;--muted:#64748b;--accent:#3B82F6;
  --gqa:#3B82F6;--gqe:#10B981;--hyper:#8B5CF6;
  --success:#22c55e;--warn:#f59e0b;--err:#ef4444;
}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;overflow-x:hidden}

/* Header */
.header{
  background:linear-gradient(135deg,#0f1629 0%,#1a2540 50%,#0f1629 100%);
  border-bottom:1px solid var(--border);padding:18px 32px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;backdrop-filter:blur(8px);
}
.header-left{display:flex;align-items:center;gap:14px}
.logo{font-size:1.4rem;font-weight:700;background:linear-gradient(135deg,#3B82F6,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.live-badge{display:flex;align-items:center;gap:6px;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:20px;padding:4px 12px;font-size:0.75rem;color:#22c55e;font-weight:600}
.live-dot{width:7px;height:7px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(1.3)}}
.header-right{font-size:0.8rem;color:var(--muted)}
#last-update{color:var(--text);font-weight:500}

/* Layout */
.main{padding:24px 32px;max-width:1600px;margin:0 auto}

/* Status bar */
.status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}
.stat-card{
  background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:18px 20px;transition:transform .2s,box-shadow .2s;
}
.stat-card:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.stat-label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.stat-value{font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--text)}
.stat-sub{font-size:0.75rem;color:var(--muted);margin-top:4px}
.stat-card.blue .stat-value{color:#3B82F6}
.stat-card.green .stat-value{color:#10B981}
.stat-card.purple .stat-value{color:#8B5CF6}
.stat-card.amber .stat-value{color:#f59e0b}

/* Progress bars */
.arch-progress{margin-bottom:28px}
.arch-progress h3{font-size:0.85rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:14px}
.progress-row{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px 18px;margin-bottom:10px;display:grid;
  grid-template-columns:140px 1fr 100px;align-items:center;gap:16px;
}
.progress-arch{font-size:0.85rem;font-weight:600}
.progress-bar-wrap{background:rgba(255,255,255,0.05);border-radius:6px;height:8px;overflow:hidden}
.progress-bar{height:100%;border-radius:6px;transition:width 1s ease;position:relative}
.progress-bar::after{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);animation:shimmer 2s infinite}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.progress-pct{font-size:0.8rem;font-weight:600;font-family:'JetBrains Mono',monospace;text-align:right}

/* Charts grid */
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
.chart-card{
  background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px;
  transition:box-shadow .2s;
}
.chart-card:hover{box-shadow:0 8px 40px rgba(0,0,0,0.4)}
.chart-card.wide{grid-column:1/-1}
.chart-title{font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:8px}
.chart-title span{width:8px;height:8px;border-radius:50%;display:inline-block}
.chart-wrap{position:relative;height:220px}

/* Legend */
.legend{display:flex;gap:20px;justify-content:center;margin-top:16px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:7px;font-size:0.78rem;color:var(--muted)}
.legend-dot{width:10px;height:10px;border-radius:50%}

/* Waiting state */
.waiting{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:300px;color:var(--muted);gap:12px;
}
.waiting svg{opacity:0.3;animation:spin 3s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ETA formatter */
.eta{font-size:0.72rem;color:var(--muted)}

/* Seed tabs */
.seed-tabs{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.seed-tab{
  background:var(--bg3);border:1px solid var(--border);border-radius:8px;
  padding:5px 14px;font-size:0.78rem;cursor:pointer;transition:all .2s;
  font-family:'JetBrains Mono',monospace;
}
.seed-tab.active{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.seed-tab:hover:not(.active){background:var(--card2)}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <div class="logo">⚡ HyperGQE Monitor</div>
    <div class="live-badge"><span class="live-dot"></span>LIVE</div>
  </div>
  <div class="header-right">Last updated: <span id="last-update">—</span> &nbsp;|&nbsp; Auto-refresh: 5s</div>
</div>

<div class="main">

  <!-- Global stats -->
  <div class="status-grid" id="stat-cards">
    <div class="stat-card blue"><div class="stat-label">Active Run</div><div class="stat-value" id="sc-arch">—</div><div class="stat-sub" id="sc-seed">Waiting...</div></div>
    <div class="stat-card"><div class="stat-label">Step</div><div class="stat-value" id="sc-step">—</div><div class="stat-sub" id="sc-step-pct">of 15,259</div></div>
    <div class="stat-card green"><div class="stat-label">Train Loss</div><div class="stat-value" id="sc-loss">—</div><div class="stat-sub">current</div></div>
    <div class="stat-card purple"><div class="stat-label">Val PPL</div><div class="stat-value" id="sc-ppl">—</div><div class="stat-sub">last eval</div></div>
    <div class="stat-card amber"><div class="stat-label">Tokens Seen</div><div class="stat-value" id="sc-tok">—</div><div class="stat-sub">of 500M</div></div>
    <div class="stat-card"><div class="stat-label">Speed</div><div class="stat-value" id="sc-speed">—</div><div class="stat-sub">tok/sec</div></div>
    <div class="stat-card"><div class="stat-label">ETA</div><div class="stat-value" id="sc-eta" style="font-size:1.1rem">—</div><div class="stat-sub">this run</div></div>
  </div>

  <!-- Progress per architecture -->
  <div class="arch-progress">
    <h3>Training Progress</h3>
    <div id="progress-rows"></div>
  </div>

  <!-- Seed selector -->
  <div class="seed-tabs" id="seed-tabs">
    <div class="seed-tab active" onclick="setSeed(42)">Seed 42</div>
    <div class="seed-tab" onclick="setSeed(123)">Seed 123</div>
    <div class="seed-tab" onclick="setSeed(456)">Seed 456</div>
  </div>

  <!-- Charts -->
  <div class="charts-grid">
    <div class="chart-card wide">
      <div class="chart-title"><span style="background:#3B82F6"></span>Training Loss — All Architectures</div>
      <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span style="background:#8B5CF6"></span>Validation Perplexity</div>
      <div class="chart-wrap"><canvas id="pplChart"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span style="background:#f59e0b"></span>Learning Rate Schedule</div>
      <div class="chart-wrap"><canvas id="lrChart"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span style="background:#10B981"></span>Tokens Seen</div>
      <div class="chart-wrap"><canvas id="tokChart"></canvas></div>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#3B82F6"></div>GQA</div>
    <div class="legend-item"><div class="legend-dot" style="background:#10B981"></div>GQE</div>
    <div class="legend-item"><div class="legend-dot" style="background:#8B5CF6"></div>HyperGQE</div>
  </div>
</div>

<script>
const COLORS = {gqa:'#3B82F6', gqe:'#10B981', hyper_gqe:'#8B5CF6'};
const FILLS  = {gqa:'rgba(59,130,246,0.1)', gqe:'rgba(16,185,129,0.1)', hyper_gqe:'rgba(139,92,246,0.1)'};
const LABELS = {gqa:'GQA', gqe:'GQE', hyper_gqe:'HyperGQE'};
let activeSeed = 42;
let charts = {};
let allData = {};

function setSeed(s) {
  activeSeed = s;
  document.querySelectorAll('.seed-tab').forEach((t,i) => t.classList.toggle('active', [42,123,456][i]===s));
  if(allData) updateCharts(allData);
}

function fmtNum(n) {
  if(n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if(n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if(n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return n.toString();
}
function fmtEta(secs) {
  if(!secs) return '—';
  const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60);
  if(h > 0) return h+'h '+m+'m';
  return m+'m';
}

function mkChart(id, type, opts={}) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type, data:{datasets:[]},
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:600},
      interaction:{mode:'index', intersect:false},
      plugins:{legend:{display:false}, tooltip:{
        backgroundColor:'rgba(15,22,41,0.95)', borderColor:'#243154', borderWidth:1,
        titleColor:'#e2e8f0', bodyColor:'#94a3b8', padding:10,
      }},
      scales:{
        x:{grid:{color:'rgba(255,255,255,0.04)'}, ticks:{color:'#64748b', maxTicksLimit:8, font:{size:10}}},
        y:{grid:{color:'rgba(255,255,255,0.04)'}, ticks:{color:'#64748b', font:{size:10}}},
      },
      ...opts
    }
  });
}

function initCharts() {
  charts.loss = mkChart('lossChart','line');
  charts.ppl  = mkChart('pplChart','line');
  charts.lr   = mkChart('lrChart','line',{scales:{y:{ticks:{callback:v=>v.toExponential(1)}}}});
  charts.tok  = mkChart('tokChart','line',{scales:{y:{ticks:{callback:v=>fmtNum(v)}}}});
}

function updateCharts(data) {
  allData = data;
  const seed = String(activeSeed);

  ['loss','ppl','lr','tok'].forEach(k => {
    charts[k].data.datasets = [];
  });

  let anyRunning = false;
  let activeArch = '—', activeStep = 0, activeLoss = '—', activePpl = '—';
  let activeTok = 0, activeSpeed = 0, activeEta = 0;

  // Progress rows
  const pr = document.getElementById('progress-rows');
  pr.innerHTML = '';

  for (const [arch, seeds] of Object.entries(data)) {
    const d = seeds[seed] || {};
    const color = COLORS[arch];
    const label = LABELS[arch];
    const pct = d.pct || 0;

    // Progress row
    pr.innerHTML += `
      <div class="progress-row">
        <div class="progress-arch" style="color:${color}">${label}</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar" style="width:${pct}%;background:linear-gradient(90deg,${color},${color}cc)"></div>
        </div>
        <div class="progress-pct" style="color:${color}">${pct}%</div>
      </div>`;

    if (d.running) {
      anyRunning = true;
      if (d.step > activeStep) {
        activeArch  = label; activeStep = d.step;
        activeLoss  = d.train.loss.at(-1) || '—';
        activePpl   = d.eval.ppl.at(-1) || '—';
        activeTok   = d.tok_seen;
        activeSpeed = d.speed;
        activeEta   = d.remaining_s;
      }
    }

    // Subsample for chart performance
    const sub = (arr) => {
      if(!arr || arr.length === 0) return {x:[],y:[]};
      const step = Math.max(1, Math.floor(arr.length/200));
      const x=[], y=[];
      for(let i=0;i<arr.length;i+=step){x.push(arr[i]);}
      return x;
    };
    const trainSteps = d.train?.steps || [];
    const trainLoss  = d.train?.loss  || [];
    const trainLr    = d.train?.lr    || [];
    const trainTok   = d.train?.tokens|| [];
    const evalSteps  = d.eval?.steps  || [];
    const evalPpl    = d.eval?.ppl    || [];

    const step2 = Math.max(1,Math.floor(trainSteps.length/200));
    const ssSteps=[], ssLoss=[], ssLr=[], ssTok=[];
    for(let i=0;i<trainSteps.length;i+=step2){
      ssSteps.push(trainSteps[i]); ssLoss.push(trainLoss[i]);
      ssLr.push(trainLr[i]); ssTok.push(trainTok[i]);
    }

    if(ssSteps.length){
      charts.loss.data.datasets.push({label,data:ssSteps.map((s,i)=>({x:s,y:ssLoss[i]})),
        borderColor:color,backgroundColor:FILLS[arch],fill:true,tension:0.3,pointRadius:0,borderWidth:2});
      charts.lr.data.datasets.push({label,data:ssSteps.map((s,i)=>({x:s,y:ssLr[i]})),
        borderColor:color,backgroundColor:'transparent',tension:0.3,pointRadius:0,borderWidth:2});
      charts.tok.data.datasets.push({label,data:ssSteps.map((s,i)=>({x:s,y:ssTok[i]})),
        borderColor:color,backgroundColor:FILLS[arch],fill:true,tension:0.3,pointRadius:0,borderWidth:2});
    }
    if(evalSteps.length){
      charts.ppl.data.datasets.push({label,data:evalSteps.map((s,i)=>({x:s,y:evalPpl[i]})),
        borderColor:color,backgroundColor:FILLS[arch],fill:true,tension:0.3,pointRadius:3,borderWidth:2});
    }
  }

  charts.loss.options.scales.x.type='linear';
  charts.ppl.options.scales.x.type='linear';
  charts.lr.options.scales.x.type='linear';
  charts.tok.options.scales.x.type='linear';

  Object.values(charts).forEach(c=>c.update('none'));

  // Stats
  document.getElementById('sc-arch').textContent  = activeArch;
  document.getElementById('sc-seed').textContent  = anyRunning ? `Seed ${activeSeed}` : 'No runs active';
  document.getElementById('sc-step').textContent  = activeStep.toLocaleString();
  document.getElementById('sc-step-pct').textContent = `of ${(15259).toLocaleString()} (${((activeStep/15259)*100).toFixed(1)}%)`;
  document.getElementById('sc-loss').textContent  = typeof activeLoss === 'number' ? activeLoss.toFixed(4) : activeLoss;
  document.getElementById('sc-ppl').textContent   = typeof activePpl === 'number' ? activePpl.toFixed(2) : activePpl;
  document.getElementById('sc-tok').textContent   = fmtNum(activeTok);
  document.getElementById('sc-speed').textContent = fmtNum(activeSpeed);
  document.getElementById('sc-eta').textContent   = fmtEta(activeEta);
  document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
}

async function fetchAndUpdate() {
  try {
    const r = await fetch('/api/logs');
    if(!r.ok) return;
    const data = await r.json();
    updateCharts(data);
  } catch(e) { console.warn('Fetch failed:', e); }
}

initCharts();
fetchAndUpdate();
setInterval(fetchAndUpdate, 5000);
</script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/dashboard'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif path == '/api/logs':
            data = aggregate_data()
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    print(f"[Monitor] Reading logs from: {CHECKPOINTS_DIR}")
    print(f"[Monitor] Dashboard: http://localhost:{PORT}")
    print(f"[Monitor] Press Ctrl+C to stop\n")
    server = HTTPServer(('localhost', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Monitor] Stopped.")


if __name__ == '__main__':
    main()
