"""
FastAPI server wrapping the Incident Response RL environment.

Start with:
    uvicorn src.server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, ValidationError

from src.environment import IncidentResponseEnv, SCENARIO_MAP
from src.models import IncidentAction, IncidentObservation, IncidentState
from src.server.graders import grade

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("incident-env")

# ──────────────────────────────────────────────
# Global environment holder
# ──────────────────────────────────────────────

_env: IncidentResponseEnv | None = None


def _get_env() -> IncidentResponseEnv:
    """Return the current environment; raise 503 if not initialised."""
    if _env is None:
        raise HTTPException(
            status_code=503,
            detail="Environment not initialised. Server is starting up.",
        )
    return _env


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise the environment on startup, clean up on shutdown."""
    global _env
    task_name = os.environ.get("INCIDENT_TASK", "single_service_failure")
    logger.info("Initialising environment with task: %s", task_name)

    try:
        _env = IncidentResponseEnv(task_name)
        _env.reset()
        logger.info(
            "Environment ready  │  task=%s  episode=%s",
            _env.task_name,
            _env.episode_id,
        )
    except ValueError as exc:
        logger.error("Failed to initialise environment: %s", exc)
        raise

    yield  # ← app is running

    logger.info("Shutting down environment.")
    _env = None


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Incident Response Environment",
    description=(
        "OpenEnv-compliant RL environment for production incident response. "
        "An AI agent receives system alerts and must diagnose and remediate "
        "production failures across 3 difficulty levels."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins (needed for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Request logging middleware
# ──────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s  →  %d  (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ──────────────────────────────────────────────
# Validation error handler
# ──────────────────────────────────────────────

@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
        },
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚨 Incident Response RL Env</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0a; --surface: #111111; --card: #1a1a1a; --border: #2a2a2a;
            --accent: #ef4444; --accent-glow: rgba(239, 68, 68, 0.2);
            --green: #22c55e; --yellow: #f59e0b;
            --text-primary: #f0f0f0; --text-secondary: #888888;
            --terminal-bg: #0d1117; --terminal-text: #22c55e;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background-color: var(--bg); color: var(--text-primary); font-family: 'Inter', sans-serif; line-height: 1.6; overflow-x: hidden; scroll-behavior: smooth; }
        .container { max-width: 1100px; margin: 0 auto; padding: 0 1.5rem; }
        section { padding: 4rem 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239,68,68,0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(239,68,68,0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239,68,68,0); } }
        .fade-in { animation: fadeIn 0.6s ease-out forwards; }
        
        .hero { height: 85vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; border-bottom: 1px solid var(--border); background: radial-gradient(circle at 50% 50%, #1a0a0a 0%, #0a0a0a 100%); }
        .live-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; animation: pulse 2s infinite; display: inline-block; margin-right: 8px; }
        .badge { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.2); padding: 5px 12px; border-radius: 99px; color: var(--accent); font-size: 0.8rem; font-weight: 600; margin-bottom: 2rem; }
        .hero h1 { font-size: 3.5rem; font-weight: 800; letter-spacing: -0.05em; margin-bottom: 1rem; color: #fff; }
        .hero p { font-size: 1.1rem; color: var(--text-secondary); max-width: 600px; margin-bottom: 2rem; }
        .btn { padding: 0.8rem 1.5rem; border-radius: 6px; font-weight: 600; text-decoration: none; transition: var(--transition); cursor: pointer; border: none; display: inline-flex; align-items: center; }
        .btn-primary { background: var(--accent); color: white; box-shadow: 0 4px 15px var(--accent-glow); }
        .btn-outline { background: transparent; color: white; border: 1px solid var(--border); margin-left:10px; }
        
        .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
        .step-card { background: var(--surface); border: 1px solid var(--border); padding: 2rem; border-radius: 12px; text-align: center; }
        .step-icon { font-size: 2rem; margin-bottom: 1rem; display: block; }

        .task-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
        .task-card { background: var(--card); border: 1px solid var(--border); padding: 2rem; border-radius: 12px; transition: var(--transition); border-top: 4px solid var(--border); }
        .easy { border-top-color: var(--green); }
        .medium { border-top-color: var(--yellow); }
        .hard { border-top-color: var(--accent); }
        .dif-badge { font-size: 0.7rem; font-weight: 800; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; margin-bottom: 10px; display: inline-block; }
        .easy .dif-badge { background: rgba(34,197,94,0.1); color: var(--green); }
        .medium .dif-badge { background: rgba(245,158,11,0.1); color: var(--yellow); }
        .hard .dif-badge { background: rgba(239,68,68,0.1); color: var(--accent); }

        .demo-window { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; display: grid; grid-template-columns: 1fr 320px; height: 650px; }
        .demo-main { padding: 1.5rem; display: flex; flex-direction: column; background: #000; min-height: 0; }
        .terminal { background: var(--terminal-bg); border-radius: 8px; padding: 1rem; font-family: 'Fira Code', monospace; font-size: 0.85rem; color: var(--terminal-text); overflow-y: auto; flex-grow: 1; margin-bottom: 1rem; border: 1px solid #222; white-space: pre-wrap; word-break: break-all; }
        .demo-sidebar { padding: 1.5rem; border-left: 1px solid var(--border); display: flex; flex-direction: column; gap: 1rem; overflow-y: auto; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.8rem; }
        .reward-val { font-size: 1.2rem; font-weight: 800; color: var(--green); }
        .progress-bg { height: 6px; background: #222; border-radius: 3px; overflow: hidden; }
        .progress-fill { height: 100%; width: 0%; background: var(--green); transition: width 0.5s; }
        .action-btn { background: #1a1a1a; border: 1px solid var(--border); color: #fff; padding: 8px; border-radius: 4px; font-size: 0.75rem; cursor: pointer; text-align: left; }
        .action-btn:hover { background: #222; border-color: #444; }
        .input-box { width: 100%; background: #0a0a0a; border: 1px solid var(--border); color: #fff; padding: 8px; border-radius: 4px; font-size: 0.8rem; }
        
        .api-table { width: 100%; border-collapse: collapse; border: 1px solid var(--border); }
        .api-table th { background: var(--surface); padding: 12px; text-align: left; font-size: 0.9rem; }
        .api-table td { padding: 12px; border-top: 1px solid var(--border); font-size: 0.85rem; }
        .m-post { color: #3b82f6; font-weight: 800; }
        .m-get { color: var(--green); font-weight: 800; }

        .arch { display: flex; align-items: center; justify-content: center; gap: 20px; padding: 2rem 0; }
        .arch-node { background: var(--surface); border: 1px solid var(--border); padding: 15px; border-radius: 8px; font-size: 0.8rem; text-align: center; width: 140px; }
        .arch-env { border: 2px dashed var(--accent); width: 300px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 15px; }
        .arch-svc { background: #1a1a1a; padding: 5px; border-radius: 4px; font-size: 0.7rem; }

        @media (max-width: 850px) { .task-grid, .steps { grid-template-columns: 1fr; } .demo-window { grid-template-columns: 1fr; height: auto; } .demo-sidebar { border-left: none; border-top: 1px solid var(--border); } }
    </style>
</head>
<body>
    <section class="hero">
        <div class="container fade-in">
            <div class="badge"><span class="live-dot"></span> LIVE ENVIRONMENT</div>
            <h1>Incident Response RL</h1>
            <p>A production-grade RL environment for training SRE agents. Real services, real metrics, real incidents.</p>
            <div style="display:flex; justify-content:center; gap:10px; margin-bottom:30px;">
                <div class="arch-node" style="width:auto; padding:5px 15px;">3 Tasks</div>
                <div class="arch-node" style="width:auto; padding:5px 15px;">5 Services</div>
                <div class="arch-node" style="width:auto; padding:5px 15px;">9 Actions</div>
            </div>
            <a href="#demo" class="btn btn-primary">Try Demo →</a>
            <a href="/docs" class="btn btn-outline">API Docs</a>
        </div>
    </section>

    <div class="container">
        <section id="how">
            <div class="steps">
                <div class="step-card">
                    <span class="step-icon">🚨</span>
                    <h4>1. Alert</h4>
                    <p style="font-size:0.85rem; color:var(--text-secondary)">Agent receives a P1 alert with symptoms.</p>
                </div>
                <div class="step-card">
                    <span class="step-icon">🔍</span>
                    <h4>2. Investigate</h4>
                    <p style="font-size:0.85rem; color:var(--text-secondary)">Inspect logs and metrics to find root cause.</p>
                </div>
                <div class="step-card">
                    <span class="step-icon">🛠️</span>
                    <h4>3. Remediate</h4>
                    <p style="font-size:0.85rem; color:var(--text-secondary)">Apply fixes and declare resolution.</p>
                </div>
            </div>
        </section>

        <section id="demo">
            <h2 style="margin-bottom:2rem; text-align:center;">Interactive Demo</h2>
            <div class="demo-window">
                <div class="demo-main">
                    <div id="term" class="terminal">// Select a task and click 'Start'...</div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <div style="display:flex; gap:15px;">
                            <div style="font-size:0.75rem;"><span style="color:var(--text-secondary)">DIAGNOSED:</span> <span id="diag-stat" style="font-weight:800; color:var(--text-secondary)">NO</span></div>
                            <div style="font-size:0.75rem;"><span style="color:var(--text-secondary)">RESOLVED:</span> <span id="res-stat" style="font-weight:800; color:var(--text-secondary)">NO</span></div>
                        </div>
                        <div style="font-size:0.8rem;"><span style="color:var(--text-secondary)">REWARD:</span> <span id="rew" class="reward-val">0.00</span></div>
                    </div>
                    <div class="progress-bg"><div id="p-bar" class="progress-fill"></div></div>
                </div>
                <div class="demo-sidebar">
                    <div>
                        <label style="font-size:0.7rem; font-weight:800; color:var(--text-secondary)">SCENARIO</label>
                        <select id="t-sel" class="input-box" style="margin-top:5px;">
                            <option value="single_service_failure">Single Service (Easy)</option>
                            <option value="database_latency">DB Latency (Medium)</option>
                            <option value="cascade_failure">Cascade Failure (Hard)</option>
                        </select>
                        <button id="s-btn" class="btn btn-primary" style="width:100%; margin-top:10px; justify-content:center;">Start Incident</button>
                    </div>
                    <div id="svc-st" style="font-size:0.75rem;">
                        <label style="font-weight:800; color:var(--text-secondary)">SERVICES</label>
                        <div style="margin-top:5px;" id="svc-list"></div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:8px;">
                        <label style="font-weight:800; color:var(--text-secondary)">ACTIONS</label>
                        <input id="tg-in" class="input-box" placeholder="Target (e.g. user-service)">
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:5px;">
                            <button class="action-btn" data-a="check_all_services">Check Svc</button>
                            <button class="action-btn" data-a="check_recent_deploys">Deploys</button>
                            <button class="action-btn" data-a="read_logs">Logs</button>
                            <button class="action-btn" data-a="check_metrics">Metrics</button>
                            <button class="action-btn" data-a="check_db_queries">DB Qs</button>
                            <button class="action-btn" data-a="restart_service">Restart</button>
                            <button class="action-btn" data-a="rollback">Rollback</button>
                            <button class="action-btn" data-a="scale_up">Scale</button>
                            <button class="action-btn" data-a="declare_resolved" style="grid-column:span 2; color:var(--green)">RESOLVE</button>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section id="rewards">
            <h2 style="text-align: center; margin-bottom: 2rem;">Reward System</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                <div class="arch-node" style="width: auto; height: 140px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="font-size:0.6rem; color:var(--text-secondary)">DIAGNOSIS</span>
                    <span style="color:var(--green); font-size:1.5rem; font-weight:800;">+0.2</span>
                    <p style="font-size:0.7rem; padding:0 10px;">Correct investigative step</p>
                </div>
                <div class="arch-node" style="width: auto; height: 140px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="font-size:0.6rem; color:var(--text-secondary)">FIX</span>
                    <span style="color:var(--green); font-size:1.5rem; font-weight:800;">+0.3</span>
                    <p style="font-size:0.7rem; padding:0 10px;">Correct remediation action</p>
                </div>
                <div class="arch-node" style="width: auto; height: 140px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="font-size:0.6rem; color:var(--text-secondary)">RESOLVE</span>
                    <span style="color:var(--green); font-size:1.5rem; font-weight:800;">+1.0</span>
                    <p style="font-size:0.7rem; padding:0 10px;">Successful incident closure</p>
                </div>
                <div class="arch-node" style="width: auto; height: 140px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="font-size:0.6rem; color:var(--text-secondary)">ERROR</span>
                    <span style="color:var(--accent); font-size:1.5rem; font-weight:800;">-0.1</span>
                    <p style="font-size:0.7rem; padding:0 10px;">Wrong or repeated action</p>
                </div>
                <div class="arch-node" style="width: auto; height: 140px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="font-size:0.6rem; color:var(--text-secondary)">PREMATURE</span>
                    <span style="color:var(--accent); font-size:1.5rem; font-weight:800;">-0.2</span>
                    <p style="font-size:0.7rem; padding:0 10px;">Resolving while failing</p>
                </div>
            </div>
        </section>

        <section id="architecture">
            <h2 style="text-align:center; margin-bottom:2rem;">Architecture</h2>
            <div class="arch">
                <div class="arch-node">AI Agent</div>
                <span>→</span>
                <div class="arch-env">
                    <div class="arch-svc">api-gateway</div>
                    <div class="arch-svc">payment-service</div>
                    <div class="arch-svc">user-service</div>
                    <div class="arch-svc">db-primary</div>
                    <div class="arch-svc">cache-redis</div>
                </div>
                <span>→</span>
                <div class="arch-node">Env Observation</div>
            </div>
        </section>
    </div>

    <footer style="text-align:center; padding:3rem; border-top:1px solid var(--border); color:var(--text-secondary); font-size:0.8rem;">
        <p>Built for OpenEnv Hackathon • <a href="https://github.com/sahilshingate01/incident-response-env" style="color:var(--accent)">GitHub</a></p>
    </footer>

    <script>
        const BASE = window.location.origin;
        let cTask = ""; let score = 0; let active = false;
        
        const term = document.getElementById('term');
        const sBtn = document.getElementById('s-btn');
        const rewEl = document.getElementById('rew');
        const pFill = document.getElementById('p-bar');
        const tgIn = document.getElementById('tg-in');
        const diagStat = document.getElementById('diag-stat');
        const resStat = document.getElementById('res-stat');

        function addLog(m, c='#22c55e'){
            const d = document.createElement('div');
            d.innerHTML = `<span style="color:#444">[${new Date().toLocaleTimeString()}]</span> <span style="color:${c}">${m}</span>`;
            term.appendChild(d); term.scrollTop = term.scrollHeight;
        }

        async function refresh(){
            try {
                const r = await fetch(`${BASE}/state`);
                const s = await r.json();
                const l = document.getElementById('svc-list'); l.innerHTML = "";
                Object.entries(s.services).forEach(([k,v])=>{
                    const cl = v.status==='healthy'?'#22c55e':'#ef4444';
                    l.innerHTML += `<div style="display:flex; justify-content:space-between; margin-bottom:2px;"><span>${k}</span><span style="color:${cl}; font-weight:bold;">${v.status.toUpperCase()}</span></div>`;
                });
            } catch(e){}
        }

        sBtn.onclick = async () => {
            const t = document.getElementById('t-sel').value;
            cTask = t; term.innerHTML = ""; score = 0; active = true;
            rewEl.innerText = "0.00"; pFill.style.width = "0%";
            diagStat.innerText = "NO"; diagStat.style.color = "var(--text-secondary)";
            resStat.innerText = "NO"; resStat.style.color = "var(--text-secondary)";
            addLog(`Initializing ${t}...`, '#888');
            try {
                const r = await fetch(`${BASE}/reset/${t}`, {method:'POST'});
                const d = await r.json();
                addLog(`ALERT: ${d.observation_text}`, '#ef4444');
                await refresh();
            } catch(e){ addLog(`Error: ${e.message}`, '#ef4444'); }
        };

        document.querySelectorAll('.action-btn').forEach(b => {
            b.onclick = async () => {
                if(!active) return addLog("Start incident first!", "#ef4444");
                const a = b.getAttribute('data-a');
                const t = tgIn.value || null;
                b.disabled = true;
                try {
                    const r = await fetch(`${BASE}/step`, {
                        method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({action_type:a, target:t, task_name:cTask})
                    });
                    const d = await r.json();
                    addLog(`Action: ${a} ${t||''}`);
                    addLog(`Obs: ${d.observation_text}`, '#aaa');
                    if(d.reward!==0) addLog(`Reward: ${d.reward>0?'+':''}${d.reward}`, d.reward>0?'#22c55e':'#ef4444');
                    score += d.reward; rewEl.innerText = score.toFixed(2);
                    pFill.style.width = `${Math.min(100, Math.max(0, (score/1.5)*100))}%`;
                    
                    if(d.metadata.correctly_diagnosed) {
                        diagStat.innerText = "YES ✅";
                        diagStat.style.color = "var(--green)";
                    }
                    if(d.metadata.resolved) {
                        resStat.innerText = "YES ✅";
                        resStat.style.color = "var(--green)";
                    }

                    await refresh();
                    if(d.done){ active = false; addLog("SESSION COMPLETE", "#22c55e"); }
                } catch(e){ addLog(`Error: ${e.message}`, '#ef4444'); }
                finally { b.disabled = false; tgIn.value = ""; }
            }
        });
        refresh();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health():
    env = _get_env()
    return {
        "status": "ok",
        "task": env.task_name,
        "episode_id": env.episode_id,
    }


@app.get("/tasks")
async def list_tasks():
    return [
        {
            "name": "single_service_failure",
            "difficulty": "easy",
            "description": (
                "A bad deploy caused one service to fail. "
                "Identify the service and roll back."
            ),
        },
        {
            "name": "database_latency",
            "difficulty": "medium",
            "description": (
                "DB overload is causing API latency. "
                "Trace the root cause across 3 services and remediate."
            ),
        },
        {
            "name": "cascade_failure",
            "difficulty": "hard",
            "description": (
                "A cascading failure across 4 services. "
                "Find root cause and fix in correct order or make it worse."
            ),
        },
    ]


@app.post("/reset")
async def reset_env():
    env = _get_env()
    obs = env.reset()
    logger.info("Episode reset  │  episode=%s  task=%s", env.episode_id, env.task_name)
    return obs.model_dump()


@app.post("/reset/{task_name}")
async def reset_with_task(task_name: str):
    global _env
    if task_name not in SCENARIO_MAP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown task '{task_name}'. "
                f"Available: {list(SCENARIO_MAP.keys())}"
            ),
        )
    _env = IncidentResponseEnv(task_name)
    obs = _env.reset()
    logger.info(
        "Task switched & reset  │  task=%s  episode=%s",
        _env.task_name,
        _env.episode_id,
    )
    return obs.model_dump()


@app.post("/step")
async def step_env(action: IncidentAction):
    env = _get_env()
    obs = env.step(action)
    logger.info(
        "Step %d  │  action=%s  reward=%+.2f  done=%s",
        env.step_count,
        action.action_key(),
        obs.reward,
        obs.done,
    )
    return obs.model_dump()


@app.get("/state")
async def get_state():
    env = _get_env()
    return env.state().model_dump()


# ──────────────────────────────────────────────
# Grading endpoint
# ──────────────────────────────────────────────

class GradeRequest(BaseModel):
    task_name: str


@app.post("/grade")
async def grade_episode(req: GradeRequest):
    env = _get_env()

    if req.task_name not in SCENARIO_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task_name}'. Available: {list(SCENARIO_MAP.keys())}",
        )

    # Use current episode state for grading
    current_state = env.state()

    # Ensure we're grading the right task
    if current_state.task_name != req.task_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Current episode is task '{current_state.task_name}', "
                f"but grading was requested for '{req.task_name}'. "
                f"Switch tasks with POST /reset/{req.task_name} first."
            ),
        )

    result = grade(
        task_name=req.task_name,
        state=current_state,
        actions_taken=current_state.actions_taken,
        total_reward=env.cumulative_reward,
    )

    logger.info(
        "Graded episode  │  task=%s  score=%.4f  passed=%s",
        req.task_name,
        result["score"],
        result["passed"],
    )
    return result


# ──────────────────────────────────────────────
# Inline smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import subprocess
    import sys
    import httpx

    async def _smoke_test():
        """Quick async smoke test that runs a full episode via HTTP."""

        # Start server as a subprocess using uvicorn CLI
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.server.app:app",
                "--host", "127.0.0.1",
                "--port", "7860",
                "--log-level", "warning",
            ],
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give server time to start
        await asyncio.sleep(2.5)
        base = "http://127.0.0.1:7860"

        try:
            async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
                print("\n" + "=" * 60)
                print("  SMOKE TEST — Incident Response Environment API")
                print("=" * 60)

                # Health check
                r = await client.get("/health")
                assert r.status_code == 200, f"Health failed: {r.text}"
                print(f"\n✅ GET /health → {r.json()}")

                # List tasks
                r = await client.get("/tasks")
                assert r.status_code == 200
                tasks = r.json()
                print(f"✅ GET /tasks → {len(tasks)} tasks")
                for t in tasks:
                    print(f"   • {t['name']} ({t['difficulty']})")

                # Reset
                r = await client.post("/reset")
                assert r.status_code == 200
                obs = r.json()
                print(f"\n✅ POST /reset → step={obs['step_count']}, done={obs['done']}")
                print(f"   Alert: {obs['observation_text'][:80]}...")

                # Play a correct sequence for task 1
                task = "single_service_failure"
                steps = [
                    {"action_type": "check_recent_deploys", "target": None, "task_name": task},
                    {"action_type": "read_logs", "target": "user-service", "task_name": task},
                    {"action_type": "rollback", "target": "dep-evil-123", "task_name": task},
                    {"action_type": "declare_resolved", "target": None, "task_name": task},
                ]

                print(f"\n  Playing {len(steps)} steps for '{task}':")
                for step_data in steps:
                    r = await client.post("/step", json=step_data)
                    assert r.status_code == 200, f"Step failed: {r.text}"
                    obs = r.json()
                    print(
                        f"   Step {obs['step_count']}: "
                        f"{step_data['action_type']:25s}  "
                        f"reward={obs['reward']:+.1f}  "
                        f"done={obs['done']}"
                    )

                # State
                r = await client.get("/state")
                assert r.status_code == 200
                state = r.json()
                print(f"\n✅ GET /state → diagnosed={state['correctly_diagnosed']}, resolved={state['resolved']}")

                # Grade
                r = await client.post("/grade", json={"task_name": task})
                assert r.status_code == 200
                grade_result = r.json()
                print(f"✅ POST /grade → score={grade_result['score']}, passed={grade_result['passed']}")
                print(f"   Breakdown: {grade_result['breakdown']}")

                # Switch task
                r = await client.post("/reset/cascade_failure")
                assert r.status_code == 200
                obs = r.json()
                print(f"\n✅ POST /reset/cascade_failure → new episode started")
                print(f"   Alert: {obs['observation_text'][:80]}...")

                print(f"\n{'='*60}")
                print("  ALL SMOKE TESTS PASSED ✅")
                print(f"{'='*60}\n")

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    asyncio.run(_smoke_test())
