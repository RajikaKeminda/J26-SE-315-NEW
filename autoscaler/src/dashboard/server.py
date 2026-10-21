"""
server.py

Live dashboard backend (FastAPI + WebSocket). The control engines it serves live in
src/inference/engines.py.

Two modes:
  --mode demo   runs against the built-in calibrated simulator. No cluster needed -
                useful for development, demos and presentations.
  --mode live   reads real Prometheus / Kubernetes state and applies real scaling and
                resizing actions (or only logs them with --shadow).

Both modes emit the SAME payload shape, so the frontend is identical.

    python -m src.dashboard.server --mode demo
    open http://localhost:8080

One background task ticks the engine and broadcasts each payload to every connected
WebSocket client, so opening N browser tabs does not run the control loop N times.

HTTP API
    GET  /healthz                    liveness (always 200 once the app is up)
    GET  /api/state                  mode, controller, preference, paused, scenario, policy, error
    GET  /api/results                ablation / baseline comparison (comparison panel)
    GET  /api/pareto                 preference-sweep Pareto front
    GET  /api/calibration            simulator-vs-real-cluster fit report
    GET  /api/training               learning curves of the trained arms
    GET  /api/clusters               monitored clusters stored in the history database (names only)
    GET  /api/history/{what}         steps | metrics | actions | violations | critical-path | topology | events | summary | runs
                                     of ONE cluster (?cluster=<name or id>, default: the one being controlled; ?hours=24, ?service=, ?limit=)
    POST /api/pause | /api/resume    pause / resume the control loop
    POST /api/reset                  restart the episode (demo) / counters (live)
    POST /api/scenario/{name}        demo only: burst | diurnal | rampup | normal | random
    POST /api/controller/{name}      mo_td3 | hpa | vpa | threshold | static
    POST /api/preference             body {"w": [slo, cost, balance]}  (re-weights the agent live)
    WS   /ws                         payload stream
"""

import argparse
import asyncio
import glob
import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.inference.engines import CONTROLLERS, DemoEngine, LiveEngine, load_agent

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SCENARIOS = ["burst", "diurnal", "rampup", "normal", "random"]

STATE = {
    "mode": "demo", "paused": False, "agent": None, "policy_name": None,
    "engine": None, "latest": None, "error": None, "scenario": "burst",
    "interval": 1.0, "clients": set(), "lock": threading.Lock(),
}


# ---------------------------------------------------------------------------
# Background ticker + routes
# ---------------------------------------------------------------------------
async def _broadcast(payload):
    dead = []
    for ws in list(STATE["clients"]):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE["clients"].discard(ws)


def _locked_tick():
    with STATE["lock"]:
        return STATE["engine"].tick()


async def _ticker():
    while True:
        if not STATE["paused"] and STATE["engine"] is not None:
            try:
                payload = await asyncio.to_thread(_locked_tick)
                payload["paused"] = False
                STATE["latest"], STATE["error"] = payload, None
                await _broadcast(payload)
            except Exception as e:                        # keep the loop alive
                STATE["error"] = f"{type(e).__name__}: {e}"
                print("[ticker] error:", STATE["error"])
        await asyncio.sleep(STATE["interval"])


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_ticker())
    try:
        yield
    finally:
        task.cancel()
        if STATE["engine"] is not None:
            await asyncio.to_thread(STATE["engine"].close)         # flushes the history to the database


app = FastAPI(title="Topology-Aware Autoscaler — Control Room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/state")
async def state():
    eng = STATE["engine"]
    return {
        "mode": STATE["mode"], "paused": STATE["paused"],
        "scenario": STATE["scenario"] if STATE["mode"] == "demo" else None,
        "scenarios": SCENARIOS if STATE["mode"] == "demo" else [],
        "controllers": CONTROLLERS if STATE["agent"] is not None else [c for c in CONTROLLERS if c != "mo_td3"],
        "controller": getattr(eng, "controller_name", None),
        "preference": getattr(eng, "w", list(config.DEFAULT_PREFERENCE)),
        "objectives": list(config.OBJECTIVES),
        "policy": STATE["policy_name"] or "none",
        "shadow": bool(getattr(eng, "shadow", False)),
        "step": getattr(eng, "step_i", 0),
        "error": STATE["error"],
        "persistence": eng.persistence_status() if hasattr(eng, "persistence_status") else {"enabled": False, "state": "off"},
    }


def _read_json(path):
    if not os.path.exists(path):
        return {"available": False}
    with open(path) as f:
        return {"available": True, "results": json.load(f)}


@app.get("/api/results")
async def results():
    """Serves the controller comparison (ablation study) for the dashboard's panel."""
    return _read_json(config.RESULTS_PATH)


@app.get("/api/pareto")
async def pareto():
    """Serves the preference-sweep Pareto front (see src/evaluation/pareto.py)."""
    return _read_json(config.PARETO_PATH)


@app.get("/api/calibration")
async def calibration():
    """Serves the simulator fit report (see src/simulation/calibration.py)."""
    return _read_json(config.SIM_PARAMS_PATH)


@app.get("/api/training")
async def training():
    curves = {}
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "training", "*.json"))):
        with open(path) as f:
            blob = json.load(f)
        curves[os.path.splitext(os.path.basename(path))[0]] = blob.get("history", [])
    return {"available": bool(curves), "results": curves}


# ---------------------------------------------------------------------------
# History (PostgreSQL, src/persistence). Sync handlers: FastAPI runs them in its thread pool, so a slow query never blocks the event loop.
# ---------------------------------------------------------------------------
def _history_conn():
    if not config.DATABASE_URL:
        raise HTTPException(503, "history is off: set AUTOSCALER_DATABASE_URL (see docs/PERSISTENCE.md)")
    from src.persistence import db
    try:
        return db.connect(config.DATABASE_URL, autocommit=True)
    except Exception as e:
        raise HTTPException(503, f"history database unreachable: {type(e).__name__}") from e


def _has_schema(conn):
    """False on a database the agent has not initialised yet (it creates the tables on its first start)."""
    return conn.execute("SELECT to_regclass('clusters') IS NOT NULL").fetchone()[0]


def _current_cluster_id():
    eng = STATE["engine"]
    w = getattr(getattr(eng, "sink", None), "writer", None)
    return str(w.cluster_id) if w is not None and w.cluster_id else None


@app.get("/api/clusters")
def clusters():
    from src.persistence import queries
    with _history_conn() as conn:
        return {"current": _current_cluster_id(), "clusters": queries.clusters(conn) if _has_schema(conn) else []}


@app.get("/api/history/{what}")
def history(what: str, cluster: str | None = None, hours: float = Query(24.0, gt=0, le=24 * 400), service: str | None = None,
            outcome: str | None = None, changed_only: bool = False, limit: int = Query(2000, ge=1, le=20000)):
    from datetime import datetime, timedelta, timezone
    from src.persistence import queries
    kinds = ["steps", "metrics", "actions", "violations", "critical-path", "topology", "events", "summary", "runs"]
    if what not in kinds:
        raise HTTPException(404, f"unknown history {what!r}; choose from {kinds}")
    with _history_conn() as conn:
        cid = (queries.resolve_cluster(conn, cluster) if _has_schema(conn) else None) if cluster else _current_cluster_id()
        if not cid:
            raise HTTPException(404, "no such cluster (or none is being controlled): pick one from /api/clusters")
        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=hours)
        args = (conn, cid, since, until)
        if what == "steps":
            data = queries.steps(*args, limit=limit)
        elif what == "metrics":
            data = queries.metrics(*args, service=service, limit=limit)
        elif what == "actions":
            data = queries.actions(*args, service=service, outcome=outcome, limit=limit)
        elif what == "violations":
            data = queries.violations(*args, limit=limit)
        elif what == "critical-path":
            data = queries.critical_paths(*args, changed_only=changed_only, limit=limit)
        elif what == "topology":
            data = queries.topology(*args, limit=limit)
        elif what == "events":
            data = queries.events(*args, limit=limit)
        elif what == "runs":
            data = queries.runs(conn, cid, limit)
        else:
            data = queries.summary(*args)
    return {"cluster": cid, "hours": hours, "data": data}


@app.post("/api/pause")
async def pause():
    STATE["paused"] = True
    return {"paused": True}


@app.post("/api/resume")
async def resume():
    STATE["paused"] = False
    return {"paused": False}


@app.post("/api/reset")
async def reset():
    with STATE["lock"]:
        STATE["engine"].reset()
    STATE["latest"] = None
    return {"reset": True}


@app.post("/api/scenario/{name}")
async def scenario(name: str):
    if STATE["mode"] != "demo":
        raise HTTPException(400, "scenarios are only available in demo mode")
    if name not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario {name!r}; choose from {SCENARIOS}")
    with STATE["lock"]:
        STATE["engine"].reset(name)
    STATE["scenario"] = name
    STATE["latest"] = None
    return {"scenario": name}


@app.post("/api/controller/{name}")
async def controller(name: str):
    if name not in CONTROLLERS:
        raise HTTPException(404, f"unknown controller {name!r}; choose from {CONTROLLERS}")
    try:
        with STATE["lock"]:
            STATE["engine"].set_controller(name)
            if STATE["mode"] == "demo":
                STATE["engine"].reset()
    except ValueError as e:
        raise HTTPException(400, str(e))
    STATE["latest"] = None
    return {"controller": name}


@app.post("/api/preference")
async def preference(body: dict = Body(...)):
    w = body.get("w")
    if not (isinstance(w, list) and len(w) == config.N_OBJECTIVES
            and all(isinstance(x, (int, float)) for x in w) and sum(w) > 0):
        raise HTTPException(400, f"body must be {{\"w\": [{', '.join(config.OBJECTIVES)}]}} (non-negative numbers)")
    with STATE["lock"]:
        STATE["engine"].set_preference(w)
    return {"preference": STATE["engine"].w}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    STATE["clients"].add(websocket)
    try:
        if STATE["latest"] is not None:
            await websocket.send_json(STATE["latest"])
        while True:                      # ticker pushes; we just keep the socket open
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[ws] error:", e)
    finally:
        STATE["clients"].discard(websocket)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def configure(mode="demo", shadow=False, arm=None, scenario="burst", interval=None, seed=None,
              controller="mo_td3", preference=config.DEFAULT_PREFERENCE, replay=None, replay_frames=None):
    """Builds the engine and stores it in STATE (used by main() and tests)."""
    agent, name = load_agent(arm)
    STATE.update(mode=mode, paused=False, latest=None, error=None, scenario=scenario,
                 agent=agent, policy_name=name)
    if mode == "demo":
        STATE["engine"] = DemoEngine(agent, name, scenario=scenario, seed=seed,
                                     controller=controller, w=preference)
        STATE["interval"] = 1.0 if interval is None else interval
    elif mode == "replay":
        from src.inference.engines import ReplayEngine
        STATE["engine"] = ReplayEngine(replay, agent, name, frames=replay_frames)
        STATE["interval"] = 2.0 if interval is None else interval
    else:
        STATE["engine"] = LiveEngine(agent, name, shadow=shadow, controller=controller, w=preference)
        STATE["interval"] = float(config.POLL_INTERVAL_SECONDS if interval is None else interval)
    return STATE["engine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live", "replay"], default="demo")
    ap.add_argument("--replay", default=None, help="replay mode: JSONL recorded with src/dashboard/record.py")
    ap.add_argument("--replay-db", default=None, metavar="CLUSTER",
                    help="replay mode: play back the frames stored in the history database for this cluster (AUTOSCALER_STORE_FRAMES=1)")
    ap.add_argument("--replay-hours", type=float, default=24.0, help="with --replay-db: how far back")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--shadow", action="store_true",
                    default=os.environ.get("AUTOSCALER_SHADOW", "0") == "1",
                    help="live mode: compute and show decisions but never scale "
                         "(or set AUTOSCALER_SHADOW=1)")
    ap.add_argument("--arm", default=None, help="serve models/<arm>.pt (default mo_td3_full)")
    ap.add_argument("--controller", choices=CONTROLLERS, default=os.environ.get("AUTOSCALER_CONTROLLER", "mo_td3"))
    ap.add_argument("--preference", type=float, nargs=config.N_OBJECTIVES, default=list(config.DEFAULT_PREFERENCE),
                    metavar=tuple(o.upper() for o in config.OBJECTIVES))
    ap.add_argument("--scenario", choices=SCENARIOS, default="burst", help="demo traffic")
    ap.add_argument("--interval", type=float, default=None, help="seconds between ticks")
    args = ap.parse_args()
    if args.mode == "replay" and not (args.replay or args.replay_db):
        ap.error("--mode replay needs --replay FILE (record one with python -m src.dashboard.record) or --replay-db CLUSTER")
    frames = None
    if args.mode == "replay" and args.replay_db:
        from src.persistence.replay import load_frames
        frames = load_frames(args.replay_db, args.replay_hours)

    configure(args.mode, args.shadow, args.arm, args.scenario, args.interval,
              controller=args.controller, preference=args.preference, replay=args.replay, replay_frames=frames)

    import uvicorn
    print(f"[server] {args.mode} mode{' (shadow)' if args.shadow else ''} "
          f"-> http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
