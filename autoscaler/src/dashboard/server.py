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

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
              controller="mo_td3", preference=config.DEFAULT_PREFERENCE, replay=None):
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
        STATE["engine"] = ReplayEngine(replay, agent, name)
        STATE["interval"] = 2.0 if interval is None else interval
    else:
        STATE["engine"] = LiveEngine(agent, name, shadow=shadow, controller=controller, w=preference)
        STATE["interval"] = float(config.POLL_INTERVAL_SECONDS if interval is None else interval)
    return STATE["engine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live", "replay"], default="demo")
    ap.add_argument("--replay", default=None, help="replay mode: JSONL recorded with src/dashboard/record.py")
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

    configure(args.mode, args.shadow, args.arm, args.scenario, args.interval,
              controller=args.controller, preference=args.preference, replay=args.replay)

    import uvicorn
    print(f"[server] {args.mode} mode{' (shadow)' if args.shadow else ''} "
          f"-> http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
