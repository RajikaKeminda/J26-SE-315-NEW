"""
record.py

Records the payloads of a running control room (the same stream the browser gets) to a JSONL file, so that run can be
replayed later with `python -m src.dashboard.server --mode replay --replay FILE` - e.g. as a fallback in a presentation.

    python -m src.dashboard.record --out results/replays/trainticket_run.jsonl --seconds 900
"""

import argparse
import asyncio
import os
import time


async def record(url, out, seconds):
    import websockets
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    n, end = 0, time.time() + seconds
    with open(out, "a") as f:
        async with websockets.connect(url, ping_interval=None) as ws:
            while time.time() < end:
                try:
                    payload = await asyncio.wait_for(ws.recv(), timeout=max(1.0, end - time.time()))
                except asyncio.TimeoutError:
                    break
                f.write(payload + "\n")
                f.flush()
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8080/ws")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()
    print("recorded", asyncio.run(record(a.url, a.out, a.seconds)), "payloads ->", a.out)
