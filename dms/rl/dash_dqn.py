"""Live DQN training dashboard: run a training job from the browser, watch the
learning curve stream in, and inspect the final grade + per-room routes.

    python -m dms.rl.dash_dqn                    # http://127.0.0.1:8767/
    python -m dms.rl.dash_dqn --port 8767 --host 0.0.0.0

Serves a self-contained page (dms/rl/web/dash_template.html). POST /train
starts one DQN job in a background thread (second request while busy -> 409);
GET /state streams status/progress/final report back for the page to poll.
Only the standard-library HTTP server is used.

Device guidance: the page defaults to `auto`, which resolves to the RTX 4050
(CUDA) when available, so the GPU is always exercised. Honesty note: at the
current problem size (176-node graph, tiny MLP) the rollout loop is
Python-bound, so CPU wall-time is about equal; CUDA pays off once the
crowd-density / smoke features from the social-force and dispersion modules
scale the state and batches.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from ..geometry.loader import load_site
from ..grid.layers import DynamicOverlay
from .dqn import DQN, DQNConfig, calibration_features
from .evaluate import optimal_times
from .graph_env import GraphEvacEnv
from .graph_q import graph_walker_optimum
from .train_dqn import route_label, spawn_cells

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
TEMPLATE_PATH = Path(__file__).resolve().parent / "web" / "dash_template.html"
LIVE_PATH = Path(__file__).resolve().parents[2] / "out" / "rl_live.html"
MAX_PROGRESS = 1200


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "idle"                      # idle | running | grading | done | error
        self.busy = False
        self.progress: list[dict] = []
        self.done: dict | None = None
        self.error: str | None = None
        self.active_device: str | None = None     # device the running/last job resolved to
        self.devices = {"default": "cuda" if torch.cuda.is_available() else "cpu",
                        "cuda_available": torch.cuda.is_available(),
                        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def snapshot(self) -> dict:
        with self.lock:
            return dict(status=self.status, busy=self.busy, devices=self.devices,
                        active_device=self.active_device,
                        progress=self.progress[-MAX_PROGRESS:], done=self.done, error=self.error)


state = State()


def run_job(cfg: dict) -> None:
    with state.lock:
        state.busy = True
        state.status = "running"
        state.progress.clear()
        state.done = None
        state.error = None
        state.active_device = None
    t0 = time.monotonic()

    def on_progress(row: dict) -> None:
        with state.lock:
            state.progress.append(dict(row, t_s=round(time.monotonic() - t0, 1)))

    try:
        episodes = int(cfg.get("episodes", 4000))
        dc = DQNConfig(episodes=episodes,
                       batch_size=int(cfg.get("batch_size", 256)),
                       device=str(cfg.get("device", "cpu")),
                       seed=int(cfg.get("seed", 0)))
        block = [str(x) for x in cfg.get("block", [])]
        close_stair = [str(x) for x in cfg.get("close_stair", [])]

        site = load_site(CONFIG_PATH)
        building = site.get_building("B1")
        overlay = DynamicOverlay(door_states={x: "UNAVAILABLE" for x in block})
        env = GraphEvacEnv(site, "B1", overlay=overlay, closed_stairs=close_stair)
        t_cell = optimal_times(env.cell_env)
        t_graph = graph_walker_optimum(env)

        dqn = DQN(env, dc)
        with state.lock:
            state.active_device = str(dqn.device)
        dqn.train(progress_every=max(1, episodes // 200), on_progress=on_progress)

        with state.lock:
            state.status = "grading"
        attr = calibration_features()
        e = dqn.evaluate(t_cell, attr)
        n = np.asarray([int(s) for s in env.start_states if t_cell[s] > 0])
        design = float((t_graph[n] / t_cell[n] - 1.0).mean())
        starts = spawn_cells(env, building, t_cell)
        routes = []
        for room, s in starts.items():
            ro = dqn.greedy_rollout(s, attr)
            routes.append(dict(room=room,
                               route=route_label(env, ro.nodes),
                               time=f"{ro.time_s:.1f}s" if ro.reached_exit else None,
                               best_graph=f"{t_graph[s]:.1f}s",
                               cell_opt=f"{t_cell[s]:.1f}s"))
        with state.lock:
            state.done = dict(
                train_s=round(time.monotonic() - t0, 1),
                config=dict(episodes=dc.episodes, batch_size=dc.batch_size, device=str(dqn.device),
                            seed=dc.seed, n_attr=dqn.n_attr, state_dim=env.n_nodes + dqn.n_attr,
                            hidden=dc.hidden),
                scenario=dict(block=block, close_stair=close_stair),
                grade=dict(n_starts=e.n_starts, success=round(e.success_rate, 4),
                           within_5pct=round(e.within_5pct, 4),
                           mean_excess=round(e.mean_excess, 4),
                           p90_excess=round(e.p90_excess, 4),
                           max_excess=round(e.max_excess, 4),
                           design=round(design, 4)),
                routes=routes,
            )
            state.status = "done"
    except Exception:
        with state.lock:
            state.error = traceback.format_exc()
            state.status = "error"
    finally:
        with state.lock:
            state.busy = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    def _send(self, code: int, body, ctype: str) -> None:
        if isinstance(body, bytes):
            data = body
        elif ctype.startswith("application/json"):
            data = json.dumps(body).encode()
        else:                                   # html/text bodies go out raw
            data = str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path in ("/", "/index.html"):
            self._send(200, TEMPLATE_PATH.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif path == "/live":
            try:
                self._send(200, LIVE_PATH.read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, {"error": "out/rl_live.html missing — rebuild with: python -m dms.rl.live_page"},
                           "application/json")
        elif path == "/state":
            self._send(200, state.snapshot(), "application/json")
        else:
            self._send(404, {"error": "not found"}, "application/json")

    def do_POST(self) -> None:
        if self.path != "/train":
            return self._send(404, {"error": "not found"}, "application/json")
        with state.lock:
            if state.busy:
                return self._send(409, {"error": "training already running"}, "application/json")
        run_cfg = self._read_json()
        threading.Thread(target=run_job, args=(run_cfg,), daemon=True).start()
        self._send(200, {"started": True}, "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()
    devices = state.devices
    gpu = devices["gpu_name"] or "no CUDA GPU"
    print(f"Device check: CPU ok | CUDA {'on' if devices['cuda_available'] else 'off'} ({gpu})")
    print(f"  current scale is CPU-bound: CPU is as fast as the GPU until the "
          f"crowd/smoke features scale the state.")
    print(f"Dashboard: http://{args.host}:{args.port}/")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()