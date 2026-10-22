"""
CALM liquidation replay: a small local web server (Python standard library; runs need requirements.txt).

    python server.py [--port 8050]        then open http://localhost:8050

Run: the page posts its settings (model, seed index, liquidation) to /api/run. The server runs
scenario.py twice in parallel (with the liquidation, and the counterfactual: same seed, nothing
liquidated), then replay.py turns both tapes into cache/<id>/replay.json, which the page plays.

Cache: cache/<id> keeps both runs' tapes, where <id> hashes the settings and every file a run
depends on (scenario.py, protocol.py, models.py, the model's simulators/<code>, inputs/), so the
same settings replay at once and changed code runs them again. replay.json is rebuilt from the
kept tapes when replay.py changes. At start the server prepares the default settings for every
model that is not cached yet.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import protocol as P
import replay
from models import MODELS

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DEFAULTS = {"model": "extended_santafe_none", "seed_index": 0, "minutes": 15, "side": "sell", "quantity": 3712.8,
            "start_min": 5, "duration_min": 5, "slices": 60}
jobs: dict[str, dict] = {}          # id -> {"state": running/done/error, "progress": 0..1, "message": str}
lock = threading.Lock()


def files_hash(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for f in paths:
        h.update(f.relative_to(HERE).as_posix().encode() + f.read_bytes())
    return h.hexdigest()[:12]


def job_id_of(s: dict) -> str:
    """Cache key: the settings and every file the two runs depend on."""
    code = MODELS[s["model"]].code
    files = ([HERE / n for n in ("scenario.py", "protocol.py", "models.py")]
             + sorted((HERE / "simulators" / code).rglob("*.py")) + sorted((HERE / "inputs").glob("*.csv")))
    return hashlib.sha1(json.dumps([s, files_hash(files)], sort_keys=True).encode()).hexdigest()[:12]


def replay_ready(job_dir: Path) -> bool:
    """replay.json exists and was built by the current replay.py."""
    version = job_dir / "replay.version"
    return ((job_dir / "replay.json").exists() and version.exists()
            and version.read_text() == files_hash([HERE / "replay.py"]))


def clean(settings: dict) -> dict:
    """The run settings, typed and checked."""
    s = {**DEFAULTS, **{k: v for k, v in settings.items() if k in DEFAULTS}}
    s = {"model": str(s["model"]), "seed_index": int(s["seed_index"]), "minutes": float(s["minutes"]),
         "side": str(s["side"]), "quantity": float(s["quantity"]), "start_min": float(s["start_min"]),
         "duration_min": float(s["duration_min"]), "slices": int(s["slices"])}
    if s["model"] not in MODELS:
        raise ValueError(f"unknown model {s['model']}")
    if not 0 <= s["seed_index"] < P.N_SEEDS:
        raise ValueError(f"the seed index must be 0..{P.N_SEEDS - 1}")
    if s["side"] not in ("sell", "buy"):
        raise ValueError("side must be sell or buy")
    if not 0 < s["minutes"] <= P.MAX_SIM_S / 60:
        raise ValueError(f"simulated time must be between 0 and {P.MAX_SIM_S / 60:g} minutes")
    if s["quantity"] <= 0 or s["slices"] < 1 or s["duration_min"] <= 0 or s["start_min"] < 0:
        raise ValueError("quantity, slices and duration must be positive")
    if s["start_min"] + s["duration_min"] > s["minutes"]:
        raise ValueError("the liquidation must end before the simulation does")
    return s


def run_job(job_id: str, s: dict) -> None:
    """The runs not in the cache yet (in parallel), then the replay file.
    Progress comes from the kernel's progress bar."""
    job_dir = CACHE / job_id
    try:
        base = [sys.executable, str(HERE / "scenario.py"), "--model", s["model"], "--seed-index", str(s["seed_index"]),
                "--minutes", str(s["minutes"]), "--side", s["side"], "--quantity", str(s["quantity"]),
                "--start-min", str(s["start_min"]), "--duration-min", str(s["duration_min"]),
                "--slices", str(s["slices"])]
        # a run is complete once scenario.py has written its meta.json (its last file)
        progress = {name: 0.0 for name in ("liquidation", "counterfactual")
                    if not (job_dir / name / "meta.json").exists()}
        procs = []
        for name in progress:
            cmd = base + ["--out", str(job_dir / name)] + (["--counterfactual"] if name == "counterfactual" else [])
            proc = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            reader = threading.Thread(target=follow, args=(proc, job_dir / f"{name}.log", name, progress, job_id),
                                      daemon=True)
            reader.start()
            procs.append((name, proc, reader))
        for name, proc, reader in procs:
            reader.join()
            if proc.wait() != 0:
                raise RuntimeError(f"the {name} run failed, see {job_dir / f'{name}.log'}")
        with lock:
            jobs[job_id].update(progress=1.0, message="building the replay")
        (job_dir / "replay.json").write_text(json.dumps(replay.build(job_dir), separators=(",", ":")))
        (job_dir / "replay.version").write_text(files_hash([HERE / "replay.py"]))
        with lock:
            jobs[job_id].update(state="done", message="done")
    except Exception as e:  # reported to the page
        traceback.print_exc()
        with lock:
            jobs[job_id].update(state="error", message=str(e))


def follow(proc: subprocess.Popen, log_path: Path, name: str, progress: dict, job_id: str) -> None:
    """Copy a run's output to its log and read the percentage of its progress bar."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        while chunk := proc.stdout.read1(4096):
            log.write(chunk)
            pct = re.findall(rb"(\d+)%\|", chunk)
            if pct:
                with lock:
                    progress[name] = int(pct[-1]) / 100
                    jobs[job_id]["progress"] = 0.95 * sum(progress.values()) / len(progress)


def start(s: dict) -> str:
    """The job for these settings: from the cache, already running, or started now."""
    job_id = job_id_of(s)
    with lock:
        if replay_ready(CACHE / job_id):
            jobs[job_id] = {"state": "done", "progress": 1.0, "message": "cached"}
        elif jobs.get(job_id, {}).get("state") != "running":
            jobs[job_id] = {"state": "running", "progress": 0.0, "message": "simulating"}
            threading.Thread(target=run_job, args=(job_id, s), daemon=True).start()
    return job_id


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE / "static"), **kw)

    def log_message(self, fmt, *a):            # keep the console quiet
        pass

    def send_body(self, body: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code: int = 200) -> None:
        self.send_body(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        if self.path == "/api/options":
            return self.send_json({
                "models": [{"key": k, "label": m.label} for k, m in MODELS.items()],
                "defaults": DEFAULTS, "n_seeds": P.N_SEEDS, "max_minutes": P.MAX_SIM_S / 60,
                "market": {"venue": "Coinbase", "asset": "SOL", "price": P.Run(0).init_price, "daily_volume": P.V_D}})
        if self.path.startswith("/api/status/"):
            with lock:
                job = jobs.get(self.path.rsplit("/", 1)[1])
                return self.send_json(dict(job) if job else {"state": "unknown"})
        if self.path.startswith("/api/replay/"):
            f = CACHE / re.sub(r"[^0-9a-f]", "", self.path.rsplit("/", 1)[1]) / "replay.json"
            if not f.exists():
                return self.send_json({"error": "no such run"}, 404)
            return self.send_body(f.read_bytes(), "application/json")
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        try:
            s = clean(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"))
        except (ValueError, TypeError, KeyError) as e:
            return self.send_json({"error": str(e)}, 400)
        return self.send_json({"id": start(s)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CALM liquidation replay")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-prepare", action="store_true", help="do not prepare the default runs at start")
    cli = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    if not cli.no_prepare:
        for key in MODELS:
            job_id = start(clean({**DEFAULTS, "model": key}))
            print(f"{key:<28} default run {'cached' if jobs[job_id]['state'] == 'done' else 'being prepared'}")
    print(f"open http://localhost:{cli.port}")
    ThreadingHTTPServer(("127.0.0.1", cli.port), Handler).serve_forever()
