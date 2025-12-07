# fl_server.py — CPU-only FL server with clear switches and sticky/adaptive modes
#
# What we do here:
#   1) Wait for clients (HTTP).
#   2) Build assignments (who trains on which rows).
#   3) Send global weights + each client's shard.
#   4) Receive updated weights from each client.
#   5) Aggregate with FedAvg.   <-- swap to CUDA C++ later here.
#   6) Repeat for N rounds, then evaluate.
#
# Split modes (single knob: SPLIT_MODE):
#   - "sticky_calibrated"   : 1 warm-up timing pass, then build stratified fixed shards once (fast + stable).
#   - "sticky_equal"        : no warm-up, just equal stratified fixed shards once.
#   - "adaptive_each_round" : re-size and reshuffle every round from measured speed (closer to classic adaptive).
#

# from __future__ import annotations

import os, logging, time, threading
from typing import Dict, List, Tuple, Optional

# Force CPU + quiet logs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, request, jsonify
import numpy as np

import fl_core

app = Flask(__name__)

# ======================== Config (one place) ========================
TOTAL_ROUNDS: int = 3
LOCAL_EPOCHS: int = 1
LOCAL_BATCH:  int = 64

# One knob for how we split training data
#   "sticky_calibrated" | "sticky_equal" | "adaptive_each_round"
SPLIT_MODE: str = "adaptive_each_round"

# Minimal number of connected remote clients to start
MIN_CLIENTS: int = 4

# Warm-up (only for "sticky_calibrated")
CALIB_SAMPLES_PER_HOST: int = 4000
CALIB_MIN_SAMPLES: int = 512
SPEED_EMA_BETA: float = 0.7      # smoothing after real rounds

# Stratified shard constraints (helps FedAvg match centralized behavior)
MIN_SAMPLES_PER_HOST: int = 512
ALLOCATION_EXP: float = 1.0       # >1 gives more work to faster hosts

# Bandwidth saver for sticky modes: send shard only in round 1, clients cache for later
ENABLE_CLIENT_SHARD_CACHE: bool = True

# Let the server also train on a shard (acts like a local client)
SERVER_DOES_TRAIN: bool = True

# Aggregation choice:
#   "fedavg"  : plain FedAvg
AGG_MODE: str = "fedavg"
SERVER_MOMENTUM: float = 0.9

# Whether to use GPU-based aggregator (CUDA) instead of pure CPU
USE_GPU_AGG: bool = True


# ======================== Global in-memory state ========================
LOCK = threading.Lock()

ACTIVE_CLIENTS: Dict[str, dict] = {}        # client_id -> {'connected': True, 'last_seen': ts}
ROUND_NUM: int = 0
ROUND_IS_OPEN: bool = False
TRAINING_IS_DONE: bool = False

GLOBAL_WEIGHTS: Optional[List[np.ndarray]] = None
NUM_CLASSES: Optional[int] = None

# Per-round payloads/status/updates
SHARD_PAYLOADS: Dict[str, Tuple[Optional[str], int]] = {}  # client_id -> (b64(X,y) or None, n)
TASK_STATUS: Dict[str, dict] = {}                           # client_id -> {'status','n','start','train_s'}
CLIENT_UPDATES: Dict[str, Tuple[List[np.ndarray], int]] = {}

# Measured speed (samples/sec) by host id (including 'server')
SAMPLES_PER_SEC: Dict[str, float] = {}

# When sticky, we precompute one split and reuse it every round
STICKY_SPLITS: Optional[Dict[str, np.ndarray]] = None


# ======================== Data prep once ========================
print("[SRV] prepping data...")
X_train_all, X_test, y_train_all, y_test, num_classes = fl_core.load_data()
y_train_int = np.argmax(y_train_all, axis=1)  # used for stratified splits


def _build_global_model():
    return fl_core.build_model(num_classes)


def _ensure_global_weights():
    """
    Build model once and grab initial weights (and momentum buffer if using FedAvgM).
    """
    global GLOBAL_WEIGHTS, SERVER_VELOCITY
    m = _build_global_model()
    fl_core.prime_model(m)
    GLOBAL_WEIGHTS = m.get_weights()

# ======================== Split helpers ========================
def _split_even_indices(participants: List[str], n_total: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Even split without stratification. Used only for warm-up timing.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X_train_all.shape[0])[:n_total]
    chunks = np.array_split(idx, len(participants))
    return {pid: chunk for pid, chunk in zip(participants, chunks)}


def _sizes_from_speed(participants: List[str]) -> Dict[str, int]:
    """
    Convert measured speed (samples/sec) into per-host sizes that sum to the full training set.
    Faster hosts get more rows.
    """
    n_total = X_train_all.shape[0]
    v = np.array([max(SAMPLES_PER_SEC.get(pid, 1.0), 1e-6) for pid in participants], dtype=np.float64)
    v = np.power(v, ALLOCATION_EXP)    # optional nonlinearity
    props = v / v.sum()

    sizes = (props * n_total).astype(int)
    sizes = np.maximum(sizes, MIN_SAMPLES_PER_HOST)

    # Fix rounding so the sizes add to n_total exactly
    while sizes.sum() > n_total:
        j = int(np.argmax(sizes)); sizes[j] -= 1
    while sizes.sum() < n_total:
        j = int(np.argmax(props)); sizes[j] += 1
    return {pid: int(sz) for pid, sz in zip(participants, sizes)}


def _build_stratified_indices(participants: List[str], sizes: Dict[str, int], seed: int) -> Dict[str, np.ndarray]:
    """
    Build stratified splits once so label proportions per host are close to global.
    This usually makes FedAvg behave more like centralized.
    """
    rng = np.random.default_rng(seed)

    # Per-class shuffled pools
    per_class: Dict[int, np.ndarray] = {}
    for c in np.unique(y_train_int):
        ids = np.where(y_train_int == c)[0]
        per_class[c] = rng.permutation(ids)

    # Global proportions
    class_counts = {c: len(per_class[c]) for c in per_class}
    total = sum(class_counts.values())
    class_prop = {c: class_counts[c] / total for c in class_counts}

    out: Dict[str, np.ndarray] = {}
    remain = {c: per_class[c] for c in per_class}

    for pid in participants:
        target = sizes[pid]
        want = {c: int(round(target * class_prop[c])) for c in class_prop}
        # Fix rounding so sum == target
        diff = target - sum(want.values())
        order = sorted(class_prop.keys(), key=lambda c: class_prop[c], reverse=True)
        k = 0
        while diff != 0:
            c = order[k % len(order)]
            want[c] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            k += 1

        # Pull rows from each class pool
        taken = []
        for c in want:
            cnt = min(want[c], len(remain[c]))
            if cnt > 0:
                taken.append(remain[c][-cnt:])
                remain[c] = remain[c][:-cnt]
        out[pid] = np.concatenate(taken) if taken else np.array([], dtype=np.int64)

    # Hand leftovers (if any) to the largest shard so nothing is wasted
    leftovers = np.concatenate(list(remain.values())) if any(len(remain[c]) for c in remain) else np.array([], dtype=np.int64)
    if leftovers.size > 0:
        big = max(participants, key=lambda p: out[p].size)
        out[big] = np.concatenate([out[big], leftovers])

    # Unique/sorted for safety
    return {pid: np.unique(out[pid]) for pid in participants}


def _update_speed_from_task_status():
    """
    After each round (or calibration), update samples/sec from timings in TASK_STATUS.
    """
    with LOCK:
        for cid, st in TASK_STATUS.items():
            n = max(1, int(st.get("n", 1)))
            t = max(1e-6, float(st.get("train_s", 0.0)))
            thr = n / t
            prev = SAMPLES_PER_SEC.get(cid)
            if prev is None:
                SAMPLES_PER_SEC[cid] = thr
            else:
                # EMA smooth so one noisy round doesn't swing the allocation too much
                SAMPLES_PER_SEC[cid] = SPEED_EMA_BETA * prev + (1.0 - SPEED_EMA_BETA) * thr


# ======================== HTTP endpoints (clients call these) ========================
@app.post("/register")
def register():
    cid = (request.json or {}).get("client_id")
    if not cid:
        return jsonify({"ok": False, "error": "missing client_id"}), 400
    with LOCK:
        ACTIVE_CLIENTS[cid] = {"connected": True, "last_seen": time.time()}
    return jsonify({"ok": True, "round": ROUND_NUM})


@app.get("/pull_task")
def pull_task():
    """
    Clients poll here to get "train on this shard using these global weights".
    """
    cid = request.args.get("client_id")
    if not cid:
        return jsonify({"ok": False, "error": "missing client_id"}), 400

    with LOCK:
        if cid in ACTIVE_CLIENTS:
            ACTIVE_CLIENTS[cid]["last_seen"] = time.time()

        # Finish signal
        if TRAINING_IS_DONE:
            return jsonify({"ok": True, "status": "done", "round": ROUND_NUM})

        # If round isn't ready yet or this client has nothing pending
        if not ROUND_IS_OPEN or cid not in TASK_STATUS:
            return jsonify({"ok": True, "status": "wait", "round": ROUND_NUM})

        # Another poll while waiting is fine
        if TASK_STATUS[cid]["status"] != "pending":
            return jsonify({"ok": True, "status": "wait", "round": ROUND_NUM})

        # Global weights must exist
        if GLOBAL_WEIGHTS is None or len(GLOBAL_WEIGHTS) == 0:
            _ensure_global_weights()

        # Mark as claimed and build the payload
        TASK_STATUS[cid]["status"] = "claimed"
        TASK_STATUS[cid]["start"]  = time.perf_counter()
        weights_b64 = fl_core.weights_to_b64(GLOBAL_WEIGHTS)

        shard_b64, n_samples = SHARD_PAYLOADS[cid]

        # In sticky modes, after round 1, we can skip resending the shard if cache is enabled.
        send_shard_now = True
        if ENABLE_CLIENT_SHARD_CACHE and SPLIT_MODE in ("sticky_calibrated", "sticky_equal") and ROUND_NUM > 1:
            send_shard_now = False
            shard_b64 = None

    print(f"[SRV][R{ROUND_NUM}] -> {cid}: n={n_samples}{'' if send_shard_now else ' (cached shard)'}")
    return jsonify({
        "ok": True, "status": "ready", "round": ROUND_NUM,
        "weights_b64": weights_b64, "shard_b64": shard_b64, "n_samples": n_samples
    })


@app.post("/submit_update")
def submit_update():
    """
    Clients post back their updated weights and the number of rows they trained on.
    """
    p = request.json or {}
    cid = p.get("client_id"); rnd = p.get("round")
    weights_b64 = p.get("weights_b64"); n_samples = int(p.get("n_samples", 0))
    if not all([cid, weights_b64]) or rnd is None:
        return jsonify({"ok": False, "error": "missing fields"}), 400

    updated_weights = fl_core.weights_from_b64(weights_b64)

    with LOCK:
        if rnd != ROUND_NUM:
            return jsonify({"ok": True, "note": "stale round ignored"})
        if cid not in TASK_STATUS:
            return jsonify({"ok": True, "note": "unknown client this round (ignored)"})
        if TASK_STATUS[cid]["status"] == "done":
            return jsonify({"ok": True, "note": "duplicate ignored"})

        TASK_STATUS[cid]["status"]  = "done"
        TASK_STATUS[cid]["train_s"] = (time.perf_counter() - TASK_STATUS[cid].get("start", time.perf_counter()))
        CLIENT_UPDATES[cid] = (updated_weights, n_samples)
        t_s = TASK_STATUS[cid]["train_s"]

    print(f"[SRV][R{ROUND_NUM}] <- {cid}: n={n_samples}, train={t_s:.2f}s")
    return jsonify({"ok": True})


# ======================== Server local training ========================
def _server_train_on_shard(start_weights, X, y, epochs=1, batch=LOCAL_BATCH):
    """
    Server trains on its own shard like a client, then reports back into CLIENT_UPDATES.
    """
    m = _build_global_model()
    fl_core.prime_model(m)
    m.set_weights(start_weights)

    t0 = time.perf_counter()
    m.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)
    train_s = time.perf_counter() - t0

    upd = m.get_weights()
    with LOCK:
        TASK_STATUS["server"]["status"]  = "done"
        TASK_STATUS["server"]["train_s"] = train_s
        CLIENT_UPDATES["server"] = (upd, X.shape[0])
    print(f"[SRV][R{ROUND_NUM}] (server) done: n={X.shape[0]}, train={train_s:.2f}s")


# ======================== Calibration (only for sticky_calibrated) ========================
def _run_one_time_calibration(participants: List[str]) -> None:
    """
    One warm-up timing pass. We do NOT change global weights from this pass.
    Purpose: measure samples/sec per host so we can size sticky shards once.
    """
    global ROUND_NUM, ROUND_IS_OPEN, SHARD_PAYLOADS, TASK_STATUS, CLIENT_UPDATES

    print("[SRV] calibration start...")

    per_host = max(CALIB_SAMPLES_PER_HOST, CALIB_MIN_SAMPLES)
    total    = min(per_host * len(participants), X_train_all.shape[0])

    parts = _split_even_indices(participants, n_total=total, seed=999)

    with LOCK:
        SHARD_PAYLOADS = {}
        TASK_STATUS    = {}
        CLIENT_UPDATES.clear()
        for pid in participants:
            idxs = parts[pid]
            Xp, yp = X_train_all[idxs], y_train_all[idxs]
            SHARD_PAYLOADS[pid] = (fl_core.arrays_to_b64(Xp, yp), int(Xp.shape[0]))
            TASK_STATUS[pid]    = {"status": "pending", "n": int(Xp.shape[0])}
        ROUND_IS_OPEN = True
        ROUND_NUM = 0  # log as round 0

    # Server can participate in calibration too
    if SERVER_DOES_TRAIN:
        Xs, ys = fl_core.arrays_from_b64(SHARD_PAYLOADS["server"][0])
        with LOCK:
            TASK_STATUS["server"]["status"] = "claimed"
            TASK_STATUS["server"]["start"]  = time.perf_counter()
        threading.Thread(
            target=_server_train_on_shard,
            args=(GLOBAL_WEIGHTS, Xs, ys, 1, LOCAL_BATCH),
            daemon=True
        ).start()

    # Wait for everyone (in calibration, we expect all non-server participants; include server if enabled)
    while True:
        with LOCK:
            done = sum(1 for s in TASK_STATUS.values() if s["status"] == "done")
            # need = len(TASK_STATUS) if SERVER_DOES_TRAIN else (len(TASK_STATUS) - 1)
            need = len(TASK_STATUS)
        if done >= need:
            break
        time.sleep(0.2)

    _update_speed_from_task_status()
    with LOCK:
        ROUND_IS_OPEN = False
        print("[SRV] calibration done | " + " | ".join(
            f"{cid}: thr={SAMPLES_PER_SEC[cid]:.1f} samp/s" for cid in participants
            if cid in SAMPLES_PER_SEC
        ))


# ======================== Aggregation (swap to CUDA here later) ========================
def _aggregate(updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Aggregate client updates into a new set of global weights.
    """
    if not updates:
        raise ValueError("No client updates provided to _aggregate()")

    # Choose backend for FedAvg (CPU vs GPU)
    if USE_GPU_AGG:
        # GPU implementation 
        fedavg_weights = fl_core.fedavg_weighted_average_gpu(updates)
    else:
        # Pure NumPy implementation (CPU)
        fedavg_weights = fl_core.fedavg_weighted_average(updates)

    # Algorithm choice 
    if AGG_MODE == "fedavg":
        # Plain FedAvg: just return the averaged weights
        return fedavg_weights
    else:
        raise ValueError(f"Unknown AGG_MODE: {AGG_MODE}")



# ======================== Orchestration loop ========================
def run_rounds():
    global ROUND_NUM, ROUND_IS_OPEN, GLOBAL_WEIGHTS, SHARD_PAYLOADS, TASK_STATUS, TRAINING_IS_DONE, STICKY_SPLITS

    _ensure_global_weights()

    # Wait for enough clients before we start
    print("[SRV] waiting for clients...")
    t_wait = time.perf_counter()
    while True:
        with LOCK:
            ready = [c for c, info in ACTIVE_CLIENTS.items() if info.get("connected")]
        if len(ready) >= MIN_CLIENTS:
            break
        time.sleep(0.3)
    print(f"[SRV] clients ready: {len(ready)} in {time.perf_counter() - t_wait:.2f}s")

    with LOCK:
        participants = [*ACTIVE_CLIENTS.keys()]
        if SERVER_DOES_TRAIN:
            participants.append("server")

    # --- Precompute splits if we are sticky ---
    if SPLIT_MODE == "sticky_calibrated":
        _run_one_time_calibration(participants)
        sizes = _sizes_from_speed(participants)
        STICKY_SPLITS = _build_stratified_indices(participants, sizes, seed=1234)
    elif SPLIT_MODE == "sticky_equal":
        equal = {pid: X_train_all.shape[0] // len(participants) for pid in participants}
        # hand out remainder rows
        rem = X_train_all.shape[0] - sum(equal.values())
        for j in range(rem):
            equal[participants[j % len(participants)]] += 1
        STICKY_SPLITS = _build_stratified_indices(participants, equal, seed=1234)
        # assume equal speed initially
        for pid in participants:
            SAMPLES_PER_SEC[pid] = 1.0
    else:
        # adaptive_each_round
        for pid in participants:
            SAMPLES_PER_SEC[pid] = 1.0
        STICKY_SPLITS = None

    if STICKY_SPLITS is not None:
        print("[SRV] sticky shard sizes: " + " | ".join(f"{p}: {STICKY_SPLITS[p].size}" for p in participants))

    t_total = time.perf_counter()

    # --- Main FL loop ---
    for r in range(TOTAL_ROUNDS):
        with LOCK:
            ROUND_NUM = r + 1
            TASK_STATUS = {}
            CLIENT_UPDATES.clear()

        # 1) Build per-round payloads
        if STICKY_SPLITS is not None:
            # Sticky: reuse the same rows every round
            with LOCK:
                SHARD_PAYLOADS = {}
                for pid in participants:
                    idxs = STICKY_SPLITS[pid]
                    Xp, yp = X_train_all[idxs], y_train_all[idxs]
                    SHARD_PAYLOADS[pid] = (fl_core.arrays_to_b64(Xp, yp), int(Xp.shape[0]))
                    TASK_STATUS[pid]    = {"status": "pending", "n": int(Xp.shape[0])}
                ROUND_IS_OPEN = True
                start_weights_this_round = GLOBAL_WEIGHTS
        else:
            # Adaptive: compute sizes from current speeds → make a fresh stratified split
            sizes_now   = _sizes_from_speed(participants)
            splits_now  = _build_stratified_indices(participants, sizes_now, seed=1000 + ROUND_NUM)
            with LOCK:
                SHARD_PAYLOADS = {}
                for pid in participants:
                    idxs = splits_now[pid]
                    Xp, yp = X_train_all[idxs], y_train_all[idxs]
                    SHARD_PAYLOADS[pid] = (fl_core.arrays_to_b64(Xp, yp), int(Xp.shape[0]))
                    TASK_STATUS[pid]    = {"status": "pending", "n": int(Xp.shape[0])}
                ROUND_IS_OPEN = True
                start_weights_this_round = GLOBAL_WEIGHTS

        round_t0 = time.perf_counter()
        print(f"\n[SRV][R{ROUND_NUM}] start")

        # 2) Server trains on its own shard (if enabled)
        if SERVER_DOES_TRAIN:
            Xs, ys = fl_core.arrays_from_b64(SHARD_PAYLOADS["server"][0])
            with LOCK:
                TASK_STATUS["server"]["status"] = "claimed"
                TASK_STATUS["server"]["start"]  = time.perf_counter()
            threading.Thread(
                target=_server_train_on_shard,
                args=(start_weights_this_round, Xs, ys, LOCAL_EPOCHS, LOCAL_BATCH),
                daemon=True
            ).start()

        # 3) Wait for all participants of this round
        while True:
            with LOCK:
                done = sum(1 for s in TASK_STATUS.values() if s["status"] == "done")
                # need = len(TASK_STATUS) if SERVER_DOES_TRAIN else (len(TASK_STATUS) - 1)
                need = len(TASK_STATUS)
            if done >= need:
                break
            time.sleep(0.2)

        # 4) Aggregate updates (this is where CUDA C++ will drop in)
        t_agg = time.perf_counter()
        with LOCK:
            updates_list = list(CLIENT_UPDATES.values())
            new_weights  = _aggregate(updates_list)  # <-- CUDA aggregator will replace this
            GLOBAL_WEIGHTS = new_weights
            ROUND_IS_OPEN  = False
            summary = " | ".join(f"{cid}:n={st['n']},t={st.get('train_s',0):.2f}s" for cid, st in TASK_STATUS.items())
        agg_s = time.perf_counter() - t_agg

        print(f"[SRV][R{ROUND_NUM}] agg={agg_s:.2f}s | round={time.perf_counter() - round_t0:.2f}s")
        print(f"[SRV][R{ROUND_NUM}] hosts {{ {summary} }}")

        # 5) Refresh speed estimates so adaptive mode stays accurate (harmless for sticky)
        _update_speed_from_task_status()

    with LOCK:
        TRAINING_IS_DONE = True

    # Final evaluation (same held-out test split as centralized)
    total_s = time.perf_counter() - t_total
    m = _build_global_model(); fl_core.prime_model(m); m.set_weights(GLOBAL_WEIGHTS)
    loss, acc = m.evaluate(X_test, y_test, verbose=0)
    print(f"\n[SRV] done | rounds={TOTAL_ROUNDS} | total={total_s/60:.2f} min | acc={acc:.4f}")


# ======================== Entrypoint ========================
def _run_api():
    # threaded=True lets multiple clients hit the server concurrently
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

if __name__ == "__main__":
    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    run_rounds()
