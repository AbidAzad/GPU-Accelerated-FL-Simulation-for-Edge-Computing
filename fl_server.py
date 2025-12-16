# fl_server.py — FL server with sticky/adaptive splits, optional GPU aggregation,
# early stopping, and dataset-specific metrics (cyber + network traffic)

from __future__ import annotations

import os, logging, time, threading, zlib
from typing import Dict, List, Tuple, Optional

# Force CPU for TF + quiet logs (CUDA is only used in our custom aggregator)
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, request, jsonify
import numpy as np

import fl_core

app = Flask(__name__)

# ======================== Config (one place) ========================
TOTAL_ROUNDS: int = 100
LOCAL_EPOCHS: int = 2
LOCAL_BATCH:  int = 64

# One knob for how we split training data
#   "sticky_calibrated" | "sticky_equal" | "adaptive_each_round"
SPLIT_MODE: str = "sticky_calibrated"

# Data distribution across clients:
#   "iid_stratified"     : each client sees roughly the global label mix (IID-ish)
#   "noniid_dirichlet"   : label-skew Non-IID via Dirichlet allocation (lower alpha => more skew)
#   "noniid_label_shards": classic label-shards (each client gets a few label-heavy shards)
DATA_DISTRIBUTION: str = "iid_stratified"

# ---- Non-IID knobs ----
# Dirichlet concentration (typical: 0.1–1.0). Smaller => each client sees fewer classes.
NONIID_DIRICHLET_ALPHA: float = 0.3

# For label-shards mode: how many shards per client (each shard is label-heavy).
NONIID_LABEL_SHARDS_PER_CLIENT: int = 2

# ---------------- Knowledgeable Client Insertion (KCI) ----------------
# KCI inserts 1+ "knowledgeable" clients that train on a larger (or full) slice of the
# *global* training set to counteract the accuracy drop as K grows.

KCI_ENABLED: bool = True
KCI_CLIENTS: Tuple[str, ...] = ("NANO",)   # ids must be in `participants` (e.g., "server", "client1", ...)
KCI_LAMBDA: float = 0.50                      # fraction of full training set given to each KCI client (0<λ<=1)
KCI_STRATIFIED: bool = True                  # keep the KCI shard label-mix close to global
KCI_VERBOSE: bool = True

# Minimal number of connected remote clients to start
MIN_CLIENTS: int = 9

# Warm-up (only for "sticky_calibrated")
CALIB_SAMPLES_PER_HOST: int = 5000
CALIB_MIN_SAMPLES: int = 512
SPEED_EMA_BETA: float = 0.7      # smoothing after real rounds

# Stratified shard constraints (helps FedAvg match centralized behavior)
MIN_SAMPLES_PER_HOST: int = 512
ALLOCATION_EXP: float = 1.0       # >1 gives more work to faster hosts

# Bandwidth saver for sticky modes: send shard only in round 1, clients cache for later
ENABLE_CLIENT_SHARD_CACHE: bool = True

# Let the server also train on a shard (acts like a local client)
SERVER_DOES_TRAIN: bool = False

# Aggregation choice:
#   "fedavg"   : plain FedAvg (with optional GPU backend)
#   "fedadam"  : FedAvg (CPU/GPU) to get an averaged model, then a *server-side*
#               Adam step on the FedAvg delta (FedOpt / FedAdam)
AGG_MODE: str = "fedavg"
SERVER_MOMENTUM: float = 0.9  # reserved for future FedAvgM / momentum variants

# ---- FedOpt / FedAdam server hyperparams (used when AGG_MODE == "fedadam") ----
FEDOPT_LR: float = 0.3
FEDOPT_BETA1: float = 0.9
FEDOPT_BETA2: float = 0.999
FEDOPT_EPS: float = 1e-8
FEDOPT_WEIGHT_DECAY: float = 0.0

# Whether to use GPU-based aggregator (CUDA) instead of pure CPU NumPy
USE_GPU_AGG: bool = True  # safer default; flip to True once libfedavg_gpu is ready

# TEST AGGREGATION BENCHMARK FLAG
TEST_BENCHMARK: bool = True  # only can be done with GPU servers

# ---------------- Early stopping (server-side) ----------------
EARLY_STOP_ENABLED: bool    = True    # flip to False to disable
EARLY_STOP_PATIENCE: int    = 15       # rounds with no improvement before stopping
EARLY_STOP_MIN_DELTA: float = 1e-4    # required improvement to reset patience
EARLY_STOP_MONITOR: str     = "acc"   # "acc" or "loss"
EARLY_STOP_MODE: str        = "max"   # "max" for acc, "min" for loss


# ======================== Global in-memory state ========================
LOCK = threading.Lock()

ACTIVE_CLIENTS: Dict[str, dict] = {}        # client_id -> {'connected': True, 'last_seen': ts}
ROUND_NUM: int = 0
ROUND_IS_OPEN: bool = False
TRAINING_IS_DONE: bool = False

GLOBAL_WEIGHTS: Optional[List[np.ndarray]] = None
NUM_CLASSES: Optional[int] = None

# Server-side optimizer state for FedOpt/FedAdam (persistent across rounds)
SERVER_ADAM_M: Optional[List[np.ndarray]] = None
SERVER_ADAM_V: Optional[List[np.ndarray]] = None
SERVER_ADAM_T: int = 0

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
    Build model once and grab initial weights.
    """
    global GLOBAL_WEIGHTS
    m = _build_global_model()
    fl_core.prime_model(m)
    GLOBAL_WEIGHTS = m.get_weights()


# ======================== Split helpers ========================
def _split_even_indices(participants: List[str], n_total: int,
                        seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Even split without stratification. Used only for warm-up timing.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X_train_all.shape[0])[:n_total]
    chunks = np.array_split(idx, len(participants))
    return {pid: chunk for pid, chunk in zip(participants, chunks)}



def _kci_sample_indices(n_samples: int, seed: int, stratified: bool = True) -> np.ndarray:
    """Sample indices from the *global* training set for a knowledgeable client.

    This intentionally allows overlap with other clients (the "knowledgeable" client is assumed
    to obtain additional data through outside channels, per the KCI framing).
    """
    n_total = int(X_train_all.shape[0])
    n = int(max(1, min(n_total, n_samples)))
    rng = np.random.default_rng(seed)

    if not stratified:
        return np.sort(rng.choice(n_total, size=n, replace=False).astype(np.int64))

    # Stratified sample to approximate global label distribution
    classes = np.unique(y_train_int)
    per_class = {int(c): np.where(y_train_int == c)[0] for c in classes}
    class_counts = {c: len(per_class[c]) for c in per_class}
    total = float(sum(class_counts.values()))
    class_prop = {c: (class_counts[c] / total) for c in class_counts}

    want = {c: int(round(n * class_prop[c])) for c in class_prop}
    diff = n - sum(want.values())
    order = sorted(class_prop.keys(), key=lambda c: class_prop[c], reverse=True)
    k = 0
    while diff != 0:
        c = order[k % len(order)]
        want[c] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        k += 1

    parts = []
    for c in order:
        cnt = max(0, int(want[c]))
        if cnt == 0:
            continue
        pool = per_class[c]
        replace = cnt > len(pool)
        parts.append(rng.choice(pool, size=cnt, replace=replace))

    out = np.concatenate(parts).astype(np.int64) if parts else np.array([], dtype=np.int64)
    rng.shuffle(out)
    return np.sort(out)


def _apply_kci(splits: Dict[str, np.ndarray], participants: List[str], seed: int) -> Dict[str, np.ndarray]:
    """Apply KCI by overriding selected participants' shards to size ~= λ*|D_train|."""
    if not KCI_ENABLED:
        return splits

    out = dict(splits)  # shallow copy
    n_total = int(X_train_all.shape[0])

    for pid in KCI_CLIENTS:
        if pid not in participants:
            continue
        # stable per-id seed (Python hash is randomized per process)
        pid_seed = (zlib.crc32(pid.encode("utf-8")) & 0x7fffffff)
        n_kci = int(round(float(KCI_LAMBDA) * n_total))
        n_kci = max(1, min(n_total, n_kci))
        out[pid] = _kci_sample_indices(
            n_samples=n_kci,
            seed=int(seed) + pid_seed,
            stratified=bool(KCI_STRATIFIED),
        )

    if KCI_VERBOSE:
        msg = " | ".join(
            f"{p}: {out[p].size}{' (KCI)' if (KCI_ENABLED and p in KCI_CLIENTS) else ''}"
            for p in participants
        )
        print("[SRV] KCI shard sizes: " + msg)

    return out


def _sizes_from_speed(participants: List[str]) -> Dict[str, int]:
    """
    Convert measured speed (samples/sec) into per-host sizes that sum to the full training set.
    Faster hosts get more rows.
    """
    n_total = X_train_all.shape[0]
    v = np.array(
        [max(SAMPLES_PER_SEC.get(pid, 1.0), 1e-6) for pid in participants],
        dtype=np.float64,
    )
    v = np.power(v, ALLOCATION_EXP)    # optional nonlinearity
    props = v / v.sum()

    sizes = (props * n_total).astype(int)
    sizes = np.maximum(sizes, MIN_SAMPLES_PER_HOST)

    # Fix rounding so the sizes add to n_total exactly
    while sizes.sum() > n_total:
        j = int(np.argmax(sizes))
        sizes[j] -= 1
    while sizes.sum() < n_total:
        j = int(np.argmax(props))
        sizes[j] += 1
    return {pid: int(sz) for pid, sz in zip(participants, sizes)}


def _build_stratified_indices(participants: List[str],
                              sizes: Dict[str, int],
                              seed: int) -> Dict[str, np.ndarray]:
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
        order = sorted(class_prop.keys(),
                       key=lambda c: class_prop[c],
                       reverse=True)
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
        out[pid] = (np.concatenate(taken)
                    if taken else np.array([], dtype=np.int64))

    # Hand leftovers (if any) to the largest shard so nothing is wasted
    leftovers = (np.concatenate(list(remain.values()))
                 if any(len(remain[c]) for c in remain)
                 else np.array([], dtype=np.int64))
    if leftovers.size > 0:
        big = max(participants, key=lambda p: out[p].size)
        out[big] = np.concatenate([out[big], leftovers])

    # Unique/sorted for safety
    return {pid: np.unique(out[pid]) for pid in participants}


def _build_noniid_dirichlet_indices(participants: List[str],
                                    sizes: Dict[str, int],
                                    seed: int,
                                    alpha: float) -> Dict[str, np.ndarray]:
    """
    Non-IID (label-skew) split using a Dirichlet distribution.

    For each class, we draw proportions across clients from Dirichlet(alpha),
    then allocate that class's samples accordingly, respecting each client's
    target size (sizes[pid]). Lower alpha => more skew.
    """
    rng = np.random.default_rng(seed)

    # Prepare per-class shuffled pools
    classes = np.unique(y_train_int)
    per_class: Dict[int, np.ndarray] = {}
    for c in classes:
        ids = np.where(y_train_int == c)[0]
        per_class[int(c)] = rng.permutation(ids)

    # Remaining capacity per client (we respect sizes[] as much as possible)
    remaining: Dict[str, int] = {pid: int(sizes[pid]) for pid in participants}
    out_lists: Dict[str, List[np.ndarray]] = {pid: [] for pid in participants}
    unassigned_parts: List[np.ndarray] = []

    k_clients = len(participants)
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("NONIID_DIRICHLET_ALPHA must be > 0.")

    alpha_vec = np.full(k_clients, alpha, dtype=np.float64)

    for c in classes:
        ids = per_class[int(c)]
        n = int(ids.size)
        if n == 0:
            continue

        # Dirichlet proportions and integer allocations summing to n
        p = rng.dirichlet(alpha_vec)
        alloc = np.floor(p * n).astype(int)

        # Fix rounding so alloc.sum() == n
        diff = n - int(alloc.sum())
        if diff != 0:
            order = np.argsort(p)[::-1]
            j = 0
            step = 1 if diff > 0 else -1
            while diff != 0:
                alloc[order[j % k_clients]] += step
                diff -= step
                j += 1

        # Allocate in the order of participants (respecting remaining capacity)
        cursor = 0
        for i, pid in enumerate(participants):
            want = int(alloc[i])
            if want <= 0:
                continue
            chunk = ids[cursor:cursor + want]
            cursor += want

            cap = remaining[pid]
            if cap <= 0:
                unassigned_parts.append(chunk)
                continue

            take = min(want, cap)
            if take > 0:
                out_lists[pid].append(chunk[:take])
                remaining[pid] -= take
            if take < want:
                unassigned_parts.append(chunk[take:])

        if cursor < n:
            unassigned_parts.append(ids[cursor:])

    # Fill any remaining capacity using unassigned leftovers (shuffled)
    if unassigned_parts:
        pool = np.concatenate(unassigned_parts)
        pool = rng.permutation(pool)
    else:
        pool = np.array([], dtype=np.int64)

    pool_cursor = 0
    for pid in participants:
        need = int(remaining[pid])
        if need <= 0:
            continue
        take = min(need, int(pool.size - pool_cursor))
        if take > 0:
            out_lists[pid].append(pool[pool_cursor:pool_cursor + take])
            pool_cursor += take
            remaining[pid] -= take

    # Safety: if anything is still unfilled (shouldn't happen if sizes sum correctly),
    # just top-up from a fresh permutation of all indices.
    if any(remaining[pid] > 0 for pid in participants):
        all_idx = rng.permutation(np.arange(X_train_all.shape[0], dtype=np.int64))
        used = set()
        for pid in participants:
            if out_lists[pid]:
                used.update(np.concatenate(out_lists[pid]).tolist())
        all_idx = np.array([i for i in all_idx.tolist() if i not in used], dtype=np.int64)

        cursor = 0
        for pid in participants:
            need = int(remaining[pid])
            if need <= 0:
                continue
            take = min(need, int(all_idx.size - cursor))
            if take > 0:
                out_lists[pid].append(all_idx[cursor:cursor + take])
                cursor += take
                remaining[pid] -= take

    out: Dict[str, np.ndarray] = {}
    for pid in participants:
        if out_lists[pid]:
            out[pid] = np.unique(np.concatenate(out_lists[pid]).astype(np.int64))
        else:
            out[pid] = np.array([], dtype=np.int64)
    return out


def _build_noniid_label_shards_indices(participants: List[str],
                                       seed: int,
                                       shards_per_client: int) -> Dict[str, np.ndarray]:
    """
    Non-IID (label-shards) split.

    We sort training indices by label, then cut into (K * num_clients) shards.
    Each client gets `shards_per_client` random shards. This creates strong
    label skew per client.
    """
    rng = np.random.default_rng(seed)
    shards_per_client = int(shards_per_client)
    if shards_per_client <= 0:
        raise ValueError("NONIID_LABEL_SHARDS_PER_CLIENT must be >= 1.")

    # Sort indices by label so contiguous chunks are label-heavy
    idx = np.arange(X_train_all.shape[0], dtype=np.int64)
    idx = idx[np.argsort(y_train_int)]

    n_clients = len(participants)
    n_shards = n_clients * shards_per_client
    shards = np.array_split(idx, n_shards)
    rng.shuffle(shards)

    out: Dict[str, np.ndarray] = {}
    s = 0
    for pid in participants:
        take = shards[s:s + shards_per_client]
        s += shards_per_client
        out[pid] = np.unique(np.concatenate(take).astype(np.int64))
    return out


def _build_train_indices(participants: List[str],
                         sizes: Dict[str, int],
                         seed: int) -> Dict[str, np.ndarray]:
    """
    One wrapper so the rest of the server can switch between IID-ish and Non-IID
    data splits by changing DATA_DISTRIBUTION only.
    """
    mode = DATA_DISTRIBUTION.strip().lower()

    if mode == "iid_stratified":
        return _build_stratified_indices(participants, sizes, seed)

    if mode == "noniid_dirichlet":
        return _build_noniid_dirichlet_indices(
            participants, sizes, seed, NONIID_DIRICHLET_ALPHA
        )

    if mode == "noniid_label_shards":
        return _build_noniid_label_shards_indices(
            participants, seed, NONIID_LABEL_SHARDS_PER_CLIENT
        )

    raise ValueError(f"Unknown DATA_DISTRIBUTION: {DATA_DISTRIBUTION}")


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
                SAMPLES_PER_SEC[cid] = (
                    SPEED_EMA_BETA * prev +
                    (1.0 - SPEED_EMA_BETA) * thr
                )


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
            return jsonify({"ok": True,
                            "status": "done",
                            "round": ROUND_NUM})

        # If round isn't ready yet or this client has nothing pending
        if not ROUND_IS_OPEN or cid not in TASK_STATUS:
            return jsonify({"ok": True,
                            "status": "wait",
                            "round": ROUND_NUM})

        # Another poll while waiting is fine
        if TASK_STATUS[cid]["status"] != "pending":
            return jsonify({"ok": True,
                            "status": "wait",
                            "round": ROUND_NUM})

        # Global weights must exist
        if GLOBAL_WEIGHTS is None or len(GLOBAL_WEIGHTS) == 0:
            _ensure_global_weights()

        # Mark as claimed and build the payload
        TASK_STATUS[cid]["status"] = "claimed"
        TASK_STATUS[cid]["start"] = time.perf_counter()
        weights_b64 = fl_core.weights_to_b64(GLOBAL_WEIGHTS)

        shard_b64, n_samples = SHARD_PAYLOADS[cid]

        # In sticky modes, after round 1, we can skip resending the shard if cache is enabled.
        send_shard_now = True
        if (ENABLE_CLIENT_SHARD_CACHE and
                SPLIT_MODE in ("sticky_calibrated", "sticky_equal") and
                ROUND_NUM > 1):
            send_shard_now = False
            shard_b64 = None

    print(f"[SRV][R{ROUND_NUM}] -> {cid}: n={n_samples}"
          f"{'' if send_shard_now else ' (cached shard)'}")
    return jsonify({
        "ok": True,
        "status": "ready",
        "round": ROUND_NUM,
        "weights_b64": weights_b64,
        "shard_b64": shard_b64,
        "n_samples": n_samples,
    })


@app.post("/submit_update")
def submit_update():
    """
    Clients post back their updated weights and the number of rows they trained on.
    """
    p = request.json or {}
    cid = p.get("client_id")
    rnd = p.get("round")
    weights_b64 = p.get("weights_b64")
    n_samples = int(p.get("n_samples", 0))

    if not all([cid, weights_b64]) or rnd is None:
        return jsonify({"ok": False, "error": "missing fields"}), 400

    updated_weights = fl_core.weights_from_b64(weights_b64)

    with LOCK:
        if rnd != ROUND_NUM:
            return jsonify({"ok": True, "note": "stale round ignored"})
        if cid not in TASK_STATUS:
            return jsonify({"ok": True,
                            "note": "unknown client this round (ignored)"})
        if TASK_STATUS[cid]["status"] == "done":
            return jsonify({"ok": True,
                            "note": "duplicate ignored"})

        TASK_STATUS[cid]["status"] = "done"
        TASK_STATUS[cid]["train_s"] = (
            time.perf_counter() -
            TASK_STATUS[cid].get("start", time.perf_counter())
        )
        CLIENT_UPDATES[cid] = (updated_weights, n_samples)
        t_s = TASK_STATUS[cid]["train_s"]

    print(f"[SRV][R{ROUND_NUM}] <- {cid}: n={n_samples}, train={t_s:.2f}s")
    return jsonify({"ok": True})


# ======================== Server local training ========================
def _server_train_on_shard(start_weights, X, y,
                           epochs: int = 1,
                           batch: int = LOCAL_BATCH):
    """
    Server trains on its own shard like a client,
    then reports back into CLIENT_UPDATES.
    """
    m = _build_global_model()
    fl_core.prime_model(m)
    m.set_weights(start_weights)

    t0 = time.perf_counter()
    m.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)
    train_s = time.perf_counter() - t0

    upd = m.get_weights()
    with LOCK:
        TASK_STATUS["server"]["status"] = "done"
        TASK_STATUS["server"]["train_s"] = train_s
        CLIENT_UPDATES["server"] = (upd, X.shape[0])
    print(f"[SRV][R{ROUND_NUM}] (server) done: n={X.shape[0]}, "
          f"train={train_s:.2f}s")


# ======================== Calibration (only for sticky_calibrated) ========================
def _run_one_time_calibration(participants: List[str]) -> None:
    """
    One warm-up timing pass. We do NOT change global weights from this pass.
    Purpose: measure samples/sec per host so we can size sticky shards once.
    """
    global ROUND_NUM, ROUND_IS_OPEN, SHARD_PAYLOADS, TASK_STATUS, CLIENT_UPDATES

    print("[SRV] calibration start...")

    per_host = max(CALIB_SAMPLES_PER_HOST, CALIB_MIN_SAMPLES)
    total = min(per_host * len(participants), X_train_all.shape[0])

    parts = _split_even_indices(participants, n_total=total, seed=999)

    with LOCK:
        SHARD_PAYLOADS = {}
        TASK_STATUS = {}
        CLIENT_UPDATES.clear()
        for pid in participants:
            idxs = parts[pid]
            Xp, yp = X_train_all[idxs], y_train_all[idxs]
            SHARD_PAYLOADS[pid] = (fl_core.arrays_to_b64(Xp, yp),
                                   int(Xp.shape[0]))
            TASK_STATUS[pid] = {"status": "pending", "n": int(Xp.shape[0])}
        ROUND_IS_OPEN = True
        ROUND_NUM = 0  # log as round 0

    # Server can participate in calibration too
    if SERVER_DOES_TRAIN and "server" in participants:
        Xs, ys = fl_core.arrays_from_b64(SHARD_PAYLOADS["server"][0])
        with LOCK:
            TASK_STATUS["server"]["status"] = "claimed"
            TASK_STATUS["server"]["start"] = time.perf_counter()
        threading.Thread(
            target=_server_train_on_shard,
            args=(GLOBAL_WEIGHTS, Xs, ys, 1, LOCAL_BATCH),
            daemon=True,
        ).start()

    # Wait for everyone (all entries in TASK_STATUS)
    while True:
        with LOCK:
            done = sum(1 for s in TASK_STATUS.values()
                       if s["status"] == "done")
            need = len(TASK_STATUS)
        if done >= need:
            break
        time.sleep(0.2)

    _update_speed_from_task_status()
    with LOCK:
        ROUND_IS_OPEN = False
        print("[SRV] calibration done | " + " | ".join(
            f"{cid}: thr={SAMPLES_PER_SEC[cid]:.1f} samp/s"
            for cid in participants
            if cid in SAMPLES_PER_SEC
        ))


# ======================== Aggregation (CPU or GPU) ========================
# ======================== Server FedOpt (FedAdam) ========================
def _ensure_server_adam_state(ref_weights: List[np.ndarray]) -> None:
    """Init/resize server Adam buffers if weights change shape."""
    global SERVER_ADAM_M, SERVER_ADAM_V, SERVER_ADAM_T

    if SERVER_ADAM_M is None or SERVER_ADAM_V is None:
        SERVER_ADAM_M = [np.zeros_like(w, dtype=np.float32) for w in ref_weights]
        SERVER_ADAM_V = [np.zeros_like(w, dtype=np.float32) for w in ref_weights]
        SERVER_ADAM_T = 0
        return

    if len(SERVER_ADAM_M) != len(ref_weights):
        SERVER_ADAM_M = [np.zeros_like(w, dtype=np.float32) for w in ref_weights]
        SERVER_ADAM_V = [np.zeros_like(w, dtype=np.float32) for w in ref_weights]
        SERVER_ADAM_T = 0
        return

    for m, v, w in zip(SERVER_ADAM_M, SERVER_ADAM_V, ref_weights):
        if m.shape != w.shape or v.shape != w.shape:
            SERVER_ADAM_M = [np.zeros_like(w2, dtype=np.float32) for w2 in ref_weights]
            SERVER_ADAM_V = [np.zeros_like(w2, dtype=np.float32) for w2 in ref_weights]
            SERVER_ADAM_T = 0
            return


def _apply_fedadam(base_weights: List[np.ndarray],
                   fedavg_weights: List[np.ndarray]) -> List[np.ndarray]:
    """FedAdam: apply a server-side Adam update using the FedAvg delta.

    We treat the pseudo-gradient as:
        g = base_weights - fedavg_weights
    so that an SGD step would move weights *towards* the FedAvg solution.
    """
    global SERVER_ADAM_T

    if len(base_weights) != len(fedavg_weights):
        raise ValueError("Weight list length mismatch in _apply_fedadam().")

    _ensure_server_adam_state(base_weights)
    assert SERVER_ADAM_M is not None and SERVER_ADAM_V is not None

    SERVER_ADAM_T += 1
    t = SERVER_ADAM_T

    b1 = float(FEDOPT_BETA1)
    b2 = float(FEDOPT_BETA2)
    lr = float(FEDOPT_LR)
    eps = float(FEDOPT_EPS)
    wd = float(FEDOPT_WEIGHT_DECAY)

    b1_t = b1 ** t
    b2_t = b2 ** t

    out: List[np.ndarray] = []
    for i, (w0, wavg) in enumerate(zip(base_weights, fedavg_weights)):
        w0_f = w0.astype(np.float32, copy=False)
        wavg_f = wavg.astype(np.float32, copy=False)

        # Pseudo-gradient (points from FedAvg -> base); subtracting it moves towards FedAvg.
        g = (w0_f - wavg_f)

        m = SERVER_ADAM_M[i] = (b1 * SERVER_ADAM_M[i] + (1.0 - b1) * g)
        v = SERVER_ADAM_V[i] = (b2 * SERVER_ADAM_V[i] + (1.0 - b2) * (g * g))

        m_hat = m / (1.0 - b1_t)
        v_hat = v / (1.0 - b2_t)

        step = lr * m_hat / (np.sqrt(v_hat) + eps)

        # Optional decoupled weight decay (AdamW-style)
        if wd != 0.0:
            step = step + (lr * wd * w0_f)

        w_new = (w0_f - step).astype(w0.dtype, copy=False)
        out.append(w_new)

    return out

def _aggregate(base_weights: List[np.ndarray],
               updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Aggregate client updates into a new set of global weights.
    """
    if not updates:
        raise ValueError("No client updates provided to _aggregate().")

    # Choose backend for FedAvg (CPU vs GPU)
    if USE_GPU_AGG:
        fedavg_weights = fl_core.fedavg_weighted_average_gpu(updates)
    else:
        fedavg_weights = fl_core.fedavg_weighted_average(updates)

    if AGG_MODE == "fedavg":
        return fedavg_weights

    if AGG_MODE == "fedadam":
        return _apply_fedadam(base_weights, fedavg_weights)

    raise ValueError(f"Unknown AGG_MODE: {AGG_MODE}")


def _test_benchmark_aggregate(
    updates: List[Tuple[List[np.ndarray], int]]
) -> Tuple[float, str]:
    """
    Run the *other* aggregation backend purely for timing comparison.

    - If USE_GPU_AGG is True (main path uses GPU), benchmark uses CPU.
    - If USE_GPU_AGG is False (main path uses CPU), benchmark uses GPU.
    - Does NOT touch GLOBAL_WEIGHTS (caller ignores the result).
    - Returns (elapsed_seconds, backend_name).
    """
    if not updates:
        raise ValueError("No client updates provided to "
                         "_test_benchmark_aggregate().")

    t0 = time.perf_counter()
    if USE_GPU_AGG:
        # Main path is GPU → benchmark the CPU implementation
        fl_core.fedavg_weighted_average(updates)
        backend = "cpu"
    else:
        # Main path is CPU → benchmark the GPU implementation
        fl_core.fedavg_weighted_average_gpu(updates)
        backend = "gpu"
    elapsed = time.perf_counter() - t0
    return elapsed, backend


# ======================== Orchestration loop ========================
def run_rounds():
    global ROUND_NUM, ROUND_IS_OPEN, GLOBAL_WEIGHTS
    global SHARD_PAYLOADS, TASK_STATUS, TRAINING_IS_DONE, STICKY_SPLITS

    _ensure_global_weights()

    # Wait for enough clients before we start
    print("[SRV] waiting for clients...")
    t_wait = time.perf_counter()
    while True:
        with LOCK:
            ready = [c for c, info in ACTIVE_CLIENTS.items()
                     if info.get("connected")]
        if len(ready) >= MIN_CLIENTS:
            break
        time.sleep(0.3)
    print(f"[SRV] clients ready: {len(ready)} in "
          f"{time.perf_counter() - t_wait:.2f}s")

    with LOCK:
        participants = [*ACTIVE_CLIENTS.keys()]
        if SERVER_DOES_TRAIN:
            participants.append("server")

    # KCI injection: do NOT include KCI clients in the *base* split so the remaining
    # participants still cover the full training set. KCI clients get their own
    # extra shard (which may overlap), on top of the base split.
    kci_set = set(KCI_CLIENTS) if KCI_ENABLED else set()
    kci_present = [pid for pid in participants if pid in kci_set]
    base_participants = [pid for pid in participants if pid not in kci_set]

    if KCI_ENABLED and kci_present:
        print(
            "[SRV] KCI injection enabled; excluding from base split: "
            + ", ".join(kci_present)
            + f" | base_split_clients={len(base_participants)}"
        )

    if not base_participants:
        raise RuntimeError("KCI config removed all participants from the base split.")

    # --- Precompute splits if we are sticky ---
    if SPLIT_MODE == "sticky_calibrated":
        _run_one_time_calibration(participants)

        # Base split covers the *entire* training set across non-KCI participants only.
        sizes = _sizes_from_speed(base_participants)
        STICKY_SPLITS = _build_train_indices(
            base_participants,
            sizes,
            seed=1234,
        )

        # Then inject/override KCI clients with their own larger shard.
        STICKY_SPLITS = _apply_kci(STICKY_SPLITS, participants, seed=1234)

    elif SPLIT_MODE == "sticky_equal":
        # Base split covers the *entire* training set across non-KCI participants only.
        equal = {
            pid: X_train_all.shape[0] // len(base_participants)
            for pid in base_participants
        }
        # hand out remainder rows
        rem = X_train_all.shape[0] - sum(equal.values())
        for j in range(rem):
            equal[base_participants[j % len(base_participants)]] += 1

        STICKY_SPLITS = _build_train_indices(
            base_participants,
            equal,
            seed=1234,
        )

        # Then inject/override KCI clients with their own larger shard.
        STICKY_SPLITS = _apply_kci(STICKY_SPLITS, participants, seed=1234)

        # assume equal speed initially
        for pid in participants:
            SAMPLES_PER_SEC[pid] = 1.0
    else:
        # adaptive_each_round
        for pid in participants:
            SAMPLES_PER_SEC[pid] = 1.0
        STICKY_SPLITS = None

    if STICKY_SPLITS is not None:
        print("[SRV] sticky shard sizes: " + " | ".join(
            f"{p}: {STICKY_SPLITS[p].size}" for p in participants
        ))

    t_total = time.perf_counter()
    total_train_s = 0.0   # sum of per-round training times (server+clients+agg, no eval)
    total_eval_s = 0.0    # sum of per-round eval times (+ final eval at the end)

    # Early stopping tracking
    best_metric: Optional[float] = None
    best_round: int = 0
    best_weights: Optional[List[np.ndarray]] = None
    rounds_without_improve: int = 0

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
                    SHARD_PAYLOADS[pid] = (fl_core.arrays_to_b64(Xp, yp),
                                           int(Xp.shape[0]))
                    TASK_STATUS[pid] = {"status": "pending",
                                        "n": int(Xp.shape[0])}
                ROUND_IS_OPEN = True
                start_weights_this_round = GLOBAL_WEIGHTS
        else:
            # Adaptive: compute sizes from current speeds → fresh stratified split
            sizes_now = _sizes_from_speed(base_participants)
            splits_now = _build_train_indices(base_participants,
                                     sizes_now,
                                     seed=1000 + ROUND_NUM)
            splits_now = _apply_kci(splits_now, participants, seed=1000 + ROUND_NUM)
            with LOCK:
                SHARD_PAYLOADS = {}
                for pid in participants:
                    idxs = splits_now[pid]
                    Xp, yp = X_train_all[idxs], y_train_all[idxs]
                    SHARD_PAYLOADS[pid] = (fl_core.arrays_to_b64(Xp, yp),
                                           int(Xp.shape[0]))
                    TASK_STATUS[pid] = {"status": "pending",
                                        "n": int(Xp.shape[0])}
                ROUND_IS_OPEN = True
                start_weights_this_round = GLOBAL_WEIGHTS

        # Round timing starts here (server+clients training + waiting + aggregation, no eval)
        round_t0 = time.perf_counter()
        print(f"\n[SRV][R{ROUND_NUM}] start")

        # 2) Server trains on its own shard (if enabled)
        if SERVER_DOES_TRAIN and "server" in SHARD_PAYLOADS:
            Xs, ys = fl_core.arrays_from_b64(SHARD_PAYLOADS["server"][0])
            with LOCK:
                TASK_STATUS["server"]["status"] = "claimed"
                TASK_STATUS["server"]["start"] = time.perf_counter()
            threading.Thread(
                target=_server_train_on_shard,
                args=(start_weights_this_round,
                      Xs,
                      ys,
                      LOCAL_EPOCHS,
                      LOCAL_BATCH),
                daemon=True,
            ).start()

        # 3) Wait for all participants of this round to finish training
        while True:
            with LOCK:
                done = sum(1 for s in TASK_STATUS.values()
                           if s["status"] == "done")
                need = len(TASK_STATUS)
            if done >= need:
                break
            time.sleep(0.2)

        # 4) Aggregate updates (this is where CUDA C++ will drop in)
        t_agg = time.perf_counter()
        with LOCK:
            updates_list = list(CLIENT_UPDATES.values())
            new_weights = _aggregate(start_weights_this_round, updates_list)
            GLOBAL_WEIGHTS = new_weights
            ROUND_IS_OPEN = False
            summary = " | ".join(
                f"{cid}:n={st['n']},t={st.get('train_s', 0):.2f}s"
                for cid, st in TASK_STATUS.items()
            )
        agg_s = time.perf_counter() - t_agg

        # End of training-phase timing for this round (server+clients+agg, no eval)
        round_train_end = time.perf_counter()
        round_train_s = round_train_end - round_t0
        total_train_s += round_train_s

        print(
            f"[SRV][R{ROUND_NUM}] agg={agg_s:.5f}s | "
            f"train_round={round_train_s:.2f}s "
            f"(server+clients+agg, no eval)"
        )
        print(f"[SRV][R{ROUND_NUM}] hosts {{ {summary} }}")

        # 4b) Optional benchmark of the *other* aggregation backend.
        # This happens AFTER train_round timing is captured, so it does NOT
        # affect train_round or total_train_s, and it does NOT touch GLOBAL_WEIGHTS.
        if TEST_BENCHMARK:
            with LOCK:
                updates_list_bench = list(CLIENT_UPDATES.values())
            bench_s, bench_backend = _test_benchmark_aggregate(
                updates_list_bench
            )
            print(
                f"[SRV][R{ROUND_NUM}] benchmark agg({bench_backend})="
                f"{bench_s:.5f}s "
                f"(NOT used for GLOBAL_WEIGHTS; not counted in train_round)"
            )

        # 5) Refresh speed estimates (harmless for sticky)
        _update_speed_from_task_status()

        # 6) Per-round evaluation on held-out test set (timed separately, NOT in train_round)
        eval_t0 = time.perf_counter()
        eval_model = _build_global_model()
        fl_core.prime_model(eval_model)
        eval_model.set_weights(GLOBAL_WEIGHTS)
        loss, acc = eval_model.evaluate(X_test, y_test, verbose=0)
        eval_s = time.perf_counter() - eval_t0
        total_eval_s += eval_s

        print(
            f"[SRV][R{ROUND_NUM}] eval: loss={loss:.4f}, "
            f"acc={acc:.4f}, eval_time={eval_s:.2f}s"
        )

        # Dataset-specific detailed metrics every 5 rounds
        if fl_core.DATASET_TYPE == "cyber_threat" and (ROUND_NUM % 5 == 0):
            try:
                fl_core.print_cyber_threat_metrics(
                    eval_model, X_test, y_test, ROUND_NUM
                )
            except Exception as e:
                print(
                    f"[SRV][R{ROUND_NUM}] detailed cyber metrics "
                    f"failed: {e}"
                )

        if fl_core.DATASET_TYPE == "network_traffic" and (ROUND_NUM % 5 == 0):
            try:
                fl_core.print_network_traffic_metrics(
                    eval_model, X_test, y_test, ROUND_NUM
                )
            except Exception as e:
                print(
                    f"[SRV][R{ROUND_NUM}] detailed network metrics "
                    f"failed: {e}"
                )

        # 7) Early stopping check (server-side, based on eval metric)
        if EARLY_STOP_ENABLED:
            current = acc if EARLY_STOP_MONITOR == "acc" else loss

            if EARLY_STOP_MODE == "max":
                def improved(curr, best):
                    return curr > best + EARLY_STOP_MIN_DELTA
            else:  # "min"
                def improved(curr, best):
                    return curr < best - EARLY_STOP_MIN_DELTA

            if best_metric is None:
                # First round: initialize best
                best_metric = current
                best_round = ROUND_NUM
                best_weights = [w.copy() for w in GLOBAL_WEIGHTS]
                rounds_without_improve = 0
                print(
                    f"[SRV][R{ROUND_NUM}] early-stop baseline: "
                    f"{EARLY_STOP_MONITOR}={best_metric:.4f}"
                )
            else:
                if improved(current, best_metric):
                    best_metric = current
                    best_round = ROUND_NUM
                    best_weights = [w.copy() for w in GLOBAL_WEIGHTS]
                    rounds_without_improve = 0
                    print(
                        f"[SRV][R{ROUND_NUM}] early-stop monitor "
                        f"improved: "
                        f"{EARLY_STOP_MONITOR}={best_metric:.4f}"
                    )
                else:
                    rounds_without_improve += 1
                    print(
                        f"[SRV][R{ROUND_NUM}] no {EARLY_STOP_MONITOR} "
                        f"improvement "
                        f"({rounds_without_improve}/"
                        f"{EARLY_STOP_PATIENCE})"
                    )
                    if rounds_without_improve >= EARLY_STOP_PATIENCE:
                        print(
                            f"[SRV] early stopping triggered at round "
                            f"{ROUND_NUM} "
                            f"(best {EARLY_STOP_MONITOR}="
                            f"{best_metric:.4f} at R{best_round})"
                        )
                        # Restore best weights seen so far
                        if best_weights is not None:
                            GLOBAL_WEIGHTS = [w.copy()
                                              for w in best_weights]
                        break  # break out of the main rounds loop

    with LOCK:
        TRAINING_IS_DONE = True

    # Final evaluation (same held-out test split as centralized)
    final_eval_t0 = time.perf_counter()
    m = _build_global_model()
    fl_core.prime_model(m)
    m.set_weights(GLOBAL_WEIGHTS)
    loss, acc = m.evaluate(X_test, y_test, verbose=0)
    final_eval_s = time.perf_counter() - final_eval_t0
    total_eval_s += final_eval_s

    total_wall_s = time.perf_counter() - t_total
    rounds_ran = ROUND_NUM  # last round index actually executed

    print(
        f"\n[SRV] done | rounds_run={rounds_ran} "
        f"(requested={TOTAL_ROUNDS}) | "
        f"train_total={total_train_s/60:.2f} min "
        f"(all rounds: server+clients+agg, no eval) | "
        f"eval_total={total_eval_s:.2f}s "
        f"(per-round + final eval) | "
        f"wall={total_wall_s/60:.2f} min (overall wall-clock) | "
        f"acc={acc:.4f}"
    )


# ======================== Entrypoint ========================
def _run_api():
    # threaded=True lets multiple clients hit the server concurrently
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    run_rounds()
