# fl_sim.py — single-machine Federated Learning simulator (CPU-only)
# Project framing:
#   • We keep the SAME LSTM model as our centralized script.
#   • We simulate N clients by running N processes in parallel on one machine.
#   • We cap TensorFlow threads per process so all clients share CPU fairly.
#   • Each process builds its model once and reuses it across rounds.
#   • After each round, the server performs FedAvg on CPU (for now).
#   • Later, we'll replace FedAvg with a C++/CUDA aggregator.

from __future__ import annotations

import os
import warnings
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------
# Make TensorFlow quiet
# ---------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # keep GPU free; we’ll use it later for CUDA aggregation
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # silence TF info/warn logs
warnings.filterwarnings("ignore", message="Argument `input_length` is deprecated", category=UserWarning)

# We reuse the same data prep and model as the centralized baseline.
from Centralized_LSTM import (
    load_dataset,
    prepare_data,
    build_lstm_model,
    DATA_PATH,
    MAX_NB_WORDS,
    MAX_SEQ_LENGTH,
)

# ============================
# Basic configuration
# ============================
num_clients = 4                 # number of simulated clients (processes)
num_rounds = 15                 # how many FL rounds to run
local_epochs = 1                # epochs per client per round
local_batch = 64                # batch size per client
reshuffle_each_round = False     # re-split data every round to keep shards fresh
reset_optimizer_each_round = False  # recompile each round so Adam moments don’t carry over between rounds

# When the C++/CUDA aggregator is ready, we flip this to True and implement cuda_fedavg().
use_cuda_agg = False

# ---------------------------
# Thread caps per worker
# Why this matters:
#   If every client tries to use ALL CPU cores, they fight each other and stall.
#   We cap TF threads per worker so multiple clients actually run in parallel and share CPU fairly.
# ---------------------------
cpu_cores = os.cpu_count() or 16
threads_intra = 4   # math threads per worker; Sweet Spot seems to be NumClients * threads_intra <= cpu_cores
threads_inter = 1   # graph scheduling threads per worker

print(f"[FL] CPU cores={cpu_cores} | clients={num_clients} | threads/worker: intra={threads_intra}, inter={threads_inter}")

# ============================
# Small helpers (main process)
# ============================

def warm_up_once(model, seq_len: int = MAX_SEQ_LENGTH):
    """
    We run one forward pass to allocate variables.
    This makes set_weights/get_weights safe on the model.
    """
    _ = model(np.zeros((1, seq_len), dtype="int32"), training=False)
    return model

def build_global_model(num_classes: int):
    """
    We build the SAME LSTM as our centralized baseline, then warm it once.
    """
    m = build_lstm_model(
        max_words=MAX_NB_WORDS,
        embedding_dim=100,
        input_length=MAX_SEQ_LENGTH,
        num_classes=num_classes,
    )
    warm_up_once(m)
    return m

def split_iid(X, y, k: int):
    """
    We create a simple IID-like split:
      1) shuffle all indices
      2) cut into k chunks
      3) return (X_chunk, y_chunk) for each client
    """
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    parts = np.array_split(idx, k)
    return [(X[p], y[p]) for p in parts]

# ============================
# Worker process state + code
# ============================

# Each worker keeps a model in RAM across rounds to avoid rebuild overhead.
_worker_model = None
_worker_num_classes = None

def worker_startup(num_classes: int, intra_threads: int, inter_threads: int):
    """
    This runs ONCE when a worker process starts.
    We:
      • Force CPU mode + quiet logs inside the worker.
      • Cap TF threads so all clients share CPU fairly.
      • Build the SAME LSTM once, warm it, and compile it.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # We also cap common native thread pools so we don’t oversubscribe the CPU.
    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["MKL_NUM_THREADS"] = str(intra_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(intra_threads)

    # Apply TF thread limits inside this worker.
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)

    global _worker_model, _worker_num_classes
    _worker_num_classes = int(num_classes)

    _worker_model = build_lstm_model(
        max_words=MAX_NB_WORDS,
        embedding_dim=100,
        input_length=MAX_SEQ_LENGTH,
        num_classes=_worker_num_classes,
    )
    warm_up_once(_worker_model)

    # Compile once; we can recompile each round to reset the optimizer if desired.
    _worker_model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        steps_per_execution=64,   # reduces Python overhead
    )

    print(f"[Worker] threads set: intra={intra_threads}, inter={inter_threads}")

def worker_train_round(args) -> Tuple[List[np.ndarray], int, float]:
    """
    One client’s work for ONE round (runs in a worker process).
    Inputs (packed in args):
      - global_weights: current global weights to start from
      - X_local, y_local: this client’s shard
      - local_epochs, local_batch: training settings
      - reset_opt: whether we reset optimizer state by recompiling
    Steps:
      1) (optional) recompile to reset optimizer moments
      2) set global weights
      3) train on (X_local, y_local)
      4) return new weights + sample count + train seconds
    """
    (global_weights, X_local, y_local, _epochs, _batch, reset_opt) = args

    if _worker_model is None:
        worker_startup(y_local.shape[1], threads_intra, threads_inter)

    if reset_opt:
        _worker_model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            steps_per_execution=64,
        )

    _worker_model.set_weights(global_weights)

    t0 = perf_counter()
    _worker_model.fit(X_local, y_local, epochs=_epochs, batch_size=_batch, verbose=0)
    train_sec = perf_counter() - t0

    new_weights = _worker_model.get_weights()
    return new_weights, int(X_local.shape[0]), train_sec

# ============================
# FedAvg (weighted) on the server
# ============================

def fedavg(client_updates: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    CPU FedAvg:
      new_global = sum_k (n_k * w_k) / sum_k n_k

    Inputs:
      client_updates = [(weights_k: List[np.ndarray], n_k: int), ...]
    Output:
      List[np.ndarray] with the same shapes/order as a single client’s weights.

    CUDA plan:
      We will REPLACE this function with a C++ CUDA version that performs
      the same weighted averaging on GPU (see cuda_fedavg below).
    """
    weights_list = [w for (w, n) in client_updates]                 # List[clients] of List[layer arrays]
    counts = np.array([n for (w, n) in client_updates], np.float32) # shape (K,)

    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("Total client samples is zero.")

    out: List[np.ndarray] = []
    for layer_idx in range(len(weights_list[0])):
        stack = np.stack([w[layer_idx] for w in weights_list], axis=0)  # (K, ...)
        flat  = stack.reshape(stack.shape[0], -1)                       # (K, P)
        avg   = np.dot(counts, flat) / total                            # (P,)
        out.append(avg.reshape(stack.shape[1:]))
    return out

# ---------------------------
# CUDA FedAvg stub (C++ path)
# ---------------------------
def cuda_fedavg(weights_list: List[List[np.ndarray]], counts: np.ndarray) -> List[np.ndarray]:
    """
    Placeholder for our **C++/CUDA** aggregator (pybind11/ctypes).
    Inputs:
      - weights_list: List over clients; each item is List over layers (np.ndarray per layer).
      - counts:       float32 vector of sample counts per client, shape = (K,)
    Output:
      - List[np.ndarray] in the same shapes/order as weights_list[0].

    Swap-in plan (see run_fl): we build `weights_list` and `counts`, then call:
        new_global = cuda_fedavg(weights_list, counts)
    """
    raise NotImplementedError("Implement C++/CUDA FedAvg and set use_cuda_agg=True")

# ============================
# Main FL loop
# ============================

def run_fl():
    t_total_start = perf_counter()

    # 1) Load + prep data once (tokenize, pad, split, label encode).
    print("[FL] Loading and preparing data...")
    t_prep_start = perf_counter()
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, num_classes, tokenizer, label_encoder = prepare_data(df)
    prep_sec = perf_counter() - t_prep_start
    print(f"[FL] Data ready | prep={prep_sec:.2f}s | train_n={X_train.shape[0]} | test_n={X_test.shape[0]}")

    # 2) Build the global model (SAME as centralized), warm it, and get weights.
    t_build_start = perf_counter()
    global_model = build_global_model(num_classes)
    global_weights = global_model.get_weights()
    build_sec = perf_counter() - t_build_start
    print(f"[FL] Global model built | {build_sec:.2f}s | layers={len(global_weights)}")

    # 3) Start a process pool; each worker keeps a model in RAM for reuse.
    with ProcessPoolExecutor(
        max_workers=num_clients,
        initializer=worker_startup,
        initargs=(num_classes, threads_intra, threads_inter),
    ) as pool:

        shards = None
        for r in range(num_rounds):
            t_round_start = perf_counter()
            print(f"\n[FL][R{r + 1}] start")

            # 3a) Split or reshuffle the training data among clients.
            t_split_start = perf_counter()
            shards = split_iid(X_train, y_train, num_clients) if (r == 0 or reshuffle_each_round) else shards
            split_sec = perf_counter() - t_split_start

            # 3b) Build the client jobs (one job = one client's local training).
            jobs = [
                (global_weights, X_c, y_c, local_epochs, local_batch, reset_optimizer_each_round)
                for (X_c, y_c) in shards
            ]

            # 3c) Run all clients in parallel.
            t_clients_start = perf_counter()
            results = list(pool.map(worker_train_round, jobs))
            clients_sec = perf_counter() - t_clients_start

            # quick per-client timing
            for cid, (_, n_k, t_k) in enumerate(results):
                print(f"[FL][R{r + 1}] client{cid}: n={n_k}, train={t_k:.2f}s")

            # 3d) Aggregate updates on the server.
            # We assemble the exact tensors that the C++/CUDA path will need:
            weights_list = [w for (w, _n, _t) in results]                       # List[clients] of List[layer arrays]
            counts = np.array([n for (_w, n, _t) in results], dtype=np.float32) # (K,)

            t_agg_start = perf_counter()
            if use_cuda_agg:
                # === Future CUDA path (C++ implementation) ===
                new_global = cuda_fedavg(weights_list, counts)
            else:
                # === Current CPU path ===
                new_global = fedavg(list(zip(weights_list, counts.astype(int))))
            global_weights = new_global
            global_model.set_weights(global_weights)
            agg_sec = perf_counter() - t_agg_start

            round_sec = perf_counter() - t_round_start
            print(f"[FL][R{r + 1}] agg={agg_sec:.2f}s | split={split_sec:.2f}s | clients={clients_sec:.2f}s | round={round_sec:.2f}s")

    # 4) Final global evaluation (same test split as centralized).
    t_eval_start = perf_counter()
    loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
    eval_sec = perf_counter() - t_eval_start
    total_sec = perf_counter() - t_total_start
    print(f"\n[FL] Final — Test Loss: {loss:.4f}, Test Acc: {acc:.4f} | eval={eval_sec:.2f}s | total={total_sec/60:.2f} min")

# ---------------------------
# Entrypoint (Windows needs spawn)
# ---------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on Windows for ProcessPoolExecutor
    run_fl()
