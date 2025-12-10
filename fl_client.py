# fl_client.py — FL client aligned with fl_server.py
# CPU-only local training (server may use CUDA for aggregation)
#
# Flow per round:
#   - Register with server.
#   - Poll /pull_task for "ready" work.
#   - Receive global weights + shard (or reuse cached shard for sticky modes).
#   - Train locally.
#   - Post back updated weights + sample count.
#   - Repeat until server says "done".

from __future__ import annotations

import os
import logging
import time
import requests
import numpy as np  # noqa: F401 (imported for consistency / future use)

# Force CPU locally – server handles GPU aggregation
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

import fl_core


# ============================================================
# CONFIGURATION
# ============================================================

# Defaults come from the original client, but can be overridden via env vars.
SERVER_URL = os.getenv("FL_SERVER_URL", "INSERT URL HERE")
CLIENT_ID  = os.getenv("CLIENT_ID", "INSERT NAME HERE")

LOCAL_EPOCHS = 1
CLIENT_BATCH = 64


# ============================================================
# CLIENT ENTRY
# ============================================================

def main() -> None:
    # -----------------------------------------------
    # 1) Register with the server (retry until success)
    # -----------------------------------------------
    while True:
        try:
            requests.post(
                f"{SERVER_URL}/register",
                json={"client_id": CLIENT_ID},
                timeout=5,
            ).raise_for_status()
            print(f"[CLI {CLIENT_ID}] registered with server.")
            break
        except requests.exceptions.RequestException:
            time.sleep(1)

    # -----------------------------------------------
    # Persistent model + shard cache
    # -----------------------------------------------
    model = None          # built once and reused across rounds
    cached_X = None       # cached shard features (for sticky modes)
    cached_y = None       # cached shard labels

    # -----------------------------------------------
    # 2) Main training loop
    # -----------------------------------------------
    while True:
        # Poll server for work
        try:
            resp = requests.get(
                f"{SERVER_URL}/pull_task",
                params={"client_id": CLIENT_ID},
                timeout=300,
            ).json()
        except requests.exceptions.RequestException:
            time.sleep(0.5)
            continue

        status = resp.get("status")

        # -------------------
        # Training finished
        # -------------------
        if status == "done":
            print(f"[CLI {CLIENT_ID}] training complete. Exiting.")
            break

        # -------------------
        # No task yet
        # -------------------
        if status != "ready":
            time.sleep(0.2)
            continue

        # -------------------
        # Extract assignment
        # -------------------
        rnd = resp["round"]
        n_rows = int(resp["n_samples"])
        global_weights = fl_core.weights_from_b64(resp["weights_b64"])
        shard_b64 = resp.get("shard_b64")

        # -------------------
        # Sticky shard caching
        # -------------------
        if shard_b64 is None:
            # Server expects us to reuse our cached shard
            if cached_X is None or cached_y is None:
                print(
                    f"[CLI {CLIENT_ID}][R{rnd}] "
                    "Expected cached shard but none exists. Waiting."
                )
                time.sleep(0.5)
                continue
            X, y = cached_X, cached_y
        else:
            # New shard received → decode & cache
            X, y = fl_core.arrays_from_b64(shard_b64)
            cached_X, cached_y = X, y

        # -------------------
        # Build local model ONCE per client
        # -------------------
        if model is None:
            num_classes = y.shape[1]
            model = fl_core.build_model(num_classes)
            fl_core.prime_model(model)

        # -------------------
        # Load global weights into local model
        # -------------------
        model.set_weights(global_weights)

        # -------------------
        # Local training
        # -------------------
        t0 = time.perf_counter()
        print(f"[CLI {CLIENT_ID}][R{rnd}] training start on n={n_rows}")
        model.fit(X, y, epochs=LOCAL_EPOCHS, batch_size=CLIENT_BATCH, verbose=0)
        train_s = time.perf_counter() - t0

        # -------------------
        # Send model update back
        # -------------------
        upd_b64 = fl_core.weights_to_b64(model.get_weights())

        try:
            requests.post(
                f"{SERVER_URL}/submit_update",
                json={
                    "client_id": CLIENT_ID,
                    "round": rnd,
                    "weights_b64": upd_b64,
                    "n_samples": n_rows,
                },
                timeout=60,
            ).raise_for_status()
            print(
                f"[CLI {CLIENT_ID}][R{rnd}] submitted update "
                f"(train={train_s:.2f}s)"
            )
        except requests.exceptions.RequestException as e:
            print(f"[CLI {CLIENT_ID}] submission failed: {e}")
            time.sleep(1)

        # Small pause to reduce polling pressure
        time.sleep(0.2)


# ============================================================
# Start Client
# ============================================================

if __name__ == "__main__":
    main()
