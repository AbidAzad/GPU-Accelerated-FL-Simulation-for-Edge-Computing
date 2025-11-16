# client.py — CPU-only FL client with persistent model and optional shard caching
# Flow per round:
#   - Poll server for "ready" task.
#   - Receive global weights + shard (or reuse cached shard for sticky modes).
#   - Train locally.
#   - Post back updated weights + sample count.
#   - Repeat until server says "done".

from __future__ import annotations

import os, logging, time, requests, numpy as np
# Force CPU + keep logs down for a quieter console
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from fl_core import (
    MAX_NB_WORDS, MAX_SEQ_LENGTH, EMBEDDING_DIM, BATCH_SIZE,
    build_lstm_model, prime_model, arrays_from_b64, weights_from_b64, weights_to_b64,
)

# Point to our server
SERVER_URL = "SERVER_URL"                       # <-- set server IP/port
CLIENT_ID  = "ADD NAME HERE"                    # <-- unique per device

# Small knobs for this client
LOCAL_EPOCHS = 1
CLIENT_BATCH = 64


def main():
    # 1) Register until server is reachable
    while True:
        try:
            requests.post(f"{SERVER_URL}/register", json={"client_id": CLIENT_ID}, timeout=10).raise_for_status()
            print(f"[CLI {CLIENT_ID}] registered")
            break
        except requests.exceptions.RequestException:
            time.sleep(1)

    # Build-once model (we keep it persistent across rounds)
    model = None

    # For sticky modes with shard caching (server may skip resending after R1)
    cached_X = None
    cached_y = None

    # 2) Main polling/training loop
    while True:
        # Ask the server for work
        try:
            resp = requests.get(f"{SERVER_URL}/pull_task", params={"client_id": CLIENT_ID}, timeout=30).json()
        except requests.exceptions.RequestException:
            time.sleep(0.5); continue

        status = resp.get("status")
        if status == "done":
            print(f"[CLI {CLIENT_ID}] server signaled completion; exiting.")
            break
        if status != "ready":
            # Not ready yet; try again
            time.sleep(0.2); continue

        # Extract task info
        rnd            = resp["round"]
        n_rows_to_train= int(resp["n_samples"])
        global_weights = weights_from_b64(resp["weights_b64"])

        # Either decode fresh shard or reuse our cached one (sticky modes with caching)
        if resp.get("shard_b64") is None:
            if cached_X is None or cached_y is None:
                # Server expects us to reuse data, but we don't have it (first time or cache lost)
                print(f"[CLI {CLIENT_ID}][R{rnd}] no cached shard yet; waiting for a resend...")
                time.sleep(0.5); continue
            X, y = cached_X, cached_y
        else:
            X, y = arrays_from_b64(resp["shard_b64"])
            cached_X, cached_y = X, y  # save for later rounds if server uses caching

        # Build the local model once (identical architecture to centralized)
        if model is None:
            num_classes = y.shape[1]
            model = build_lstm_model(MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQ_LENGTH, num_classes)
            prime_model(model)

        # Load the current global weights for this round
        model.set_weights(global_weights)

        # Train locally
        t0 = time.perf_counter()
        print(f"[CLI {CLIENT_ID}][R{rnd}] train start n={n_rows_to_train}")
        model.fit(X, y, epochs=LOCAL_EPOCHS, batch_size=CLIENT_BATCH, verbose=0)
        train_s = time.perf_counter() - t0

        # Send updated weights back to the server
        upd = weights_to_b64(model.get_weights())
        try:
            requests.post(
                f"{SERVER_URL}/submit_update",
                json={"client_id": CLIENT_ID, "round": rnd, "weights_b64": upd, "n_samples": n_rows_to_train},
                timeout=60
            ).raise_for_status()
            print(f"[CLI {CLIENT_ID}][R{rnd}] submitted train={train_s:.2f}s")
        except requests.exceptions.RequestException as e:
            print(f"[CLI {CLIENT_ID}] submit failed: {e}")
            time.sleep(1)

        # Small pause to avoid hammering the server
        time.sleep(0.2)


if __name__ == "__main__":
    main()
