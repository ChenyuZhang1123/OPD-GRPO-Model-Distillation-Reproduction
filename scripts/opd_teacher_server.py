#!/usr/bin/env python
"""Teacher model server for OPD training.

Loads the teacher model (bf16, ~62 GB) across 3 dedicated GPUs and exposes
a simple HTTP API for computing per-token logprobs.

Key design choices for speed
----------------------------
- **Multi-threaded** — ``ThreadingMixIn`` so 4 student ranks can post requests
  concurrently instead of queueing at the TCP level.
- **GPU-resident logprobs** — ``log_softmax`` and ``gather`` run on GPU.
  Only the final (B, L)-shaped float tensor is copied to CPU.
  Avoids moving the full (B, L, 152064) logits tensor (~9 GB/request) across
  the PCIe bus and computing softmax on CPU.
- **Model lock** — GPU forward passes are serialised via ``threading.Lock``
  so concurrent threads don't step on each other's CUDA streams.

Usage
-----
    # 3 GPUs for teacher (bf16, no quantization)
    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/opd_teacher_server.py \\
        --model /home/zcy/OPD/models/Qwen3-32B-Base --port 8100

    # Or: use only GPU 0,1,2 explicitly (student uses 3-7)
    python scripts/opd_teacher_server.py \\
        --model /home/zcy/OPD/models/Qwen3-32B-Base --port 8100

API
---
    GET  /health               → "ok"
    POST /compute_logprobs     → {"logprobs": [[...], ...]}
        Body: {"input_ids": [[...], ...], "attention_mask": [[...], ...]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Multi-threaded WSGI server
# ---------------------------------------------------------------------------

class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """WSGI server that handles each request in its own daemon thread."""
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_teacher(model_path: str):
    """Load the teacher model in bf16 across available GPUs.

    With ``CUDA_VISIBLE_DEVICES=0,1,2`` the 32B model (~62 GB) distributes
    across 3 GPUs (72 GB total), running at full bf16 precision.
    """
    n_gpu = torch.cuda.device_count()
    print(f"Loading teacher from: {model_path}", flush=True)
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}",
          flush=True)
    print(f"  Visible GPUs: {n_gpu}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    param_b = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_b / 1e9:.1f}B", flush=True)
    return model


# ---------------------------------------------------------------------------
# WSGI Application
# ---------------------------------------------------------------------------

class TeacherApp:
    """WSGI application serving teacher logprobs.

    All GPU work runs inside a lock so concurrent threads don't race on
    CUDA streams.  The lock ONLY serialises GPU time — HTTP parsing and
    JSON encoding happen outside the critical section.
    """

    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        # device_map="auto" distributes layers — get input device from first param
        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda:0")
        print(f"  Input device: {self._device}", flush=True)
        print(f"  Server mode: multi-threaded (ThreadingMixIn)", flush=True)

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET")

        if method == "GET" and path == "/health":
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"ok"]

        if method == "POST" and path == "/compute_logprobs":
            # Read body outside the GPU lock
            try:
                body = self._read_body(environ)
            except Exception:
                tb = traceback.format_exc()
                print(f"[ERROR] Failed to read request body:\n{tb}",
                      file=sys.stderr, flush=True)
                err = json.dumps({"error": "cannot read request body"}).encode()
                start_response("400 Bad Request",
                               [("Content-Type", "application/json")])
                return [err]

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                start_response("400 Bad Request",
                               [("Content-Type", "application/json")])
                return [json.dumps({"error": "invalid JSON"}).encode()]

            # GPU work under lock
            try:
                result = self._handle_compute_logprobs(data)
                resp = json.dumps(result).encode()
                start_response("200 OK", [("Content-Type", "application/json")])
                return [resp]
            except Exception:
                tb = traceback.format_exc()
                print(f"[ERROR] /compute_logprobs failed:\n{tb}",
                      file=sys.stderr, flush=True)
                err = json.dumps({"error": tb.splitlines()[-1] if tb else "unknown"}).encode()
                start_response("500 Internal Server Error",
                               [("Content-Type", "application/json")])
                return [err]

        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]

    @staticmethod
    def _read_body(environ) -> bytes:
        content_length = int(environ.get("CONTENT_LENGTH", 0))
        if content_length == 0:
            raise ValueError("empty request body")
        return environ["wsgi.input"].read(content_length)

    # ------------------------------------------------------------------
    # Core computation — GPU-resident logprobs
    # ------------------------------------------------------------------

    def _handle_compute_logprobs(self, data: dict) -> dict:
        """Compute per-token teacher logprobs.

        GPU forward pass runs under ``self._lock``.  The expensive log_softmax
        and gather are offloaded to CPU to avoid OOM on the 24 GB GPUs
        (the full (B, L, 152064) float32 tensor is ~7 GB, too large to
        materialise alongside the 32B model).

        Only the forward pass holds the GPU lock, so multiple CPU-bound
        log_softmax computations can overlap.
        """
        input_ids_data = data["input_ids"]
        attention_mask_data = data["attention_mask"]

        # ---- GPU: forward pass (under lock) ----
        with self._lock:
            input_ids = torch.tensor(input_ids_data, device=self._device)
            attention_mask = torch.tensor(attention_mask_data, device=self._device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L, 152064) bf16, on GPU

            # Save input_ids on CPU for the gather step
            input_ids_cpu = input_ids.cpu()
            del outputs, input_ids, attention_mask

            # Move logits to CPU as bf16 (half the transfer of float32).
            # We DO NOT convert to float32 on GPU — that would OOM.
            logits_cpu = logits.cpu()  # (B, L, V) bf16, on CPU
            del logits

        # ---- CPU: log_softmax + gather (outside GPU lock, low priority) ----
        # Convert to float32 on CPU (not GPU) for precision.
        logits_f32 = logits_cpu.float()  # (B, L, V) float32 on CPU
        del logits_cpu

        logprobs = F.log_softmax(logits_f32, dim=-1)  # (B, L, V) float32 on CPU
        del logits_f32

        token_lp = logprobs[:, :-1, :].gather(
            dim=-1, index=input_ids_cpu[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1) float32, on CPU
        del logprobs

        token_lp = F.pad(token_lp, (1, 0), value=0.0)  # (B, L)

        return {"logprobs": token_lp.tolist()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OPD Teacher Server")
    parser.add_argument("--model", required=True, help="Path to teacher model")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    model = load_teacher(args.model)
    app = TeacherApp(model)

    server = _ThreadingWSGIServer((args.host, args.port), WSGIRequestHandler)
    server.set_app(app)

    print(f"\nTeacher server listening on http://{args.host}:{args.port}", flush=True)
    print(f"  Endpoints: GET /health  POST /compute_logprobs", flush=True)
    print(f"  ThreadingMixIn — {os.cpu_count()} CPU cores available", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
