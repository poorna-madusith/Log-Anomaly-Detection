"""Profile single-sequence DualChannelBiLSTM inference on cached embeddings.

This measures checkpoint loading separately from inference. Timed samples include
tensor transfer to the selected device and the model forward pass; they exclude
raw-log parsing, Drain matching, MiniLM embedding, and serving/network overhead.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[1]


def project_path(value, default):
    path = value if value is not None else default
    path = Path(path)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure one-sequence DualChannelBiLSTM inference latency using precomputed semantic embeddings."
    )
    parser.add_argument("--dataset", choices=["HDFS", "BGL", "Thunderbird"], default="HDFS")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if args.warmup < 0 or args.max_samples <= 0:
        raise SystemExit("warmup must be nonnegative and max_samples must be positive")

    data_dir = project_path(args.data_dir, BASE_DIR / "artifacts" / args.dataset / "data")
    model_path = project_path(
        args.model_path,
        BASE_DIR / "artifacts" / args.dataset / "checkpoints" / "best_model.pth",
    )
    npz_path = data_dir / f"{args.split}_data.npz"
    matrix_path = data_dir / "string_matrix.npy"
    for required in (model_path, npz_path, matrix_path):
        if not required.is_file():
            raise SystemExit(f"Required file not found: {required}")

    # The module path is added explicitly so this script works from any CWD.
    import sys

    sys.path.insert(0, str(BASE_DIR))
    from models.dataset import create_dataloader
    from models.model import DualChannelBiLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device, weights_only=True)
    vocab_size = state["struct_embedding.weight"].shape[0]
    model = DualChannelBiLSTM(vocab_size=vocab_size).to(device)
    model.load_state_dict(state)
    model.eval()

    loader, _ = create_dataloader(npz_path, matrix_path, batch_size=1, shuffle=False)
    iterator = iter(loader)

    with torch.inference_mode():
        warmed = 0
        while warmed < args.warmup:
            try:
                struct_seq, sem_seq, _ = next(iterator)
            except StopIteration:
                break
            logits, _ = model(struct_seq.to(device), sem_seq.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced a non-finite logit during warmup.")
            synchronize(device)
            warmed += 1

        # Start a fresh ordered pass so small datasets can still be profiled
        # after consuming all examples during warmup.
        iterator = iter(loader)
        latencies_ms = []
        while len(latencies_ms) < args.max_samples:
            try:
                struct_seq, sem_seq, _ = next(iterator)
            except StopIteration:
                break
            synchronize(device)
            start = time.perf_counter()
            logits, _ = model(struct_seq.to(device), sem_seq.to(device))
            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced a non-finite logit during profiling.")
            latencies_ms.append(elapsed_ms)

    if not latencies_ms:
        raise SystemExit("No inference samples were available after warmup.")

    values = np.asarray(latencies_ms, dtype=np.float64)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": str(model_path),
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "batch_size": 1,
        "warmup_samples": warmed,
        "measured_samples": int(values.size),
        "latency_ms_per_sequence": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
        "timed_scope": (
            "single-sequence tensor transfer plus DualChannelBiLSTM forward pass; "
            "semantic embeddings are precomputed"
        ),
        "excluded_scope": [
            "raw log parsing",
            "Drain matching",
            "MiniLM embedding generation",
            "API, queue, and network overhead",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if args.output:
        output_path = project_path(args.output, args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = BASE_DIR / "artifacts" / args.dataset / "latency" / f"{args.split}_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    print(f"[*] Device: {device} | Warmup: {warmed} | Measured: {values.size}")
    print("[*] Scope: single sequence, precomputed embeddings, transfer + model forward only")
    print(
        "[+] Latency ms/sequence: "
        f"mean={result['latency_ms_per_sequence']['mean']:.3f}, "
        f"p50={result['latency_ms_per_sequence']['p50']:.3f}, "
        f"p95={result['latency_ms_per_sequence']['p95']:.3f}"
    )
    print(f"[+] Saved profile to {output_path}")


if __name__ == "__main__":
    main()
