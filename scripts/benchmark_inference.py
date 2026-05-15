"""Benchmark inference speed for a given backbone + checkpoint.

Measures preprocessing time, model forward time, and total inference time.

Usage:
    python scripts/benchmark_inference.py \
        --config configs/experiments/pointnet2_ce_supcon_newdata.yaml \
        --checkpoint outputs/runs/newdata_pointnet2_ce_supcon_v1/checkpoints/best.pt \
        --input-dir dataset/test \
        --num-samples 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.model_factory import build_model_from_checkpoint
from src.utils.config import load_config


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_gpu_info():
    """Get GPU info if available."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_available": True,
        "gpu_name": props.name,
        "gpu_memory_gb": round(props.total_mem / 1e9, 1),
        "cuda_version": torch.version.cuda,
    }


def load_sample_pointclouds(input_dir, num_samples):
    """Load sample point clouds from .npy files."""
    import glob
    npy_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.npy"), recursive=True))
    if not npy_files:
        print(f"[ERROR] No .npy files found in {input_dir}")
        sys.exit(1)

    samples = []
    for f in npy_files[:num_samples]:
        pc = np.load(f)
        if pc.ndim == 2:
            samples.append(pc)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with test .npy files")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to benchmark")
    parser.add_argument("--num-points", type=int, default=None, help="Override num_points from config")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (not counted)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    num_points = args.num_points or cfg.get("num_points", 2048)

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    model, _ = build_model_from_checkpoint(cfg, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Parameter count
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info["gpu_available"]:
        print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB, CUDA {gpu_info['cuda_version']})")
    else:
        print("GPU: not available, using CPU")

    # Load samples
    samples = load_sample_pointclouds(args.input_dir, args.num_samples)
    if not samples:
        print("[ERROR] No valid point cloud samples loaded")
        sys.exit(1)

    print(f"Benchmarking on {len(samples)} samples, {num_points} points each")

    # Prepare tensors
    tensors = []
    for pc in samples:
        # Sample/fill to num_points
        if len(pc) >= num_points:
            idx = np.random.choice(len(pc), num_points, replace=False)
            pc = pc[idx]
        else:
            idx = np.random.choice(len(pc), num_points, replace=True)
            pc = pc[idx]
        t = torch.tensor(pc, dtype=torch.float32)
        tensors.append(t)

    # Warmup
    with torch.no_grad():
        for i in range(min(args.warmup, len(tensors))):
            t = tensors[i % len(tensors)].unsqueeze(0).to(device)
            _ = model(t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    preprocess_times = []
    forward_times = []
    total_times = []

    with torch.no_grad():
        for i in range(len(tensors)):
            t0 = time.perf_counter()

            # Preprocessing: numpy -> tensor -> sample -> to device
            pc = samples[i]
            if len(pc) >= num_points:
                idx = np.random.choice(len(pc), num_points, replace=False)
                pc = pc[idx]
            else:
                idx = np.random.choice(len(pc), num_points, replace=True)
                pc = pc[idx]
            t = torch.tensor(pc, dtype=torch.float32).unsqueeze(0)
            t1 = time.perf_counter()

            t = t.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            out = model(t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            preprocess_times.append(t2 - t0)
            forward_times.append(t3 - t2)
            total_times.append(t3 - t0)

    # Results
    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "backbone": cfg.get("model", {}).get("backbone", "pointnet"),
        "num_samples": len(tensors),
        "num_points": num_points,
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "gpu": gpu_info,
        "timing_ms": {
            "preprocessing": {
                "mean": round(np.mean(preprocess_times) * 1000, 3),
                "std": round(np.std(preprocess_times) * 1000, 3),
                "median": round(np.median(preprocess_times) * 1000, 3),
            },
            "model_forward": {
                "mean": round(np.mean(forward_times) * 1000, 3),
                "std": round(np.std(forward_times) * 1000, 3),
                "median": round(np.median(forward_times) * 1000, 3),
            },
            "total": {
                "mean": round(np.mean(total_times) * 1000, 3),
                "std": round(np.std(total_times) * 1000, 3),
                "median": round(np.median(total_times) * 1000, 3),
            },
        },
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"Inference Benchmark Results")
    print(f"{'='*50}")
    print(f"Backbone:      {results['backbone']}")
    print(f"Parameters:    {total_params:,}")
    print(f"Samples:       {len(tensors)} x {num_points} points")
    print(f"")
    print(f"Preprocessing: {results['timing_ms']['preprocessing']['mean']:.2f} ms "
          f"(+/- {results['timing_ms']['preprocessing']['std']:.2f})")
    print(f"Model forward: {results['timing_ms']['model_forward']['mean']:.2f} ms "
          f"(+/- {results['timing_ms']['model_forward']['std']:.2f})")
    print(f"Total:         {results['timing_ms']['total']['mean']:.2f} ms "
          f"(+/- {results['timing_ms']['total']['std']:.2f})")
    print(f"Throughput:    {1000 / results['timing_ms']['total']['mean']:.1f} samples/sec")

    # Save
    output_path = args.output
    if output_path is None:
        backbone = results["backbone"]
        output_dir = os.path.join("outputs", "reports", "benchmarks")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"benchmark_{backbone}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
