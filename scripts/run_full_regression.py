"""Run full regression pipeline: data -> train -> eval -> report.

Executes the complete 15-step regression pipeline for the PointNet baseline,
outputting all artifacts to outputs/runs/{run_name}/.

Usage:
    python scripts/run_full_regression.py \
        --config configs/config.yaml \
        --raw-root label \
        --run-name newdata_pointnet_baseline_v1

    # Skip training (reuse existing checkpoint):
    python scripts/run_full_regression.py \
        --config configs/config.yaml \
        --raw-root label \
        --run-name newdata_pointnet_baseline_v1 \
        --skip-train
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def step(cmd, name, run_dir, log_dir, step_num, total_steps):
    """Execute a pipeline step, logging output. Returns (success, elapsed)."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}/{total_steps}: {name}")
    print(f"{'='*70}")
    print(f"CMD: {' '.join(cmd)}")
    print()

    log_path = os.path.join(log_dir, f"step{step_num:02d}_{name.replace(' ', '_')}.log")

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=7200,  # 2h max per step
            )
            log_f.write(result.stdout)
            # Also print to console
            print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[FAIL] Step {step_num} ({name}) failed with return code {result.returncode}")
            print(f"  Log: {log_path}")
            return False, elapsed

        print(f"[OK] Step {step_num} ({name}) completed in {elapsed:.1f}s")
        return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[FAIL] Step {step_num} ({name}) timed out after {elapsed:.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL] Step {step_num} ({name}) exception: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run full regression pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--raw-root", type=str, default="label",
                        help="Raw PCD data root (e.g., label/)")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Run name (e.g., newdata_pointnet_baseline_v1)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, reuse existing checkpoint")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Skip data preparation (prepare_pcd_dataset + rebuild_split)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    # Paths
    run_dir = os.path.join("outputs", "runs", args.run_name)
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    proto_dir = os.path.join(run_dir, "prototypes")
    gallery_dir = os.path.join(run_dir, "gallery")
    report_dir = os.path.join(run_dir, "reports")
    embed_dir = os.path.join(run_dir, "embeddings")
    deploy_dir = os.path.join(run_dir, "deploy")

    for d in [run_dir, log_dir, ckpt_dir, proto_dir, gallery_dir, report_dir, embed_dir, deploy_dir]:
        os.makedirs(d, exist_ok=True)

    # Also ensure legacy dirs exist (some scripts write there too)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/prototypes", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # Paths for artifacts
    dataset_all = "dataset_all"
    dataset = "dataset"
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    # Also link to legacy location
    best_ckpt_legacy = "outputs/checkpoints/best.pt"
    prototypes = os.path.join(proto_dir, "baseline_prototypes.pt")
    threshold_json = os.path.join(proto_dir, "baseline_threshold.json")
    gallery = os.path.join(gallery_dir, "baseline_train_gallery.pt")

    python = os.environ.get("REGRESSION_PYTHON", None)
    if python is None:
        # Try uv run first, fall back to sys.executable
        import shutil
        if shutil.which("uv"):
            python = "uv"
        else:
            python = sys.executable

    # If using uv, prepend "run" to all commands
    uv_prefix = [python, "run"] if os.path.basename(python) == "uv" else [python]

    steps = []
    step_num = 0
    total_steps = 15

    # ---- Step 1: prepare_pcd_dataset ----
    if not args.skip_data_prep:
        step_num += 1
        steps.append((step_num, "prepare_pcd_dataset", [
            *uv_prefix, "scripts/prepare_pcd_dataset.py",
            "--flat",
            "--input-root", args.raw_root,
            "--output-root", dataset_all,
            "--config", args.config,
        ]))
    else:
        total_steps -= 1

    # ---- Step 2: rebuild_split ----
    if not args.skip_data_prep:
        step_num += 1
        steps.append((step_num, "rebuild_split", [
            *uv_prefix, "scripts/rebuild_split.py",
            "--input-root", dataset_all,
            "--output-root", dataset,
            "--config", args.config,
            "--train-ratio", str(args.train_ratio),
            "--val-ratio", str(args.val_ratio),
            "--test-ratio", str(args.test_ratio),
        ]))
    else:
        total_steps -= 1

    # ---- Step 3: validate_dataset ----
    step_num += 1
    steps.append((step_num, "validate_dataset", [
        *uv_prefix, "scripts/validate_dataset.py",
        "--dataset-root", dataset,
        "--config", args.config,
    ]))

    # ---- Step 4: check_class_coverage (pre-training) ----
    step_num += 1
    steps.append((step_num, "check_class_coverage_pre", [
        *uv_prefix, "scripts/check_class_coverage.py",
        "--dataset-root", dataset,
        "--config", args.config,
    ]))

    # ---- Step 5: train ----
    if not args.skip_train:
        step_num += 1
        steps.append((step_num, "train", [
            *uv_prefix, "scripts/train.py",
            "--config", args.config,
        ]))

        # After training, copy checkpoint to run dir
        # We'll do this after the step succeeds
    else:
        total_steps -= 1
        print(f"\n[SKIP] Training step skipped (--skip-train)")
        # Verify checkpoint exists
        if not os.path.exists(best_ckpt_legacy):
            print(f"[ERROR] --skip-train but no checkpoint at {best_ckpt_legacy}")
            sys.exit(1)

    # ---- Step 6: evaluate ----
    step_num += 1
    steps.append((step_num, "evaluate", [
        *uv_prefix, "scripts/evaluate.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--split", "test",
        "--output-dir", os.path.join(report_dir, "classification"),
    ]))

    # ---- Step 7: extract test embeddings ----
    step_num += 1
    test_emb_path = os.path.join(embed_dir, "test_embeddings.npz")
    steps.append((step_num, "extract_test_embeddings", [
        *uv_prefix, "scripts/extract_embeddings.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--split", "test",
        "--output", test_emb_path,
    ]))

    step_num += 1
    steps.append((step_num, "evaluate_embeddings", [
        *uv_prefix, "scripts/evaluate_embeddings.py",
        "--input", test_emb_path,
        "--output-dir", os.path.join(report_dir, "embeddings"),
        "--negative-label", "19",
        "--class-names", "configs/class_names.json",
        "--config", args.config,
    ]))

    # ---- Step 8: build_prototypes ----
    step_num += 1
    steps.append((step_num, "build_prototypes", [
        *uv_prefix, "scripts/build_prototypes.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--split", "train",
        "--output", prototypes,
    ]))

    # ---- Step 9: search_threshold ----
    step_num += 1
    steps.append((step_num, "search_threshold", [
        *uv_prefix, "scripts/search_threshold.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--prototypes", prototypes,
        "--split", "val",
        "--output", threshold_json,
    ]))

    # ---- Step 10: evaluate_ood with threshold sweep ----
    step_num += 1
    steps.append((step_num, "evaluate_ood", [
        *uv_prefix, "scripts/evaluate_ood.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--prototypes", prototypes,
        "--threshold-json", threshold_json,
        "--split", "test",
        "--output-dir", os.path.join(report_dir, "ood"),
        "--thresholds", "0.68,0.75,0.80,0.85,0.90,0.91,0.92,0.94,0.96",
    ]))

    # ---- Step 11: build_gallery ----
    step_num += 1
    steps.append((step_num, "build_gallery", [
        *uv_prefix, "scripts/build_gallery.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--split", "train",
        "--output", gallery,
    ]))

    # ---- Step 12: evaluate_retrieval ----
    step_num += 1
    steps.append((step_num, "evaluate_retrieval", [
        *uv_prefix, "scripts/evaluate_retrieval.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--gallery", gallery,
        "--threshold-json", threshold_json,
        "--split", "test",
        "--output-dir", os.path.join(report_dir, "retrieval"),
        "--thresholds", "0.68,0.75,0.80,0.85,0.90,0.91,0.92,0.94,0.96",
    ]))

    # ---- Step 13: check_class_coverage (post-training, with prototypes/gallery) ----
    step_num += 1
    steps.append((step_num, "check_class_coverage_post", [
        *uv_prefix, "scripts/check_class_coverage.py",
        "--dataset-root", dataset,
        "--config", args.config,
        "--prototypes", prototypes,
        "--gallery", gallery,
    ]))

    # ---- Step 14: export_inference_bundle ----
    step_num += 1
    bundle_dir = os.path.join(deploy_dir, "inference_bundle")
    steps.append((step_num, "export_bundle", [
        *uv_prefix, "scripts/export_inference_bundle.py",
        "--config", args.config,
        "--checkpoint", best_ckpt_legacy,
        "--prototypes", prototypes,
        "--threshold-json", threshold_json,
        "--gallery", gallery,
        "--coverage-report", "outputs/reports/class_coverage_report.json",
        "--output", bundle_dir,
    ]))

    # ---- Step 15: generate_final_report ----
    step_num += 1
    steps.append((step_num, "generate_final_report", [
        *uv_prefix, "scripts/generate_final_report.py",
        "--config", args.config,
        "--run-name", args.run_name,
        "--run-dir", run_dir,
        "--checkpoint", best_ckpt_legacy,
        "--prototypes", prototypes,
        "--threshold-json", threshold_json,
        "--gallery", gallery,
        "--output-dir", run_dir,
    ]))

    total_steps = len(steps)

    # =========================================================================
    # Execute pipeline
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# FULL REGRESSION: {args.run_name}")
    print(f"# Steps: {total_steps}")
    print(f"# Config: {args.config}")
    print(f"# Raw data: {args.raw_root}")
    print(f"# Skip train: {args.skip_train}")
    print(f"# Skip data prep: {args.skip_data_prep}")
    print(f"# Output: {run_dir}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#'*70}")

    regression_results = {
        "run_name": args.run_name,
        "config": args.config,
        "raw_root": args.raw_root,
        "started_at": datetime.now().isoformat(),
        "steps": [],
        "status": "RUNNING",
    }

    pipeline_start = time.time()

    for step_num, step_name, cmd in steps:
        # Special handling after training: copy checkpoint to run dir
        if step_name == "train":
            ok, elapsed = step(cmd, step_name, run_dir, log_dir, step_num, total_steps)
            regression_results["steps"].append({
                "step": step_num, "name": step_name,
                "success": ok, "elapsed": round(elapsed, 1),
            })
            if not ok:
                regression_results["status"] = "FAILED"
                regression_results["failed_step"] = f"{step_num}: {step_name}"
                _save_summary(regression_results, run_dir, pipeline_start)
                sys.exit(1)
            # Copy best checkpoint to run dir
            import shutil
            src = best_ckpt_legacy
            if os.path.exists(src):
                shutil.copy2(src, best_ckpt)
                print(f"  Copied checkpoint to {best_ckpt}")
            else:
                print(f"  [WARN] Checkpoint not found at {src}")

        else:
            ok, elapsed = step(cmd, step_name, run_dir, log_dir, step_num, total_steps)
            regression_results["steps"].append({
                "step": step_num, "name": step_name,
                "success": ok, "elapsed": round(elapsed, 1),
            })
            if not ok:
                regression_results["status"] = "FAILED"
                regression_results["failed_step"] = f"{step_num}: {step_name}"
                _save_summary(regression_results, run_dir, pipeline_start)
                sys.exit(1)

    # All steps completed
    regression_results["status"] = "COMPLETED"
    _save_summary(regression_results, run_dir, pipeline_start)

    print(f"\n{'#'*70}")
    print(f"# REGRESSION COMPLETE: {args.run_name}")
    print(f"# Status: {regression_results['status']}")
    print(f"# Total time: {regression_results['total_elapsed']:.1f}s")
    print(f"# Output: {run_dir}")
    print(f"# Final report: {os.path.join(run_dir, 'final_report.html')}")
    print(f"# Comparison: {os.path.join(run_dir, 'comparison_to_previous.html')}")
    print(f"{'#'*70}")


def _save_summary(regression_results, run_dir, pipeline_start):
    regression_results["completed_at"] = datetime.now().isoformat()
    regression_results["total_elapsed"] = round(time.time() - pipeline_start, 1)

    summary_path = os.path.join(run_dir, "regression_summary.json")
    with open(summary_path, "w") as f:
        json.dump(regression_results, f, indent=2, ensure_ascii=False)
    print(f"\nRegression summary saved to {summary_path}")


if __name__ == "__main__":
    main()
