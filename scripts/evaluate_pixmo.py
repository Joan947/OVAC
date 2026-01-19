#!/usr/bin/env python3

"""

SAM3 Image Counting - Pixmo Benchmark Evaluation Script with IoM-based NMS

Evaluates SAM3 counting performance on Pixmo benchmark dataset.

Pixmo format uses dictionary with IDs as keys and uses 'label' and 'count' fields.

"""

import os
import json
import argparse
import time
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
from io import BytesIO
import urllib.request
from tqdm import tqdm

# Import counting functions from the NMS-enabled script
from count_in_images_ovac import (
    build_sam3_image_predictor,
    run_sam3_single_pass,
    run_sam3_two_pass,
    apply_nms_and_filter,
)


# ============================================================
# Image Loading (URL or Local Path)
# ============================================================

def download_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """
    Download image from URL and return PIL Image.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        image_data = response.read()

    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def load_image(image_source: str, cache_dir: str = None) -> Tuple[Image.Image, str]:
    """
    Load image from URL or local path. Optionally cache downloaded images.
    """
    # Check if it's a local file
    if os.path.isfile(image_source):
        image = Image.open(image_source).convert("RGB")
        return image, image_source

    # Assume it's a URL
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Create filename from URL
        filename = image_source.split('/')[-1]
        if not filename or '.' not in filename:
            import hashlib
            url_hash = hashlib.md5(image_source.encode()).hexdigest()
            filename = f"{url_hash}.jpg"

        cache_path = os.path.join(cache_dir, filename)

        # Check if cached
        if os.path.exists(cache_path):
            print(f"    Loading from cache: {cache_path}")
            image = Image.open(cache_path).convert("RGB")
            return image, cache_path

        # Download and cache
        print(f"    Downloading: {image_source}")
        image = download_image_from_url(image_source)
        image.save(cache_path)
        print(f"    Cached to: {cache_path}")
        return image, cache_path
    else:
        # Download without caching
        print(f"    Downloading: {image_source}")
        image = download_image_from_url(image_source)
        return image, None


# ============================================================
# Counting Pipeline with NMS
# ============================================================

def count_objects_from_image(
    image: Image.Image,
    text_prompt: str,
    model,
    device: torch.device,
    use_two_pass: bool = False,
    confidence_threshold: float = 0.5,
    max_exemplars: int = 3,
    min_obj_area: int = 0,
    iom_threshold: float = 0.5,
    apply_nms: bool = True,
) -> int:
    """
    Count objects in PIL Image using SAM3 with IoM-based NMS.

    Args:
        image: PIL Image object
        text_prompt: Text description of objects to count
        model: Pre-loaded SAM3 model
        device: Torch device
        use_two_pass: Use 2-pass refinement
        confidence_threshold: Minimum confidence score for final predictions
        max_exemplars: Number of exemplars for 2-pass
        min_obj_area: Minimum pixel area to count
        iom_threshold: IoM threshold for NMS (default: 0.5)
        apply_nms: Whether to apply IoM-based NMS (default: True)

    Returns:
        Number of objects detected after NMS and confidence filtering
    """
    # Run SAM3 (single or two-pass) with low initial threshold to get all detections
    initial_threshold = 0.0 if apply_nms else confidence_threshold

    if use_two_pass:
        inference_state, processor = run_sam3_two_pass(
            model=model,
            device=device,
            image=image,
            text_prompt=text_prompt,
            confidence_threshold=initial_threshold,
            max_exemplars=max_exemplars,
        )
    else:
        inference_state, processor = run_sam3_single_pass(
            model=model,
            device=device,
            image=image,
            text_prompt=text_prompt,
            confidence_threshold=initial_threshold,
        )

    # Apply NMS and filter
    num_objects, detections = apply_nms_and_filter(
        inference_state=inference_state,
        iom_threshold=iom_threshold,
        confidence_threshold=confidence_threshold,
        min_obj_area=min_obj_area,
        apply_nms=apply_nms,
    )

    return num_objects


# ============================================================
# Pixmo Benchmark Evaluation
# ============================================================

def evaluate_pixmo_benchmark(
    benchmark_file: str,
    output_file: str,
    device_str: str = "cuda",
    use_two_pass: bool = False,
    confidence_threshold: float = 0.5,
    max_exemplars: int = 3,
    min_obj_area: int = 0,
    iom_threshold: float = 0.5,
    apply_nms: bool = True,
    bpe_path: str = None,
    cache_dir: str = None,
    save_incremental: bool = True,
    failures_file: str = None,
) -> Dict[str, Any]:
    """
    Run SAM3 counting on Pixmo benchmark dataset and save predictions.
    Pixmo format: Dictionary with IDs as keys, uses 'label' and 'count' fields.
    """
    # Load benchmark data
    print(f"\n Loading Pixmo benchmark: {benchmark_file}")
    with open(benchmark_file, 'r') as f:
        benchmark_dict = json.load(f)

    print(f"Loaded {len(benchmark_dict)} Pixmo benchmark entries\n")

    # Build SAM3 model
    print("Building SAM3 model...")
    model, device = build_sam3_image_predictor(device_str, bpe_path)
    print("")

    # Initialize results
    predictions_dict = {}  # Store as dict to match Pixmo format
    failed_samples = []
    errors = []
    ground_truths = []
    predicted_counts = []
    processing_times = []

    # Convert dict to list for processing with progress bar
    benchmark_items = list(benchmark_dict.items())

    # Process each benchmark entry
    mode_str = '2-pass' if use_two_pass else 'single-pass'
    nms_str = f'with IoM-NMS (threshold={iom_threshold})' if apply_nms else 'without NMS'
    print(f" Starting evaluation ({mode_str} mode, {nms_str})\n")
    print("=" * 80)

    for idx, (entry_id, entry) in enumerate(tqdm(benchmark_items, desc="Processing images")):
        image_url = entry.get("image_url", "")
        label = entry.get("label", "")  # ← Pixmo uses 'label' instead of 'text'
        ground_truth = entry.get("count", 0)  # ← Pixmo uses 'count' instead of 'number'

        print(f"\n[{idx + 1}/{len(benchmark_items)}] Processing:")
        print(f"   Entry ID: {entry_id}")
        print(f"   URL: {image_url}")
        print(f"   Label: {label}")
        print(f"   Ground Truth Count: {ground_truth}")

        try:
            # Load image
            start_time = time.time()
            image, local_path = load_image(image_url, cache_dir=cache_dir)

            # Count objects with NMS
            predicted_count = count_objects_from_image(
                image=image,
                text_prompt=label,  # Use label as text prompt
                model=model,
                device=device,
                use_two_pass=use_two_pass,
                confidence_threshold=confidence_threshold,
                max_exemplars=max_exemplars,
                min_obj_area=min_obj_area,
                iom_threshold=iom_threshold,
                apply_nms=apply_nms,
            )

            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time)

            # Calculate error
            error = abs(predicted_count - ground_truth)
            print(f"   Predicted: {predicted_count} (GT: {ground_truth}, Error: {error})")
            print(f"    Time: {elapsed_time:.2f}s")

            # Store successful results (maintain Pixmo format)
            prediction_entry = {
                "image_url": image_url,
                "label": label,
                "count": predicted_count,  # ← Predicted count
            }

            # Optionally include other Pixmo fields
            if "image_sha256" in entry:
                prediction_entry["image_sha256"] = entry["image_sha256"]
            if "points" in entry:
                prediction_entry["points"] = entry["points"]

            predictions_dict[entry_id] = prediction_entry

            # Add to metrics
            errors.append(error)
            ground_truths.append(ground_truth)
            predicted_counts.append(predicted_count)

            # Save incrementally
            if save_incremental:
                with open(output_file, 'w') as f:
                    json.dump(predictions_dict, f, indent=2)

        except Exception as e:
            print(f"    Error: {str(e)}")

            # Track failed sample
            failed_entry = {
                "benchmark_index": idx,
                "benchmark_id": entry_id,
                "image_url": image_url,
                "label": label,
                "ground_truth": ground_truth,
                "error_message": str(e),
                "error_type": type(e).__name__,
            }
            failed_samples.append(failed_entry)

            # Add to predictions with failure marker
            prediction_entry = {
                "image_url": image_url,
                "label": label,
                "count": 0,
                "failed": True,
            }
            predictions_dict[entry_id] = prediction_entry

        print("-" * 80)

    # Save final predictions
    print(f"\n Saving predictions to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(predictions_dict, f, indent=2)

    # Save failures separately
    if failures_file and failed_samples:
        print(f"  Saving {len(failed_samples)} failed samples to: {failures_file}")
        with open(failures_file, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        print(f"Failures saved! You can trace back using 'benchmark_id' field.")

    # Calculate metrics
    print(f"\n Calculating evaluation metrics...\n")

    total_attempted = len(benchmark_dict)
    total_successful = len(errors)
    total_failed = len(failed_samples)

    if total_successful == 0:
        print(" No successful predictions to evaluate!")
        return {
            "total_attempted": total_attempted,
            "total_successful": 0,
            "total_failed": total_failed,
            "error": "No successful predictions"
        }

    errors_array = np.array(errors)
    ground_truths_array = np.array(ground_truths)
    predicted_counts_array = np.array(predicted_counts)

    mae = np.mean(errors_array)
    rmse = np.sqrt(np.mean(errors_array ** 2))

    # MAPE
    non_zero_gt = ground_truths_array > 0
    if non_zero_gt.sum() > 0:
        mape = np.mean(errors_array[non_zero_gt] / ground_truths_array[non_zero_gt]) * 100
    else:
        mape = float('inf')

    # Accuracy metrics
    exact_matches = np.sum(errors_array == 0)
    within_1 = np.sum(errors_array <= 1)
    within_2 = np.sum(errors_array <= 2)
    within_5 = np.sum(errors_array <= 5)

    exact_accuracy = (exact_matches / total_successful) * 100
    within_1_accuracy = (within_1 / total_successful) * 100
    within_2_accuracy = (within_2 / total_successful) * 100
    within_5_accuracy = (within_5 / total_successful) * 100

    # Timing stats
    avg_time = np.mean(processing_times) if processing_times else 0
    total_time = np.sum(processing_times)

    metrics = {
        "total_attempted": total_attempted,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "success_rate": (total_successful / total_attempted) * 100,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "exact_accuracy": float(exact_accuracy),
        "within_1_accuracy": float(within_1_accuracy),
        "within_2_accuracy": float(within_2_accuracy),
        "within_5_accuracy": float(within_5_accuracy),
        "avg_processing_time": float(avg_time),
        "total_processing_time": float(total_time),
        "mode": "2-pass" if use_two_pass else "single-pass",
        "confidence_threshold": confidence_threshold,
        "max_exemplars": max_exemplars if use_two_pass else None,
        "min_obj_area": min_obj_area,
        "iom_threshold": iom_threshold if apply_nms else None,
        "nms_applied": apply_nms,
    }

    return metrics


# ============================================================
# Results Display
# ============================================================

def print_evaluation_results(metrics: Dict[str, Any]):
    """
    Pretty print evaluation metrics.
    """
    print("\n" + "=" * 80)
    print(" PIXMO EVALUATION RESULTS")
    print("=" * 80)

    # Show success/failure stats
    print(f"\n Dataset Statistics:")
    print(f"   Total Samples: {metrics['total_attempted']}")
    print(f"   Successful: {metrics['total_successful']} ({metrics['success_rate']:.2f}%)")
    print(f"   Failed: {metrics['total_failed']}")
    if metrics['total_failed'] > 0:
        print(f"     {metrics['total_failed']} samples failed (excluded from accuracy metrics)")

    print(f"\nConfiguration:")
    print(f"   Mode: {metrics['mode']}")
    print(f"   Confidence Threshold: {metrics['confidence_threshold']}")
    if metrics.get('nms_applied'):
        print(f"   NMS: IoM-based (threshold={metrics['iom_threshold']})")
    else:
        print(f"   NMS: Disabled")
    if metrics['max_exemplars']:
        print(f"   Max Exemplars: {metrics['max_exemplars']}")
    print(f"   Min Object Area: {metrics['min_obj_area']} pixels")

    print(f"\n Error Metrics (based on {metrics['total_successful']} successful predictions):")
    print(f"   MAE (Mean Absolute Error): {metrics['mae']:.3f}")
    print(f"   RMSE (Root Mean Squared Error): {metrics['rmse']:.3f}")
    if metrics['mape'] != float('inf'):
        print(f"   MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")

    print(f"\n Accuracy Metrics (based on {metrics['total_successful']} successful predictions):")
    print(f"   Exact Match: {metrics['exact_accuracy']:.2f}%")
    print(f"   Within ±1: {metrics['within_1_accuracy']:.2f}%")
    print(f"   Within ±2: {metrics['within_2_accuracy']:.2f}%")
    print(f"   Within ±5: {metrics['within_5_accuracy']:.2f}%")

    print(f"\n  Performance:")
    print(f"   Avg Time/Image: {metrics['avg_processing_time']:.2f}s")
    print(f"   Total Time: {metrics['total_processing_time']:.2f}s ({metrics['total_processing_time']/60:.2f} min)")

    print("\n" + "=" * 80)


# ============================================================
# CLI Interface
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser(
        "SAM3 Counting - Pixmo Benchmark Evaluation with IoM-NMS",
        description="Evaluate SAM3 counting performance on Pixmo benchmark dataset with IoM-based NMS",
    )

    # Input/Output
    parser.add_argument(
        "--benchmark_file",
        type=str,
        required=True,
        help="Path to Pixmo benchmark JSON file (dict format with 'label' and 'count' fields)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save predictions JSON (same Pixmo format with predicted counts)",
    )

    parser.add_argument(
        "--metrics_file",
        type=str,
        default="",
        help="Optional path to save evaluation metrics as JSON",
    )

    parser.add_argument(
        "--failures_file",
        type=str,
        default="",
        help="Optional file to save failed samples (download/prediction errors)",
    )

    # Model parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda', 'cpu', or 'mps'",
    )

    parser.add_argument(
        "--bpe_path",
        type=str,
        default="/project/advdls25/jowusu1/CountVid/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        help="Path to BPE vocabulary file",
    )

    # Counting parameters
    parser.add_argument(
        "--use_two_pass",
        action="store_true",
        help="Enable 2-pass refinement (text → exemplars → refined predictions)",
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to keep detection (default: 0.5)",
    )

    parser.add_argument(
        "--max_exemplars",
        type=int,
        default=3,
        help="Number of exemplar boxes for 2-pass (only used if --use_two_pass)",
    )

    parser.add_argument(
        "--min_obj_area",
        type=int,
        default=0,
        help="Minimum pixel area to count as object",
    )

    # NMS parameters
    parser.add_argument(
        "--iom_threshold",
        type=float,
        default=0.5,
        help="IoM threshold for NMS post-processing (default: 0.5)",
    )

    parser.add_argument(
        "--no_nms",
        action="store_true",
        help="Disable IoM-based NMS post-processing",
    )

    # Cache options
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./image_cache",
        help="Directory to cache downloaded images (set to empty string to disable)",
    )

    parser.add_argument(
        "--no_incremental_save",
        action="store_true",
        help="Disable incremental saving (only save at end)",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.benchmark_file):
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark_file}")

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    else:
        device_str = args.device

    # Handle cache directory
    cache_dir = args.cache_dir if args.cache_dir else None

    # Print configuration
    print("\n" + "=" * 80)
    print(" SAM3 Counting - Pixmo Benchmark Evaluation (with IoM-NMS)")
    print("=" * 80)
    print(f" Benchmark : {args.benchmark_file}")
    print(f" Output    : {args.output_file}")
    if args.failures_file:
        print(f"  Failures  : {args.failures_file}")
    print(f" Device    : {device_str}")
    print(f" Mode      : {'2-pass refinement' if args.use_two_pass else 'Single-pass'}")
    print(f"  Confidence: {args.confidence_threshold}")
    if not args.no_nms:
        print(f" NMS (IoM) : Enabled (threshold={args.iom_threshold})")
    else:
        print(f" NMS (IoM) : Disabled")
    if args.use_two_pass:
        print(f" Exemplars : {args.max_exemplars}")
    print(f" Min Area  : {args.min_obj_area} pixels")
    if cache_dir:
        print(f" Cache Dir : {cache_dir}")
    print("=" * 80)

    # Run evaluation
    try:
        metrics = evaluate_pixmo_benchmark(
            benchmark_file=args.benchmark_file,
            output_file=args.output_file,
            device_str=device_str,
            use_two_pass=args.use_two_pass,
            confidence_threshold=args.confidence_threshold,
            max_exemplars=args.max_exemplars,
            min_obj_area=args.min_obj_area,
            iom_threshold=args.iom_threshold,
            apply_nms=not args.no_nms,
            bpe_path=args.bpe_path,
            cache_dir=cache_dir,
            save_incremental=not args.no_incremental_save,
            failures_file=args.failures_file,
        )

        # Print results
        print_evaluation_results(metrics)

        # Save metrics if requested
        if args.metrics_file:
            print(f"\n Saving metrics to: {args.metrics_file}")
            os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)
            with open(args.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print("Metrics saved!")

        print("\nEvaluation complete!\n")

    except KeyboardInterrupt:
        print("\n\n  Evaluation interrupted by user.")
        print(f" Partial results saved to: {args.output_file}")

    except Exception as e:
        print(f"\n Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
