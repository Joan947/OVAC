
#!/usr/bin/env python3

"""
SAM3 Image Counting Benchmark Evaluation Script with IoM-based NMS

Evaluates SAM3 counting performance on benchmark datasets ("CountBench").
Supports both URL-based and local image paths and integrates robust
multi-strategy downloading so failed URLs are retried internally
(before being written to failures.json).
"""

import os
import json
import argparse
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from io import BytesIO
import urllib.request
import urllib.error
from tqdm import tqdm

# Optional: requests for more robust downloading
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print(" 'requests' library not found. Install with: pip install requests")

# Import counting functions from the NMS-enabled script
from count_in_images_ovac import (
    build_sam3_image_predictor,
    run_sam3_single_pass,
    run_sam3_two_pass,
    apply_nms_and_filter,
)


# ============================================================
# Robust Download Utilities (inlined from retry_failed_downloads.py)
# ============================================================

def get_cache_filename(url: str) -> str:
    """Generate cache filename from URL (same logic as retry_failed_downloads)."""
    filename = url.split("/")[-1].split("?")[0]
    if not filename or "." not in filename:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        filename = f"{url_hash}.jpg"
    return filename


def download_with_urllib(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Strategy 1: Download using urllib with various user agents."""
    user_agents = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
        "Gecko/20100101 Firefox/121.0",
        # Safari on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        # Chrome on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Mobile Chrome
        "Mozilla/5.0 (Linux; Android 10; SM-G973F) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    ]
    for user_agent in user_agents:
        try:
            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                          "image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
            print(f"   urllib succeeded (UA: {user_agent[:30]}...)")
            return image
        except Exception:
            continue
    return None


def download_with_requests(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Strategy 2: Download using requests library with headers."""
    if not HAS_REQUESTS:
        return None
    try:
        session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": url,
            "DNT": "1",
            "Connection": "keep-alive",
        }
        response = session.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            verify=True,
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("   requests succeeded")
        return image
    except Exception:
        return None


def download_with_requests_no_verify(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Strategy 3: Download using requests with SSL verification disabled."""
    if not HAS_REQUESTS:
        return None
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36",
        }
        response = session.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            verify=False,
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("   requests (no SSL verify) succeeded")
        return image
    except Exception:
        return None


def download_image_robust(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Try all download strategies in order until one succeeds."""
    strategies = [
        ("urllib (multiple UAs)", lambda: download_with_urllib(url, timeout)),
        ("requests library", lambda: download_with_requests(url, timeout)),
        (
            "requests (no SSL verify)",
            lambda: download_with_requests_no_verify(url, timeout),
        ),
    ]
    for strategy_name, fn in strategies:
        try:
            image = fn()
            if image is not None:
                return image
        except Exception:
            continue
    return None


# ============================================================
# Image Loading (URL or Local Path, with robust retries)
# ============================================================

def load_image(
    image_source: str,
    cache_dir: str = None,
    download_timeout: int = 30,
    max_download_retries: int = 3,
) -> Tuple[Image.Image, Optional[str]]:
    """Load image from URL or local path with robust retry + caching.

    Args:
        image_source: URL or local file path
        cache_dir: Directory to cache downloaded images (optional)
        download_timeout: Timeout per download attempt
        max_download_retries: Max retry attempts per image

    Returns:
        (PIL Image, local_path or cached_path)
    """
    # Local file path
    if os.path.isfile(image_source):
        image = Image.open(image_source).convert("RGB")
        return image, image_source

    # URL case
    url = image_source
    local_path = None

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        filename = get_cache_filename(url)
        cache_path = os.path.join(cache_dir, filename)

        # If already cached, load directly
        if os.path.exists(cache_path):
            print(f"    Loading from cache: {cache_path}")
            image = Image.open(cache_path).convert("RGB")
            return image, cache_path

        # Not cached: try robust download with retries
        print(f"    Downloading (robust) : {url}")
        image = None
        for attempt in range(max_download_retries):
            if attempt > 0:
                print(f"    Retry {attempt}/{max_download_retries - 1}")
                time.sleep(1.0)
            image = download_image_robust(url, timeout=download_timeout)
            if image is not None:
                break

        if image is None:
            raise RuntimeError(f"All download strategies failed for URL: {url}")

        # Save to cache
        image.save(cache_path)
        print(f"    Cached to: {cache_path}")
        return image, cache_path

    # No cache_dir: still use robust strategies, but do not save
    print(f"    Downloading (no cache, robust) : {url}")
    image = None
    for attempt in range(max_download_retries):
        if attempt > 0:
            print(f"    Retry {attempt}/{max_download_retries - 1}")
            time.sleep(1.0)
        image = download_image_robust(url, timeout=download_timeout)
        if image is not None:
            break

    if image is None:
        raise RuntimeError(f"All download strategies failed for URL: {url}")

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
    """Count objects in PIL Image using SAM3 with IoM-based NMS."""
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
# Benchmark Evaluation
# ============================================================

def evaluate_benchmark(
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
    download_timeout: int = 30,
    max_download_retries: int = 3,
) -> Dict[str, Any]:
    """Run SAM3 counting on CountBench benchmark and save predictions."""
    print(f"\n Loading benchmark: {benchmark_file}")
    with open(benchmark_file, "r") as f:
        benchmark_data = json.load(f)

    print(f" Loaded {len(benchmark_data)} benchmark entries\n")

    # Build SAM3 model
    print(" Building SAM3 model...")
    model, device = build_sam3_image_predictor(device_str, bpe_path)
    print("")

    # Initialize results
    predictions: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []
    errors: List[float] = []
    ground_truths: List[float] = []
    predicted_counts: List[float] = []
    processing_times: List[float] = []

    mode_str = "2-pass" if use_two_pass else "single-pass"
    nms_str = f"with IoM-NMS (threshold={iom_threshold})" if apply_nms else "without NMS"
    print(f" Starting evaluation ({mode_str} mode, {nms_str})\n")
    print("=" * 80)

    for idx, entry in enumerate(tqdm(benchmark_data, desc="Processing images")):
        image_url = entry.get("image_url", "")
        text_prompt = entry.get("text", "")
        ground_truth = entry.get("number", 0)
        benchmark_id = entry.get("id", idx)

        print(f"\n[{idx + 1}/{len(benchmark_data)}] Processing:")
        print(f"   Benchmark Index: {idx}")
        if "id" in entry:
            print(f"   Benchmark ID: {benchmark_id}")
        print(f"   URL: {image_url}")
        print(f"   Text: {text_prompt}")
        print(f"   Ground Truth: {ground_truth}")

        try:
            # Load image (robust download + cache)
            start_time = time.time()
            image, local_path = load_image(
                image_url,
                cache_dir=cache_dir,
                download_timeout=download_timeout,
                max_download_retries=max_download_retries,
            )

            # Count objects with NMS
            predicted_count = count_objects_from_image(
                image=image,
                text_prompt=text_prompt,
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

            # Error
            error = abs(predicted_count - ground_truth)
            print(f"   Predicted: {predicted_count} (GT: {ground_truth}, Error: {error})")
            print(f"    Time: {elapsed_time:.2f}s")

            prediction_entry: Dict[str, Any] = {
                "image_url": image_url,
                "text": text_prompt,
                "number": predicted_count,
            }
            if "id" in entry:
                prediction_entry["id"] = benchmark_id

            predictions.append(prediction_entry)

            errors.append(error)
            ground_truths.append(ground_truth)
            predicted_counts.append(predicted_count)

            if save_incremental:
                with open(output_file, "w") as f:
                    json.dump(predictions, f, indent=2)

        except Exception as e:
            print(f"    Error: {str(e)}")

            failed_entry = {
                "benchmark_index": idx,
                "benchmark_id": benchmark_id,
                "image_url": image_url,
                "text": text_prompt,
                "ground_truth": ground_truth,
                "error_message": str(e),
                "error_type": type(e).__name__,
            }
            failed_samples.append(failed_entry)

            prediction_entry = {
                "image_url": image_url,
                "text": text_prompt,
                "number": 0,
                "failed": True,
            }
            if "id" in entry:
                prediction_entry["id"] = benchmark_id
            predictions.append(prediction_entry)

        print("-" * 80)

    # Save final predictions
    print(f"\n Saving predictions to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    # Save failures
    if failures_file and failed_samples:
        print(f"  Saving {len(failed_samples)} failed samples to: {failures_file}")
        with open(failures_file, "w") as f:
            json.dump(failed_samples, f, indent=2)
        print(" Failures saved! You can trace back using 'benchmark_index' field.")

    # Metrics
    print("\n Calculating evaluation metrics...\n")

    total_attempted = len(benchmark_data)
    total_successful = len(errors)
    total_failed = len(failed_samples)

    if total_successful == 0:
        print(" No successful predictions to evaluate!")
        return {
            "total_attempted": total_attempted,
            "total_successful": 0,
            "total_failed": total_failed,
            "error": "No successful predictions",
        }

    errors_array = np.array(errors)
    ground_truths_array = np.array(ground_truths)
    predicted_counts_array = np.array(predicted_counts)

    mae = np.mean(errors_array)
    rmse = np.sqrt(np.mean(errors_array ** 2))

    non_zero_gt = ground_truths_array > 0
    if non_zero_gt.sum() > 0:
        mape = (
            np.mean(errors_array[non_zero_gt] / ground_truths_array[non_zero_gt])
            * 100
        )
    else:
        mape = float("inf")

    exact_matches = np.sum(errors_array == 0)
    within_1 = np.sum(errors_array <= 1)
    within_2 = np.sum(errors_array <= 2)
    within_5 = np.sum(errors_array <= 5)

    exact_accuracy = (exact_matches / total_successful) * 100
    within_1_accuracy = (within_1 / total_successful) * 100
    within_2_accuracy = (within_2 / total_successful) * 100
    within_5_accuracy = (within_5 / total_successful) * 100

    avg_time = np.mean(processing_times) if processing_times else 0
    total_time = np.sum(processing_times)

    metrics: Dict[str, Any] = {
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
        "download_timeout": download_timeout,
        "max_download_retries": max_download_retries,
    }

    return metrics


# ============================================================
# Results Display
# ============================================================

def print_evaluation_results(metrics: Dict[str, Any]):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS")
    print("=" * 80)

    print("\n Dataset Statistics:")
    print(f"   Total Samples: {metrics['total_attempted']}")
    print(
        f"   Successful: {metrics['total_successful']} "
        f"({metrics['success_rate']:.2f}%)"
    )
    print(f"   Failed: {metrics['total_failed']}")
    if metrics["total_failed"] > 0:
        print(
            f"     {metrics['total_failed']} samples failed "
            f"(excluded from accuracy metrics)"
        )

    print("\n Configuration:")
    print(f"   Mode: {metrics['mode']}")
    print(f"   Confidence Threshold: {metrics['confidence_threshold']}")
    if metrics.get("nms_applied"):
        print(f"   NMS: IoM-based (threshold={metrics['iom_threshold']})")
    else:
        print("   NMS: Disabled")
    if metrics.get("max_exemplars"):
        print(f"   Max Exemplars: {metrics['max_exemplars']}")
    print(f"   Min Object Area: {metrics['min_obj_area']} pixels")
    print(f"   Download Timeout: {metrics['download_timeout']}s")
    print(f"   Max Download Retries: {metrics['max_download_retries']}")

    print(
        f"\n Error Metrics (based on {metrics['total_successful']} "
        f"successful predictions):"
    )
    print(f"   MAE (Mean Absolute Error): {metrics['mae']:.3f}")
    print(f"   RMSE (Root Mean Squared Error): {metrics['rmse']:.3f}")
    if metrics["mape"] != float("inf"):
        print(f"   MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")

    print(
        f"\n Accuracy Metrics (based on {metrics['total_successful']} "
        f"successful predictions):"
    )
    print(f"   Exact Match: {metrics['exact_accuracy']:.2f}%")
    print(f"   Within ±1: {metrics['within_1_accuracy']:.2f}%")
    print(f"   Within ±2: {metrics['within_2_accuracy']:.2f}%")
    print(f"   Within ±5: {metrics['within_5_accuracy']:.2f}%")

    print("\n  Performance:")
    print(f"   Avg Time/Image: {metrics['avg_processing_time']:.2f}s")
    print(
        f"   Total Time: {metrics['total_processing_time']:.2f}s "
        f"({metrics['total_processing_time'] / 60:.2f} min)"
    )

    print("\n" + "=" * 80)


# ============================================================
# CLI Interface
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser(
        "SAM3 Counting Benchmark Evaluation with IoM-NMS",
        description=(
            "Evaluate SAM3 counting performance on benchmark datasets "
            "with IoM-based NMS and robust URL downloading"
        ),
    )

    # Input/Output
    parser.add_argument(
        "--benchmark_file",
        type=str,
        required=True,
        help=(
            "Path to benchmark JSON file with format: "
            "[{image_url, text, number, (optional) id}, ...]"
        ),
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help=(
            "Path to save predictions JSON (same format as input, "
            "with predicted counts)"
        ),
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

    # Model
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda', 'cpu', or 'mps'",
    )

    parser.add_argument(
        "--bpe_path",
        type=str,
        default=(
            "/project/advdls25/jowusu1/CountVid/sam3/assets/"
            "bpe_simple_vocab_16e6.txt.gz"
        ),
        help="Path to BPE vocabulary file",
    )

    # Counting
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

    # NMS
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

    # Cache
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./image_cache",
        help=(
            "Directory to cache downloaded images (set to empty string to "
            "disable caching)"
        ),
    )

    parser.add_argument(
        "--no_incremental_save",
        action="store_true",
        help="Disable incremental saving (only save at end)",
    )

    # Robust download params
    parser.add_argument(
        "--download_timeout",
        type=int,
        default=30,
        help="Download timeout in seconds for each attempt (default: 30)",
    )

    parser.add_argument(
        "--max_download_retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts per image URL (default: 3)",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.benchmark_file):
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark_file}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    else:
        device_str = args.device

    cache_dir = args.cache_dir if args.cache_dir else None

    print("\n" + "=" * 80)
    print(" SAM3 Counting Benchmark Evaluation (with IoM-NMS + robust downloads)")
    print("=" * 80)
    print(f" Benchmark : {args.benchmark_file}")
    print(f" Output    : {args.output_file}")
    if args.failures_file:
        print(f"  Failures  : {args.failures_file}")
    print(f" Device    : {device_str}")
    print(
        f" Mode      : "
        f"{'2-pass refinement' if args.use_two_pass else 'Single-pass'}"
    )
    print(f"  Confidence: {args.confidence_threshold}")
    if not args.no_nms:
        print(f" NMS (IoM) : Enabled (threshold={args.iom_threshold})")
    else:
        print(" NMS (IoM) : Disabled")
    if args.use_two_pass:
        print(f" Exemplars : {args.max_exemplars}")
    print(f" Min Area  : {args.min_obj_area} pixels")
    if cache_dir:
        print(f" Cache Dir : {cache_dir}")
    print(f"  Download Timeout   : {args.download_timeout}s")
    print(f" Max Download Retries: {args.max_download_retries}")
    print("=" * 80)

    try:
        metrics = evaluate_benchmark(
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
            download_timeout=args.download_timeout,
            max_download_retries=args.max_download_retries,
        )

        print_evaluation_results(metrics)

        if args.metrics_file:
            print(f"\n Saving metrics to: {args.metrics_file}")
            os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)
            with open(args.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(" Metrics saved!")

        print("\n Evaluation complete!\n")

    except KeyboardInterrupt:
        print("\n\n  Evaluation interrupted by user.")
        print(f" Partial results saved to: {args.output_file}")

    except Exception as e:
        print(f"\n Evaluation failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
