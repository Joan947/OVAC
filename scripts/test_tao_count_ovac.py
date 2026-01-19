import json
import argparse
import subprocess
import shlex
import os
import time
from datetime import datetime

parser = argparse.ArgumentParser("Testing OVAC on TAO-Count", add_help=False)

parser.add_argument(
    "--output_file",
    type=str,
    default="tao-count-predicted-ovac.json",
    help="file where to save predicted counts",
)

parser.add_argument(
    "--data_dir",
    type=str,
    default="data/VideoCount/TAO-Count",
    help="path to TAO-Count dataset",
)

# Text control
parser.add_argument("--no_text", action="store_true", help="whether or not to drop the text")

# OVAC args (aligned with count_in_videos_reid_adaptive.py)
parser.add_argument(
    "--sam3_checkpoint",
    type=str,
    default="countvid/lib/python3.10/site-packages/assets/sam3.pt",
    help="path to SAM3 checkpoint",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="device to use for inference, e.g. 'cuda', 'cpu', 'mps'",
)

parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="min SAM3 score to keep a detection",
)

parser.add_argument(
    "--min_obj_area",
    type=int,
    default=0,
    help="min pixel area for mask to be considered a distinct object",
)

parser.add_argument(
    "--reid_similarity_threshold",
    type=float,
    default=0.75,
    help="similarity threshold for re-identification",
)

parser.add_argument(
    "--max_lost_frames",
    type=int,
    default=50,
    help="maximum frames before lost track is removed",
)

parser.add_argument(
    "--mode",
    type=str,
    default="balanced",
    choices=["sequential", "crowd", "static", "balanced"],
    help="Tracking mode: sequential(cars), crowd(penguins), static(clothes), balanced(general)",
)

parser.add_argument(
    "--sample_frames",
    type=int,
    default=0,
    help="number of video frames to sample; 0 means use all frames",
)

parser.add_argument(
    "--downsample_factor",
    type=float,
    default=3.0,
    help="downsample total number of frames by this factor when sample_frames=0 (default: 3.0 for efficiency)",
)

# Optional visualization passthrough
parser.add_argument(
    "--save_final_video",
    action="store_true",
    help="save final tracking visualization as an MP4 video",
)

parser.add_argument(
    "--output_fps",
    type=float,
    default=30.0,
    help="frames per second of output video",
)

parser.add_argument(
    "--font_size",
    type=int,
    default=8,
    help="font size for visualization",
)

parser.add_argument(
    "--save_T",
    action="store_true",
    help="save tracking data as .npz file",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="directory where to save visualization outputs",
)

# Re-evaluation features
parser.add_argument(
    "--skipped_log",
    type=str,
    default="tao_skipped_prompts.json",
    help="file to log skipped prompts for re-evaluation",
)

parser.add_argument(
    "--retry_skipped",
    action="store_true",
    help="retry previously skipped prompts from skipped_log",
)

parser.add_argument(
    "--max_retries",
    type=int,
    default=3,
    help="maximum number of retries for each skipped prompt",
)

parser.add_argument(
    "--timeout",
    type=int,
    default=600,
    help="timeout in seconds for each video/prompt processing (default: 600s)",
)

args = parser.parse_args()

# Dataset paths (same structure as TAO scripts)
gt_file = os.path.join(args.data_dir, "anno", "TAO-count-gt.json")
video_dir = os.path.join(args.data_dir, "frames")

# Load ground truth
with open(gt_file) as tao_count_json:
    tao_count_gt = json.load(tao_count_json)

# Load or create skipped prompts log
skipped_log = {}
if os.path.exists(args.skipped_log):
    with open(args.skipped_log, 'r') as f:
        try:
            skipped_log = json.load(f)
        except json.JSONDecodeError:
            skipped_log = {}

# Initialize or load existing output file
os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
existing_results = {}
if os.path.exists(args.output_file):
    with open(args.output_file, 'r') as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = {}

def save_skipped_log():
    """Save the current skipped prompts log"""
    with open(args.skipped_log, 'w') as f:
        json.dump(skipped_log, f, indent=2)

def update_output_json(video, prompt, count):
    """Update the output JSON file with new result"""
    if video not in existing_results:
        existing_results[video] = {}
    existing_results[video][prompt] = count

    # Write immediately to prevent data loss
    with open(args.output_file, 'w') as f:
        json.dump(existing_results, f, indent=2)

def is_already_processed(video, video_path, prompt):
    """Check if video+prompt combination is already in output"""
    return video_path in existing_results and prompt in existing_results[video_path]

def process_video_prompt(video, video_path, input_text, original_prompt):
    """Process a single video+prompt combination"""

    # Build command for count_in_videos_reid_adaptive.py
    command_parts = [
        'python', 'count_in_videos_ovac.py',
        '--video_dir', video_path,
        '--input_text', input_text,
        '--sam3_checkpoint', args.sam3_checkpoint,
        '--device', args.device,
        '--confidence_threshold', str(args.confidence_threshold),
        '--min_obj_area', str(args.min_obj_area),
        '--reid_similarity_threshold', str(args.reid_similarity_threshold),
        '--max_lost_frames', str(args.max_lost_frames),
        '--mode', args.mode,
        '--sample_frames', str(args.sample_frames),
        '--downsample_factor', str(args.downsample_factor),
        '--output_file', args.output_file,
        '--font_size', str(args.font_size),
        '--output_fps', str(args.output_fps),
    ]

    # Optional visualization and tracking data
    if args.save_final_video:
        command_parts.append('--save_final_video')

    if args.save_T:
        command_parts.append('--save_T')

    # Create per-video output directory if needed
    if args.output_dir:
        # Use safer filename for prompt
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in original_prompt)[:50]
        per_vid_out = os.path.join(args.output_dir, video, safe_prompt)
        os.makedirs(per_vid_out, exist_ok=True)
        command_parts.extend(['--output_dir', per_vid_out])

    print(f"  Command: python count_in_videos_reid_adaptive.py --video_dir '{video_path}' --input_text '{input_text}' ...")

    try:
        # Run the command
        start_time = time.time()
        result = subprocess.run(command_parts, capture_output=True, text=True, timeout=args.timeout)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Read the updated output file to get the count
            with open(args.output_file, 'r') as f:
                updated_data = json.load(f)

            if video_path in updated_data and original_prompt in updated_data[video_path]:
                count = updated_data[video_path][original_prompt]
                print(f"   Successfully processed in {elapsed:.1f}s (count: {count})")

                # Remove from skipped log if it was there
                if video in skipped_log and original_prompt in skipped_log[video]:
                    del skipped_log[video][original_prompt]
                    if not skipped_log[video]:
                        del skipped_log[video]
                    save_skipped_log()

                return True, None
            else:
                error_msg = "Output not found in JSON after processing"
                print(f"   {error_msg}")
                return False, error_msg
        else:
            error_msg = f"Return code: {result.returncode}"
            if result.stderr:
                error_msg += f" | Error: {result.stderr[:200]}"
            print(f"   Failed: {error_msg}")
            return False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = f"Timeout expired (>{args.timeout}s)"
        print(f"    {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        print(f"   {error_msg}")
        return False, error_msg

# Determine which prompts to process
prompts_to_process = []

if args.retry_skipped:
    # Retry mode: only process skipped prompts
    print(f"\n{'='*70}")
    print(f"RETRY MODE: Re-evaluating skipped prompts")
    print(f"{'='*70}\n")

    for video in skipped_log:
        for prompt in skipped_log[video]:
            prompts_to_process.append((video, prompt))

    if not prompts_to_process:
        print("No skipped prompts found to retry. Exiting.")
        exit(0)
else:
    # Normal mode: process all prompts (skip already completed ones)
    print(f"\n{'='*70}")
    print(f"Starting TAO-Count evaluation for OVAC")
    print(f"Downsample factor: {args.downsample_factor}x (for efficiency)")
    print(f"{'='*70}\n")

    for video in tao_count_gt:
        for prompt in tao_count_gt[video]:
            prompts_to_process.append((video, prompt))

# Track processing status
total_prompts = len(prompts_to_process)
processed_prompts = 0
skipped_prompts = 0
already_done = 0

print(f"Total prompts to process: {total_prompts}")
print(f"Already completed: {sum(1 for v, p in prompts_to_process if is_already_processed(v, os.path.join(video_dir, v), p))}")
print(f"{'='*70}\n")

# Process each video and prompt
for idx, (video, original_prompt) in enumerate(prompts_to_process, 1):
    video_path = os.path.join(video_dir, video)

    # Check if already processed
    if is_already_processed(video, video_path, original_prompt):
        print(f"[{idx}/{total_prompts}] {video} / '{original_prompt}' - ⏭️  Already completed")
        already_done += 1
        continue

    # Check if video directory exists
    if not os.path.exists(video_path):
        print(f"[{idx}/{total_prompts}] {video} -  Video directory not found")

        # Log as skipped
        if video not in skipped_log:
            skipped_log[video] = {}
        skipped_log[video][original_prompt] = {
            'reason': 'Video directory not found',
            'attempts': skipped_log.get(video, {}).get(original_prompt, {}).get('attempts', 0) + 1,
            'last_attempt': datetime.now().isoformat()
        }
        save_skipped_log()
        skipped_prompts += 1
        continue

    print(f"\n[{idx}/{total_prompts}] Processing: {video}")
    print(f"  Prompt: '{original_prompt}'")

    # Determine input text
    if args.no_text:
        input_text_to_use = ""
    else:
        input_text_to_use = original_prompt

    # Get retry count
    current_attempts = 0
    if video in skipped_log and original_prompt in skipped_log[video]:
        current_attempts = skipped_log[video][original_prompt].get('attempts', 0)

    # Check if we've exceeded max retries
    if current_attempts >= args.max_retries:
        print(f"   Max retries ({args.max_retries}) reached. Skipping.")
        skipped_prompts += 1
        continue

    # Process the video+prompt
    success, error_msg = process_video_prompt(video, video_path, input_text_to_use, original_prompt)

    if success:
        processed_prompts += 1
    else:
        # Log the failure
        if video not in skipped_log:
            skipped_log[video] = {}

        skipped_log[video][original_prompt] = {
            'reason': error_msg,
            'attempts': current_attempts + 1,
            'last_attempt': datetime.now().isoformat()
        }
        save_skipped_log()
        skipped_prompts += 1

# Final summary
print(f"\n{'='*70}")
print(f"TAO-Count Evaluation Complete")
print(f"{'='*70}")
print(f"Total prompts: {total_prompts}")
print(f"Already completed: {already_done}")
print(f"Newly processed: {processed_prompts}")
print(f"Skipped/Failed: {skipped_prompts}")
print(f"\nOutput file: {args.output_file}")
print(f"Skipped log: {args.skipped_log}")
print(f"{'='*70}\n")

# Verify output file
if os.path.exists(args.output_file):
    with open(args.output_file, 'r') as f:
        try:
            final_data = json.load(f)
            print(f"Final JSON contains {len(final_data)} videos")
            total_entries = sum(len(prompts) for prompts in final_data.values())
            print(f"Total entries in JSON: {total_entries}")

            # Show which videos/prompts were processed
            print(f"\nProcessed videos in output:")
            for vid in sorted(final_data.keys()):
                print(f"  {vid}: {len(final_data[vid])} prompts")
        except json.JSONDecodeError:
            print(f" Warning: Output file exists but contains invalid JSON")
else:
    print(f" Warning: Output file not found: {args.output_file}")

# Show skipped prompts summary
if skipped_prompts > 0:
    print(f"\n{'='*70}")
    print(f"SKIPPED PROMPTS SUMMARY")
    print(f"{'='*70}")

    for video in skipped_log:
        print(f"\n{video}:")
        for prompt in skipped_log[video]:
            info = skipped_log[video][prompt]
            print(f"  • '{prompt}'")
            print(f"    Reason: {info['reason']}")
            print(f"    Attempts: {info['attempts']}/{args.max_retries}")
            print(f"    Last attempt: {info['last_attempt']}")

    print(f"\n{'='*70}")
    print(f"To retry skipped prompts, run:")
    print(f"python test_tao_count_ovac.py --retry_skipped \\")
    print(f"  --output_file {args.output_file} \\")
    print(f"  --data_dir {args.data_dir}")
    print(f"{'='*70}\n")
