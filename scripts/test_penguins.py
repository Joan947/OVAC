import json
import argparse
import subprocess
import shlex
import os

parser = argparse.ArgumentParser("Testing on Penguins with ReID Adaptive Tracking", add_help=False)
parser.add_argument(
    "--output_file",
    type=str,
    default="penguins-count-predicted-reid.json",
    help="file where to save predicted counts",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/VideoCount/Penguins",
    help="path to Penguins dataset",
)
parser.add_argument(
    "--no_text",
    action="store_true",
    help="whether or not to drop the text (send empty prompt to SAM3)",
)
parser.add_argument(
    "--mode",
    type=str,
    default="crowd",
    choices=["sequential", "crowd", "static", "balanced"],
    help="Tracking mode: sequential(cars), crowd(penguins), static(clothes), balanced(general)",
)
parser.add_argument(
    "--save_final_video",
    action="store_true",
    help="whether to save the final annotated video",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="directory to save outputs (videos, tracks)",
)

args = parser.parse_args()

# Use the specified args to get inputs for main [count_in_videos_reid_adaptive.py] script.
gt_file = os.path.join(args.data_dir, "anno", "penguins-count-gt.json")
video_dir = os.path.join(args.data_dir, "frames")

with open(gt_file) as penguins_json:
    penguins_gt = json.load(penguins_json)

# Create empty output file for results.
with open(args.output_file, "w") as out_file:
    json.dump({}, out_file)

for video in penguins_gt:
    for input_text in penguins_gt[video]:
        # Optionally drop text (empty prompt)
        if args.no_text:
            input_text_to_use = ""
        else:
            input_text_to_use = input_text

        # Build command to call ReID-based counting script
        command = (
            'python count_in_videos_ovac.py'
            + ' --video_dir "' + os.path.join(video_dir, video) + '"'
            + ' --input_text "' + input_text_to_use + '"'
            + ' --output_file "' + args.output_file + '"'
            + ' --mode ' + args.mode
        )

        # Optionally add output directory for saving videos
        if args.output_dir:
            video_output_dir = os.path.join(args.output_dir, video)
            os.makedirs(video_output_dir, exist_ok=True)
            command += ' --output_dir "' + video_output_dir + '"'

        # Optionally save final video
        if args.save_final_video:
            command += ' --save_final_video'

        print("running: " + command)
        subprocess.run(
            shlex.split(command)
        )
