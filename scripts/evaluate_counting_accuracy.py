import json
import argparse
import numpy as np

parser = argparse.ArgumentParser("Evaluating Counting Accuracy", add_help=False)
parser.add_argument(
    "--ground_truth", type=str, help="file containing ground truth counts"
)
parser.add_argument(
    "--predicted", type=str, help="file containing the predicted counts"
)
parser.add_argument(
  "--parent_dir", type=str, default="", help="parent directory of the videos (prefix of pred file keys not in gt file keys)"
)
parser.add_argument(
    "--exclude_classes", nargs='+', type=str, default=[], help="exclude evaluation for certain classes (to test open-world performance for example)"
)
parser.add_argument("--only_exemplars", action="store_true", help="whether or not only exemplars were used to get the predicted counts")
parser.add_argument(
    "--frames_subdir", type=str, default="",
    help="optional subfolder inside each video dir (e.g., img1)"
)
args = parser.parse_args()

# Load GT counts.
with open(args.ground_truth) as gt_file:
    gt_counts = json.load(gt_file)

# Load predicted counts.
with open(args.predicted) as pred_file:
    pred_counts = json.load(pred_file)

print("Excluding classes " + str(args.exclude_classes) + " from evaluation.")
unique_vids = []
abs_errs = []
squared_errs = []
for video in gt_counts:
    for text in gt_counts[video]:
        if text not in args.exclude_classes:
            if video not in unique_vids:
                unique_vids.append(video)
            gt_count = gt_counts[video][text]
            if isinstance(gt_count, list):
                gt_count = len(gt_count)
            if not args.only_exemplars:
                if len(args.parent_dir) > 0:
                    # pred_count = pred_counts[args.parent_dir + "/" + video][text]
                    vid_key = args.parent_dir + "/" + video
                    if len(args.frames_subdir) > 0:
                        vid_key = vid_key + "/" + args.frames_subdir
                    pred_count = pred_counts[vid_key][text]
                else:
                    pred_count = pred_counts[video][text]
            else:
                if len(args.parent_dir) > 0:
                    # print(args.parent_dir + "/" + video)
                    pred_count = pred_counts[args.parent_dir + "/" + video][""]
                else:
                    pred_count = pred_counts[video][""]
                    
            abs_err = np.abs(gt_count - pred_count)
            print(video)
            print((abs_err/gt_count)*100)
            abs_errs.append(abs_err)
            squared_errs.append(abs_err ** 2)

print("MAE: " + str(np.mean(abs_errs)))
print("RMSE: " + str(np.sqrt(np.mean(squared_errs)))) 
print("Num vids: " + str(len(unique_vids)))
