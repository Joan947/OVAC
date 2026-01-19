#!/usr/bin/env python3

"""
SAM3 Image Counting with Direct Repo Integration + IoM-based NMS

BALANCED 2-PASS VERSION:
- Relaxed exemplar selection (less conservative)
- Option to return Pass 2 only (old behavior) OR merge Pass 1+2
- Stricter NMS threshold option for merging
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# SAM3 repo imports (not HF Transformers)
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
from sam3.sam3.model.box_ops import box_xywh_to_cxcywh, box_cxcywh_to_xyxy
from sam3.sam3.visualization_utils import normalize_bbox
from sam3.sam3.train.utils.checkpoint_utils import load_state_dict_into_model

# ============================================================
# Configuration: Toggle merging behavior
# ============================================================
USE_MERGING = False  # Set to True to merge Pass 1+2, False for Pass 2 only (old behavior)
STRICT_NMS_THRESHOLD = 0.3  # Use stricter NMS when merging (default 0.5)


# ============================================================
# IoM-based NMS Functions
# ============================================================

def compute_iom(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Minimum (IoM) between two binary masks.

    IoM = intersection_area / min(area1, area2)

    This detects whole vs. part situations: if a mask is fully covered by another,
    the IoM will be high, even if the covering mask is much bigger.

    Args:
        mask1: Binary mask (H, W) with values 0 or 1
        mask2: Binary mask (H, W) with values 0 or 1

    Returns:
        IoM value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()

    if area1 == 0 or area2 == 0:
        return 0.0

    min_area = min(area1, area2)
    iom = intersection / min_area
    return float(iom)


def nms_iom(
    masks: List[np.ndarray],
    scores: List[float],
    boxes: List[np.ndarray],
    iom_threshold: float = 0.5
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Apply Non-Maximum Suppression using IoM (Intersection over Minimum).

    Args:
        masks: List of binary masks (each is H x W numpy array)
        scores: List of confidence scores corresponding to each mask
        boxes: List of bounding boxes
        iom_threshold: IoM threshold for suppression (default: 0.5)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(masks) == 0:
        return [], [], []

    # Sort by confidence score (descending)
    indices = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()

    for i in indices:
        if i in suppressed:
            continue

        keep.append(i)

        # Suppress all masks with IoM > threshold
        for j in indices:
            if j == i or j in suppressed:
                continue

            iom = compute_iom(masks[i], masks[j])
            if iom >= iom_threshold:
                suppressed.add(j)

    # Return filtered results
    filtered_masks = [masks[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]
    filtered_boxes = [boxes[i] for i in keep]

    return filtered_masks, filtered_scores, filtered_boxes


# ============================================================
# Core SAM3 Prediction Functions
# ============================================================

def build_sam3_image_predictor(device_str: str = "cuda", bpe_path: str = None):
    """
    Build SAM3 image model and processor using direct repo integration.

    Args:
        device_str: Device to use ('cuda', 'cpu', 'mps')
        bpe_path: Path to BPE vocabulary file. If None, assumes default location.

    Returns:
        model: SAM3 model
        device: torch device
    """
    device = torch.device(device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu")

    print(f"Loading SAM3 IMAGE model on {device}...")

    # Set default BPE path if not provided
    if bpe_path is None:
        bpe_path = "/project/advdls25/jowusu1/CountVid/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    # Load checkpoint path
    checkpoint_path = "/project/advdls25/jowusu1/CountVid/fscd147/sam3.pt"
    # Build model from repo
    model = build_sam3_image_model(
        enable_inst_interactivity=False,
        enable_segmentation=True,
        bpe_path=bpe_path, 
        eval_mode=True, 
        load_from_HF=True,
        checkpoint_path = "/project/advdls25/jowusu1/CountVid/fscd147/sam3.pt", ) # only build the model
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)

    # 2) Load FSCD147 weights but ignore missing segmentation_head.* keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Ignored missing keys:", [k for k in missing if k.startswith("segmentation_head.")])
    print("Other missing keys:", [k for k in missing if not k.startswith("segmentation_head.")])
    print("Unexpected keys:", unexpected)


    model.to(device)
    model.eval()

    # Enable optimizations for better performance
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Model loaded successfully on {device}")

    return model, device


def run_sam3_single_pass(
    model,
    device: torch.device,
    image: Image.Image,
    text_prompt: str,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run SAM3 with text prompt only (single pass).

    Returns:
        inference_state with masks, boxes, scores
        processor: Sam3Processor instance
    """
    # Create processor with confidence threshold
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)

    # Set image and get inference state
    inference_state = processor.set_image(image)

    # Set text prompt and run inference
    inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    print("DEBUG raw candidates:", len(inference_state.get("scores", [])))

    return inference_state, processor


def _compute_iou_boxes(box1, box2):
    """Quick IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter

    if union <= 0.0:
        return 0.0

    return float(inter / union)


def run_sam3_two_pass(
    model,
    device: torch.device,
    image: Image.Image,
    text_prompt: str,
    confidence_threshold: float = 0.5,
    max_exemplars: int = 3,
) -> Dict[str, Any]:
    """
    BALANCED 2-pass refinement with relaxed constraints.

    Pass 1: Text prompt only → initial detections
    Pass 2: Text + quality exemplars (relaxed selection) → refined detections

    Returns either:
    - Pass 2 only (USE_MERGING=False, matches old behavior ~89%)
    - Pass 1 + Pass 2 merged (USE_MERGING=True, experimental)

    Returns:
        inference_state, processor
    """
    width, height = image.size

    # ===== PASS 1: Text-only detection =====
    print(f"   Pass 1: Text prompt only...")
    processor1 = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
    state1 = processor1.set_image(image)
    state1 = processor1.set_text_prompt(state=state1, prompt=text_prompt)

    # Extract pass-1 results
    if "masks" not in state1 or len(state1["masks"]) == 0:
        print(f"  No detections in Pass 1")
        return state1, processor1

    masks1 = state1["masks"]
    scores1 = state1["scores"]
    boxes1 = state1["boxes"]
    num_detections = len(masks1)
    print(f"   Pass 1: {num_detections} detections")

    # RELAXED: Only require 1 detection minimum (was 3)
    if num_detections < 1:
        return state1, processor1

    # Compute mask areas for quality filtering
    areas1 = []
    for m in masks1:
        if isinstance(m, torch.Tensor):
            m_np = m.squeeze().cpu().numpy()
        else:
            m_np = m.squeeze()
        areas1.append(float((m_np > 0.5).sum()))
    areas1 = torch.tensor(areas1, device=scores1.device, dtype=torch.float32)

    # ===== RELAXED Exemplar Selection =====
    # 1) RELAXED quality filter: lower thresholds
    min_conf_for_exemplar = max(0.7, confidence_threshold)  # Was 0.7 → now 0.5
    min_area_for_exemplar = 0.0  # Was 50 → now 20 pixels

    quality_mask = (scores1 >= min_conf_for_exemplar) & (areas1 >= min_area_for_exemplar)
    quality_indices = torch.nonzero(quality_mask, as_tuple=False).flatten()

    # RELAXED: If no high-quality detections, use ALL detections
    if len(quality_indices) == 0:
        print(" No high-quality exemplars (using all detections)")
        quality_indices = torch.arange(len(scores1), device=scores1.device)

    # Limit exemplar pool to top 20 by score
    quality_scores = scores1[quality_indices]
    max_pool = min(len(quality_indices), 20)
    top_quality_scores, top_quality_idx = torch.topk(quality_scores, k=max_pool)
    candidate_indices = quality_indices[top_quality_idx]

    # 2) LESS AGGRESSIVE adaptive k
    if num_detections < 5:
        k = 1
    elif num_detections < 15:
        k = 2
    else:
        k = min(3, max_exemplars)

    # 3) RELAXED spatial diversity: IoU < 0.5 (was 0.3)
    selected: List[int] = []
    for idx in candidate_indices:
        idx_int = int(idx.item())
        box_i = boxes1[idx_int].detach().cpu().numpy()

        if not selected:
            selected.append(idx_int)
            if len(selected) >= k:
                break
            continue

        # Check IoU with already selected exemplars
        too_close = False
        for sel_idx in selected:
            box_j = boxes1[sel_idx].detach().cpu().numpy()
            iou = _compute_iou_boxes(box_i, box_j)
            if iou > 0.3:  # Was 0.3 → now 0.5 (less strict)
                too_close = True
                break

        if not too_close:
            selected.append(idx_int)
            if len(selected) >= k:
                break

    # RELAXED: If diversity selection fails, just take top-k by confidence
    if not selected:
        print("   Diversity check failed, using top-k by confidence")
        k_fallback = min(k, len(candidate_indices))
        selected = [int(candidate_indices[i].item()) for i in range(k_fallback)]

    if not selected:
        # Still no exemplars (shouldn't happen), fall back to Pass 1
        print("   Could not select any exemplars, using Pass 1 only")
        return state1, processor1

    print(f"   Selected {len(selected)} exemplars for Pass 2")

    # ===== PASS 2: Text + Exemplar boxes =====
    print(f"   Pass 2: Text + {len(selected)} exemplar boxes...")
    processor2 = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
    state2 = processor2.set_image(image)

    # Set text prompt first
    state2 = processor2.set_text_prompt(state=state2, prompt=text_prompt)

    # Add exemplar boxes as geometric prompts (normalized cxcywh)
    for idx_int in selected:
        box_xyxy = boxes1[idx_int].detach().cpu().numpy()
        x1, y1, x2, y2 = box_xyxy
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        box_xywh = torch.tensor([[x, y, w, h]], dtype=torch.float32, device=device)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()

        state2 = processor2.add_geometric_prompt(
            state=state2,
            box=norm_box,
            label=True,  # positive exemplar
        )

    if "masks" not in state2 or len(state2["masks"]) == 0:
        print("   Pass 2 found nothing, using Pass 1 only")
        return state1, processor1

    masks2 = state2["masks"]
    scores2 = state2["scores"]
    boxes2 = state2["boxes"]
    num_refined = len(masks2)
    print(f"   Pass 2: {num_refined} refined detections")

    # ===== Return Strategy: Pass 2 only OR Merge =====
    if not USE_MERGING:
        # Old behavior: Return Pass 2 only
        print(f"   Returning Pass 2 only (old behavior)")
        print("DEBUG raw candidates:", len(state2.get("scores", [])))

        return state2, processor2
    else:
        # New behavior: Merge Pass 1 + Pass 2
        merged_state: Dict[str, Any] = {}
        merged_state["masks"] = list(masks1) + list(masks2)
        merged_state["scores"] = torch.cat([scores1, scores2], dim=0)
        merged_state["boxes"] = torch.cat([boxes1, boxes2], dim=0)
        print(f"   Merged: Pass1={len(masks1)}, Pass2={len(masks2)}, Total={len(merged_state['masks'])}")
        print("DEBUG raw candidates:", len(merged_state.get("scores", [])))

        return merged_state, processor2


# ============================================================
# Detection Post-Processing with NMS and Counting
# ============================================================

def apply_nms_and_filter(
    inference_state: Dict[str, Any],
    iom_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    min_obj_area: int = 0,
    apply_nms: bool = True,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Apply IoM-based NMS and filter detections by confidence and area.

    Args:
        inference_state: State dict with 'masks', 'scores', 'boxes'
        iom_threshold: IoM threshold for NMS (default: 0.5, use 0.3 for stricter)
        confidence_threshold: Minimum confidence score (default: 0.5)
        min_obj_area: Minimum pixel area to count (default: 0)
        apply_nms: Whether to apply NMS (default: True)

    Returns:
        num_objects: Final count after NMS and filtering
        detections: List of detection dicts with mask, box, score, area
    """
    if "masks" not in inference_state or len(inference_state["masks"]) == 0:
        return 0, []

    masks = inference_state["masks"]
    scores = inference_state["scores"]
    boxes = inference_state["boxes"]

    # Convert to numpy arrays
    masks_np = []
    scores_list = []
    boxes_list = []

    for i in range(len(masks)):
        mask = masks[i]

        # Convert to numpy if tensor
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask.squeeze()

        # Ensure boolean mask
        mask_bool = mask_np > 0.5 if mask_np.dtype in [np.float32, np.float64] else mask_np.astype(bool)

        # Get score
        score = float(scores[i]) if isinstance(scores[i], torch.Tensor) else float(scores[i])

        # Get box
        box = boxes[i].cpu().numpy() if isinstance(boxes[i], torch.Tensor) else boxes[i]

        masks_np.append(mask_bool)
        scores_list.append(score)
        boxes_list.append(box)

    print(f"   Detections before NMS: {len(masks_np)}")

    # Apply IoM-based NMS
    if apply_nms and len(masks_np) > 0:
        # Use stricter threshold if merging is enabled
        nms_threshold = STRICT_NMS_THRESHOLD if USE_MERGING else iom_threshold

        masks_np, scores_list, boxes_list = nms_iom(
            masks=masks_np,
            scores=scores_list,
            boxes=boxes_list,
            iom_threshold=nms_threshold
        )
        print(f"   Detections after NMS (IoM={nms_threshold}): {len(masks_np)}")

    # Filter by confidence threshold and area
    filtered_detections = []
    for i in range(len(masks_np)):
        mask = masks_np[i]
        score = scores_list[i]
        box = boxes_list[i]

        # Check confidence
        if score <= confidence_threshold:
            continue

        # Check area
        area = int(mask.sum())
        if area < min_obj_area:
            continue

        filtered_detections.append({
            "mask": mask,
            "box": box,
            "score": score,
            "area": area
        })

    print(f"   Final count (conf>{confidence_threshold}, area>={min_obj_area}): {len(filtered_detections)}")

    return len(filtered_detections), filtered_detections


def filter_detections_by_area(
    inference_state: Dict[str, Any],
    min_obj_area: int = 0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Legacy function for backward compatibility.
    Filter detections by minimum area only (no NMS).

    Returns:
        num_objects: count after filtering
        detections: list of detection dicts with mask, box, score
    """
    return apply_nms_and_filter(
        inference_state=inference_state,
        iom_threshold=0.5,
        confidence_threshold=0.0,  # No confidence filtering in legacy mode
        min_obj_area=min_obj_area,
        apply_nms=False  # NMS disabled for backward compatibility
    )


def count_objects_in_image(
    image_path: str,
    text_prompt: str,
    use_two_pass: bool = False,
    device_str: str = "cuda",
    confidence_threshold: float = 0.5,
    max_exemplars: int = 3,
    min_obj_area: int = 0,
    iom_threshold: float = 0.5,
    apply_nms: bool = True,
    bpe_path: str = None,
) -> Tuple[int, List[Dict[str, Any]], np.ndarray]:
    """
    Main function to count objects in a single image with IoM-based NMS.

    Args:
        image_path: Path to input image
        text_prompt: Text description of objects to count
        use_two_pass: If True, use balanced 2-pass refinement; else single pass
        device_str: Device to use
        confidence_threshold: Minimum confidence score for final predictions
        max_exemplars: Number of exemplars for 2-pass (if enabled)
        min_obj_area: Minimum pixel area to count
        iom_threshold: IoM threshold for NMS (default: 0.5)
        apply_nms: Whether to apply IoM-based NMS (default: True)
        bpe_path: Path to BPE vocab file

    Returns:
        num_objects: Final count after NMS and filtering
        detections: List of detection dictionaries
        frame_np: Original image as numpy array
    """
    # Build model
    model, device = build_sam3_image_predictor(device_str, bpe_path)

    # Load image
    image = Image.open(image_path).convert("RGB")
    frame_np = np.array(image)

    # Run SAM3 (single or two-pass) with low initial threshold to get all detections
    initial_threshold = 0.0 if apply_nms else confidence_threshold

    if use_two_pass:
        print(f"\n Running BALANCED 2-pass SAM3 (USE_MERGING={USE_MERGING})...")
        inference_state, processor = run_sam3_two_pass(
            model=model,
            device=device,
            image=image,
            text_prompt=text_prompt,
            confidence_threshold=initial_threshold,
            max_exemplars=max_exemplars,
        )
    else:
        print(f"\n Running single-pass SAM3...")
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

    return num_objects, detections, frame_np


# ============================================================
# Visualization
# ============================================================

def visualize_detections_on_image(
    frame_np: np.ndarray,
    detections: List[Dict[str, Any]],
    count: int,
    save_path: str,
    show_id: bool = False,
    id_font_size: int = 48,
):
    """
    Overlay masks and boxes on image and save visualization.

    Args:
        frame_np: Image as numpy array
        detections: List of detection dictionaries
        count: Total object count
        save_path: Path to save visualization
        show_id: If True, overlay ID number on each mask (default: False)
        id_font_size: Font size for ID numbers (default: 48, increased from 24)
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    img = frame_np.copy().astype(np.uint8)
    overlay = img.copy()

    # Generate colors for each detection
    rng = np.random.default_rng(0)
    colors = [rng.integers(0, 256, size=3, dtype=np.uint8) for _ in range(len(detections))]

    # Draw masks
    for i, det in enumerate(detections):
        mask = det["mask"]
        box = det.get("box", None)

        if mask is None:
            continue

        color = colors[i]

        # Apply color overlay where mask is True
        overlay[mask] = (0.3 * overlay[mask] + 0.7 * color).astype(np.uint8)

        # Draw bounding box
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, img.shape[1] - 1)
            y2 = min(y2, img.shape[0] - 1)

            # Draw white border
            overlay[y1:y1+2, x1:x2+1] = [0,0,0]
            overlay[y2-1:y2+1, x1:x2+1] = [0,0,0]
            overlay[y1:y2+1, x1:x1+2] = [0,0,0]
            overlay[y1:y2+1, x2-1:x2+1] = [0,0,0]

    # Blend original + overlay
    vis = (0.4 * img + 0.6 * overlay).astype(np.uint8)

    # Convert to PIL for text drawing
    pil_vis = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_vis)

    # Overlay ID numbers on each mask if requested
    if show_id:
        try:
            id_font = ImageFont.truetype("arial.ttf", id_font_size)
        except Exception:
            try:
                # Try to use a larger default font
                id_font = ImageFont.truetype("DejaVuSans-Bold.ttf", id_font_size)
            except Exception:
                # Fallback to default (will be small)
                id_font = ImageFont.load_default()

        for i, det in enumerate(detections):
            mask = det["mask"]
            box = det.get("box", None)

            if mask is None:
                continue

            # Calculate centroid of mask for ID placement
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
            elif box is not None:
                # Fallback to box center if mask centroid fails
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
            else:
                continue

            # Draw ID number
            id_text = str(i + 1)
            bbox = draw.textbbox((0, 0), id_text, font=id_font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Calculate text position (centered on cx, cy)
            text_x = cx - text_w // 2
            text_y = cy - text_h // 2

            # Draw background rectangle for better visibility (larger padding)
            # padding = 8
            # draw.rectangle(
            #     [text_x - padding, text_y - padding, 
            #      text_x + text_w + padding, text_y + text_h + padding],
            #     fill=(0, 0, 0, 200)
            # )

            # Draw ID text in bright yellow for maximum visibility
            draw.text((text_x, text_y), id_text, fill=(0, 0, 0), font=id_font)

    # Add count text at top-left corner
    text = f"Count: {count}"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4

    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0, 160))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)

    pil_vis.save(save_path)
    print(f" Saved visualization: {save_path}")
# ============================================================
# JSON Output (compatible with count_in_videos_reid_adaptive)
# ============================================================

def update_counts_json(
    output_file: str,
    image_path: str,
    input_text: str,
    num_objects: int,
):
    """
    Update JSON file with counts in format: {image_path: {text: count}}
    Compatible with count_in_videos_reid_adaptive format.
    """
    if not output_file:
        return

    data = {}
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass

    if image_path not in data:
        data[image_path] = {}

    data[image_path][input_text] = int(num_objects)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f" Updated counts JSON: {output_file}")


# ============================================================
# CLI Interface
# ============================================================

def get_args_parser():
    parser = argparse.ArgumentParser(
        "SAM3 Image Counting with Balanced 2-Pass + IoM-NMS",
        add_help=True,
    )

    # Input/Output
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Text prompt describing object to count (e.g., 'penguin', 'car')",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Optional JSON file to save counts",
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
        default=None,
        help="Path to BPE vocabulary file (default: auto-detect from sam3 package)",
    )

    # Counting parameters
    parser.add_argument(
        "--use_two_pass",
        action="store_true",
        help="Enable balanced 2-pass refinement (relaxed exemplar selection)",
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
        help="Maximum number of exemplar boxes for 2-pass (only used if --use_two_pass)",
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

    # Visualization
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save visualization with masks/boxes overlaid",
    )

    parser.add_argument(
        "--vis_path",
        type=str,
        default="",
        help="Path for visualization output (default: <image>_vis.png)",
    )

    parser.add_argument(
        "--show_id",
        action="store_true",
        help="Overlay ID numbers on each detected mask for counting visualization",
    )

    parser.add_argument(
        "--id_font_size",
        type=int,
        default=48,
        help="Font size for ID numbers when --show_id is enabled (default: 48, try 60-80 for larger)",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    else:
        device_str = args.device

    # Print configuration
    print("\n" + "=" * 70)
    print("SAM3 Image Counting with BALANCED 2-Pass + IoM-NMS")
    print("=" * 70)
    print(f"Image     : {args.image_path}")
    print(f" Prompt    : '{args.input_text}'")
    print(f" Device    : {device_str}")
    print(f" Mode      : {'Balanced 2-pass' if args.use_two_pass else 'Single-pass'}")
    print(f"  Confidence: {args.confidence_threshold}")

    if not args.no_nms:
        print(f" NMS (IoM) : Enabled (threshold={args.iom_threshold})")
    else:
        print(f" NMS (IoM) : Disabled")

    if args.use_two_pass:
        print(f"Max Exemplars: {args.max_exemplars}")
        print(f" Merging   : {'Enabled (stricter NMS=' + str(STRICT_NMS_THRESHOLD) + ')' if USE_MERGING else 'Disabled (Pass 2 only)'}")

    print(f" Min Area  : {args.min_obj_area} pixels")
    print("=" * 70)

    # Run counting
    num_objects, detections, frame_np = count_objects_in_image(
        image_path=args.image_path,
        text_prompt=args.input_text,
        use_two_pass=args.use_two_pass,
        device_str=device_str,
        confidence_threshold=args.confidence_threshold,
        max_exemplars=args.max_exemplars,
        min_obj_area=args.min_obj_area,
        iom_threshold=args.iom_threshold,
        apply_nms=not args.no_nms,
        bpe_path=args.bpe_path,
    )

    # Print results
    print("\n" + "=" * 70)
    print(f" Final Count: {num_objects}")
    print("=" * 70)

    # Print detection details
    if detections:
        print(f"\nDetection Details:")
        for i, det in enumerate(detections, 1):
            print(f"   #{i}: score={det['score']:.3f}, area={det['area']} pixels")

    # Update JSON output
    if args.output_file:
        update_counts_json(
            output_file=args.output_file,
            image_path=args.image_path,
            input_text=args.input_text,
            num_objects=num_objects,
        )

    # Save visualization
    if args.save_vis:
        if args.vis_path:
            vis_path = args.vis_path
        else:
            base, ext = os.path.splitext(args.image_path)
            vis_path = base + "_vis.png"

        visualize_detections_on_image(
            frame_np=frame_np,
            detections=detections,
            count=num_objects,
            save_path=vis_path,
            show_id=args.show_id,
            id_font_size=args.id_font_size,
        )
    print("\n Done!\n")


if __name__ == "__main__":
    main()