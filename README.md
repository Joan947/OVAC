
# OVAC: Open Vocabulary Adaptive Counting
<img width="907" height="438" alt="arc_f11" src="https://github.com/user-attachments/assets/3932d121-aa5e-418f-a4f0-3b4640fb0c96" />


**OVAC** (Open Vocabulary Adaptive Counting) is a text-prompted counting framework for images and videos that achieves state-of-the-art performance on open-vocabulary counting tasks. Built on SAM3, OVAC introduces a multimodal re-identification tracker that maintains consistent object identities across frames, reducing double-counting and ID fragmentation under occlusion, deformation, and scale changes. OVAC addresses the challenges of open-vocabulary counting in images and videos by combining SAM3's zero-shot detection capabilities with a lightweight re-identification tracker. The framework maintains per-instance appearance banks, uses short-term motion cues, and performs conservative online association through multi-stage matching to preserve identities under occlusions and deformations. An optional two-pass prompt refinement converts high-confidence detections into exemplar prompts to reduce missed instances in crowded or low-contrast scenes. 
## Installation

OVAC is built on SAM3. Follow the exact installation instructions from the official SAM3 repository:

**SAM3 Repository**: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/segment-anything-model-3)

```bash
# Clone the repository
git clone https://github.com/yourusername/ovac.git
cd ovac

# Follow SAM3 installation instructions from their repository
# Install dependencies as specified in SAM3 documentation
```

## Datasets

Follow the intructions from the [**CountVid** ](https://github.com/niki-amini-naieni/CountVid)repository to download the TAO-Count and Penguins video dataset. 
Follow the these various repositories to download the [**Pixmo-Count**](https://huggingface.co/datasets/allenai/pixmo-count) and [CountBench](https://github.com/teaching-clip-to-count/teaching-clip-to-count.github.io/blob/main/CountBench.json) . For the Countbench benchmark, modify the
question sentence to the simple noun phrase. For the Pixmo run the "parquet_to_json.py" file to convert to json before using it.You can also use our preprocessed version:
- **PixMo-Count**: [Download](https://drive.google.com/drive/folders/1_QBtW7inZHIq5zIvC4i-PVO0mVbmXABR?usp=drive_link)
- **CountBench**: [Download](https://drive.google.com/file/d/15H2tUB0ZCX-QSeTLSPOY6RmcD5BcTqcI/view?usp=drive_link)

  
Organize datasets as follows:
```
data/
├── VideoCount/
│   ├── TAO-Count/
│   ├── Penguins/
│   └── MOT20-Count/
├── CountBench.json
└── pixmo/
```

## Reproducing Paper Results

### Video Counting Benchmarks

**TAO-Count**:
```bash
python scripts/test_tao_count_ovac.py \
  --output_file results/tao_results/tao-count-ovac-predicted.json \
  --data_dir data/VideoCount/TAO-Count \
  --mode balanced \
  --downsample_factor 2 \
  --output_dir results/tao_results
```

**Penguins**:
```bash
python scripts/test_penguins.py \
  --data_dir data/VideoCount/Penguins \
  --output_dir results/penguins_dataset \
  --output_file results/penguins-count-predicted.json
```

### Image Counting Benchmarks

**CountBench**:
```bash
python scripts/evaluate_countbench.py \
  --benchmark_file data/CountBench.json \
  --output_file results/countbench/predictions.json \
  --failures_file results/countbench/failures.json \
  --metrics_file results/countbench/metrics.json \
  --use_two_pass \
  --max_exemplars 3 \
  --confidence_threshold 0.5 \
  --iom_threshold 0.5 \
  --cache_dir results/countbench/image_cache \
  --device cuda
```

**PixMo (Test Set)**:
```bash
python scripts/evaluate_pixmo.py \
  --benchmark_file results/pixmo/pixmo_test.json \
  --output_file results/pixmo/test/pixmo_predictions.json \
  --metrics_file results/pixmo/test/pixmo_metrics.json \
  --failures_file results/pixmo/pixmo_failures.json \
  --use_two_pass \
  --confidence_threshold 0.5 \
  --iom_threshold 0.5 \
  --max_exemplars 3 \
  --cache_dir results/pixmo/pixmo_cache
```

**PixMo (Validation Set)**:
```bash
python scripts/evaluate_pixmo.py \
  --benchmark_file results/pixmo/pixmo_val.json \
  --output_file results/pixmo/val/pixmo_predictions.json \
  --metrics_file results/pixmo/val/pixmo_metrics.json \
  --failures_file results/pixmo/pixmo_failures.json \
  --use_two_pass \
  --confidence_threshold 0.5 \
  --iom_threshold 0.5 \
  --max_exemplars 3 \
  --cache_dir results/pixmo/pixmo_cache
```

## Demo Usage

### Image Counting Demo

Count objects in a single image:

```bash
python count_in_images_ovac.py \
  --image_path data/VideoCount/MOT20-Count/frames/MOT20-05/img1/000001.jpg \
  --input_text "person" \
  --use_two_pass \
  --max_exemplars 2 \
  --confidence_threshold 0.05 \
  --iom_threshold 0.5 \
  --save_vis \
  --show_id \
  --id_font_size 28 \
  --vis_path ./results/sam_img/demo_pic.png \
  --output_file ./results/sam_img/counts_img.json
```

### Video Counting Demo

Count objects in a video:

```bash
python count_in_videos_ovac.py \
  --mode balanced \
  --video_dir examples/scc \
  --input_text "car" \
  --output_dir results/vid_count \
  --downsample_factor 1 \
  --output_file results/vid_count/vid_count_ovac.json
```

## Performance

### Image Benchmarks

| Method | PixMo-Val Acc | PixMo-Test Acc | CountBench Acc |
|--------|---------------|----------------|----------------|
| SAM3   | 85.6%         | 87.5%          | 93.23%         |
| OVAC   | **88.8%**     | **89.18%**     | **93.43%**     |

### Video Benchmarks

| Method   | TAO-Count MAE | Penguins MAE |
|----------|---------------|--------------|
| CountVid | 2.6           | 4.0          |
| OVAC     | **0.78**      | **2.3**      |

All metrics are reported in the paper. 

## Acknowledgments

This work builds upon [SAM3](https://github.com/facebookresearch/segment-anything-model-3) for zero-shot detection and segmentation. 
## License

MIT License

---
