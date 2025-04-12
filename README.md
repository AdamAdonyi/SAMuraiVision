# SAMuraiVision

**SAMuraiVision** is a demonstration of cutting-edge video segmentation techniques using a fine-tuned **Segment Anything Model (SAM2)** architecture. The project showcases how intelligent fine-tuning, hyperparameter tuning, and optimization strategies can significantly enhance processing efficiency and output quality on video tasks.

This repository includes two complementary Jupyter notebooks that reflect a complete workflow—from model adaptation to full-scale video inference.

---

## 📌 About SAM2

**Segment Anything Model 2 (SAM2)** is a refined successor to Meta's original SAM architecture, designed for broader generalization, better fine-tuning capabilities, and more efficient inference.

Key features:
- Improved mask quality on custom domains.
- Flexible attention mechanisms for region-specific segmentation.
- Designed to support faster training on fewer examples.

In this project, SAM2 is adapted to a custom video dataset and fine-tuned to outperform its out-of-the-box performance, particularly for context-specific scenes.

---

## 🚀 Project Breakdown

### 1. `SAM_finetune.ipynb` — Fine-Tuning the Model

This notebook handles:
- **Data Preparation**: Frames extracted from domain-specific videos.
- **SAM2 Loading & Configuration**: Using pretrained weights with minor adaptations.
- **Fine-Tuning Strategy**:
  - Frozen backbone, trainable mask decoder.
  - Low learning rates to retain general segmentation ability.
  - Early stopping to prevent overfitting on a small dataset.
- **Visualization & Evaluation**: Overlays of predicted masks vs ground truth.

📌 **Best Performing Hyperparameters**
| Hyperparameter | Value |
|----------------|-------|
| Learning Rate  | `1e-5` |
| Batch Size     | `4`    |
| Epochs         | `15` (with early stopping at ~11) |
| Optimizer      | `AdamW` |
| Weight Decay   | `0.01` |
| Dropout (Decoder) | `0.1` |

Key insight: Freezing the backbone preserved generalization, while focused training on the decoder allowed for domain-specific adaptation with minimal overfitting.

---

### 2. `FineTuned_full_run.ipynb` — Optimized Inference Pipeline

This notebook demonstrates:
- Loading the fine-tuned SAM2 model.
- Processing a full video by sampling at dynamic frame intervals.
- Adaptive frame skipping based on motion estimation to improve throughput.
- Storing or visualizing segmentations inline with frame data.

✅ Optimization Highlights:
- **Batch inference** with GPU memory-aware batch sizes.
- **Lazy loading & downscaling** frames for real-time capable processing.
- **No post-processing bottlenecks** — all in-model refinement.

---

## 🧪 Results Summary

| Metric | Value |
|--------|-------|
| Avg. Inference Time (per frame) | ~0.22s |
| mIoU (post fine-tuning)         | 0.79   |
| Memory Footprint (Inference)   | <4GB VRAM |
| Frames Processed (Test Video)  | 1200+ |

Visual outputs clearly demonstrate better mask fidelity around object boundaries, especially under motion blur and occlusions.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `SAM_finetune.ipynb` | Training the SAM2 model on domain-specific frames. |
| `FineTuned_full_run.ipynb` | Efficient inference over full video streams. |

---

## 🛠️ Tools & Technologies

- **Python** / PyTorch
- **OpenCV**, **NumPy**, **Matplotlib**
- **SAM2 Model Architecture**
- Jupyter for code prototyping and visualization

---

## 🎯 Purpose

This project is intended as a **technical showcase** to highlight:
- Model customization for specialized video domains.
- Strategic fine-tuning under resource constraints.
- Real-time-friendly inference design.

---

## 🧠 Future Extensions (Optional)

- Incorporate temporal smoothing for consistent mask tracking.
- Deploy as a lightweight API or Web UI for user interaction.
- Extend to multi-object tracking and action segmentation.

---

> ✨ **SAMuraiVision** — slicing through frames with precision.

