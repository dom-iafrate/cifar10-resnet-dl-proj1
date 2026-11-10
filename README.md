# CIFAR-10 ResNet Architecture Challenge

Deep Learning Project 1 — Fall 2026

This repository contains our PyTorch code for training three **custom residual networks from scratch** on the course's CIFAR-10 Kaggle competition. The requirement was to stay below **5 million trainable parameters** and use only the competition-provided training images.

The final reported single-model Kaggle score was **86.1%**. The models and training methods are described below. We do **not** use a multi-model ensemble as the final submission.

## Results

| Model | Best recorded validation accuracy | Kaggle public accuracy | Training |
| --- | ---: | ---: | --- |
| V1 baseline ResNet | 94.04% | 82.4% (83.2% with flip TTA) | 60 epochs |
| V2: stronger augmentation + Mixup | 95.56% | 86.0% with flip TTA | 120 epochs |
| V3: pre-activation ResNet | Not recorded | 86.1% | 120 epochs |

These are **historical scores observed in our competition runs**, not outputs recreated in the repository. Kaggle scores cannot be recomputed locally because the competition's test labels are hidden. Rerunning training can lead to somewhat different results. Parameter counts are printed and checked in every notebook.

## Project structure

```text
cifar10-resnet-project/
├── README.md                    # Overview and instructions
├── requirements.txt
├── .gitignore
├── src/
│   ├── dataset.py               # Competition dataset, transforms, loaders
│   ├── models.py                # Standard and pre-activation ResNets
│   ├── training.py              # Training, validation, resume, plots
│   └── submission.py            # One-model TTA predictions and CSV
├── notebooks/
│   ├── README.md
│   ├── v1_resnet.ipynb
│   ├── v2_resnet.ipynb
│   └── v3_resnet.ipynb
├── data/README.md               # Where to put the original Kaggle images
├── checkpoints/README.md        # Local saved-model files
├── submissions/README.md        # Generated Kaggle CSVs
└── figures/README.md            # Training plots
```

## Setup

This project was developed on NYU HPC with a CUDA-capable NVIDIA A100 GPU, Python 3.9, and PyTorch 2.8. A GPU is recommended for training.

1. Clone or download the repository.
2. Create a Python environment and install a compatible build of **PyTorch + torchvision** for your machine (see [PyTorch installation instructions](https://pytorch.org/get-started/locally/)).
3. Install the remaining dependencies: `pip install -r requirements.txt`.
4. Launch JupyterLab **in the repository root** with `jupyter lab` and open `notebooks/v1_resnet.ipynb`, `v2_resnet.ipynb`, or `v3_resnet.ipynb`.
5. Run the cells in order; training automatically creates checkpoint files. The last cell generates a single-model Kaggle submission.

On NYU HPC, make sure JupyterLab is using the Python kernel with PyTorch, torchvision, and the other required packages installed.

## Dataset setup

**The dataset is not included in the repository.** Download the official images and CSV files from the course [Kaggle competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl). Do not substitute another version of CIFAR-10.

Place the files like this:

```text
cifar10-resnet-project/
├── train_labels.csv
├── test.csv
├── sample_submission.csv
└── data/
    ├── train_*.png             # 50,000 files; may be in subfolders
    └── test_*.png              # 2,000 files; may be in subfolders
```

Expected CSV column names: `train_labels.csv`: `id, filename, label`; `test.csv`: `id, filename`; `sample_submission.csv`: `id, label`. The code matches submission predictions to Kaggle IDs, rather than relying on row position alone.

## Models and methods

**V1 — Baseline.** Four feature stages (48 → 96 → 192 → 256 channels), eight ordinary residual blocks with skip connections, SGD with momentum, random crop/flip, label smoothing, and cosine learning-rate decay over 60 epochs.

**V2 — Augmented training.** The V1 network architecture, trained for 120 epochs with RandAugment, random erasing, Mixup on approximately half of batches, five-epoch learning-rate warmup, and cosine decay. The same 45k/5k split is retained for a fair validation comparison.

**V3 — Pre-activation network.** Nine residual blocks with normalization and activation before convolution, channel widths 96 → 144 → 192, and the same augmented training approach as V2. This architecture also uses skip connections.

All three networks are trained **from randomly initialized weights**, using the 50,000 supplied labeled images only. A 10% stratified hold-out is used for validation. The notebooks save the best model based on validation accuracy.

### Test-time augmentation

The notebook's final cell averages predictions for each image and its horizontal flip before taking the most likely class. This is a **single-model** inference strategy; no other network is used.

## Checkpoints and reproducibility

Training automatically saves `checkpoints/checkpoint_vN.pth` after each epoch, plus `checkpoints/best_resnet_vN.pth` whenever validation accuracy improves. If Jupyter disconnects, run the same notebook again; it resumes from the last completed epoch. Existing files are intentionally not committed to Git.

Notebook figures are written to `figures/` after training. To preserve evidence of an actual run for grading, keep the real notebook outputs and plots after rerunning on your machine; the supplied notebooks do not include invented execution outputs.

## Limitations

- We observed a sizable difference between validation and Kaggle accuracy, suggesting the held-out Kaggle images may be more challenging, but this was not conclusively established.
- Kaggle test labels are hidden; only Kaggle can report competition test accuracy.
- Validation accuracy is computed on 5,000 training-set images reserved before training.
- Single-model Kaggle results are reported; an experimental multi-model ensemble is deliberately **not included** in this codebase because of the assignment's intended parameter restriction.

## References

- Kaiming He et al., *Deep Residual Learning for Image Recognition* (2015/2016).
- Kaiming He et al., *Identity Mappings in Deep Residual Networks* (2016).
- Hongyi Zhang et al., *mixup: Beyond Empirical Risk Minimization* (2017/2018).
- Ekin D. Cubuk et al., *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space* (2020).

All network code in this repository is written specifically for this course project; no pretrained weights or external training datasets are required.
