# Notebooks

You can open JupyterLab from the repository root or open the notebooks from their own directory. Each notebook finds the project root and imports `src` automatically.

- `v1_resnet.ipynb`: baseline architecture, 60 epochs.
- `v2_resnet.ipynb`: same architecture, improved data augmentation and Mixup, 120 epochs.
- `v3_resnet.ipynb`: pre-activation architecture, same enhanced training, 120 epochs.

Each notebook includes: dataset setup, parameter counting, checkpointed training, validation accuracy, loss/accuracy plots, and generation of a **single-model** Kaggle CSV.

The notebooks are distributed without saved execution outputs, since the original HPC sessions and outputs were not available to export. Rerun them to generate real execution results before final submission; do not invent output cells. See the performance table in the root README for the historical results recorded during the experiments.
