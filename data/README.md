# Dataset placement

Download the **competition-provided** data yourself from Kaggle. Do not upload the training/test images into the public repository.

Extract the images into this directory, e.g. `data/train_00001.png` and `data/test_00001.png`. The loader searches recursively, so subfolders are fine. Place `train_labels.csv`, `test.csv`, and `sample_submission.csv` in the repository root (one level above this folder).

Expected totals: **50,000** training PNGs and **2,000** test PNGs. The notebooks check these counts. Do not replace these with another CIFAR-10 dataset: it may not match the competition filenames, test IDs, or labels.

I have not uploaded the full dataset here due to size constrains and access issues. View and download the dataset at https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/data.

