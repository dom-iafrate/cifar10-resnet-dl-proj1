# Saved model weights

The training notebooks save weights and resumable checkpoints here:

- `best_resnet_v1.pth` / `checkpoint_v1.pth`
- `best_resnet_v2.pth` / `checkpoint_v2.pth`
- `best_resnet_v3.pth` / `checkpoint_v3.pth`

These binary files are excluded from Git. The code can train every model from scratch, as required. If you already have weights in the older `~/deep-learning-project1` folder, copy them here and rename the original `best_resnet.pth` to `best_resnet_v1.pth`. The V2 and V3 filenames are `best_resnet_v2.pth` and `best_resnet_v3.pth`. The **V1 and V2 architecture is the same**, while V3 needs the V3 class.
