"""Data loading and transforms for our CIFAR-10 Kaggle competition."""
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

class CIFARCompetitionDataset(Dataset):
    def __init__(self, dataframe, image_paths, transform=None, labeled=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(self.image_paths[row["filename"]]) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labeled:
            return image, int(row["label"])
        return image


def get_transforms(version):
    augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if version in ("v2", "v3"):
        augmentation.append(transforms.RandAugment(num_ops=2, magnitude=7))
    augmentation += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if version in ("v2", "v3"):
        augmentation.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    train_transform = transforms.Compose(augmentation)
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train_transform, eval_transform


def get_loaders(project, version, workers=4):
    project = Path(project)
    data = project / "data"
    train_df = pd.read_csv(project / "train_labels.csv")
    test_df = pd.read_csv(project / "test.csv")
    train_paths = {p.name: p for p in data.rglob("train_*.png")}
    test_paths = {p.name: p for p in data.rglob("test_*.png")}
    assert len(train_paths) == 50000, f"Expected 50,000 training PNGs; found {len(train_paths)}"
    assert len(test_paths) == 2000, f"Expected 2,000 test PNGs; found {len(test_paths)}"
    assert train_df["filename"].isin(train_paths).all(), "Some training images were not found"
    assert test_df["filename"].isin(test_paths).all(), "Some test images were not found"
    train_split, val_split = train_test_split(train_df, test_size=0.1,
        random_state=42, stratify=train_df["label"])
    train_tf, eval_tf = get_transforms(version)
    train_ds = CIFARCompetitionDataset(train_split, train_paths, train_tf)
    val_ds = CIFARCompetitionDataset(val_split, train_paths, eval_tf)
    test_ds = CIFARCompetitionDataset(test_df, test_paths, eval_tf, labeled=False)
    return {
        "train": DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True),
        "train_df": train_df, "test_df": test_df,
    }
