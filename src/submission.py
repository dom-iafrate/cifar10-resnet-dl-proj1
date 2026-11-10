"""Build one-model Kaggle submissions with horizontal-flip TTA."""
from pathlib import Path
import pandas as pd
import torch


def make_submission(model, test_loader, test_df, project, version, device, tta=True):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                original = model(images)
                if tta:
                    flipped = model(torch.flip(images, dims=[3]))
            probs = original.float().softmax(1)
            if tta:
                probs = (probs + flipped.float().softmax(1)) / 2
            predictions.extend(probs.argmax(1).cpu().tolist())

    project = Path(project)
    pred_df = test_df[["id"]].copy()
    pred_df["label"] = predictions
    submission = pd.read_csv(project / "sample_submission.csv")[["id"]]
    submission = submission.merge(pred_df, on="id", how="left", validate="one_to_one")
    assert len(submission) == len(test_df)
    assert submission["label"].notna().all()
    submission["label"] = submission["label"].astype(int)
    destination = project / "submissions" / f"submission_{version}{'_tta' if tta else ''}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    print("Saved:", destination)
    return destination
