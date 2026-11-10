"""Training code used for each version of our ResNet."""
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim


def mixup(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[indices], labels, labels[indices], lam


def setup_training(model, version, device):
    epochs = 60 if version == "v1" else 120
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    if version == "v1":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-5, eta_min=0.0001)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                           schedulers=[warmup, cosine], milestones=[5])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    return epochs, criterion, optimizer, scheduler, scaler


def resume_if_available(project, version, model, optimizer, scheduler, scaler, device):
    checkpoint_path = Path(project) / "checkpoints" / f"checkpoint_{version}.pth"
    start_epoch, best_accuracy = 0, 0.0
    losses, accuracies = [], []
    if checkpoint_path.exists():
        # Only use weights_only=False with a checkpoint you created and trust.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        losses = checkpoint["train_losses"]
        accuracies = checkpoint["val_accuracies"]
        print(f"Resuming {version} from completed epoch {start_epoch}")
    return start_epoch, best_accuracy, losses, accuracies


def evaluate(model, loader, device, tta=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                if tta:
                    flip_logits = model(torch.flip(images, dims=[3]))
            if tta:
                probs = (logits.float().softmax(1) + flip_logits.float().softmax(1)) / 2
                predicted = probs.argmax(1)
            else:
                predicted = logits.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def train(model, loaders, project, version, device):
    project = Path(project)
    checkpoint_dir = project / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs, criterion, optimizer, scheduler, scaler = setup_training(model, version, device)
    start_epoch, best_accuracy, losses, accuracies = resume_if_available(
        project, version, model, optimizer, scheduler, scaler, device)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            use_mixup = (version != "v1") and (np.random.rand() < 0.5)
            if use_mixup:
                images, labels_a, labels_b, lam = mixup(images, labels)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = (lam * criterion(outputs, labels_a) +
                        (1-lam) * criterion(outputs, labels_b)) if use_mixup else criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)

        train_loss = total_loss / len(loaders["train"].dataset)
        val_accuracy = evaluate(model, loaders["val"], device)
        scheduler.step()
        losses.append(train_loss)
        accuracies.append(val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_dir / f"best_resnet_{version}.pth")
        torch.save({
            "epoch": epoch+1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_accuracy": best_accuracy,
            "train_losses": losses, "val_accuracies": accuracies,
        }, checkpoint_dir / f"checkpoint_{version}.pth")
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | Best: {best_accuracy:.2f}%", flush=True)
    return losses, accuracies, best_accuracy


def plot_training(losses, accuracies, save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(losses)+1), losses)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(range(1, len(accuracies)+1), accuracies)
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
