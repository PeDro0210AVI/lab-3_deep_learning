"""Utilidades de entrenamiento y evaluación compartidas por los modelos de torch."""
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ASLTensorDataset(Dataset):
    """Dataset en memoria a partir de arreglos uint8 [N,H,W,3] y labels int64.
    Normaliza a [0,1] float32 y permuta a canal-primero (C,H,W)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        label = int(self.y[idx])
        return img, label


def make_loaders(X_train, y_train, X_val, y_val, batch_size=128, transform=None, num_workers=0):
    train_ds = ASLTensorDataset(X_train, y_train, transform=transform)
    val_ds = ASLTensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, weight_decay=1e-4, verbose=True):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(
                f"epoch {epoch+1}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"({time.time()-t0:.1f}s)"
            )
    return history


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        running_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        preds.append(out.argmax(1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
