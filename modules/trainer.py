"""Fine-tune a model on your own dataset.

Expected folder layout (classes MUST be named benign / malignant):
    data/train/benign/*.png   data/train/malignant/*.png
    data/val/benign/*.png     data/val/malignant/*.png
"""
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import settings as cfg
from modules import metrics
from modules.classifier import build_model

log = logging.getLogger(__name__)


def make_loaders(data_dir, size, batch):
    norm = transforms.Normalize(cfg.MEAN, cfg.STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    train = datasets.ImageFolder(f"{data_dir}/train", train_tf)
    val = datasets.ImageFolder(f"{data_dir}/val", val_tf)
    if train.class_to_idx != {"benign": 0, "malignant": 1}:
        raise ValueError(f"Folders must be 'benign' and 'malignant', found {train.class_to_idx}")
    return (DataLoader(train, batch, shuffle=True, num_workers=2),
            DataLoader(val, batch, num_workers=2), np.bincount(train.targets))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += torch.softmax(model(x.to(device)), 1)[:, 1].cpu().tolist()
        ys += y.tolist()
    return np.array(ys), np.array(ps)


def train(cancer, data_dir, epochs=cfg.TRAIN_EPOCHS, batch=cfg.TRAIN_BATCH, lr=cfg.TRAIN_LR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = cfg.CANCER_TYPES[cancer]["arch"]
    size = cfg.INPUT_SIZE.get(arch, cfg.IMAGE_SIZE)
    train_dl, val_dl, counts = make_loaders(data_dir, size, batch)
    log.info("%s: %d train images, class counts %s, device %s", cancer, counts.sum(), counts.tolist(), device)

    model = build_model(arch, pretrained=True).to(device)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)  # class imbalance
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    cfg.WEIGHTS_DIR.mkdir(exist_ok=True)
    best_auc, stale, history = -1.0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()

        y_true, y_prob = evaluate(model, val_dl, device)
        m = metrics.summary(y_true, y_prob)
        history.append({"epoch": ep, "loss": total / len(train_dl.dataset), **m})
        log.info("epoch %d/%d loss %.4f val AUC %.3f sens %.2f spec %.2f",
                 ep, epochs, history[-1]["loss"], m["auc"], m["sensitivity"], m["specificity"])

        if m["auc"] > best_auc:
            best_auc, stale = m["auc"], 0
            torch.save(model.state_dict(), cfg.WEIGHTS_DIR / f"{cancer}.pt")
            thr = metrics.youden_threshold(y_true, y_prob)
            (cfg.WEIGHTS_DIR / f"{cancer}.json").write_text(
                json.dumps({"arch": arch, "val_auc": best_auc, "suggested_threshold": thr, "epoch": ep}, indent=2))
        else:
            stale += 1
            if stale >= cfg.EARLY_STOP_PATIENCE:
                log.info("Early stopping (no AUC gain for %d epochs)", stale)
                break
    log.info("Best validation AUC: %.3f -> weights/%s.pt", best_auc, cancer)
    return history
