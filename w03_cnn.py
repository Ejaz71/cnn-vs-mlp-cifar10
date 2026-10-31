"""CSE752 Week 3: small CNN vs flatten-and-MLP on CIFAR-10 (Steps 1-7 + optional stretch).

Usage: python w03_cnn.py [--quick] [--full] [--epochs N] [--stretch] [--data-dir DIR]
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 0
BATCH_SIZE = 128
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLORS = {"MLP": "#0072B2", "CNN": "#D55E00", "CNN+aug": "#009E73", "CNN-3block": "#CC79A7"}

History = Dict[str, List[float]]


# ----------------------------------------------------------------------------- helpers
def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (incl. cuda if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def base_transform() -> transforms.Compose:
    """ToTensor + Normalize (test transform, and the no-augmentation train transform)."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_loaders(
    train_tf: transforms.Compose,
    subset_frac: float,
    data_dir: str = "./data",
    test_frac: float = 1.0,
    seed: int = SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Train/test loaders. The train subset is a fixed seeded random sample, identical for every run."""
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=base_transform())
    if subset_frac < 1.0:
        idx = np.random.RandomState(seed).permutation(len(train))[: round(subset_frac * len(train))]
        train = Subset(train, idx.tolist())
    if test_frac < 1.0:
        test = Subset(test, list(range(round(test_frac * len(test)))))
    gen = torch.Generator().manual_seed(seed)
    loader_tr = DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, generator=gen,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    loader_te = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    return loader_tr, loader_te


def count_params(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def param_breakdown(model: nn.Module) -> Dict[str, int]:
    """Print a per-layer parameter table; return {'conv': n, 'fc': n, 'total': n}."""
    split = {"conv": 0, "fc": 0}
    for name, m in model.named_modules():
        own = sum(p.numel() for p in m.parameters(recurse=False))
        if own == 0:
            continue
        print(f"    {name:<8} {type(m).__name__:<8} {tuple(m.weight.shape)!s:<20} {own:>9,}")
        split["conv" if isinstance(m, nn.Conv2d) else "fc"] += own
    split["total"] = count_params(model)
    print(f"    conv total = {split['conv']:,}   fc total = {split['fc']:,}   all = {split['total']:,}")
    return split


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy in eval mode."""
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return correct / n


def train_eval(
    model: nn.Module, opt: torch.optim.Optimizer, loader_tr: DataLoader, loader_te: DataLoader,
    epochs: int, device: torch.device, name: str = "",
) -> History:
    """Train with CrossEntropyLoss on raw logits; record per-epoch metrics."""
    crit = nn.CrossEntropyLoss()
    hist: History = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_time": []}
    model.to(device)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, n, nb = 0.0, 0, 0, 0
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            nb += 1
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        test_acc = evaluate(model, loader_te, device)
        hist["train_loss"].append(loss_sum / nb)
        hist["train_acc"].append(correct / n)
        hist["test_acc"].append(test_acc)
        hist["epoch_time"].append(time.time() - t0)
        print(f"  [{name}] epoch {ep:>2}/{epochs}  loss {hist['train_loss'][-1]:.4f}  "
              f"train_acc {hist['train_acc'][-1]:.4f}  test_acc {test_acc:.4f}  {hist['epoch_time'][-1]:.1f}s")
    return hist


# ----------------------------------------------------------------------------- models
def make_mlp() -> nn.Module:
    """Flatten -> Linear(3072,256) -> ReLU -> Linear(256,10)."""
    return nn.Sequential(nn.Flatten(), nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, 10))


class SmallCNN(nn.Module):
    """VGG-style two-block CNN, exactly as in the assignment PDF."""

    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class SmallCNN3(nn.Module):
    """Stretch: SmallCNN plus a third 128-channel block, pooled to 4x4."""

    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def fit(
    name: str, model_fn: Callable[[], nn.Module], device: torch.device, train_tf: transforms.Compose,
    cfg: dict, num_workers: int = 0,
) -> Tuple[nn.Module, History]:
    """One reproducible run: seed -> loaders -> model -> optimizer -> train."""
    seed_everything(SEED)
    loader_tr, loader_te = get_loaders(train_tf, cfg["subset_frac"], cfg["data_dir"], num_workers=num_workers)
    model = model_fn()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = train_eval(model, opt, loader_tr, loader_te, cfg["epochs"], device, name)
    return model, hist


# ----------------------------------------------------------------------------- plots
def plot_curves(runs: Dict[str, History], path: str, title: str) -> None:
    """Two subplots: train loss (left), test accuracy (right)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in runs.items():
        ep = range(1, len(h["train_loss"]) + 1)
        a1.plot(ep, h["train_loss"], marker="o", color=COLORS[name], label=name)
        a2.plot(ep, [100 * a for a in h["test_acc"]], marker="o", color=COLORS[name], label=name)
    a1.set(xlabel="Epoch", ylabel="Train loss (mean minibatch CE)", title="Train loss per epoch")
    a2.set(xlabel="Epoch", ylabel="Test accuracy (%)", title="Test accuracy per epoch")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- steps
def run_step1(device: torch.device, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Print one batch shape and sanity-check conv output sizes against the formula."""
    loader_tr, _ = get_loaders(base_transform(), cfg["subset_frac"], cfg["data_dir"])
    x, y = next(iter(loader_tr))
    print(f"[Step 1] batch images {tuple(x.shape)}, labels {tuple(y.shape)}")
    assert tuple(x.shape) == (BATCH_SIZE, 3, 32, 32), x.shape
    H, k, p = x.shape[-1], 3, 1
    for s in (1, 2):
        out = nn.Conv2d(3, 16, kernel_size=k, padding=p, stride=s)(x)
        pred = (H + 2 * p - k) // s + 1
        print(f"  stride={s}: output {tuple(out.shape)}   H' = floor((H + 2p - k)/s) + 1 "
              f"= floor(({H} + {2 * p} - {k})/{s}) + 1 = {pred}")
        assert out.shape[-2:] == (pred, pred), (out.shape, pred)
    return x, y


def run_step2() -> int:
    """MLP parameter count."""
    n = count_params(make_mlp())
    print(f"[Step 2] MLP params = {n:,}")
    assert n == 789_258, n
    return n


def run_step3(mlp_params: int) -> Dict[str, int]:
    """CNN parameter count and breakdown."""
    model = SmallCNN()
    n = count_params(model)
    print(f"[Step 3] SmallCNN params = {n:,}")
    assert n == 591_274, n
    split = param_breakdown(model)
    print(f"  MLP / CNN parameter ratio = {mlp_params / n:.3f}")
    return split


def run_step4(device: torch.device, cfg: dict) -> Tuple[Dict[str, History], nn.Module]:
    """Train MLP and CNN (no augmentation); save the comparison plot and CNN weights."""
    print("[Step 4] training MLP and CNN (Adam 1e-3, no augmentation)")
    _, h_mlp = fit("MLP", make_mlp, device, base_transform(), cfg)
    cnn, h_cnn = fit("CNN", SmallCNN, device, base_transform(), cfg)
    runs = {"MLP": h_mlp, "CNN": h_cnn}
    for name, h in runs.items():
        if h["test_acc"][-1] < 0.15:
            raise SystemExit(f"{name} ended near chance ({h['test_acc'][-1]:.3f}); check loss/logits.")
    plot_curves(runs, "w03_mlp_vs_cnn.png", "MLP vs CNN on CIFAR-10")
    cnn = cnn.cpu()
    torch.save(cnn.state_dict(), "cnn_baseline.pt")
    verdict = "YES" if h_cnn["test_acc"][-1] > h_mlp["test_acc"][-1] else "NO"
    print(f"  final test acc: MLP {h_mlp['test_acc'][-1]:.4f}  CNN {h_cnn['test_acc'][-1]:.4f}  "
          f"-> CNN beats MLP: {verdict}")
    return runs, cnn


def run_step5(cnn: nn.Module, cfg: dict) -> dict:
    """Plot image, first 8 conv1 activation maps, and the matching 3x3 kernels."""
    cnn.eval().cpu()
    test = datasets.CIFAR10(cfg["data_dir"], train=False, download=True, transform=base_transform())
    chosen = None
    for cls in (3, 1):  # cat, then automobile
        for idx in [i for i, t in enumerate(test.targets) if t == cls]:
            x, _ = test[idx]
            with torch.no_grad():
                if cnn(x[None]).argmax(1).item() == cls:
                    chosen = (idx, cls, x)
                    break
        if chosen:
            break
    if chosen is None:
        raise SystemExit("Step 5: CNN classified no cat/automobile test image correctly.")
    idx, cls, x = chosen
    print(f"[Step 5] test image index {idx}, class {cls} ({CLASSES[cls]}), correctly classified")
    conv1 = cnn.f[0]
    with torch.no_grad():
        acts = conv1(x[None])[0, :8]  # pre-ReLU
    kern = conv1.weight.detach()[:8]
    img = (x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(3, 8)
    ax = fig.add_subplot(gs[0, 3:5])
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"test #{idx} ({CLASSES[cls]})")
    ax.axis("off")
    for c in range(8):
        a = fig.add_subplot(gs[1, c])
        a.imshow(acts[c].numpy(), cmap="gray")
        a.set_title(f"ch {c}")
        a.axis("off")
        k = kern[c]
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        b = fig.add_subplot(gs[2, c])
        b.imshow(k.permute(1, 2, 0).numpy(), interpolation="nearest")
        b.set_title(f"kernel {c}")
        b.axis("off")
    fig.suptitle("Conv1 (first 8 of 32 channels): row 2 = activations (pre-ReLU), row 3 = learned 3x3 kernels (min-max normalised)")
    fig.tight_layout()
    fig.savefig("w03_maps.png", dpi=150)
    plt.close(fig)
    return {"index": idx, "class": CLASSES[cls]}


def rf_table(layers: List[Tuple[str, int, int]]) -> List[dict]:
    """RF_l = RF_{l-1} + (k_l - 1) * prod_{i<l} s_i. Rows: layer, k, s, jump (prod s_i, i<=l), rf."""
    rows = [{"layer": "input", "k": "-", "s": "-", "jump": 1, "rf": 1}]
    rf, jump_in = 1, 1
    for name, k, s in layers:
        rf = rf + (k - 1) * jump_in
        jump_in *= s
        rows.append({"layer": name, "k": k, "s": s, "jump": jump_in, "rf": rf})
    return rows


def run_step6() -> List[dict]:
    """Print/save the receptive-field table for the two SmallCNN blocks."""
    layers = [("conv3x3 s1", 3, 1), ("conv3x3 s1", 3, 1), ("maxpool2x2 s2", 2, 2),
              ("conv3x3 s1", 3, 1), ("conv3x3 s1", 3, 1), ("maxpool2x2 s2", 2, 2)]
    rows = rf_table(layers)
    assert rows[3]["rf"] == 6, rows[3]  # hand-computed: 1 -> 3 -> 5 -> 6
    lines = [f"{'layer':<16}{'k':>3}{'s':>3}{'jump':>6}{'RF':>5}"]
    lines += [f"{r['layer']:<16}{r['k']!s:>3}{r['s']!s:>3}{r['jump']:>6}{r['rf']:>5}" for r in rows]
    lines.append("(jump = cumulative stride product after the layer; the recurrence uses the jump before it)")
    text = "\n".join(lines)
    print("[Step 6] receptive field\n" + text)
    with open("w03_rf.txt", "w") as f:
        f.write(text + "\n")
    return rows


def run_step7(device: torch.device, cfg: dict) -> History:
    """Retrain the CNN with train-only augmentation (Normalize last)."""
    print("[Step 7] retraining CNN with RandomCrop(32, pad=4) + RandomHorizontalFlip")
    aug_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    _, h = fit("CNN+aug", SmallCNN, device, aug_tf, cfg, num_workers=2)
    return h


def run_stretch(device: torch.device, cfg: dict) -> Tuple[History, int]:
    """Three-block CNN, no augmentation."""
    model = SmallCNN3()
    n = count_params(model)
    print(f"[Stretch] 3-block CNN params = {n:,}")
    param_breakdown(model)
    _, h = fit("CNN-3block", SmallCNN3, device, base_transform(), cfg)
    return h, n


def selftest_determinism(cfg: dict) -> bool:
    """Two identical 1-epoch CPU runs on 1,000 images must give identical loss histories."""
    cpu = torch.device("cpu")
    small = dict(cfg, subset_frac=0.02, epochs=1)
    for name, fn in (("MLP", make_mlp), ("CNN", SmallCNN)):
        losses = []
        for _ in range(2):
            seed_everything(SEED)
            tr, te = get_loaders(base_transform(), small["subset_frac"], small["data_dir"], test_frac=0.05)
            m = fn()
            h = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), tr, te, 1, cpu, f"selftest-{name}")
            losses.append(h["train_loss"])
        assert losses[0] == losses[1], (name, losses)
    print("[Self-test] determinism OK (identical loss histories for MLP and CNN)")
    return True


# ----------------------------------------------------------------------------- reporting
def write_summary(runs: Dict[str, History], params: Dict[str, int], path: str = "w03_summary.md") -> List[str]:
    """Write a plain markdown table; return a list of anomaly flags."""
    lines = ["| Model | Params | Final train loss | Final train acc | Final test acc | Train-test acc gap |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, h in runs.items():
        lines.append(f"| {name} | {params[name]:,} | {h['train_loss'][-1]:.4f} | {h['train_acc'][-1]:.4f} | "
                     f"{h['test_acc'][-1]:.4f} | {h['train_acc'][-1] - h['test_acc'][-1]:+.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    flags = []
    if any(not math.isfinite(v) for h in runs.values() for v in h["train_loss"]):
        flags.append("NaN/inf in train loss")
    t = {k: h["test_acc"][-1] for k, h in runs.items()}
    if t["CNN"] <= t["MLP"]:
        flags.append("CNN did not beat MLP on test accuracy")
    if "CNN+aug" in t and t["CNN+aug"] < t["CNN"]:
        flags.append("augmentation lowered test accuracy vs no-aug CNN")
    if "CNN-3block" in t and t["CNN-3block"] < t["CNN"]:
        flags.append("3-block CNN scored below the 2-block CNN")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, 2,000 train images")
    ap.add_argument("--full", action="store_true", help="full training set, 10 epochs, even on CPU/MPS")
    ap.add_argument("--epochs", type=int, default=None, help="override epoch count")
    ap.add_argument("--stretch", action="store_true", help="also run the 3-block CNN")
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    t_start = time.time()
    device = get_device()
    gpu = device.type == "cuda"
    if args.quick:
        frac, epochs = 0.04, 2
    elif args.full or gpu:
        frac, epochs = 1.0, 10
    else:
        frac, epochs = 0.2, 5
    if args.epochs is not None:
        epochs = args.epochs
    cfg = {"subset_frac": frac, "epochs": epochs, "data_dir": args.data_dir, "seed": SEED,
           "batch_size": BATCH_SIZE, "lr": 1e-3, "device": str(device), "quick": args.quick,
           "full": args.full, "stretch": args.stretch, "n_train": round(frac * 50000), "n_test": 10000}
    print(f"Device: {device} | train images: {cfg['n_train']:,} ({frac:.0%}) | test images: 10,000 | epochs: {epochs}")

    selftest_determinism(cfg)
    run_step1(device, cfg)
    mlp_params = run_step2()
    split = run_step3(mlp_params)
    runs, cnn = run_step4(device, cfg)
    maps_info = run_step5(cnn, cfg)
    rf_rows = run_step6()
    runs["CNN+aug"] = run_step7(device, cfg)
    params = {"MLP": mlp_params, "CNN": split["total"], "CNN+aug": split["total"]}
    plot_curves({k: runs[k] for k in ("CNN", "CNN+aug")}, "w03_aug.png", "CNN vs CNN + light augmentation")
    h_cnn, h_aug = runs["CNN"], runs["CNN+aug"]
    print(f"  CNN     final test {h_cnn['test_acc'][-1]:.4f}  train-test gap {h_cnn['train_acc'][-1] - h_cnn['test_acc'][-1]:+.4f}")
    print(f"  CNN+aug final test {h_aug['test_acc'][-1]:.4f}  train-test gap {h_aug['train_acc'][-1] - h_aug['test_acc'][-1]:+.4f}")
    if args.stretch:
        runs["CNN-3block"], params["CNN-3block"] = run_stretch(device, cfg)
        print(f"  2-block CNN test {h_cnn['test_acc'][-1]:.4f} vs 3-block CNN test {runs['CNN-3block']['test_acc'][-1]:.4f}")
        plot_curves({k: runs[k] for k in ("CNN", "CNN-3block")}, "w03_stretch.png", "2-block vs 3-block CNN")

    flags = write_summary(runs, params)
    with open("w03_results.json", "w") as f:
        json.dump({"config": cfg, "params": params, "param_split_cnn": split, "step5": maps_info,
                   "rf_table": rf_rows, "runs": runs}, f, indent=2)
    print("\n" + open("w03_summary.md").read())
    print("Anomalies:", flags if flags else "none")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
