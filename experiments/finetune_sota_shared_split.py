"""
Fine-tune MobileNetV2 and ShuffleNetV2 on the same 70/15/15 split as Models A and B.

Recipe matches experiments/train_sota.py (ImageNet init, Adam lr=1e-3, weight decay 1e-4,
cosine schedule, batch 32, 20 epochs) but runs on MPS/CUDA/CPU, keeps the best-validation
checkpoint in weights/ and writes test metrics to results/sota_metrics.json.

Usage: uv run python -m experiments.finetune_sota_shared_split
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data import get_realwaste_dataloaders, CLASSES
from src.models.sota_models import get_mobilenet_v2, get_shufflenet_v2
OUT = "weights"; EPOCHS = int(os.environ.get("SOTA_EPOCHS", 20))
LR, WD, BATCH, SEED = 1e-3, 1e-4, 32, 42
WEIGHT_FILES = {"MobileNetV2": "mobilenet_v2.pth", "ShuffleNetV2": "shufflenet_v2.pth"}
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
tr, va, te, _ = get_realwaste_dataloaders(batch_size=BATCH, seed=SEED, num_workers=0, download=False)
out = {}
for name, fn in [("MobileNetV2", get_mobilenet_v2), ("ShuffleNetV2", get_shufflenet_v2)]:
    path = f"{OUT}/{WEIGHT_FILES[name]}"; torch.manual_seed(SEED)
    model = fn(num_classes=9, pretrained=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS); crit = nn.CrossEntropyLoss()
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_times")}; best = -1
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tl = tc = n = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); o = model(x); loss = crit(o, y); loss.backward(); opt.step()
            tl += loss.item() * len(y); tc += (o.argmax(1) == y).sum().item(); n += len(y)
        sched.step(); model.eval(); vl = vc = vn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev); o = model(x); vl += crit(o, y).item() * len(y); vc += (o.argmax(1) == y).sum().item(); vn += len(y)
        dt = time.time() - t0
        for k, v in zip(hist, (tl/n, tc/n*100, vl/vn, vc/vn*100, dt)): hist[k].append(v)
        if hist["val_acc"][-1] > best: best = hist["val_acc"][-1]; torch.save(model.state_dict(), path)
        print(f"{name} ep{ep:02d} train {tc/n*100:5.2f}% val {vc/vn*100:5.2f}% ({dt:.1f}s)", flush=True)
    model.load_state_dict(torch.load(path, map_location=dev)); model.eval(); P, T = [], []
    with torch.no_grad():
        for x, y in te: P += model(x.to(dev)).argmax(1).cpu().tolist(); T += y.tolist()
    P, T = np.array(P), np.array(T)
    out[name] = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "size_mb": os.path.getsize(path)/1024**2, "epochs": EPOCHS, "best_val_acc": best, "training_history": hist,
        "test_metrics": {"accuracy": accuracy_score(T,P)*100, "precision_macro": precision_score(T,P,average="macro"),
        "recall_macro": recall_score(T,P,average="macro"), "f1_macro": f1_score(T,P,average="macro"),
        "per_class_precision": precision_score(T,P,average=None).tolist(), "per_class_recall": recall_score(T,P,average=None).tolist(),
        "confusion_matrix": confusion_matrix(T,P).tolist(), "classes": list(CLASSES)}}
    print(f"== {name} TEST acc {out[name]['test_metrics']['accuracy']:.2f}%", flush=True)
    os.makedirs("results", exist_ok=True); json.dump(out, open("results/sota_metrics.json", "w"), indent=2)
