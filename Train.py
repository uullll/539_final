from torch.amp import autocast, GradScaler
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train(net, train_dl, val_dl, criterion, opt, epochs = 3, lr = 1e-4,
          device = "mps", best_model_name = "best.pt", threshold = 0.5):
    # Set training specs
    device = device if torch.backends.mps.is_available() else "cpu"
    scaler = GradScaler(device=device)
    for g in opt.param_groups:
        g["lr"] = lr

    best_auc = 0.0
    use_meta = True if net.meta is not None else False
    hist_train_loss, hist_val_loss, hist_val_acc = [], [], []

    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(net, train_dl, scaler, device, criterion, opt, use_meta, threshold)
        ys, ps, val_loss, val_acc = val_one_epoch(net, val_dl, criterion, device, use_meta, threshold)
        ys = torch.cat(ys).numpy()
        ps = torch.cat(ps).numpy()
        auc = roc_auc_score(ys, ps)

        print(f"Epoch {ep+1}/{epochs} AUC={auc:.4f}")

        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(net.state_dict(), best_model_name)
            print(f"Saved new best model at epoch {ep+1} with AUC={auc:.4f}")

        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_val_acc.append(val_acc)

        print(f"\nEpoch {ep + 1}/{epochs}")

    print(f"Best validation AUC: {best_auc:.4f}")
    return hist_train_loss, hist_val_loss, hist_val_acc


def train_one_epoch(net, train_dl, scaler, device, criterion, opt, use_meta, threshold = 0.5):

    # Train
    net.train()
    train_acc = []
    train_loss = []
    for xb, yb, mb in tqdm(train_dl):
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device) if (use_meta and mb.numel() > 0) else None

        opt.zero_grad(set_to_none=True)

        with autocast(device_type="mps"):
            if use_meta:
                logits = net(xb, mb)
            else:
                logits = net(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        train_loss.append(loss.detach().item())
        preds = (torch.sigmoid(logits) >= threshold).float()
        train_acc.append((preds == yb).float().mean().item())


    return np.mean(train_loss), np.mean(train_acc)

def val_one_epoch(net, val_dl, criterion, device, use_meta, threshold = 0.5):
    # Validate
    net.eval()
    ys, ps = [], []
    val_loss, val_acc = [], []
    with torch.no_grad():
        for xb, yb, mb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device) if (use_meta and mb.numel() > 0) else None
            with autocast(device_type="mps"):
                logits = net(xb, mb)
                prob = torch.sigmoid(logits)
                loss = criterion(logits, yb)
            ys.append(yb.cpu())
            ps.append(prob.cpu())
            val_loss.append(loss.detach().item())
            preds = (torch.sigmoid(logits) >= threshold).float()
            val_acc.append((preds == yb).float().mean().item())

    return ys, ps, np.mean(val_loss), np.mean(val_acc)


def plot_train_hist(hist_train_loss, hist_val_loss, hist_val_acc, epochs = 3):
    plt.figure(figsize=(10, 5))
    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), hist_train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), hist_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    # Plot Val Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), hist_val_acc, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over epochs')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()