from torch.amp import autocast, GradScaler
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from Utility import TrainingMetrics, epochTrainingMetrics
import contextlib

def train(
        net, train_dl, val_dl, criterion_patient, criterion_lesion, opt,
        epochs=3, device="mps", best_model_name="best.pt",
        threshold=0.5, patience=2
):
    net.to(device)

    use_meta = True if net.meta_dim > 0 else False

    scaler = GradScaler() if "cuda" in device else None

    # Track training metrics
    metrics = TrainingMetrics(threshold=threshold, best_model_name=best_model_name)

    for epoch in range(epochs):
        # Train one epoch
        train_metrics = train_one_epoch(net, train_dl, scaler, device, criterion_patient, criterion_lesion, opt, use_meta, threshold)
        # Validate one epoch
        val_metrics = val_one_epoch(net, val_dl, criterion_patient, criterion_lesion, device, use_meta, threshold)

        # Update global metrics and optionally save the best model
        metrics.update(train_metrics, val_metrics, net=net, epoch=epoch)

        if metrics.no_improvement > patience:
            print("Early stopping triggered due to no improvement.")
            break

    print(f"Best validation positive accuracy: {metrics.best_pos_acc:.4f}")
    return metrics


def train_one_epoch(net, train_dl, scaler, device, criterion_patient, criterion_lesion, opt, use_meta, threshold=0.5):
    net.train()
    metrics = epochTrainingMetrics()

    for xb, yb_patient, yb_lesion, mb in tqdm(train_dl):
        xb, yb_patient, yb_lesion = xb.to(device).squeeze(0), yb_patient.to(device).squeeze(0), yb_lesion.to(device).squeeze(0)
        mb = mb.to(device).squeeze(0) if use_meta else None

        opt.zero_grad(set_to_none=True)

        autocast_ctx = (
            autocast(device_type="cuda")
            if "cuda" in device else
            contextlib.nullcontext()
        )
        with autocast_ctx:
            logits_patient, logits_lesion = net(xb, mb) if use_meta else net(xb)
            loss_patient = criterion_patient(logits_patient, yb_patient)
            loss_lesion = criterion_lesion(logits_lesion.squeeze(-1), yb_lesion)
            loss =  loss_lesion*0.001 + loss_patient

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        metrics.append_loss(loss.detach().item())
        metrics.compute_metrics(logits_patient, logits_lesion, yb_lesion, yb_patient)

    return metrics


def val_one_epoch(net, val_dl, criterion_patient, criterion_lesion, device, use_meta, threshold=0.5):
    net.eval()
    metrics = epochTrainingMetrics()

    with torch.no_grad():
        for xb, yb_patient, yb_lesion, mb in tqdm(val_dl):
            xb, yb_patient, yb_lesion = xb.to(device).squeeze(0), yb_patient.to(device).squeeze(0), yb_lesion.to(
                device).squeeze(0)
            mb = mb.to(device).squeeze(0) if use_meta else None

            logits_patient, logits_lesion = net(xb, mb) if use_meta else net(xb)
            loss_patient = criterion_patient(logits_patient, yb_patient)
            loss_lesion = criterion_lesion(logits_lesion.squeeze(-1), yb_lesion)
            loss = loss_patient + loss_lesion * 0.2

            metrics.append_loss(loss.detach().item())
            metrics.compute_metrics(logits_patient, logits_lesion, yb_lesion, yb_patient)

    return metrics


def plot_train_hist(train_metrics):
    epochs = train_metrics.last_epoch
    hist_train_loss = train_metrics.hist_train_loss
    hist_val_loss = train_metrics.hist_val_loss
    hist_val_acc = train_metrics.hist_val_acc

    plt.figure(figsize=(10, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    x = range(len(hist_train_loss))
    # Line plots
    plt.plot(x, hist_train_loss, label='Train Loss', color='blue')
    plt.plot(x, hist_val_loss, label='Validation Loss', color='orange')
    # Scatter plots
    plt.scatter(x, hist_train_loss, color='blue', marker='o', s=30)
    plt.scatter(x, hist_val_loss, color='orange', marker='x', s=30)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    #  Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    x = range(len(hist_val_acc))
    plt.plot(x, hist_val_acc, label='Validation Accuracy', color='green')
    plt.scatter(x, hist_val_acc, color='darkgreen', marker='o', s=30)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over epochs')
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()


def set_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

