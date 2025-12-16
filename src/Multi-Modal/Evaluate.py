import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def evaluate(net, loader, device, use_meta, threshold=0.5):
    net.eval()
    ys_patient, ps_patient, ys_lesion, ps_lesion, total_idx = [], [], [], [], []

    with torch.no_grad():
        for xb, yb_patient, yb_lesion_batch, mb in tqdm(loader):
            xb = xb.squeeze(0)
            yb_lesion_batch = yb_lesion_batch.squeeze(0)
            yb_patient = yb_patient.squeeze(0)
            if mb.numel() > 0:
                mb = mb.squeeze(0)

            xb = xb.to(device)
            yb_patient = yb_patient.to(device)
            yb_lesion_batch = yb_lesion_batch.to(device)
            mb = mb.to(device) if (use_meta and mb.numel() > 0) else None

            logits_patient,logits_lesion = net(xb, mb)

            # Convert outputs and batch data to numpy for evaluation metrics
            ys_patient.append(yb_patient.cpu().numpy().ravel())
            ps_patient.append(torch.sigmoid(logits_patient).cpu().numpy().ravel())
            ys_lesion.append(yb_lesion_batch.cpu().numpy().ravel())
            ps_lesion.append(torch.sigmoid(logits_lesion).cpu().numpy().ravel())

    ys_patient = np.concatenate(ys_patient)
    ps_patient = np.concatenate(ps_patient)
    ys_lesion = np.concatenate(ys_lesion)
    ps_lesion = np.concatenate(ps_lesion)

    # Convert probabilities to class predictions
    preds_patient = (ps_patient >= threshold).astype(int)
    preds_lesion = (ps_lesion >= threshold).astype(int)

    # Compute metrics
    auc_patient = roc_auc_score(ys_patient, ps_patient)
    auc_lesion = roc_auc_score(ys_lesion, ps_lesion)
    acc_patient = (preds_patient == ys_patient).mean()
    acc_lesion = (preds_lesion == ys_lesion).mean()

    print(f"Patient AUC: {auc_patient:.4f}, Patient Accuracy: {acc_patient:.4f}")
    print(f"Lesion AUC: {auc_lesion:.4f}, Lesion Accuracy: {acc_lesion:.4f}")

    return {
        "ys_patient": ys_patient,
        "ps_patient": ps_patient,
        "preds_patient": preds_patient,
        "ys_lesion": ys_lesion,
        "ps_lesion": ps_lesion,
        "preds_lesion": preds_lesion,
        "auc_patient": auc_patient,
        "auc_lesion": auc_lesion,
        "acc_patient": acc_patient,
        "acc_lesion": acc_lesion
    }

def plot_confusion_matrix(preds, ys, output_dir = None, title="Confusion_Matrix", save=False, dpi=300):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(ys, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)

    if save:
        plt.savefig(os.path.join(output_dir, f"{title}.jpg"), format='jpg', dpi=dpi)
    plt.show()


def probability_histogram(ps, ys, output_dir = None, title="Probability_Histogram", save=False, dpi=300):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.histplot(ps[ys == 0], color='blue', label='Class 0', kde=False, stat="density", bins=20)
    sns.histplot(ps[ys == 1], color='red', label='Class 1', kde=False, stat="density", bins=20)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()

    if save:
        plt.savefig(os.path.join(output_dir, f"{title}.jpg"), format='jpg', dpi=dpi)
    plt.show()
