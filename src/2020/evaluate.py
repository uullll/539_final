
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate(net,loader,device, use_meta, threshold=0.5):
    net.eval()
    ys, ps, total_idx = [], [],[]
    with torch.no_grad():
        for xb, yb, mb, idx in tqdm(loader):
            xb, yb = xb.to(device), yb.to(device)
            mb = mb.to(device) if (use_meta and mb.numel() > 0) else None
            logit = net(xb, mb)
            prob = torch.sigmoid(logit)
            ys.append(yb.cpu())
            ps.append(prob.cpu())
            total_idx.append(idx)

    idx = torch.cat(total_idx).numpy()
    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy().ravel()

    # Convert probabilities to class predictions
    preds = (ps >= threshold).astype(int)

    # Compute evaluation Metrics
    acc = (preds == ys).mean()
    auc = roc_auc_score(ys, ps)

    print('Test AUC: {:.4f}'.format(auc), 'Test Accuracy: {:.4f}'.format(acc))

    # Plot Confusion Matrix
    cm = confusion_matrix(ys, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)

    # Plot histogram of probability of classification
    plt.figure(figsize=(6, 4))
    sns.histplot(ps[ys == 0], color='blue', label='Class 0', kde=False, stat="density", bins=20)
    sns.histplot(ps[ys == 1], color='red', label='Class 1', kde=False, stat="density", bins=20)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Probability distribution by class")
    plt.legend()
    plt.show()
    return cm, ys, ps, idx

def plot_confusion_matrix(preds, ys, output_dir,
                          title="Evaluation_Metrics",
                          save=True,
                          dpi=300):

    os.makedirs(output_dir, exist_ok=True)

    # Compute and plot confusion matrix
    cm = confusion_matrix(ys, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)

    if save:
        base_path = f"{output_dir}/{title}"
        plt.savefig(f"{base_path}.jpg", format='jpg', dpi=dpi)
        plt.savefig(f"{base_path}.ps",  format='ps',  dpi=dpi)
        plt.savefig(f"{base_path}.eps", format='eps', dpi=dpi)

    plt.show()


    # Plot histogram of probability of classification
def probability_histogram(ps, ys, output_dir, title="Evaluation_Metrics", save=True, dpi=300):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.histplot(ps[ys == 0], color='blue', label='Class 0', kde=False, stat="density", bins=20)
    sns.histplot(ps[ys == 1], color='red', label='Class 1', kde=False, stat="density", bins=20)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    if save:
        path_base = f"{output_dir}/{title}"
        plt.savefig(f"{path_base}.jpg", format='jpg', dpi=dpi)
        plt.savefig(f"{path_base}.ps",  format='ps',  dpi=dpi)
        plt.savefig(f"{path_base}.eps", format='eps', dpi=dpi)
    plt.show()
