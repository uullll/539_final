
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def evaluate(net, loader, device, use_meta, threshold=0.5, classes=None):
    net.eval()
    ys, ps = [], []

    def ensure_2d(arr):
        return arr[:, None] if arr.ndim == 1 else arr

    with torch.no_grad():
        for xb, yb, mb in tqdm(loader):
            xb = xb.to(device).squeeze(0)
            yb = yb.to(device)
            mb = mb.to(device).squeeze(0) if (use_meta and mb.numel() > 0) else None

            logit = net(xb, mb)
            prob = torch.sigmoid(logit)

            ys.append(ensure_2d(yb.cpu()))
            ps.append(ensure_2d(prob.cpu()))


    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy()
    preds = (ps >= threshold).astype(int)

    acc = (preds == ys).mean()

    per_class_acc = (preds == ys).mean(axis=0)

    num_classes = ys.shape[1]

    if classes is None:
        if num_classes == 1:
            classes = ["MEL"]
        elif num_classes == 9:
            classes = ["MEL", "NV", "BKL", "DF", "VASC", "SCC", "AK", "BCC", "UNK"]
        elif num_classes == 4:
            classes = ["MEL", "NV", "BKL", "BMP"]

    # AUC per class
    auc_per_class = []
    for c in range(num_classes):
        if ys[:, c].max() == ys[:, c].min():
            auc_per_class.append(float("nan"))
        else:
            auc_per_class.append(roc_auc_score(ys[:, c], ps[:, c]))

    # Confusion matrices
    cms = {}
    for i, name in enumerate(classes):
        cms[name] = confusion_matrix(ys[:, i], preds[:, i], labels=[0, 1])

    # Print results
    for name, auc_c, acc_c in zip(classes, auc_per_class, per_class_acc):
        print(f"{name}: AUC={auc_c:.4f}  ACC={acc_c:.4f}")
        print(cms[name], "\n")

    return acc, per_class_acc, auc_per_class, ys, ps

def ensure_2d(arr):
    return arr if arr.ndim > 1 else np.expand_dims(arr, axis=0)


def confusion_matrices(ys, preds, class_names):
    cms = {}
    for i, name in enumerate(class_names):
        cm = confusion_matrix(ys[:, i], preds[:, i], labels=[0, 1])
        cms[name] = cm
    return cms
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
    return cm, ys, ps



def plot_confusion_matrix(preds, ys, output_dir,
                          class_names=None,
                          title="Evaluation_Metrics",
                          save=True,
                          dpi=300):

    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(ys, preds)

    if class_names is None:
        class_names = list(range(cm.shape[0]))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='d')
    plt.title(title)

    if save:
        base_path = f"{output_dir}/{title}"
        plt.savefig(f"{base_path}.jpg", format='jpg', dpi=dpi)

    plt.show()


    # Plot histogram of probability of classification
def probability_histogram(ps, ys, output_dir = None, title="Evaluation_Metrics", save=False, dpi=300):
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
        path_base = f"{output_dir}/{title}"
        plt.savefig(f"{path_base}.jpg", format='jpg', dpi=dpi)
        # plt.savefig(f"{path_base}.ps",  format='ps',  dpi=dpi)
        # plt.savefig(f"{path_base}.eps", format='eps', dpi=dpi)
    plt.show()