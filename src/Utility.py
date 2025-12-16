
import inspect
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def add_patient_labels(df, filename='GroundTruth_PatientLabels.csv'):
    patient_labels_df = df.groupby("patient_id")["target"].max().reset_index()
    patient_labels_df.rename(columns={"target": "patient_label"}, inplace=True)

    # Merge into original DataFrame
    df = df.merge(patient_labels_df, on="patient_id", how="left")
    df.to_csv(filename, index=False)


## Source from d2l package:
class HyperParameters:
    """Base class for hyper‑parameter containers."""
    def save_hyperparameters(self, ignore: list = None):
        """Save all arguments passed to the caller’s __init__ into self, except ignored."""
        if ignore is None:
            ignore = []
        # inspect the frame of the caller (should be __init__)
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        # filter out private / ignored names and 'self'
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ['self'])
               and not k.startswith('_')
        }
        for name, value in self.hparams.items():
            setattr(self, name, value)

class EpochTrainingMetrics:
    def __init__(self):
        self._reset_epoch()

    def __getattr__(self, name):
        if name in self.epoch_metrics:
            vals = self.epoch_metrics[name]
            if name in ["loss", "acc_patient", "acc_lesion"]:
                return float(np.mean(vals)) if vals else 0.0
            vals = [v if torch.is_tensor(v) else torch.tensor(v) for v in vals]
            return torch.cat(vals) if vals else torch.tensor([])
        raise AttributeError(name)

    def _reset_epoch(self):
        self.epoch_metrics = {
            "acc_patient": [],
            "acc_lesion": [],
            "loss": [],
            "ys_patient": [],
            "prob_patient": [],
            "ys_lesion": [],
            "prob_lesion": []
        }

    def _get_accuracy_prob(self, logits, yb, threshold=0.5):
        prob = torch.sigmoid(logits)
        preds = (prob >= threshold).float()
        return (preds == yb).float().mean().item(), prob

    def append_loss(self, loss):
        self.epoch_metrics["loss"].append(float(loss))

    def _save_metric(self, name, metric):
        if torch.is_tensor(metric):
            metric = metric.detach().cpu()
        self.epoch_metrics[name].append(metric)

    def compute_metrics(self, logits_patient, logits_lesion, yb_lesion, yb_patient):
        acc_lesion, prob_lesion = self._get_accuracy_prob(logits_lesion, yb_lesion)
        acc_patient, prob_patient = self._get_accuracy_prob(logits_patient, yb_patient)

        self._save_metric("acc_patient", acc_patient)
        self._save_metric("acc_lesion", acc_lesion)
        self._save_metric("prob_patient",prob_patient)
        self._save_metric("prob_lesion",prob_lesion)
        self._save_metric("ys_lesion", yb_lesion)
        self._save_metric("ys_patient",yb_patient)


class TrainingMetrics:
    def __init__(self, threshold=0.5, best_model_name='best_model.pth'):
        self.threshold = threshold
        self.best_model_name = best_model_name
        self._reset()

    def __getattr__(self, name):
        if name in self.history:
            return self.history[name] if self.history[name] else None
        raise AttributeError(name)

    def _reset(self):
        self.history = {
            "hist_train_loss": [],
            "hist_val_loss": [],
            "hist_val_patient_acc": [],
            "hist_val_lesion_acc": []
            
        }
        # Current metrics
        self.pos_patient_acc = 0.0
        self.patient_auc = None
        self.best_patient_pos_acc = 0.0
        self.no_improvement = 0
        self.loss_ratio = float("inf")

    def _compute_pos_acc(self, prob, ys):
        preds = (prob >= self.threshold).float()
        pos_mask = (ys == 1)
        return (preds[pos_mask] == 1).float().mean().item() if pos_mask.sum() > 0 else 0.0

    def _compute_auc(self, prob, ys):
        y_np = ys.cpu().numpy()
        if len(np.unique(y_np)) < 2:
            return 0.0
        return roc_auc_score(y_np, prob.cpu().numpy())

    def _compute_loss_ratio(self, val_loss, train_loss):
        return val_loss / train_loss if train_loss > 0 else float("inf")

    def _compute_patient_probability(self, logits_patient):
        return torch.sigmoid(logits_patient)

    def update(self, train_metrics, val_metrics, net=None, epoch=None):
        """Update metrics after one epoch and optionally save the best model."""
        self.history["hist_train_loss"].append(train_metrics.loss)
        self.history["hist_val_loss"].append(val_metrics.loss)
        self.history["hist_val_patient_acc"].append(val_metrics.acc_patient)
        self.history["hist_val_lesion_acc"].append(val_metrics.acc_lesion)


        self.last_epoch = epoch
        self.loss_ratio = self._compute_loss_ratio(val_metrics.loss, train_metrics.loss)
        self.pos_patient_acc = self._compute_pos_acc(val_metrics.prob_patient, val_metrics.ys_patient)
        self.pos_lesion_acc = self._compute_pos_acc(val_metrics.prob_lesion, val_metrics.ys_lesion)
        self.patient_auc = self._compute_auc(val_metrics.prob_patient, val_metrics.ys_patient)
        self.lesion_auc = self._compute_auc(val_metrics.prob_lesion, val_metrics.ys_lesion)

        # Save best model based on positive accuracy
        if self.pos_patient_acc > self.best_patient_pos_acc:
            self.best_patient_pos_acc = self.pos_patient_acc
            self.no_improvement = 0
            if net is not None and self.best_model_name is not None and epoch is not None:
                torch.save(net.state_dict(), self.best_model_name)
                print(f"Saved new best model at epoch {epoch + 1} with patient_auc={self.patient_auc:.4f}")
        else:
            self.no_improvement += 1

        print(f"Epoch {epoch + 1}: "
              f"Train Loss={train_metrics.loss:.4f}, "
              f"Val Loss={val_metrics.loss:.4f}, "
              f"patient_acc={val_metrics.acc_patient:.4f}, "
              f"patient_pos_acc={self.pos_patient_acc:.4f}, "
              f"lesion_acc={val_metrics.acc_lesion:.4f}, "
              f"lesion_pos_acc={self.pos_lesion_acc:.4f}, "
              )





