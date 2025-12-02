
import inspect
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

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

class epochTrainingMetrics:
    def __init__(self):
        self.reset_epoch()

    def reset_epoch(self):
        self.epoch_metrics = {
            "acc_patient": [],
            "acc_lesion": [],
            "loss": [],
            "ys_patient": [],
            "prob_patient": [],
            "ys_lesion": [],
            "prob_lesion": [],
            "yb_lesion": [],
            "yb_patient": []
        }

    def _get_accuracy_prob(self, logits, yb, threshold=0.5):
        prob = torch.sigmoid(logits)
        preds = (prob >= threshold).float()
        return (preds == yb).float().mean().item(), prob

    def append_loss(self, loss):
        self.epoch_metrics["loss"].append(float(loss))



    # def _get_variable_name(self, target_value):
    #     for name, value in list(locals().items()):
    #         if value is target_value:
    #             return name
    #     return None

    # def _save_metric(self, metric):
    #     metric_name = self.get_variable_name(metric)
    #     self.epoch_metrics[f"{metric_name}"].append(metric.detach().cpu())
    #     self.metric_name = self.epoch_metrics[f"{metric_name}"]

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


    @property
    def acc_patient(self):
        vals = self.epoch_metrics["acc_patient"]
        return float(np.mean(vals))

    @property
    def acc_lesion(self):
        vals = self.epoch_metrics["acc_lesion"]
        return float(np.mean(vals))

    @property
    def loss(self):
        vals = self.epoch_metrics["loss"]
        return float(np.mean(vals))
    @property
    def ys_patient(self):
        vals = self.epoch_metrics["ys_patient"]
        return torch.cat(vals)

    @property
    def prob_patient(self):
        vals = self.epoch_metrics["prob_patient"]
        return torch.cat(vals)

    @property
    def ys_lesion(self):
        vals = self.epoch_metrics["ys_lesion"]
        return torch.cat(vals)

    @property
    def prob_lesion(self):
        vals = self.epoch_metrics["prob_lesion"]
        return torch.cat(vals)

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class TrainingMetrics:
    def __init__(self, threshold=0.5, best_model_name='best_model.pth'):
        self.threshold = threshold
        self.best_model_name = best_model_name
        self.reset()

    def reset(self):
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

    def compute_pos_acc(self, prob, ys):
        preds = (prob >= self.threshold).float()
        pos_mask = (ys == 1)
        return (preds[pos_mask] == 1).float().mean().item() if pos_mask.sum() > 0 else 0.0

    def compute_auc(self, prob, ys):
        return roc_auc_score(ys.cpu().numpy(), prob.cpu().numpy())

    def compute_loss_ratio(self, val_loss, train_loss):
        return val_loss / train_loss if train_loss > 0 else float("inf")

    def compute_patient_probability(self, logits_patient):
        return torch.sigmoid(logits_patient)

    # def _save_to_history(self, metric):
    #     self.history[f"hist_{metric}"].append(metric.detach().cpu())

    def update(self, train_metrics, val_metrics, net=None, epoch=None):
        """Update metrics after one epoch and optionally save the best model."""
        self.history["hist_train_loss"].append(train_metrics.loss)
        self.history["hist_val_loss"].append(val_metrics.loss)
        self.history["hist_val_patient_acc"].append(val_metrics.acc_patient)
        self.history["hist_val_lesion_acc"].append(val_metrics.acc_lesion)
        self.hist_train_loss = self.history["hist_train_loss"]
        self.hist_val_loss = self.history["hist_val_loss"]
        self.hist_val_patient_acc = self.history["hist_val_patient_acc"]
        self.hist_val_lesion_acc = self.history["hist_val_lesion_acc"]

        self.last_epoch = epoch
        self.loss_ratio = self.compute_loss_ratio(val_metrics.loss, train_metrics.loss)
        self.pos_patient_acc = self.compute_pos_acc(val_metrics.prob_patient, val_metrics.ys_patient)
        self.pos_lesion_acc = self.compute_pos_acc(val_metrics.prob_lesion, val_metrics.ys_lesion)
        self.patient_auc = self.compute_auc(val_metrics.prob_patient, val_metrics.ys_patient)
        self.lesion_auc = self.compute_auc(val_metrics.prob_lesion, val_metrics.ys_lesion)

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
              f"Val patient_auc={self.patient_auc:.4f}, "
              f"Val Pos Acc={self.pos_patient_acc:.4f}, "
              f"No improvement count={self.no_improvement}")


# class check_model:
#     def __init__(self, model):
#         self.model = model








