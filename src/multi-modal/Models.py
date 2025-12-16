
import torch.nn as nn
import torchvision as tv
import torch
import torch.nn.functional as F
from Utility import HyperParameters
from math import sqrt

# cascade loss method adapted from https://github.com/svishwa/crowdcount-cascaded-mtl/blob/master/src/crowd_count.py
# Attention based method adapted from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/main.py

class PatientClassifier(nn.Module):
    def __init__(self, attention, resnet101meta, meta_dim = 3, dropout=0.2):
        super().__init__()
        self.attention = attention(attention_branches = 1, meta_dim = meta_dim, dropout=dropout)
        self.resnet101 = resnet101meta(dropout=dropout)
        self.meta_dim = meta_dim

    def forward(self, imgs, meta):
        logits_lesion, embeddings = self.resnet101(imgs)
        logits_patient = self.attention(embeddings, meta, logits_lesion)
        return logits_patient, logits_lesion

class Attention(nn.Module, HyperParameters):
    def __init__(self, input_dim = 1000, L=128, M=500, attention_branches=3, meta_dim=3, dropout=0.2):

        super().__init__()
        self.save_hyperparameters()

        if meta_dim > 0:
            self.meta = 18
        else: self.meta = 0

        self.fc = nn.Sequential(
            nn.Linear(input_dim, M),
            nn.Tanh())

        self.attention = nn.Sequential(
            nn.Linear(M + self.meta, L),
            nn.Tanh(),
            nn.Linear(L, attention_branches))

        self.classifier = nn.Linear(attention_branches*(M + self.meta), 1)

        self.process_patient_meta = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.layernorm = nn.LayerNorm(attention_branches*(M + self.meta))

        self.dropout = nn.Dropout(dropout)

    def separate_meta(self, meta):
        # meta: (K, 3) => col0=sex, col1=site, col2=age
        sex_age = meta[:, [0, 2]].mean(dim=0, keepdim=True)  # (1,2)
        K = meta.size(0)
        sex_age = sex_age.expand(K, -1)  # (K,2)
        anatom_site = meta[:, [1]]  # (K,1) one numeric site embedding
        return sex_age, anatom_site

    def get_patient_meta(self, logits_lesion, meta):
        sex_age, anatom_site = self.separate_meta(meta)
        logits_lesion_anatom = torch.cat((logits_lesion, anatom_site), dim=1)  # (K,2)
        patient_meta = self.process_patient_meta(logits_lesion_anatom)  # (K,16)
        return torch.cat((patient_meta, sex_age), dim=1)  # (K,18)

    def forward(self, embeddings, meta, logits_lesion):
        H = self.fc(embeddings)  # [K, M]

        H = self.dropout(H)

        if self.meta_dim > 0:
            patient_meta = self.get_patient_meta(logits_lesion, meta)  # [K, meta_features]
            H = torch.cat((H, patient_meta), dim=1)

        # Attention
        A = self.attention(H/sqrt(H.size(1)))  # [K, attention_branches]
        A = A.transpose(0, 1)  # [attention_branches, K]
        A = F.softmax(A, dim=1) + 1e-6 # softmax over instances


        # Pool embeddings using attention
        Z = torch.mm(A, H)  # [attention_branches, M + meta]
        Z = Z.view(1, -1)  # collapse branches
        Z = self.layernorm(Z)
        # if self.meta_dim > 0:
        #     patient_meta = self.get_patient_meta(logits_lesion, meta)  # [K, meta_features]
        #     Z = torch.cat((Z, patient_meta), dim=1)
        Z = self.dropout(Z)



        logits_patient = self.classifier(Z)  # [1, 1]
        return logits_patient.squeeze(-1)

    # def forward(self, embeddings, meta, logits_lesion):
    #     H = self.fc(embeddings)
    #     print("H after fc", H.shape)
    #     print("embeddings", embeddings.shape)
    #
    #
    #
    #     # Process meta if exists
    #     if self.meta_dim > 0:
    #         patient_meta = self.get_patient_meta(logits_lesion, meta)
    #         print("patient_meta", patient_meta.shape)
    #
    #         # Add processed resnet output to embedding
    #         H = torch.cat((H, patient_meta), dim=1)
    #         print("H after concat", H.shape)
    #
    #
    #     H = self.dropout(H)
    #     A = self.attention(H)  # (K, attention_branches)
    #     print("A raw", A.shape)
    #
    #     A = A.transpose(0, 1)  # (attention_branches, K)
    #
    #     A = F.softmax(A, dim=1)  # softmax over instances
    #     print("A softmax", A.shape)
    #
    #
    #     # Pool embeddings using attention
    #     Z = torch.mm(A, H)  # (attention_branches, M + meta)
    #     print("Z", Z.shape)
    #
    #
    #     # Could additionally or instead concat patient_meta here
    #     # Z = torch.cat((Z, patient_meta), dim=1)
    #
    #     # Final patient-level classification
    #     logits_patient = self.classifier(Z)  # (attention_branches,1)
    #     print("logits_patient raw", self.classifier(Z).shape)
    #     logits_patient = logits_patient.mean(dim=0, keepdim=True)
    #
    #
    #     return logits_patient.squeeze(-1)


class ResNet101Meta(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = tv.models.resnet101(weights=tv.models.ResNet101_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head1 = nn.Sequential(
            nn.Linear(in_feats, 1000),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head2 = nn.Linear(1000, 1)

    def forward(self, x, m=None):
        f = self.backbone(x)
        embeddings = self.head1(f)
        logits = self.head2(embeddings)
        return logits, embeddings


class Efficientnet_b6(nn.Module):
    def __init__(self, meta_dim=0):
        super().__init__()

        self.backbone = tv.models.efficientnet_b6(
            weights=tv.models.EfficientNet_B6_Weights.IMAGENET1K_V1
        )
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Identity()

        if meta_dim > 0:
            self.meta = nn.Sequential(
                nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.head = nn.Linear(in_feats + 16, 1)
            self.head2 = nn.Linear(in_feats + 16, 500)
        else:
            self.meta = None
            self.head = nn.Linear(in_feats, 1)
            self.head2 = nn.Linear(in_feats, 500)

    def forward(self, x, m=None):
        f = self.backbone(x)

        if self.meta is not None and m is not None and m.numel() > 0:
            meta_out = self.meta(m)
            f = torch.cat([f, meta_out], dim=1)
        return self.head(f), self.head2(f)


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.attention_branches = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.attention_branches) # matrix w (or vector w if self.attention_branches==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.attention_branches, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # Kxattention_branches
        A = torch.transpose(A, 1, 0)  # attention_branchesxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # attention_branchesxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A




    # class ResNetExtractFeatures(nn.Module):
    #     def __init__(self, resnet_model, meta_dim=0):
    #         super().__init__()
    #         self.backbone = resnet_model
    #         in_feats = self.backbone.fc.in_features
    #         self.backbone.fc = nn.Identity()
    #
    #         if meta_dim and meta_dim > 0:
    #             self.meta = nn.Sequential(
    #                 nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
    #                 nn.Linear(32, 16), nn.ReLU()
    #             )
    #             meta_feat = 16
    #         else:
    #             self.meta = None
    #             meta_feat = 0
    #
    #         in_size = in_feats + meta_feat
    #         self.head1 = nn.Linear(in_size, 500)   # embeddings per instance (batch, H)
    #         self.head2 = nn.Linear(in_size, 1)   # per-instance logits
    #
    #     def forward(self, x, m=None):
    #         f = self.backbone(x)  # (batch, in_feats)
    #         if self.meta is not None and m is not None and m.numel() > 0:
    #             mf = self.meta(m)            # (batch, meta_feat)
    #             f = torch.cat([f, mf], dim=1)
    #         emb = self.head1(f)             # (batch, H)
    #         logits_lesion = self.head2(f).squeeze(-1)  # (batch,)
    #         return emb, logits_lesion
    #
    # class ResNetExtractFeatures_2(nn.Module):
    #     def __init__(self, resnet_model, meta_dim=0, H: int = 500):
    #         super().__init__()
    #         self.backbone = resnet_model
    #         in_feats = self.backbone.fc.in_features
    #         self.backbone.fc = nn.Identity()
    #
    #         if meta_dim and meta_dim > 0:
    #             self.meta = nn.Sequential(
    #                 nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
    #                 nn.Linear(32, 16), nn.ReLU()
    #             )
    #             meta_feat = 16
    #         else:
    #             self.meta = None
    #             meta_feat = 0
    #
    #         in_size = in_feats + meta_feat
    #         self.head = nn.Linear(in_size, 1)   # per-instance logits
    #
    #     def forward(self, x, m=None):
    #         f = self.backbone(x)  # (batch, in_feats)
    #         if self.meta is not None and m is not None and m.numel() > 0:
    #             mf = self.meta(m)            # (batch, meta_feat)
    #             f = torch.cat([f, mf], dim=1)
    #         inst = self.head(f)             # (batch, H)
    #         return inst, logits_lesion

    # class PatientDetector(nn.Module):
    #     def __init__(self,
    #                  lesion_weights=None,
    #                  patient_weights=None,
    #                  Attention=None,
    #                  meta_dim=0,
    #                  smooth_eps=0.1,
    #                  smooth_negatives_only=True,
    #                  device='mps',
    #                  H=500):
    #         super().__init__()
    #
    #         # device
    #         if device is None:
    #             self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #         else:
    #             self.device = torch.device(device)
    #
    #         # smoothing params
    #         self.smooth_eps = float(smooth_eps) if smooth_eps is not None else 0.0
    #         self.smooth_neg_only = bool(smooth_negatives_only)
    #
    #         # CNN Feature Extractor
    #         cnn_base = tv.models.resnet101(weights=tv.models.ResNet101_Weights.IMAGENET1K_V1)
    #         self.CNN = ResNetExtractFeatures(cnn_base, meta_dim=meta_dim, H=H)
    #
    #         # Attention-based MIL Pooling
    #         if Attention is None:
    #             self.Attention = AttentionPool(M=H)
    #         elif isinstance(Attention, type): # checks if the attention variable is a class
    #             self.Attention = Attention(M=H)
    #         else: # if attention is an instance of a class:
    #             self.Attention = Attention
    #
    #         # pos_weight tensors for loss
    #         lesion_pw = torch.as_tensor(lesion_weights, device=self.device) if lesion_weights is not None else None
    #         patient_pw = torch.as_tensor(patient_weights, device=self.device) if patient_weights is not None else None
    #
    #         # loss functions
    #         self.loss_lesion_fn = nn.BCEWithLogitsLoss(pos_weight=lesion_pw)
    #         self.loss_patient_fn = nn.BCEWithLogitsLoss(pos_weight=patient_pw)
    #
    #         # Convenience Parameters for training
    #         self.last_loss_lesion = None
    #         self.last_loss_patient = None
    #         self.last_total_loss = None
    #
    #     def _smooth_labels(self, labels: torch.Tensor):
    #         if not self.smooth_eps or self.smooth_eps <= 0.0:
    #             return labels.float()
    #         t = labels.float()
    #         eps = self.smooth_eps
    #         if self.smooth_neg_only:
    #             # replace label = 0 with eps, keep label = 1 as 1.0
    #             return torch.where(t == 0.0, torch.full_like(t, eps), t)
    #         else:
    #             # symmetric smoothing: y*(1-eps) + 0.5*eps
    #             return t * (1.0 - eps) + 0.5 * eps
    #
    #     def forward(self, imgs, labels_patient=None,labels_lesion=None, meta=None, lam=1e-4):
    #         embeddings, logits_lesion = self.CNN(imgs, meta)
    #         logits_patient, prob_patient, pred_patient = self.Attention(embeddings)
    #
    #         loss_lesion = loss_patient = total_loss = None
    #
    #         if labels_lesion is not None and labels_patient is not None:
    #             labels_lesion = labels_lesion.to(logits_lesion.device)
    #             labels_patient = labels_patient.to(logits_patient.device)
    #             smoothed_inst_targets = self._smooth_labels(labels_lesion)
    #             loss_lesion = self.loss_lesion_fn(logits_lesion, smoothed_inst_targets)
    #             loss_patient = self.loss_patient_fn(logits_patient, labels_patient.float())
    #             total_loss = loss_patient + lam * loss_lesion
    #
    #             self.last_loss_lesion = loss_lesion
    #             self.last_loss_patient = loss_patient
    #             self.last_total_loss = total_loss
    #
    #         return {
    #             "logits_patient": logits_patient,
    #             "prob_patient": prob_patient,
    #             "pred_patient": pred_patient,
    #             "logits_lesion": logits_lesion,
    #             "loss_lesion": loss_lesion,
    #             "loss_patient": loss_patient,
    #             "total_loss": total_loss
    #         }
    #
    #     def loss(self, lam=1e-4):
    #         if self.last_loss_patient is None or self.last_loss_lesion is None:
    #             raise RuntimeError("run forward in training mode before computing  loss")
    #         return self.last_loss_patient + lam * self.last_loss_lesion
