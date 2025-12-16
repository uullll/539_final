from torch import nn
import torchvision as tv

class ResNet101Meta_19_20(nn.Module):
    def __init__(self, meta_dim=0, out_dim = 9):
        super().__init__()
        self.backbone = tv.models.resnet101(weights=tv.models.ResNet101_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if meta_dim > 0:
            self.meta = nn.Sequential(
                nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.head = nn.Linear(in_feats + 16, out_dim)
        else:
            self.meta = None
            self.head = nn.Linear(in_feats, out_dim)
    def forward(self, x, m=None):
        f = self.backbone(x)
        if self.meta is not None and m is not None and m.numel() > 0:
            f = torch.cat([f, self.meta(m)], dim=1)
        return self.head(f)
