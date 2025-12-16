
import torch.nn as nn
import torchvision as tv
import torch
import torch.nn.functional as F
from math import sqrt
from torch.nn.utils.rnn import pad_sequence


class PatientCNNClassifier(nn.Module):
    def __init__(self, meta_dim = 0, embed_size = 9, new_embed = 4, num_heads = 1, attention_type = 'self'):
        super().__init__()
        self.resnet = ResNet101Meta_19_20(meta_dim)
        self.meta = self.resnet.meta
        state_dict = torch.load("Best_Resnet101_meta2.pt", map_location='mps')
        self.resnet.load_state_dict(state_dict)
        self.attention = Attention_Pool(embed_size=embed_size, new_embed = new_embed, num_heads = num_heads, attention_type=attention_type)
    def forward(self, imgs, m=None, mask = None):
        emb = self.resnet(imgs, m) # get embeddings
        x = self.attention(emb, mask = mask) # attention pooling
        return x.squeeze(0)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt((d_k))

    # Apply mask if provided (useful for masked self-attention in transformers)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to normalize scores, producing attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Compute the final output as weighted values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

class CustomAttention(nn.Module):
    def __init__(self, embed_size, new_embed, num_heads=1, attention_type="self"):
        super().__init__()
        self.attention_type = attention_type
        # Linear layers for Q, K, V
        self.query = nn.Linear(embed_size, new_embed)
        self.key = nn.Linear(embed_size, new_embed)
        self.value = nn.Linear(embed_size, new_embed)

        # Final linear layer after concatenating heads
        self.fc_out = nn.Linear(new_embed, new_embed)

    def forward(self, target, source=None, mask=None):
        if self.attention_type == "self":
            Q = self.query(target)
            K = self.key(target)
            V = self.value(target)
        elif self.attention_type == "cross":
            assert source is not None, "Source input required for cross-attention"
            Q = self.query(target)
            K = self.key(source)
            V = self.value(source)

        # Perform attention calculation (self or cross)
        out, self.weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.fc_out(out)

class Attention_Pool(nn.Module):
    def __init__(self, embed_size, new_embed, num_heads=1, attention_type="self", pool_branches = 1, L = 128):
        super().__init__()
        self.attention = CustomAttention(embed_size, new_embed, num_heads, attention_type)
        self.layer_norm = nn.LayerNorm(new_embed)
        self.dropout = nn.Dropout(0.1)
        self.attention_pool = nn.Sequential(
            nn.Linear(new_embed, L),  # matrix V
            nn.Tanh(),
            nn.Linear(L, pool_branches)  # w
        )

    def forward(self, target, source=None, mask=None):
        attention_out = self.attention(target, source, mask)
        # Add residual connection and layer normalization
        out = self.layer_norm(self.dropout(attention_out)) # (num_samples, embed_size)
        A = self.attention_pool(out)  # (num_samples, 1)
        A = torch.transpose(A, 1, 0)  # (1, num_samples)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, out)  # (1, embed_size)
        return Z

class ResNet101Meta_19_20(nn.Module):
    def __init__(self, meta_dim=0):
        super().__init__()
        self.backbone = tv.models.resnet101(weights=tv.models.ResNet101_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if meta_dim > 0:
            self.meta = nn.Sequential(
                nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.head = nn.Linear(in_feats + 16, 9)
        else:
            self.meta = None
            self.head = nn.Linear(in_feats, 9)
    def forward(self, x, m=None):
        f = self.backbone(x)
        if self.meta is not None and m is not None and m.numel() > 0:
            f = torch.cat([f, self.meta(m)], dim=1)
        return self.head(f)

class ResNet101Meta(nn.Module):
    def __init__(self, meta_dim=0):
        super().__init__()
        self.backbone = tv.models.resnet101(weights=tv.models.ResNet101_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if meta_dim > 0:
            self.meta = nn.Sequential(
                nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.head = nn.Linear(in_feats + 16, 1)
        else:
            self.meta = None
            self.head = nn.Linear(in_feats, 1)
    def forward(self, x, m=None):
        f = self.backbone(x)
        if self.meta is not None and m is not None and m.numel() > 0:
            f = torch.cat([f, self.meta(m)], dim=1)
        return self.head(f)


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
        else:
            self.meta = None
            self.head1 = nn.Linear(in_feats, 1)


    def forward(self, x, m=None):
        f = self.backbone(x)

        if self.meta is not None and m is not None and m.numel() > 0:
            meta_out = self.meta(m)
            f = torch.cat([f, meta_out], dim=1)
        return self.head(f).squeeze(1)


class ResNet18Meta(nn.Module):
    def __init__(self, meta_dim=0):
        super().__init__()
        self.backbone = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
        self.meta_dim = meta_dim
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if self.meta_dim > 0:
            self.meta = nn.Sequential(
                nn.Linear(meta_dim, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.ReLU()
            )
            self.head = nn.Linear(in_feats + 16, 1)
        else:
            self.meta = None
            self.head = nn.Linear(in_feats, 1)
    def forward(self, x, m=None):
        f = self.backbone(x)
        if self.meta is not None and m is not None and m.numel() > 0:
            f = torch.cat([f, self.meta(m)], dim=1)
        return self.head(f).squeeze(1)

 # concatenate batch of patients into "mega-bag"
        # lengths = [int(l) for l in lengths]
        # all_x = torch.cat(imgs, dim=0)
        # emb = self.resnet(all_x, m)
        # split embeddings into batches
        # embs = torch.split(emb, lengths, dim=0) # list of bag of embeddings (1 variable bag per patient)
        # # Pad sequences along the "num_les" dimension
        # padded_embs = pad_sequence(embs, batch_first=True)
        # # mask paddings
        # mask = torch.zeros(len(embs), padded_embs.shape[1], dtype=torch.bool)
        # mask = mask.to(emb.device)
        # for i, length in enumerate(lengths):
        #     mask[i, :length] = 1