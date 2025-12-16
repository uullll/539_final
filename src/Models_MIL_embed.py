
import torch.nn as nn
import torchvision as tv
import torch
import torch.nn.functional as F
from math import sqrt
from torch.nn.utils.rnn import pad_sequence

class PatientCNNClassifier(nn.Module):
    def __init__(self, meta_dim = 0, embed_size = 500, new_embed = 4, num_heads = 1, attention_type = 'self', pool_branches = 1):
        super().__init__()
        self.squeeze = False
        self.resnet = ResNet50_embed(meta_dim)
        self.meta = self.resnet.meta
        self.attention = Attention_Pool(embed_size=embed_size, new_embed = new_embed, num_heads = num_heads, attention_type=attention_type, pool_branches = pool_branches)
    def forward(self, imgs, m=None, mask = None):
        emb = self.resnet(imgs, m) # get embeddings
        x = self.attention(emb, mask = mask) # attention pooling
        if self.training or self.squeeze:
            return x.squeeze(0)
        else:
            return x


class ResNet50_embed(nn.Module):
    def __init__(self, meta_dim=0):
        super().__init__()
        self.backbone = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
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
            self.head = nn.Linear(in_feats, 500)
    def forward(self, x, m=None):
        f = self.backbone(x)
        if self.meta is not None and m is not None and m.numel() > 0:
            f = torch.cat([f, self.meta(m)], dim=1)
        return self.head(f)

class Attention_Pool(nn.Module):
    def __init__(self, embed_size, new_embed, num_heads, attention_type, pool_branches, L = 128):
        super().__init__()
        attention_type = attention_type.lower()
        self.attention = MultiHeadAttention(embed_size, new_embed, num_heads)
        self.layer_norm = nn.LayerNorm(new_embed)
        self.dropout = nn.Dropout(0.1)
        if attention_type is None:
            self.attention_pool = nn.Sequential(
                nn.Linear(new_embed, L),  # matrix V
                nn.Tanh(),
                nn.Linear(L, pool_branches)  # w
            )
        elif attention_type == 'gated':
            self.attention_pool = GatedAttention(new_embed, L, pool_branches)

    def forward(self, target, source=None, mask=None):
        attention_out = self.attention(target, mask)
        # Add residual connection and layer normalization
        out = self.layer_norm(self.dropout(attention_out)) # (num_samples, new_embed)
        A = self.attention_pool(out)  # (num_samples, 1)
        A = torch.transpose(A, 1, 0)  # (1, num_samples)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, out)  # (1, new_embed)
        return Z

class GatedAttention(nn.Module):
    def __init__(self, M, L, pool_branches):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(M, L),  # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(M, L),  # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(L, pool_branches)
    def forward(self, H):
        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U)
        return A

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, new_embed, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.new_embed = new_embed

        self.num_heads = num_heads
        self.head_dim = new_embed // num_heads
        assert new_embed % num_heads == 0

        # Linear layers for Q, K, V for all heads
        self.query = nn.Linear(embed_size, new_embed)
        self.key = nn.Linear(embed_size, new_embed)
        self.value = nn.Linear(embed_size, new_embed)

        # Output linear layer
        self.fc_out = nn.Sequential(nn.Linear(new_embed, new_embed), nn.Tanh())

    def forward(self, x, mask=None):
        num_imgs, embed_size = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view( num_imgs, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view( num_imgs, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view( num_imgs, self.num_heads, self.head_dim).transpose(0, 1)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(0, 1).contiguous().view( num_imgs, self.new_embed)

        return self.fc_out(out)

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








