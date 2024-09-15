import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter

class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True):
        super(HyperbolicLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights on the manifold
        self.weight = ManifoldParameter(
            torch.Tensor(out_features, in_features), manifold=manifold
        )
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = ManifoldParameter(
                torch.zeros(out_features), manifold=manifold
            )
        else:
            self.bias = None

    def forward(self, input):
        # Combine 2D and 3D cases using flexible tensor shapes
        original_shape = input.shape
        in_features = input.shape[-1]

        if in_features != self.in_features:
            raise ValueError(
                f"Incompatible shapes: input shape {input.size()} and weight shape {self.weight.size()}"
            )

        input_flat = input.reshape(-1, in_features)
        output_flat = self.manifold.mobius_matvec(self.weight, input_flat)
        output = output_flat.view(*original_shape[:-1], self.out_features)

        if self.bias is not None:
            # Compute bias_unsqueezed during forward pass to ensure correct device
            bias_unsqueezed = self.bias.view(
                *([1] * (output.dim() - 1)),
                self.out_features
            )
            output = self.manifold.mobius_add(output, bias_unsqueezed)

        return output



class HyperbolicLearnedPositionEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim, manifold):
        super(HyperbolicLearnedPositionEncoding, self).__init__()
        self.manifold = manifold
        self.position_embeddings = ManifoldParameter(
            torch.zeros(1, num_patches, embedding_dim), manifold=manifold
        )
        nn.init.xavier_uniform_(self.position_embeddings)
        self.curvature = nn.Parameter(torch.tensor(1.0))  # Learnable curvature

    def forward(self, x):
        # Fuse multiplication and addition in hyperbolic space
        scaled_embeddings = self.position_embeddings.mul(self.curvature)
        return self.manifold.mobius_add(x, scaled_embeddings)


class HyperbolicMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, manifold):
        super(HyperbolicMultiheadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.head_dim = embedding_dim // num_heads

        self.query_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.key_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.value_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.out_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        q = self.query_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose for multi-head attention
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]  # Shape: (batch_size, num_heads, seq_length, head_dim)

        # Compute hyperbolic distances efficiently
        q_norm = q.norm(dim=-1, keepdim=True).pow(2)
        k_norm = k.norm(dim=-1, keepdim=True).pow(2)
        denom = (1 - q_norm).unsqueeze(-2) * (1 - k_norm).unsqueeze(-3) + 1e-15

        diff = q.unsqueeze(-2) - k.unsqueeze(-3)
        diff_norm_sq = diff.pow(2).sum(dim=-1)

        arccosh_arg = 1 + 2 * diff_norm_sq / denom.squeeze(-1)
        arccosh_arg = arccosh_arg.clamp(min=1 + 1e-7)  # Stability

        dist = torch.log(arccosh_arg + torch.sqrt(arccosh_arg.pow(2) - 1))
        attn_scores = -dist.pow(2) / (self.head_scaling * (self.head_dim ** 0.5))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)

        return output


class HyperbolicTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, manifold):
        super(HyperbolicTransformerLayer, self).__init__()
        self.manifold = manifold

        self.self_attn = HyperbolicMultiheadAttention(embedding_dim, num_heads, manifold)
        self.linear1 = HyperbolicLinear(embedding_dim, embedding_dim * 4, manifold)
        self.linear2 = HyperbolicLinear(embedding_dim * 4, embedding_dim, manifold)
        self.dropout = nn.Dropout(dropout)
        self.layer_scaling = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Self-attention
        attn_output = self.self_attn(x)
        x = self.manifold.mobius_add(x, self.dropout(attn_output).mul(self.layer_scaling))

        # Feedforward network with Mobius ReLU
        x2 = self.linear1(x)
        x2 = self.mobius_relu(x2)
        x2 = self.linear2(x2)
        x = self.manifold.mobius_add(x, self.dropout(x2).mul(self.layer_scaling))

        return x

    def mobius_relu(self, x):
        x_euclidean = self.manifold.logmap0(x)
        x_euclidean_relu = F.relu_(x_euclidean)  # In-place ReLU
        return self.manifold.expmap0(x_euclidean_relu)


class Net(nn.Module):
    # def __init__(
    #     self,
    #     img_size=224,
    #     patch_size=16,
    #     in_channels=3,
    #     num_classes=1000,
    #     embedding_dim=768,
    #     num_heads=12,
    #     num_layers=12,
    #     dropout=0.1,
    #     manifold=None,
    # ):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 num_classes=10,
                 embedding_dim=128,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.1,
                 manifold=None):
        super(Net, self).__init__()
        if manifold is None:
            manifold = geoopt.PoincareBall(c=1.0)
        self.manifold = manifold

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embeddings = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.hyperbolic_embedding = HyperbolicLearnedPositionEncoding(
            self.num_patches, embedding_dim, manifold
        )

        self.layers = nn.ModuleList([
            HyperbolicTransformerLayer(embedding_dim, num_heads, dropout, manifold)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)  # Shape: (batch_size, embedding_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)

        x = self.manifold.expmap0(x)
        x = self.hyperbolic_embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.manifold.logmap0(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

    def clip_gradients(self, clip_value=1.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)

    def geodesic_regularization(self, x, labels):
        dist_matrix = self.manifold.dist(x.unsqueeze(1), x.unsqueeze(0))
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        return ((1 - label_matrix) * dist_matrix).mean()
