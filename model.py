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
        weight = torch.Tensor(out_features, in_features)
        nn.init.xavier_uniform_(weight)
        self.weight = ManifoldParameter(weight, manifold=manifold)
        if bias:
            bias = torch.zeros(out_features)
            self.bias = ManifoldParameter(bias, manifold=manifold)
        else:
            self.bias = None

    def forward(self, input):
        output = self.manifold.mobius_matvec(self.weight, input)

        if self.bias is not None:
            output = self.manifold.mobius_add(output, self.bias)
        return output


class HyperbolicMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, manifold):
        super(HyperbolicMultiheadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.head_dim = embedding_dim // num_heads

        assert (
            embedding_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.query_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.key_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)
        self.value_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)

        self.out_proj = HyperbolicLinear(embedding_dim, embedding_dim, manifold)

        # Initialize head_scaling as a learnable parameter
        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        # Project inputs to queries, keys, and values
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the attention scores using hyperbolic distance
        attn_scores = self._hyperbolic_attention_scores(q, k) / (self.head_scaling * (self.head_dim ** 0.5))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)

        return output

    def _hyperbolic_attention_scores(self, q, k):
        # Efficiently compute pairwise hyperbolic distances for attention scores
        # q, k: (batch_size, num_heads, seq_length, head_dim)
        batch_size, num_heads, seq_length, head_dim = q.size()

        # Normalize q and k to have norms less than 1 (ensure they are in Poincare ball)
        q = q.clamp_max(1 - 1e-5)
        k = k.clamp_max(1 - 1e-5)

        # Compute pairwise distance matrix using Poincare distance formula
        # Efficient computation using batch operations
        q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True)

        # Compute cross term q · k^T (batch matrix multiplication)
        qk_inner = torch.matmul(q, k.transpose(-2, -1))  # Shape: (B, H, S, S)

        # Compute the distances using the hyperbolic law of cosines
        numerator = 2 * (q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * qk_inner)
        denominator = (1 - q_norm_sq) * (1 - k_norm_sq).transpose(-2, -1) + 1e-15

        dist_sq = torch.acosh(1 + numerator / denominator).clamp(min=1e-7)
        attn_scores = -dist_sq.squeeze(-1)  # Shape: (B, H, S, S)
        return attn_scores


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
        x = self.manifold.mobius_add(x, self.dropout(attn_output) * self.layer_scaling)

        # Feedforward network with Möbius ReLU
        x2 = self.linear1(x)
        x2 = self.mobius_relu(x2)
        x2 = self.linear2(x2)
        x = self.manifold.mobius_add(x, self.dropout(x2) * self.layer_scaling)

        return x

    def mobius_relu(self, x):
        x_euclidean = self.manifold.logmap0(x)
        # Use in-place ReLU for efficiency
        x_euclidean.relu_()
        return self.manifold.expmap0(x_euclidean)


class HyperbolicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        c=1.0,
    ):
        super(HyperbolicConv2d, self).__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weight on the manifold
        weight = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        nn.init.xavier_uniform_(weight)
        self.weight = ManifoldParameter(weight, manifold=self.manifold)
        if bias:
            bias = torch.zeros(out_channels)
            self.bias = ManifoldParameter(bias, manifold=self.manifold)
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        # Transform the input and weights to Euclidean space
        input_euclidean = self.manifold.logmap0(input)
        weight_euclidean = self.manifold.logmap0(self.weight)

        # Perform Euclidean convolution
        output_euclidean = F.conv2d(
            input_euclidean,
            weight_euclidean,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            output_euclidean += self.bias.view(1, -1, 1, 1)

        # Map the output back to hyperbolic space
        output_hyperbolic = self.manifold.expmap0(output_euclidean)
        return output_hyperbolic


class HyperbolicLearnedPositionEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim, manifold):
        super(HyperbolicLearnedPositionEncoding, self).__init__()
        self.manifold = manifold
        position_embeddings = torch.zeros(1, num_patches, embedding_dim)
        nn.init.xavier_uniform_(position_embeddings)
        self.position_embeddings = ManifoldParameter(position_embeddings, manifold=manifold)
        self.curvature = nn.Parameter(torch.tensor(1.0))  # Learnable curvature

    def forward(self, x):
        # Efficient element-wise multiplication with broadcasting
        scaled_embeddings = self.position_embeddings * self.curvature
        return self.manifold.mobius_add(x, scaled_embeddings)


class Net(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embedding_dim=128,
        num_heads=8,
        num_layers=8,
        dropout=0.1,
        manifold=None,
        # img_size=32,
        # patch_size=4,
        # in_channels=3,
        # num_classes=10,
        # embedding_dim=128,
        # num_heads=8,
        # num_layers=8,
        # dropout=0.1,
        # manifold=None
    ):
        super(Net, self).__init__()
        if manifold is None:
            manifold = geoopt.PoincareBall(c=1.0)
        self.manifold = manifold

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embeddings = HyperbolicConv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            c=1.0,
        )

        self.hyperbolic_embedding = HyperbolicLearnedPositionEncoding(
            self.num_patches, embedding_dim, manifold
        )

        self.layers = nn.ModuleList(
            [
                HyperbolicTransformerLayer(embedding_dim, num_heads, dropout, manifold)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)  # Shape: (batch_size, embedding_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)

        x = self.hyperbolic_embedding(x)

        for layer in self.layers:
            x = layer(x)

        # Map back to Euclidean space for classification
        x = self.manifold.logmap0(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

    def geodesic_regularization(self, outputs, labels, lambda_reg=0.01):
        # Compute pairwise geodesic distances efficiently
        outputs = outputs / outputs.norm(dim=1, keepdim=True).clamp_min(1e-7)
        dist_matrix = self.manifold.dist(outputs.unsqueeze(1), outputs.unsqueeze(0))  # Shape: (B, B)
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # Shape: (B, B)
        reg_loss = ((1 - label_matrix) * dist_matrix).mean()
        return lambda_reg * reg_loss
