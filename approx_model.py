import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Define the curvature c globally
curvature = 1.0

def mobius_add_approx(x, y, c=curvature):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
    y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)

    denominator = 1 + 2 * c * xy_dot
    numerator = (1 + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y

    denominator = denominator.clamp_min(1e-7)
    result = numerator / denominator
    return result

def mobius_matvec_approx(m, x, c=curvature):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    mx = torch.matmul(x, m.transpose(-1, -2))  # Euclidean matmul

    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - c * x_sqnorm).clamp_min(1e-7)  # Approximate conformal factor
    mx_scaled = mx * lambda_x
    result = mobius_add_approx(torch.zeros_like(mx), mx_scaled, c)
    return result

def expmap0_approx(v, c=curvature):
    c = torch.as_tensor(c, dtype=v.dtype, device=v.device)
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp_min(1e-7)
    coef = 1 - (c * v_norm_sq) / 3
    return v * coef

def logmap0_approx(x, c=curvature):
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp_min(1e-7)
    coef = 1 + (c * x_norm_sq) / 3
    return x * coef

def mobius_relu_approx(x):
    x_euclidean = logmap0_approx(x)
    x_euclidean_relu = F.relu(x_euclidean)
    return expmap0_approx(x_euclidean_relu)

class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(HyperbolicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, input):
        output = mobius_matvec_approx(self.weight, input)
        if self.bias is not None:
            bias_expmap = expmap0_approx(self.bias).unsqueeze(0)
            output = mobius_add_approx(output, bias_expmap)
        return output

class HyperbolicMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(HyperbolicMultiheadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert (
            embedding_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.query_proj = HyperbolicLinear(embedding_dim, embedding_dim)
        self.key_proj = HyperbolicLinear(embedding_dim, embedding_dim)
        self.value_proj = HyperbolicLinear(embedding_dim, embedding_dim)

        self.out_proj = HyperbolicLinear(embedding_dim, embedding_dim)

        # Initialize head scaling as a learnable parameter
        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Project inputs to queries, keys, and values
        q = self.query_proj(x)  # (B, S, D)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, S, D_head)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute approximate hyperbolic attention scores
        attn_scores = self.hyperbolic_attention_scores_approx(q, k) / (self.head_dim ** 0.5)
        attn_scores = attn_scores / self.head_scaling

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # (B, H, S, D_head)

        # Reshape and project the output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)
        output = self.out_proj(attn_output)

        return output

    def hyperbolic_attention_scores_approx(self, q, k):
        c = torch.as_tensor(curvature, dtype=q.dtype, device=q.device)

        # Compute norms
        q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True)  # (B, H, S, 1)
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True)  # (B, H, S, 1)

        # Compute inner products
        inner_prod = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)

        # Compute delta
        delta = c * (q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * inner_prod)
        delta = delta.clamp_min(1e-7)

        # Approximate hyperbolic distance using sqrt(delta)
        dist = torch.sqrt(delta)

        attn_scores = -dist.squeeze(-1)  # Shape: (B, H, S, S)
        return attn_scores

class HyperbolicTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(HyperbolicTransformerLayer, self).__init__()

        self.self_attn = HyperbolicMultiheadAttention(embedding_dim, num_heads)
        self.linear1 = HyperbolicLinear(embedding_dim, embedding_dim * 4)
        self.linear2 = HyperbolicLinear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_scaling = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Use gradient checkpointing to save memory
        x = checkpoint(self._forward_impl, x, use_reentrant=False)
        return x

    def _forward_impl(self, x):
        # Self-attention
        attn_output = self.self_attn(x)
        x = mobius_add_approx(x, self.dropout(attn_output) * self.layer_scaling)

        # Feedforward network with Möbius ReLU
        x2 = self.linear1(x)
        x2 = mobius_relu_approx(x2)
        x2 = self.linear2(x2)
        x = mobius_add_approx(x, self.dropout(x2) * self.layer_scaling)

        return x

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
    ):
        super(HyperbolicConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        input_expmap = expmap0_approx(input)
        output = F.conv2d(
            input_expmap,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        output_logmap = logmap0_approx(output)
        return output_logmap

class HyperbolicLearnedPositionEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(HyperbolicLearnedPositionEncoding, self).__init__()
        position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embedding_dim))
        nn.init.xavier_uniform_(position_embeddings)
        self.position_embeddings = position_embeddings

    def forward(self, x):
        pos_embedding_expmap = expmap0_approx(self.position_embeddings)
        return mobius_add_approx(x, pos_embedding_expmap)

class Net(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
    ):
        super(Net, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embeddings = HyperbolicConv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.hyperbolic_embedding = HyperbolicLearnedPositionEncoding(
            self.num_patches, embedding_dim
        )

        # Updated transformer layers without 'manifold' parameter
        self.layers = nn.ModuleList(
            [
                HyperbolicTransformerLayer(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.hyperbolic_embedding(x)

        for layer in self.layers:
            # Apply gradient checkpointing to each layer
            x = checkpoint(layer, x, use_reentrant=False)

        x = logmap0_approx(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

    def geodesic_regularization(self, outputs, labels, lambda_reg=0.01):
        outputs = outputs / outputs.norm(dim=1, keepdim=True).clamp_min(1e-7)
        dist_matrix = torch.cdist(outputs, outputs, p=2)
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        reg_loss = ((1 - label_matrix) * dist_matrix).mean()
        return lambda_reg * reg_loss
