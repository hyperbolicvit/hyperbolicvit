import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter
import math

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


class HyperbolicLayerNorm(nn.Module):
    """
    Hyperbolic Layer Normalization as defined in Equation (18):
    LayerNorm^D(x) = exp_0(LayerNorm(log_0(x)))
    """
    def __init__(self, normalized_shape, manifold):
        super(HyperbolicLayerNorm, self).__init__()
        self.manifold = manifold
        self.normalized_shape = normalized_shape
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # Map to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        # Apply Euclidean LayerNorm
        x_norm = self.layer_norm(x_tangent)
        # Map back to manifold
        x_hyperbolic = self.manifold.expmap0(x_norm)
        return x_hyperbolic


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

        # Initialize head_scaling as a learnable parameter (alpha_h in methods)
        self.head_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Small constants for numerical stability
        self.epsilon = 1e-15  # For denominator
        self.delta = 1e-7     # For clamping acosh argument

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
        attn_scores = self._hyperbolic_attention_scores(q, k) / (self.head_scaling * math.sqrt(self.head_dim))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output using the updated _mobius_matmul method
        attn_output = self._mobius_matmul(attn_weights, v)  # Using Möbius operations

        # Reshape and project the output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)

        return output

    # Include the updated _hyperbolic_attention_scores method here from the previous fix
    def _hyperbolic_attention_scores(self, q, k):
        # Compute hyperbolic distances as per Equations (23)-(26)
        c = self.manifold.c
        c = c.type_as(q.data)
        batch_size, num_heads, seq_length, head_dim = q.size()

        # Ensure q and k have norms less than 1/sqrt(c)
        max_norm = (1 - 1e-5) / c.sqrt()
        q = q.clamp_max(max_norm)
        k = k.clamp_max(max_norm)

        # Compute norms squared and reshape for broadcasting
        q_norm_sq = c * torch.sum(q * q, dim=-1, keepdim=True)  # Shape: (B, H, S, 1)
        k_norm_sq = c * torch.sum(k * k, dim=-1, keepdim=True)  # Shape: (B, H, S, 1)

        # Expand dimensions for pairwise operations
        q_norm_sq_expanded = q_norm_sq.expand(-1, -1, -1, seq_length)  # Shape: (B, H, S, S)
        k_norm_sq_expanded = k_norm_sq.transpose(2, 3).expand(-1, -1, seq_length, -1)  # Shape: (B, H, S, S)

        # Compute Möbius subtraction q ⊕ (-k)
        # Expand q and k for pairwise operations
        q_expanded = q.unsqueeze(3)  # Shape: (B, H, S, 1, D)
        k_expanded = k.unsqueeze(2)  # Shape: (B, H, 1, S, D)
        neg_k_expanded = -k_expanded  # Shape: (B, H, 1, S, D)

        # Compute pairwise Möbius addition
        diff = self.manifold.mobius_add(neg_k_expanded, q_expanded)  # Shape: (B, H, S, S, D)

        # Compute norm squared of the difference
        diff_norm_sq = c * torch.sum(diff * diff, dim=-1)  # Shape: (B, H, S, S)

        # Compute denominator (1 - c * ||q||^2)(1 - c * ||k||^2) + epsilon
        denominator = (1 - q_norm_sq_expanded) * (1 - k_norm_sq_expanded) + self.epsilon  # Shape: (B, H, S, S)

        # Compute the argument of acosh, ensure it's >= 1 + delta
        acosh_arg = 1 + (2 * diff_norm_sq) / denominator  # Shape: (B, H, S, S)
        acosh_arg = acosh_arg.clamp_min(1 + self.delta)

        # Compute hyperbolic distances
        dist = torch.log(acosh_arg + torch.sqrt(acosh_arg * acosh_arg - 1))  # Shape: (B, H, S, S)

        # Negative distances as attention scores
        attn_scores = -dist  # Shape: (B, H, S, S)
        return attn_scores
    
    
    def _mobius_matmul(self, attn_weights, v):
        """
        Perform Möbius matrix multiplication (weighted sum) for attention.

        Args:
            attn_weights (torch.Tensor): Attention weights of shape (B, H, S_q, S_k).
            v (torch.Tensor): Value vectors of shape (B, H, S_k, D).

        Returns:
            torch.Tensor: The attention output of shape (B, H, S_q, D).
        """
        # Map v to the tangent space at the origin
        v_tangent = self.manifold.logmap0(v)  # Shape: (B, H, S_k, D)

        # Perform the weighted sum in the tangent space
        attn_output_tangent = torch.matmul(attn_weights, v_tangent)  # Shape: (B, H, S_q, D)

        # Map the result back to the manifold
        attn_output = self.manifold.expmap0(attn_output_tangent)  # Shape: (B, H, S_q, D)

        return attn_output


class HyperbolicTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, manifold):
        super(HyperbolicTransformerLayer, self).__init__()
        self.manifold = manifold

        self.self_attn = HyperbolicMultiheadAttention(embedding_dim, num_heads, manifold)
        self.linear1 = HyperbolicLinear(embedding_dim, embedding_dim * 4, manifold)
        self.linear2 = HyperbolicLinear(embedding_dim * 4, embedding_dim, manifold)
        self.dropout = nn.Dropout(dropout)
        self.layer_scaling = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))  # For residual scaling

        # Hyperbolic Layer Norm layers
        self.hyperbolic_layer_norm1 = HyperbolicLayerNorm(embedding_dim, manifold)
        self.hyperbolic_layer_norm2 = HyperbolicLayerNorm(embedding_dim, manifold)

    def forward(self, x):
        # Self-attention
        attn_output = self.self_attn(x)
        residual = self.manifold.mobius_scalar_mul(self.beta, attn_output)
        x = self.manifold.mobius_add(x, self.dropout(residual))
        # Hyperbolic Layer Norm
        x = self.hyperbolic_layer_norm1(x)

        # Feedforward network with Möbius ReLU
        x2 = self.linear1(x)
        x2 = self.mobius_relu(x2)
        x2 = self.linear2(x2)
        residual = self.manifold.mobius_scalar_mul(self.beta, x2)
        x = self.manifold.mobius_add(x, self.dropout(residual))
        # Hyperbolic Layer Norm
        x = self.hyperbolic_layer_norm2(x)

        return x

    def mobius_relu(self, x):
        x_euclidean = self.manifold.logmap0(x)
        # Apply ReLU in Euclidean space
        x_euclidean = F.relu(x_euclidean)
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
            bias_euclidean = self.manifold.logmap0(self.bias).view(1, -1, 1, 1)
            output_euclidean += bias_euclidean

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
        self.curvature = nn.Parameter(torch.tensor(1.0))  # Learnable curvature (c in the methods)

    def forward(self, x):
        # Scaled positional embeddings: E_pos = c ⊗ E_pos^0
        scaled_embeddings = self.manifold.mobius_scalar_mul(self.curvature, self.position_embeddings)
        return self.manifold.mobius_add(x, scaled_embeddings)


class Net(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embedding_dim=512,
        num_heads=8,
        num_layers=8,
        dropout=0.1,
        manifold=None,
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
            c=manifold.c.item(),
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
        # Output layer: mapping from hyperbolic to Euclidean space
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
        # Compute geodesic distances as per Equation (35)
        batch_size = outputs.size(0)
        outputs = outputs / outputs.norm(dim=1, keepdim=True).clamp_min(1e-7)
        dist_matrix = self.manifold.dist(outputs.unsqueeze(1), outputs.unsqueeze(0))  # Shape: (B, B)
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # Shape: (B, B)
        reg_loss = ((1 - label_matrix) * dist_matrix).mean()
        return lambda_reg * reg_loss
