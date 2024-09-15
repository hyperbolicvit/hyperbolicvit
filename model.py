import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import ManifoldParameter
import math

# Hyperbolic Layer Normalization
class HyperbolicLayerNorm(nn.Module):
    def __init__(self, embedding_dim, manifold, eps=1e-5):
        super(HyperbolicLayerNorm, self).__init__()
        self.manifold = manifold
        self.eps = eps
        self.normalized_shape = (embedding_dim,)  # Normalize over embedding_dim only
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        # Map to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        # Apply LayerNorm over the last dimension (embedding_dim)
        x_norm = F.layer_norm(
            x_tangent, 
            self.normalized_shape, 
            self.gamma, 
            self.beta, 
            self.eps
        )
        # Map back to manifold
        return self.manifold.expmap0(x_norm)

# Parametric ReLU in Hyperbolic Space with Shared Alpha
class MobiusPReLU(nn.Module):
    def __init__(self, manifold):
        super(MobiusPReLU, self).__init__()
        self.manifold = manifold
        # Initialize alpha as a single parameter shared across all channels
        self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Map to tangent space at origin
        x_euclidean = self.manifold.logmap0(x)
        # Apply PReLU with shared alpha
        x_relu = F.prelu(x_euclidean, self.alpha)
        # Map back to manifold
        return self.manifold.expmap0(x_relu)

# Hyperbolic Linear Layer with He Initialization
class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True):
        super(HyperbolicLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        self.weight = ManifoldParameter(
            torch.Tensor(out_features, in_features), manifold=manifold
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He Initialization
        if bias:
            self.bias = ManifoldParameter(
                torch.zeros(out_features), manifold=manifold
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, input):
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
            bias_unsqueezed = self.bias.view(
                *([1] * (output.dim() - 1)),
                self.out_features
            )
            output = self.manifold.mobius_add(output, bias_unsqueezed)

        return output

# Hyperbolic Learned Position Encoding
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
        scaled_embeddings = self.position_embeddings.mul(self.curvature)
        return self.manifold.mobius_add(x, scaled_embeddings)

# Hyperbolic Multihead Attention with DropConnect
class HyperbolicMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, manifold, dropconnect_prob=0.1):
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
        self.curvature = manifold.c if hasattr(manifold, 'c') else 1.0
        self.dropconnect_prob = dropconnect_prob

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        q = self.query_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]  # (batch_size, num_heads, seq_length, head_dim)

        q_norm = q.norm(dim=-1, keepdim=True).pow(2)
        k_norm = k.norm(dim=-1, keepdim=True).pow(2)
        denom = (1 - self.curvature * q_norm).unsqueeze(-2) * (1 - self.curvature * k_norm).unsqueeze(-3) + 1e-15

        diff = q.unsqueeze(-2) - k.unsqueeze(-3)
        diff_norm_sq = diff.pow(2).sum(dim=-1)

        arccosh_arg = 1 + 2 * self.curvature * diff_norm_sq / denom.squeeze(-1)
        arccosh_arg = arccosh_arg.clamp(min=1 + 1e-7)  # Stability

        dist = torch.log(arccosh_arg + torch.sqrt(arccosh_arg.pow(2) - 1))
        attn_scores = -dist.pow(2) / (self.head_scaling * (self.head_dim ** 0.5))

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.training and self.dropconnect_prob > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropconnect_prob, training=True)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)

        return output

# Hyperbolic Transformer Layer with Enhancements
class HyperbolicTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, manifold):
        super(HyperbolicTransformerLayer, self).__init__()
        self.manifold = manifold

        self.self_attn = HyperbolicMultiheadAttention(embedding_dim, num_heads, manifold)
        self.norm1 = HyperbolicLayerNorm(embedding_dim, manifold)  # Corrected LayerNorm
        self.norm2 = HyperbolicLayerNorm(embedding_dim, manifold)  # Corrected LayerNorm
        
        self.linear1 = HyperbolicLinear(embedding_dim, embedding_dim * 4, manifold)
        self.activation = MobiusPReLU(manifold)  # Updated to use shared alpha
        self.linear2 = HyperbolicLinear(embedding_dim * 4, embedding_dim, manifold)
        self.dropout = nn.Dropout(dropout)
        self.layer_scaling = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Self-attention with normalization
        attn_output = self.self_attn(x)
        x = self.norm1(self.manifold.mobius_add(x, self.dropout(attn_output).mul(self.layer_scaling)))
    
        # Feedforward network with Mobius PReLU and normalization
        x2 = self.linear1(x)
        x2 = self.activation(x2)  # Activation now correctly handles shape
        x2 = self.linear2(x2)
        x = self.norm2(self.manifold.mobius_add(x, self.dropout(x2).mul(self.layer_scaling)))
    
        return x

# Final Enhanced Network
class Net(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 num_classes=10,
                 embedding_dim=256,  # Increased from 128
                 num_heads=8,         # Increased from 4
                 num_layers=6,        # Increased from 4
                 dropout=0.1,
                 manifold=None):
        super(Net, self).__init__()
        if manifold is None:
            self.manifold = geoopt.PoincareBall(c=1.0)  # Use fixed curvature
        else:
            self.manifold = manifold

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embeddings = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.hyperbolic_embedding = HyperbolicLearnedPositionEncoding(
            self.num_patches, embedding_dim, self.manifold
        )

        self.layers = nn.ModuleList([
            HyperbolicTransformerLayer(embedding_dim, num_heads, dropout, self.manifold)
            for _ in range(num_layers)
        ])

        self.norm = HyperbolicLayerNorm(embedding_dim, self.manifold)  # Corrected LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.fc = HyperbolicLinear(embedding_dim, num_classes, self.manifold)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (batch_size, embedding_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embedding_dim)

        x = self.manifold.expmap0(x)
        x = self.hyperbolic_embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.manifold.logmap0(x)
        x = x.mean(dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_features(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.manifold.expmap0(x)
        x = self.hyperbolic_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.manifold.logmap0(x)

    def geodesic_regularization(self, x, labels, margin=1.0):
        dist_matrix = self.manifold.dist(x.unsqueeze(1), x.unsqueeze(0))
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        positive_dist = (label_matrix * dist_matrix).sum(dim=1) / (label_matrix.sum(dim=1) + 1e-15)
        negative_dist = ((1 - label_matrix) * dist_matrix).min(dim=1)[0]
        loss = F.relu(margin + positive_dist - negative_dist).mean()
        return loss
