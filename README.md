# Hyperbolic Transformer for Image Classification

This repository implements a Hyperbolic Transformer model in PyTorch, designed for image classification tasks. The model leverages hyperbolic geometry to learn representations on the Poincaré ball manifold. It is built upon the Transformer architecture and includes a multi-head attention mechanism, feed-forward layers, and position encodings, all adapted to work within hyperbolic space.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training on ImageNet](#training-on-imagenet)
  - [Replicating ViT Variants](#replicating-vit-variants)
- [Parameters for ViT-Base, ViT-Large, and ViT-Huge](#parameters-for-vit-base-vit-large-and-vit-huge)
- [Model Parameters](#model-parameters)
- [Citation](#citation)

## Overview

This implementation modifies the Vision Transformer (ViT) architecture to operate in hyperbolic space using the Poincaré ball model. This results in enhanced geometric expressiveness, especially for hierarchical data such as images.

The model can be used to train on datasets like ImageNet, CIFAR-10, CIFAR-100, and others. Additionally, this model includes features such as learnable curvature and gradient clipping to stabilize training.

## Model Architecture

- **HyperbolicLinear**: A linear layer that operates on the hyperbolic manifold.
- **HyperbolicMultiheadAttention**: Multi-head attention mechanism with hyperbolic distance calculations.
- **HyperbolicLearnedPositionEncoding**: Learnable position embeddings on the Poincaré ball.
- **HyperbolicTransformerLayer**: Combines hyperbolic self-attention and feed-forward layers using Möbius arithmetic.
- **Net**: The main transformer-based model for image classification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/hyperbolic-transformer.git
   cd hyperbolic-transformer
   ```
2. Install dependencies:
  ```python
  pip install torch torchvision geoopt
  ```
## Usage

### Training on ImageNet

To train the model on ImageNet, follow these steps:

1. Prepare the ImageNet dataset:
   - Download the ImageNet dataset from [here](http://www.image-net.org/).
   - Organize the dataset into a `train` and `val` folder.

2. Run the training script, replacing the GPUs in launch.sh with your GPUs
  ```bash
  bash launch.sh
  ```
### Replicating ViT Variants

To replicate the Vision Transformer (ViT) variants (Base, Large, Huge), use the following hyperparameters.

## Parameters for ViT-Base, ViT-Large, and ViT-Huge

| Variant  | Patch Size | Embedding Dim | Num Layers | Num Heads | Image Size | Num Params | Dropout |
|----------|------------|---------------|------------|-----------|------------|------------|---------|
| ViT-Base | 16         | 768           | 12         | 12        | 224        | 86M        | 0.1     |
| ViT-Large| 16         | 1024          | 24         | 16        | 224        | 307M       | 0.1     |
| ViT-Huge | 14         | 1280          | 32         | 16        | 224        | 632M       | 0.1     |

## Model Parameters

Here are some important parameters that can be tuned:

- `img_size`: Size of the input image (default: 224 for ImageNet).
- `patch_size`: Size of each image patch (default: 16 for ImageNet, smaller for CIFAR).
- `embedding_dim`: Dimensionality of the patch embeddings.
- `num_heads`: Number of attention heads in the multi-head attention mechanism.
- `num_layers`: Number of transformer layers in the model.
- `dropout`: Dropout rate applied after each transformer layer (default: 0.1).
- `manifold`: The type of manifold to use for hyperbolic geometry. By default, the Poincaré ball is used.

### Gradient Clipping

To stabilize training in hyperbolic space, you can apply gradient clipping:

```python
model.clip_gradients(clip_value=1.0)
```

### Citation


  
  
