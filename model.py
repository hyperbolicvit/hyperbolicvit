import os
import random
import numpy as np
import logging
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Net
from torch.cuda.amp import GradScaler
from geoopt.optim import RiemannianAdam
import torch.nn.utils
    
# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to save the model
def save_model(output_dir, model, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Model checkpoint saved at {model_checkpoint}")

# AverageMeter to track losses and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Function to calculate accuracy
def simple_accuracy(preds, labels):
    return (preds == labels).mean() * 100

import numpy as np

def top_5_accuracy(logits, targets):
    """
    Compute the top-5 accuracy for classification tasks using logits.

    Parameters:
    logits (numpy.ndarray): Array of shape (n_samples, n_classes) containing the logits (raw model outputs).
    targets (numpy.ndarray): Array of shape (n_samples,) containing the true labels (as integer indices).

    Returns:
    float: Top-5 accuracy score as a percentage.
    """
    # Ensure logits is a 2D array
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    # Get the indices of the top 5 logits for each sample
    top5_preds = np.argsort(logits, axis=1)[:, -5:]
    
    # Check if the true label is in the top 5 predictions for each sample
    top5_correct = np.any(top5_preds == targets.reshape(-1, 1), axis=1)
    
    # Compute the top-5 accuracy
    top5_accuracy = np.mean(top5_correct) * 100
    
    return top5_accuracy


# Geodesic regularization function
def geodesic_regularization(outputs, labels, manifold, lambda_reg=0.01):
    """
    Compute the geodesic regularization loss based on the output embeddings.
    
    Args:
        outputs (torch.Tensor): Output embeddings of shape (batch_size, embedding_dim).
        labels (torch.Tensor): Ground truth labels of shape (batch_size,).
        manifold (geoopt.Manifold): The manifold being used, e.g., PoincareBall.
        lambda_reg (float): The regularization coefficient.
    
    Returns:
        torch.Tensor: The geodesic regularization loss.
    """
    dist_matrix = manifold.dist(outputs.unsqueeze(1), outputs.unsqueeze(0))  # Pairwise geodesic distances
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # 1 if labels are the same, else 0
    reg_loss = ((1 - label_matrix) * dist_matrix).mean()  # Penalize distances between different-class points
    return lambda_reg * reg_loss


# Validation function
def validate(model, val_loader, device):
    eval_losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    logger.info("***** Running Validation *****")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_losses.update(loss.item(), inputs.size(0))

            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = simple_accuracy(all_preds, all_labels)
    top5 = top_5_accuracy(all_preds, all_labels)
    logger.info(f"Validation Accuracy: {accuracy:.4f}%, Validation Loss: {eval_losses.avg:.4f}, Top-5 Accuracy: {top5:.4f}%")
    return accuracy, top5

# Training function
def train_ddp(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    set_seed(42)  # Set a fixed seed for reproducibility

    # Set hyperparameters
    batch_size = 32
    num_epochs = 350       
    learning_rate = 0.01 / world_size 
    output_dir = './output'

    # Model setup
    model = Net(num_classes=1000).to(device)  # ImageNet has 1000 classes
    model = DDP(model, device_ids=[rank])

    # Data augmentation and normalization for ImageNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageNet(root='/data/jacob/ImageNet/', split='train', transform=transform)
    val_dataset = datasets.ImageNet(root='/data/jacob/ImageNet/', split='val', transform=transform)

    # train_dataset = datasets.CIFAR10(root='/data/jacob/cifar10/', train=True, download=True, transform=transform_cifar)
    # val_dataset = datasets.CIFAR10(root='/data/jacob/cifar10/', train=False, download=True, transform=transform_cifar)
    
    # Sampler and DataLoader
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)  # Step LR, decay at epoch 30, 60

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Gradient clipping value
    clip_value = 1.0  # Clip gradients to this value

    # Regularization weight for geodesic loss
    lambda_reg = 0.01  # Weight for the geodesic regularization

    # TensorBoard writer (only for rank 0 process)
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    file = open("log.txt", "w")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Geodesic regularization term
            reg_loss = geodesic_regularization(outputs, labels, model.module.manifold, lambda_reg=lambda_reg)
            total_loss = loss + reg_loss

            # Backward pass with gradient scaling for mixed precision
            scaler.scale(total_loss).backward()

            # Gradient clipping (after backward, before step)
            scaler.unscale_(optimizer)  # Unscales gradients to prevent issues when clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()

        logger.info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # Validation
        accuracy, accuracy5 = validate(model, val_loader, device)

        # TensorBoard writer
        if rank == 0:
            writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch+1)
            writer.add_scalar('Accuracy/val', accuracy, epoch+1)
            logger.info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.4f}, Top5: {accuracy5:.4f}")
            file.write(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy:.4f}, Top5: {accuracy5:.4f}\n")
            file.flush()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(output_dir, model, epoch+1)
    
        scheduler.step()
 
    if rank == 0: 
        writer.close()

    dist.destroy_process_group()

# Main function
def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ['RANK'])  # rank should be set by distributed launcher
    train_ddp(rank, world_size)

if __name__ == "__main__":
    main()

