import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import numpy as np
from model import Net
import gc
import thop

# Clean memory before testing
gc.collect()
torch.cuda.empty_cache()

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Calculate accuracy
def simple_accuracy(preds, labels):
    return (preds == labels).mean() * 100

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_gflops(model, input_shape):
    input = torch.randn(*input_shape)
    return thop.profile(model, inputs=(input,))

# Top-5 accuracy
def top_5_accuracy(logits, targets):
    if not logits.is_floating_point():
        logits = logits.float()

    targets = targets.long()
    
    with torch.no_grad():
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)

        top5_preds = torch.topk(logits, k=5, dim=1).indices
        correct = top5_preds.eq(targets.view(-1, 1))
        correct_any = correct.any(dim=1).float()
        top5_accuracy = correct_any.mean().item() * 100.0

        return top5_accuracy

# AverageMeter for loss tracking
class AverageMeter:
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

# Load the checkpoint
def load_checkpoint(checkpoint_path, model):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    
    # Load the checkpoint
    state_dict = torch.load(checkpoint_path, map_location=map_location)

    # Handle cases where the checkpoint was saved without DistributedDataParallel (DDP)
    # If the keys in the state dict don't have 'module.' prefix, add it
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k] = v  # No changes needed if it already has 'module.'
        else:
            new_state_dict[f'module.{k}'] = v  # Add 'module.' prefix for DDP

    # Load the modified state dict into the model
    model.load_state_dict(new_state_dict, strict=False)  # strict=False to ignore minor mismatches

    logger.info(f"Checkpoint loaded from '{checkpoint_path}'")

# Validation function for testing
def validate_ddp(model, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    eval_losses = AverageMeter()

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
    top5_acc = top_5_accuracy(outputs, labels)
    
    logger.info(f"Validation Accuracy: {accuracy:.2f}%, Validation Loss: {eval_losses.avg:.4f}, Top-5 Accuracy: {top5_acc:.2f}%")
    return accuracy, top5_acc

# Testing function
def test_ddp(rank, world_size, checkpoint_path):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    set_seed(42)

    # Model setup
    model = Net(num_classes=1000).to(device)
    model = DDP(model, device_ids=[rank])

    # Load checkpoint
    load_checkpoint(checkpoint_path, model)

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load validation dataset
    val_dataset = datasets.ImageNet(root='/data/jacob/ImageNet/', split='val', transform=transform)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, sampler=val_sampler)

    # Run validation
    accuracy, top5_acc = validate_ddp(model, val_loader, device)
    # num parameters
    num_parameters = get_num_parameters(model)
    logger.info(f"Number of parameters in the model: {num_parameters}")
    print(f"Number of parameters in the model: {num_parameters}")
    # GFLOPs calculation
    input_shape = (1, 3, 224, 224)
    gflops, _ = get_gflops(model, input_shape)
    logger.info(f"GFLOPs: {gflops / 1e9:.2f}")
    print(f"GFLOPs: {gflops / 1e9:.2f}")
    dist.barrier()  # Synchronize all processes

    if rank == 0:
        logger.info(f"Final Test Accuracy: {accuracy:.2f}%")
        logger.info(f"Final Top-5 Accuracy: {top5_acc:.2f}%")

    dist.destroy_process_group()

# Main function
def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ['RANK'])  # This should be set by the launcher (like torchrun)
    checkpoint_path = './output/checkpoint_epoch_1.pth'  # Replace with your checkpoint path
    test_ddp(rank, world_size, checkpoint_path)

if __name__ == "__main__":
    main()
