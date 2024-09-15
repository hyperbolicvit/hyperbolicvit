import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from model import Net  # Make sure the Net class is defined in model.py
from model import Net
import os

def load_checkpoint(checkpoint_path, model, device):
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' not in k:  # Add the 'module.' prefix if loading into a DDP model
                new_state_dict[f'module.{k}'] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore mismatched keys
        print(f"Checkpoint loaded from '{checkpoint_path}'")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        exit()

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_gflops(model, input_shape):
    from thop import profile
    input = torch.randn(*input_shape)
    return profile(model, inputs=(input,))

def test_model(checkpoint_path, batch_size=32):
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_dataset = datasets.CIFAR10(root='/data/jacob/cifar10/', train=False, download=True, transform=transform_cifar)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load imagenet
    test_dataset = datasets.ImageNet(root='/data/jacob/ImageNet/', split='val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = Net(num_classes=1000).to(device)
    load_checkpoint(checkpoint_path, model, device)
    
    # Print the number of parameters in the model
    print(f"Number of parameters in the model: {get_num_parameters(model)}")

    # Print the number of FLOPs
    input_shape = (1, 3, 224, 224)
    gflops, _ = get_gflops(model, input_shape)
    print(f"GFLOPs: {gflops / 1e9:.2f}")
    
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = test_loss / len(test_loader)

    print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    checkpoint_path = 'checkpoint_epoch_1.pth'  # Replace with your checkpoint path
    test_model(checkpoint_path)
