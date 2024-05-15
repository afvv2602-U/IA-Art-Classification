from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_transforms():
    # Definir transformaciones para entrenamiento y validación
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, valid_transforms

def get_dataloaders(base_dir, batch_size=32):
    train_transforms, valid_transforms = get_transforms()
    # Dataset y DataLoader
    train_dataset = datasets.ImageFolder(root=f'{base_dir}/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=f'{base_dir}/val', transform=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader