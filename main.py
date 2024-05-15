import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from PIL import ImageFile

# Clases personales
import Model.train_model as train_model
import Model.save_model as sv

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Configuración para permitir imágenes truncadas

def get_transforms():
    # Transformaciones para el conjunto de entrenamiento con aumento de datos
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transformaciones para el conjunto de validación
    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, valid_test_transforms

def get_dataloaders(base_dir, batch_size=32):
    train_transforms, valid_test_transforms = get_transforms()

    # Datasets
    train_dataset = datasets.ImageFolder(root=f'{base_dir}/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=f'{base_dir}/val', transform=valid_test_transforms)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def main():
    base_dir = 'C:/Users/afvv2/OneDrive/Documentos/Datasets'  # Ajusta esto a la ruta de tus datos
    train_loader, valid_loader = get_dataloaders(base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo y configuración
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 8)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Decremento de LR

    # Entrenamiento y validación del modelo
    for epoch in range(1, 2):  # Cambia el rango según el número deseado de épocas
        train_model.train(model, device, train_loader, criterion, optimizer, epoch)
        train_model.validation(model, device, valid_loader, criterion)
        scheduler.step()  # Actualizar el LR según el scheduler

    # Guardar el modelo
    sv.save_model_dict(model)
    sv.save_complete_model(model)

if __name__ == "__main__":
    main()
