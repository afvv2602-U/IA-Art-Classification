import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
from torch.utils.data import DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Configuración para permitir imágenes truncadas

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

def save_model_dict(model):
    base_path = "States/Dict"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        new_version = 1.0
    else:
        # Lista todas las subcarpetas en la carpeta base
        existing_versions = os.listdir(base_path)
        # Filtra para mantener solo las carpetas que se pueden convertir a float
        existing_versions = [float(folder) for folder in existing_versions if folder.replace('.', '', 1).isdigit()]
        if existing_versions:
            # Encuentra la versión más alta y suma 1.0
            new_version = max(existing_versions) + 1.0
        else:
            # Si no hay carpetas que se ajusten al formato, comienza desde 1.0
            new_version = 1.0
    
    # Crea la nueva carpeta
    new_folder_path = os.path.join(base_path, str(new_version))
    os.makedirs(new_folder_path)
    
    # Guarda el estado del modelo en la nueva carpeta
    model_path = os.path.join(new_folder_path, "model_state_dict.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in {model_path}")

def save_complete_model(model):
    base_path = "States/Model"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        new_version = 1.0
    else:
        # Lista todas las subcarpetas en la carpeta base
        existing_versions = os.listdir(base_path)
        # Filtra para mantener solo las carpetas que se pueden convertir a float
        existing_versions = [float(folder) for folder in existing_versions if folder.replace('.', '', 1).isdigit()]
        if existing_versions:
            # Encuentra la versión más alta y suma 1.0
            new_version = max(existing_versions) + 1.0
        else:
            # Si no hay carpetas que se ajusten al formato, comienza desde 1.0
            new_version = 1.0
    
    # Crea la nueva carpeta
    new_folder_path = os.path.join(base_path, str(new_version))
    os.makedirs(new_folder_path)
    
    # Guarda el modelo completo en la nueva carpeta
    model_path = os.path.join(new_folder_path, "complete_model.pth")
    torch.save(model, model_path)
    print(f"Complete model saved in {model_path}")

def train(model, device, train_loader, criterion, optimizer, epoch, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f} Avg Acc: {100. * correct / total:.2f}%')

    if scheduler:
        scheduler.step()
    print(f'End of Epoch {epoch}: Average Loss: {running_loss / len(train_loader):.4f}, Average Accuracy: {100. * correct / total:.2f}%')

def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar el modelo con pesos pre-entrenados
    weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
    model = regnet_y_128gf(weights=weights)
    model.to(device)

    # Cambiar la capa final para ajustarse al número de clases
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)

    # Configuración de optimizador y pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Carga de datos
    base_dir = ''
    train_loader, valid_loader = get_dataloaders(base_dir)

    # Entrenamiento y validación
    for epoch in range(1, 11):  # Número de épocas
        train(model, device, train_loader, criterion, optimizer, epoch)
        validation(model, device, valid_loader, criterion)

    # Guardar el modelo
    save_model_dict(model)
    save_complete_model(model)

if __name__ == "__main__":
    main()


