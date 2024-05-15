import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import Model.train_model as train_model
import Model.save_model as sv


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

    # Crea los datasets usando ImageFolder
    train_dataset = datasets.ImageFolder(root=f'{base_dir}/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=f'{base_dir}/val', transform=valid_test_transforms)

    # Crea los dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def train(model, device, train_loader, criterion, optimizer, epoch):
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

        # Acumular las pérdidas y calcular la precisión
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Calcular la precisión al final de cada epoch
    accuracy = 100. * correct / total
    print(f'\nTrain set: Average loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')


def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / total
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n')

def main():
    base_dir = ''  # Ajusta esto a la ruta de tus datos
    train_loader, valid_loader = get_dataloaders(base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo y configuración
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
    num_epochs = 25
    best_val_loss = float('inf')
    patience = 5  # Número de épocas para early stopping
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        val_loss = validation(model, device, valid_loader, criterion)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            sv.save_model_dict(model)
            sv.save_complete_model(model)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    main()
