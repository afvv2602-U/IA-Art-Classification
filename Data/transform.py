import torch
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Función para aplicar transformaciones a un lote de imágenes
def apply_transforms(dataset):
    transformed_dataset = []
    for img in dataset:
        # Añade una dimensión extra para que img tenga shape [C, H, W] esperado por torchvision transforms
        img = img.unsqueeze(0)  # Transforma de [H, W, C] a [1, H, W, C] temporalmente
        img = data_transforms(img)
        transformed_dataset.append(img)
    # Concatena la lista de imágenes transformadas en un tensor
    return torch.cat(transformed_dataset, dim=0)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
