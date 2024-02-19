import os
import io
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import Model.custom_data as  custom_data

def main():
    train_df,validation_df,test_df = load_dataset()
    device = torch_settings()

def load_dataset():
    # Cargar el dataset
    data = load_dataset("keremberke/painting-style-classification", name="full")
    # Convertir el subset de entrenamiento a DataFrame
    train_df = pd.DataFrame(data['train'])
    validation_df = pd.DataFrame(data['validation'])
    test_df = pd.DataFrame(data['test'])
    return train_df,validation_df,test_df

def torch_settings():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funcion de entrenamiento
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Funcion de validacion
def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader.dataset)
    print(f'\nTest set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')

def prepare_train(train_df):
    y_train = train_df.iloc[:,0].values
    x_train = train_df.iloc[:, 1].values

    # Tamaño deseado para las imágenes
    tamaño_deseado = (224, 224)

    # Lista para almacenar las imágenes procesadas
    imagenes_procesadas = []

    # Iterar sobre cada imagen en x_train
    for imagen in x_train:
        # Redimensionar la imagen
        imagen_redimensionada = imagen.resize(tamaño_deseado)

        # Convertir la imagen a un arreglo de NumPy
        imagen_np = np.array(imagen_redimensionada)

        # Normalizar los valores de píxeles
        imagen_np = imagen_np.astype('float32') / 255.0

        # Añadir la imagen procesada a la lista
        imagenes_procesadas.append(imagen_np)

    labels = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern',
            'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance',
            'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance',
            'Minimalism', 'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
            'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']

    label_to_index = {label: index for index, label in enumerate(labels)}

    filtered_labels = []

    for path in y_train:
        label = path.split('/')[8]  # Ajusta el índice según tu estructura de ruta
        if label in labels:
            filtered_labels.append(label_to_index[label])

    # Convertir la lista a un arreglo de NumPy
    x_train_procesado = np.array(imagenes_procesadas)
    y_train_procesado = np.array(filtered_labels)

    return x_train_procesado, y_train_procesado

def prepare_validation(validation_df):
    y_val = validation_df.iloc[:,0].values
    x_val = validation_df.iloc[:, 1].values

    # Tamaño deseado para las imágenes
    tamaño_deseado = (224, 224)

    # Lista para almacenar las imágenes procesadas
    imagenes_procesadas = []

    # Iterar sobre cada imagen en x_train
    for imagen in x_val:
        # Redimensionar la imagen
        imagen_redimensionada = imagen.resize(tamaño_deseado)

        # Convertir la imagen a un arreglo de NumPy
        imagen_np = np.array(imagen_redimensionada)

        # Normalizar los valores de píxeles
        imagen_np = imagen_np.astype('float32') / 255.0

        # Añadir la imagen procesada a la lista
        imagenes_procesadas.append(imagen_np)

    labels = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern',
            'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance',
            'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance',
            'Minimalism', 'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
            'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']

    label_to_index = {label: index for index, label in enumerate(labels)}

    filtered_labels = []

    for path in y_val:
        label = path.split('/')[8]  # Ajusta el índice según tu estructura de ruta
        if label in labels:
            filtered_labels.append(label_to_index[label])

    # Convertir la lista a un arreglo de NumPy
    x_val_procesado = np.array(imagenes_procesadas)
    y_val_procesado = np.array(filtered_labels)

    return x_val_procesado,y_val_procesado

def image_transpose(x_train_procesado,x_val_procesado):
    x_train_procesado = x_train_procesado.transpose((0, 3, 1, 2))
    x_val_procesado = x_val_procesado.transpose((0, 3, 1, 2))

    return  x_train_procesado,x_val_procesado

def tensor_transform(x_train_procesado,y_train_procesado,x_val_procesado,y_val_procesado):
    train_dataset = custom_data.CustomDataset(x_train_procesado, y_train_procesado)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)

    val_dataset = custom_data.CustomDataset(x_val_procesado, y_val_procesado)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    return  train_loader,val_loader

def train_model(device,train_loader,val_loader):
    model = models.mobilenet_v2(pretrained=True)

    # Obtenemos los parametros y hacemos que no sean actualizables por el gradiente
    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[15].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(model.last_channel, 27)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, 6):
        train(model, device, train_loader, criterion, optimizer, epoch)
        validation(model, device, val_loader, criterion)

    return model

if __name__ == "__main__":
    main()