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
    

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.encodings[idx])
        label = F.one_hot(torch.tensor(self.labels[idx]), num_classes=27).float()

        return image, label


y_train = train_df.iloc[:,0].values
x_train = train_df.iloc[:, 1].values



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

x_train_procesado = x_train_procesado.transpose((0, 3, 1, 2))
x_val_procesado = x_val_procesado.transpose((0, 3, 1, 2))

# from sklearn.model_selection import train_test_split
# x_train_procesado, x_val, y_train_procesado, y_val = train_test_split(x_train_procesado, y_train_procesado, test_size=0.2, random_state=42)

train_dataset = CustomDataset(x_train_procesado, y_train_procesado)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)

val_dataset = CustomDataset(x_val_procesado, y_val_procesado)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

import torchvision.models as models
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

y_test = test_df.iloc[:,0].values
x_test = test_df.iloc[:, 1].values

# Tamaño deseado para las imágenes
tamaño_deseado = (224, 224)

# Lista para almacenar las imágenes procesadas
imagenes_procesadas = []

# Iterar sobre cada imagen en x_train
for imagen in x_test:
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

for path in y_test:
    label = path.split('/')[8]  # Ajusta el índice según tu estructura de ruta
    if label in labels:
        filtered_labels.append(label_to_index[label])

# Convertir la lista a un arreglo de NumPy
x_test_procesado = np.array(imagenes_procesadas)
y_test_procesado = np.array(filtered_labels)
x_test_procesado = x_test_procesado.transpose((0, 3, 1, 2))

tensor = torch.tensor(x_test_procesado[0:1,:]).to(device)

model.eval()
with torch.no_grad():
    predictions = model(tensor)

predictions

predictions.argmax(axis=-1).flatten()


