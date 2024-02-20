import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import models
from datasets import load_dataset
import Model.train_model as train_model

def load_datasets():
    # Cargar el dataset
    data = load_dataset("keremberke/painting-style-classification", name="full")

    # Convertir el subset de entrenamiento a DataFrame
    train_df = pd.DataFrame(data['train'])
    validation_df = pd.DataFrame(data['validation'])
    test_df = pd.DataFrame(data['test'])
    return train_df,validation_df,test_df

def prepare_train(train_df):
    y_train = train_df.iloc[:,0].values
    x_train = train_df.iloc[:,1].values

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

        imagenes_procesadas.append(imagen_np)
        # Añadir la imagen procesada a la lista

    labels = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern',
            'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance',
            'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance',
            'Minimalism', 'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
            'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']

    label_to_index = {label: index for index, label in enumerate(labels)}

    filtered_labels = []

    for path in y_train:
        label = path.split('/')[9]
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
        label = path.split('/')[9]
        if label in labels:
            filtered_labels.append(label_to_index[label])

    # Convertir la lista a un arreglo de NumPy
    x_val_procesado = np.array(imagenes_procesadas)
    y_val_procesado = np.array(filtered_labels)

    return x_val_procesado,y_val_procesado

def create_tensors_dataloader(x_train,y_train,x_val,y_val):  
    # Crear los tensores de las imagenes y las labels
    train_images = torch.from_numpy(x_train.transpose((0, 3, 1, 2))).float()
    valid_images = torch.from_numpy(x_val.transpose((0, 3, 1, 2))).float()
    train_labels = torch.tensor(y_train, dtype=torch.long)
    valid_labels = torch.tensor(y_val, dtype=torch.long)

    # Crear los DataSet de los tensores
    train_data = TensorDataset(train_images, train_labels)
    valid_data = TensorDataset(valid_images, valid_labels)

    # Crear el Dataloader para entrenamiento y validación
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    return  train_loader,valid_loader

def main():
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carga del dataset
    train_df,validation_df,test_df = load_datasets()
    
    # Preparación de los datos
    x_train, y_train = prepare_train(train_df)
    x_val, y_val = prepare_validation(validation_df)

    # Cambiar distribucion  de las imágenes a tensor y normalizarlas
    train_loader,valid_loader = create_tensors_dataloader(x_train,y_train,x_val,y_val)
    
    # Selección y preparación del modelo
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 27)  # Ajuste para 27 clases
    model.to(device)
    
    # Configuración del optimizador y el criterio
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Entrenamiento y validación del modelo
    for epoch in range(1, 6):
        train_model.train(model, device, train_loader, criterion, optimizer, epoch)
        train_model.validation(model, device, valid_loader, criterion)

if __name__ == "__main__":
    main()