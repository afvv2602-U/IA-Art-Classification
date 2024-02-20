import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_dataset

# Clases personales
import Model.train_model as train_model
import Data.prepare_data as pdata
import Model.save_model as sv

def load_datasets():
    # Cargar el dataset
    data = load_dataset("keremberke/painting-style-classification", name="full")

    # Convertir el subset de entrenamiento a DataFrame
    train_df = pd.DataFrame(data['train'])
    validation_df = pd.DataFrame(data['validation'])
    test_df = pd.DataFrame(data['test'])
    return train_df,validation_df,test_df

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
    x_train, y_train = pdata.prepare_train(train_df)
    x_val, y_val = pdata.prepare_validation(validation_df)

    # Cambiar distribucion  de las imágenes a tensor y normalizarlas
    train_loader,valid_loader = create_tensors_dataloader(x_train,y_train,x_val,y_val)
    
    # Selección y preparación del modelo
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
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

    # Guardar el modelo en formato diccionario
    sv.save_model_dict(model)

    # Guardar el modelo completo
    sv.save_complete_model(model)

if __name__ == "__main__":
    main()