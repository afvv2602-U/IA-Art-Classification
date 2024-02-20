import os
import torch
import torchvision.models as model

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