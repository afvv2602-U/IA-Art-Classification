import os
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import ImageFile

# Imports clases 
import Model.train_model as tr
import Model.save_model as sv
import Model.validate_model as vm
import Model.transformers as tfm
import Model.evaluate_model as vlm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Configuración para permitir imágenes truncadas

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verificar la ruta del archivo de modelo
    model_path = 'IA-Art-Classification\AI\complete_model.pth'  # Ajusta esto según la versión que quieras cargar
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de modelo en la ruta: {model_path}")

    # Cargar el modelo guardado
    model = torch.load(model_path)
    model.to(device)
    print(f"Model loaded from {model_path}")

    # Configuración de optimizador y pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Puedes ajustar la tasa de aprendizaje para seguir entrenando
    criterion = nn.CrossEntropyLoss()

    # Carga de datos
    base_dir = '' 
    train_loader, valid_loader = tfm.get_dataloaders(base_dir)

    # Entrenamiento y validación
    num_epochs = 6 

    for epoch in range(1, num_epochs + 1):
        tr.train(model, device, train_loader, criterion, optimizer, epoch)
        vm.validation(model, device, valid_loader, criterion) # Este si solo quieres hacer una evaluacion normal 
        vlm.evaluate_model_per_class(model,device,valid_loader,criterion) # Este si quieres saber el porcentaje de cada una 

    # Guardar el modelo actualizado
    sv.save_complete_model(model, version=2.0)
    

if __name__ == "__main__":
    main()
