import os
import random
from shutil import copyfile

def split_data(source, training, validation, train_size=0.85):
    """
    Divide los datos en directorios de entrenamiento y validación.
    """
    files = []
    for filename in os.listdir(source):
        file = os.path.join(source, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(f'{filename} is zero length, so ignoring.')
    
    # Calculamos el número de imágenes para entrenamiento y validación
    train_length = int(len(files) * train_size)
    val_length = len(files) - train_length
    
    # Mezclamos los archivos y dividimos según los cálculos anteriores
    shuffled_files = random.sample(files, len(files))
    training_set = shuffled_files[:train_length]
    validation_set = shuffled_files[train_length:]
    
    # Copiamos los archivos a sus nuevos directorios
    for filename in training_set:
        this_file = os.path.join(source, filename)
        destination = os.path.join(training, filename)
        copyfile(this_file, destination)
        
    for filename in validation_set:
        this_file = os.path.join(source, filename)
        destination = os.path.join(validation, filename)
        copyfile(this_file, destination)

# Ruta base de tus datos
base_dir = 'C:/Users/afvv2/OneDrive/Documentos/Datasets'
art_styles = ['Barroco', 'Cubismo', 'Expresionismo', 'Impresionismo', 'Realismo', 'Renacimiento', 'Rococo', 'Romanticismo']

# Directorios para datos divididos
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Dividir datos para cada estilo de arte
for style in art_styles:
    print(f"Processing {style}")
    source_dir = os.path.join(base_dir, style)
    train_style_dir = os.path.join(train_dir, style)
    val_style_dir = os.path.join(val_dir, style)
    os.makedirs(train_style_dir, exist_ok=True)
    os.makedirs(val_style_dir, exist_ok=True)
    split_data(source_dir, train_style_dir, val_style_dir)

