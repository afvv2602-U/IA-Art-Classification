from datasets import load_dataset
import pandas as pd
import numpy as np

def painting_style_dataset():
    data = load_dataset("keremberke/painting-style-classification", name="full")
    # Convertir el subset de entrenamiento a DataFrame
    train_df = pd.DataFrame(data['train'])
    validation_df = pd.DataFrame(data['validation'])
    test_df = pd.DataFrame(data['test'])


def prepare_images(x_train,y_train):
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

    return x_train_procesado,y_train_procesado