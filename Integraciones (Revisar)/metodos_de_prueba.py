from torchvision import transforms

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0) 
scheduler = StepLR(optimizer, step_size=1, gamma=1)
num_epochs = 50
print("Training the Model...")
train_loop(train_df, validation_df, test_df, model, criterion, optimizer, scheduler, num_epochs)

def load_dataset():
        # Load DataSets for training and testing from ImageFolder folders
        data_transforms = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
            ])
        
        image_folder_path = "./images"
        train_dir = os.path.join(image_folder_path, 'training')
        valid_dir = os.path.join(image_folder_path, 'validating')
        test_dir = os.path.join(image_folder_path, 'testing ')
        train_set = ImageFolder(root=train_dir, transform=data_transforms)
        valid_set = ImageFolder(root=valid_dir, transform=data_transforms)
        test_set = ImageFolder(root=test_dir, transform=data_transforms)
    
        # Create a DataFrame with labels for each image in the dataset
        label_mapping = {label: index for index, label in enumerate(train_set.class_to_idx)}
        train_labels = [label_mapping[item[1]] for item in train_set.imgs]
        validation_labels = [label_mapping[item[1]] for item in valid_set.imgs]
        test_labels = [label_mapping[item[1]] for item in test_set.imgs]
        train_df = pd.DataFrame(list(zip(range(len(train_labels)), train_labels)))
        validation_df = pd.DataFrame(list(zip(range(len(validation_labels)), validation_labels)))
        test_df = pd.DataFrame(list(zip(range(len(test_labels)), test_labels)))
        return (train_df, validation_df, test_df)

from sklearn.model_selection import train_test_split
def separar_dataset():
    x_train_procesado, x_val, y_train_procesado, y_val = train_test_split(x_train_procesado, y_train_procesado, test_size=0.2, random_state=42)

    # Transformaciones para el entrenamiento y la validación
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])