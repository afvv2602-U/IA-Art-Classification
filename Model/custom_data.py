import torch
import torch.nn.functional as F
 
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