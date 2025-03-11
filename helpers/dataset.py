import torch
import pandas as pd
import os
import PIL.Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.labels = self.df.columns[1:]  # Extract column names excluding 'filename'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row.iloc[0]  # First column is the filename
        labels = torch.tensor(row.iloc[1:].values.astype(float), dtype=torch.float32)  # Multi-labels as tensor
        image = PIL.Image.open(os.path.join(self.images_folder, filename)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, labels
