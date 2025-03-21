from typing import Tuple
from venv import logger
import torch
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from PIL import Image

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
        image = Image.open(os.path.join(self.images_folder, filename)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, labels
    
    
def load_datasets(images: dict, data: dict, train: bool) -> Tuple[(CustomDataset, CustomDataset, CustomDataset)]:
    
    train_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomRotation(degrees=10),
                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.6544, 0.6596, 0.6643], std=[0.1878, 0.1899, 0.1975])
            ])
    
    test_val_transforms = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.6544, 0.6596, 0.6643], std=[0.1878, 0.1899, 0.1975])
            ])
    try:
        if train:
            

            training_data = CustomDataset(data['training_data'], images['training_images'], transform = train_transform)

            validation_data = CustomDataset(data['validation_data'], images['validation_images'], transform = test_val_transforms)
            print(f"[INFO]: Total training images: {len(training_data)}")
            print(f"[INFO]: Total validation images: {len(validation_data)}")
            return training_data, validation_data

        else:
            testing_data = CustomDataset(data['testing_data'], images['testing_images'], transform = test_val_transforms)
            print(f"[INFO]: Total test images: {len(testing_data)}")
            return testing_data

    except Exception as e:
        logger.error("dataset creation failed due to...", e, "is missing from dict")


def create_dataloaders(training_data: CustomDataset=None, validation_data: CustomDataset=None, testing_data: CustomDataset=None, batch_size: int=64) -> Tuple[(DataLoader, DataLoader, DataLoader)]:

        try:
            if (training_data != None and validation_data != None):
                train_dl = DataLoader(training_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)

                val_dl = DataLoader(validation_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)

                return train_dl, val_dl
            
            elif (testing_data != None):
                test_dl = DataLoader(testing_data, shuffle = False, num_workers = 0, pin_memory = True)

                return test_dl
            
            else:
                raise Exception("parameters not passed...")

        
        except Exception as e:
            logger.error("data loader creation failed due to...", e)