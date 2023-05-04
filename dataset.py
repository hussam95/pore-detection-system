import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class CellImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file, index_col='image_id')
        self.image_list = []
        for (dirpath, dirnames, filenames) in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.bmp'):
                    self.image_list.append(os.path.join(dirpath, filename))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_list[idx]
        image = Image.open(image_path)

        # Get image metadata from CSV
        image_name = os.path.splitext(os.path.basename(image_path))[0] + os.path.splitext(os.path.basename(image_path))[1]
        
        metadata = self.csv_file.loc[image_name]

        # Convert metadata to tensor
        metadata_tensor = torch.tensor(metadata.values.astype('float32'))

        # Apply transforms and convert image to tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = image.convert('L')
            image = torch.tensor(np.array(image)).unsqueeze(0).float() / 255.0

        return (image, metadata_tensor)
