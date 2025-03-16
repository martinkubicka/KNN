import os
import cv2
import lmdb
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HWDataset(Dataset):
    def __init__(self, lmdb_path, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.lmdb_txn = lmdb.open(lmdb_path, readonly=True, lock=False).begin()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data.iloc[idx]
        
        img = cv2.imdecode(np.frombuffer(self.lmdb_txn.get(image_name.encode()), dtype=np.uint8), 1)
        image = Image.fromarray(img)
        
        if self.transform:
            image = self.transform(image)
            
            

        return image, int(label)
