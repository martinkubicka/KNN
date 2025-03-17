from torch.utils.data import Dataset
import lmdb
import numpy as np
import pandas as pd
import cv2
import PIL.Image as Image
import io


class HandWrittenDataset(Dataset):
    def __init__(self, database, indices, transform=None):
        
        self.database = lmdb.open(database, readonly=True, lock=False)
        self.transform = transform
        try:
            self.indices = pd.read_csv(indices)
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error reading CSV file: {e}")
            raise e
        
        self.txn = self.database.begin()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        key = self.indices.iloc[idx, 0]
        img_data = self.txn.get(key.encode("utf-8"))
        img = Image.open(io.BytesIO(img_data))
        img = self.transform(img) if self.transform else img
        
        label = self.indices.iloc[idx, 1]
        return img, label

    def __del__(self):
        self.txn.abort()
        self.database.close()