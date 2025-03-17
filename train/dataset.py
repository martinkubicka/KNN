from torch.utils.data import Dataset
import lmdb
import numpy as np
import pandas as pd
import cv2


class HandWrittenDataset(Dataset):
    def __init__(self, database, indices, transform=None):
        
        self.database = lmdb.open(database, readonly=True, lock=False)
        self.transform = transform
        try:
            self.indices = pd.read_csv(indices)
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error reading CSV file: {e}")
            raise e

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        key = self.indicies.iloc[idx, 0]
        img_data = self.txn.get(key.encode("utf-8"))
        img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        img = self.transform(img) if self.transform else img
        
        label = self.indicies.iloc[idx, 1]
        return img, label

    def __del__(self):
        self.txn.abort()
        self.database.close()