import os
from PIL import Image
from torch.utils.data import Dataset

class HWDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        image_dir = os.path.join(dataset_path, "images")
        label_dir = os.path.join(dataset_path, "labels")
        self.transform = transform

        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        self.data = []
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.split(".")[0] + ".txt")
            self.data.append((img_path, label_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx][0])
        if self.transform:
            image = self.transform(image)
        
        with open(self.data[idx][1], 'r') as f:
            label_str = f.readline().strip()
        label = int(label_str)
        
        return image, label
