import torchvision.transforms as transforms
import torch
import os

DATASET_PATH = "./dataset/"
TRAIN_PATH = os.path.join(DATASET_PATH, "train/")
TEST_PATH = os.path.join(DATASET_PATH, "test/")
MODEL_PATH = "./model-baseline.pth"

BATCH_SIZE = 8
HEIGHT = 40
WIDTH = 300
IMG_SIZE = [HEIGHT, WIDTH],
NB_CLASS = 1
EMBEDDING_DIM = 64
LR = 0.0001
EPOCHS = 1000
PATCH_SIZE = [8, 20] # H, W (based on input H and W)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])
