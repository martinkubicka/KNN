import torchvision.transforms as transforms
import torch
import os

DATASET_PATH = "./dataset/"
DATA_MDB_PATH = os.path.join(DATASET_PATH, "lmdb_processed/")
TRAIN_CSV_PATH = os.path.join(DATASET_PATH, "train_ids.csv")
TEST_CSV_PATH = os.path.join(DATASET_PATH, "test_ids.csv")
MODEL_PATH = "./model-baseline.pth"

BATCH_SIZE = 256
HEIGHT = 48
WIDTH = 1024
IMG_SIZE = [HEIGHT, WIDTH],
NB_CLASS = 4968
NB_CLASS_TEST = 124
EMBEDDING_DIM = 64
LR = 0.0001
EPOCHS = 10000
PATCH_SIZE = [6, 32] # H, W (based on input H and W) - number of patches needs to match 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
