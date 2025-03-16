import torch
import torch.utils.data as data
from dataset import HWDataset
from globals import TRANSFORM, DEVICE, TEST_PATH, NB_CLASS, EMBEDDING_DIM, MODEL_PATH, IMG_SIZE
from model import create_model
from loss import LMCL_loss

test_dataset = HWDataset(TEST_PATH, transform=TRANSFORM)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = create_model(NB_CLASS, IMG_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
correct, total = 0, 0

lmcl_loss = LMCL_loss(num_classes=NB_CLASS, feat_dim=EMBEDDING_DIM)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        embeddings = model(images)
        logits, _ = lmcl_loss(embeddings, labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
print(f"Test accuracy: {100 * correct / total:.2f}%")
