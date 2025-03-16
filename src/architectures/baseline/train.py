import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import HWDataset
from globals import TRANSFORM, BATCH_SIZE, EMBEDDING_DIM, NB_CLASS, IMG_SIZE, DEVICE, LR, EPOCHS, DATA_MDB_PATH, TRAIN_CSV_PATH
from loss import LMCL_loss
from model import create_model

# DATASET
dataset = HWDataset(DATA_MDB_PATH, TRAIN_CSV_PATH, transform=TRANSFORM)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# TRAINING STUFF SETUP
model = create_model(NB_CLASS, IMG_SIZE)
lmcl_loss = LMCL_loss(num_classes=NB_CLASS, feat_dim=EMBEDDING_DIM)
criterion_ce = nn.CrossEntropyLoss()

device = DEVICE
model = model.to(device)
lmcl_loss = lmcl_loss.to(device)
criterion_ce = criterion_ce.to(device)

optimizer = optim.Adam(list(model.parameters()) + list(lmcl_loss.parameters()), lr=LR)

print("Training started.")

# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        
        logits, margin_logits = lmcl_loss(embeddings, labels)
        loss = criterion_ce(margin_logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # VALIDATION
    model.eval()
    correct_val = 0
    total_val = 0
    val_acc = -1
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            logits, _ = lmcl_loss(embeddings, labels)
            preds = torch.argmax(logits, dim=1)
            total_val += labels.size(0)
            correct_val += (preds == labels).sum().item()
    new_val_acc = 100 * correct_val / total_val
    if val_acc < new_val_acc:
        torch.save({
            'lmcl_loss_state_dict': lmcl_loss.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model-baseline.pth')
        
    val_acc = new_val_acc
    print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {epoch_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
