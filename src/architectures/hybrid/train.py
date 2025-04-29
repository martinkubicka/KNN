import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import HWDataset
from globals import TRANSFORM, BATCH_SIZE, EMBEDDING_DIM, NB_CLASS, IMG_SIZE, DEVICE, LR, EPOCHS, DATA_MDB_PATH, TRAIN_CSV_PATH
from loss import LMCL_loss, AdaFaceLoss, ArcFace
from model import create_model
import matplotlib.pyplot as plt

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

criterion_ada = AdaFaceLoss(
            class_num=NB_CLASS,
            embedding_size=EMBEDDING_DIM,
            device=DEVICE
        )

criterion_arc = ArcFace(
            class_num=NB_CLASS,
            embedding_size=EMBEDDING_DIM,
            device=DEVICE
        )

optimizer = optim.Adam(list(model.parameters()) + list(lmcl_loss.parameters()), lr=LR)
# optimizer = optim.SGD(
#     list(model.parameters()) + list(lmcl_loss.parameters()),
#     lr=LR,
#     momentum=0.9,
#     weight_decay=0.0001
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

train_losses = []
val_losses = []
val_accuracies = []

print("Training started.")

# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        
        # cosface
        logits, margin_logits = lmcl_loss(embeddings, labels)
        loss = criterion_ce(margin_logits, labels)
        
        # adaface
        # loss = criterion_ada(embeddings, labels)
        
        # arcface
        #loss = criterion_arc(embeddings, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    
    scheduler.step()
    
    epoch_loss = train_loss / len(train_loader.dataset)
    
    # VALIDATION
    model.eval()
    correct_val = 0
    total_val = 0
    val_acc = -1
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            logits, margin_logits = lmcl_loss(embeddings, labels)
            
            # cosface
            loss = criterion_ce(margin_logits, labels)
            preds = torch.argmax(logits, dim=1)
            
            # adaface
            # loss = criterion_ada(embeddings, labels)
            # preds = criterion_ada.get_predictions(embeddings)
            
            # arcface
            # loss = criterion_arc(embeddings, labels)
            # preds = criterion_arc.get_predictions(embeddings)
            
            total_val += labels.size(0)
            
            correct_val += (preds == labels).sum().item()
            val_loss += loss.item() * images.size(0)
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
            
    new_val_acc = 100 * correct_val / total_val
    if val_acc < new_val_acc:
        torch.save({
            'lmcl_loss_state_dict': lmcl_loss.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model.pth')
        
    val_acc = new_val_acc
    
    train_losses.append(epoch_loss)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(train_losses, linestyle='-')
    ax1.plot(val_losses, linestyle='-')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, linestyle='-', color='green')
    ax2.set_ylabel("Accuracy (%)")
    plt.savefig("./training_results.png")
    