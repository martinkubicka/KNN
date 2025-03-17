from .models import get_model
from .dataset import HandWrittenDataset
from .loss import AdaFaceLoss
import json
import argparse
import torch
from pathlib import Path


def get_dataloader(config: dict, indices: str):

    transforms = transforms.Compose([
        transforms.Resize(config["input_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2,
                                contrast=0.2,
                                saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = HandWrittenDataset(config["database"], indices, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, indices, shuffle=True) 

    return dataloader

                          

 
def train(config: dict):
    model = get_model(config).to(config["device"])
    dataloader_training = get_dataloader(config, config["indices"]["train"])
    dataloader_validation = get_dataloader(config, config["indices"]["val"]) #TODO vylepšiť

    optimizer = torch.optim.SGD( # Vyskúšame neskôr aj Adama ale na toto mám celkom dobre odskúšaný SGD
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["step_size"],
        gamma=config["gamma"]
    )

    criterion = AdaFaceLoss(
        class_num=config["num_classes"],
        embedding_size=config["embedding_size"],
        device=config["device"]
    )

    t_losses = []
    v_losses = []
    t_accs = []
    v_accs = []

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print("-" * 10)

        model.train()
        t_loss = 0.0
        t_correct = 0

        for i, (inputs, labels) in enumerate(dataloader_training):
            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                
                if isinstance(criterion, AdaFaceLoss): # AdaFaceLoss berie embedding
                    embeddings = model.get_embedding(inputs)
                    loss = criterion(embeddings, labels)
                    preds = criterion.get_predictions(embeddings)
                else:
                    raise NotImplementedError(f"Criterion {criterion} not implemented") # Ak by sme chceli pridať iný loss napr CrossEntropyLoss
                
                loss.backward()
                optimizer.step()
            
            t_loss += loss.item() * inputs.size(0)
            t_correct += torch.sum(preds == labels.data)

        epoch_loss = t_loss / len(dataloader_training.dataset)
        epoch_acc = t_correct.double() / len(dataloader_training.dataset)
        t_losses.append(epoch_loss)
        t_accs.append(epoch_acc)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        scheduler.step()

        model.eval()
        v_loss = 0.0
        v_correct = 0
        for inputs, labels in dataloader_validation:

            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])

            with torch.no_grad():
                if isinstance(criterion, AdaFaceLoss):
                    embeddings = model.get_embedding(inputs)
                    loss = criterion(embeddings, labels)
                    preds = criterion.get_predictions(embeddings)
                else:
                    raise NotImplementedError(f"Criterion {criterion} not implemented")

            v_loss += loss.item() * inputs.size(0)
            v_correct += torch.sum(preds == labels.data)

        val_loss = v_loss / len(dataloader_validation.dataset)
        val_acc = v_correct.double() / len(dataloader_validation.dataset)

        v_losses.append(val_loss)
        v_accs.append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f} LR: {current_lr}")
        print("-" * 10)
        print("-" * 10)
    

def main(config: Path):
    with open(config) as f:
        config = json.load(f)

    train(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model on the Handwritten Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main(Path(args.config))