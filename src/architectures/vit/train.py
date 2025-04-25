from models import get_model
from dataset import HandWrittenDataset
from loss.adaface import AdaFaceLoss
from loss.arcface import ArcFaceLoss
from loss.cosface import CosFaceLoss
from torchvision import transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
import torch
from pathlib import Path
import os
import sys
import cv2
import numpy as np


def get_dataloader(config: dict, indices: str):

    data_transforms = transforms.Compose([
        transforms.Resize([config["input_size"][1], config["input_size"][0]]),
        transforms.Grayscale(num_output_channels=3), # Toto tu musi byt lebo by nesedeli pocty kanalov
        transforms.ColorJitter(brightness=0.2,
                                contrast=0.2,
                                saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = HandWrittenDataset(config["data_dir"], indices, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=config["batch_size"]) 

    return dataloader

def get_basic_info(model, writer, config):
    writer.add_text("Model", str(model))
    writer.add_text("Model Parameters", str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    writer.add_text("Model Layers", str(len(list(model.parameters()))))
    writer.add_text("Model Configuration", json.dumps(config, indent=4))

    gpu_info = torch.cuda.get_device_name(config["device"]) if torch.cuda.is_available() else "CPU"
    writer.add_text("System Info", f"Using device: {gpu_info}")
    writer.add_text("System Info", f"PyTorch Version: {torch.__version__}")
    writer.add_text("System Info", f"Torchvision Version: {torchvision.__version__}")
    writer.add_text("System Info", f"Python Version: {sys.version}")

                          

 
def train(config: dict):
    model = get_model(config).to(config["device"])
    dataloader_training = get_dataloader(config, config["indicies"]["train"])
    dataloader_validation = get_dataloader(config, config["indicies"]["val"]) #TODO vylepšiť
    writer = SummaryWriter(log_dir=os.path.join(config["output"], config["log_dir"]))
    get_basic_info(model, writer, config)

    dummy_input = torch.randn(1, 3, config["input_size"][1], config["input_size"][0]).to(config["device"])
    writer.add_graph(model, dummy_input)

    try:
        if config["optimizer"]["name"].lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                momentum=config["optimizer"]["momentum"],
                weight_decay=config["optimizer"]["weight_decay"]
            )

        if config["optimizer"]["name"].lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["optimizer"].get("lr", 0.001),
                betas=config["optimizer"].get("betas", (0.9, 0.999)),
                weight_decay=config["optimizer"].get("weight_decay", 0),
                eps=config["optimizer"].get("eps", 1e-8)
            )

    except KeyError:
        optimizer = torch.optim.SGD(  # default optimizer at to funguje i na stare configy
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["scheduler"]["step_size"],
        gamma=config["scheduler"]["gamma"]
    )

    if config["loss"]["name"].lower() == "adaface":
        criterion = AdaFaceLoss(  # TODO dynamic loss
            class_num=config["architecture"]["num_classes"],
            embedding_size=config["embedding"]["dim"],
            device=config["device"]
        )
    elif config["loss"]["name"].lower() == "arcface":
        criterion = ArcFaceLoss(
            embedding_size=config["embedding"]["dim"],
            class_num=config["architecture"]["num_classes"],
            device=config["device"]
        )
    elif config["loss"]["name"].lower() == "cosface":
        criterion = CosFaceLoss(
            embedding_size=config["embedding"]["dim"],
            class_num=config["architecture"]["num_classes"],
            device=config["device"]
        )
    else:
        raise NotImplementedError(f"Criterion {config['loss']} not implemented")

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
                
                if isinstance(criterion, AdaFaceLoss) or isinstance(criterion, ArcFaceLoss) or isinstance(criterion, CosFaceLoss): # AdaFaceLoss berie embedding
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
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc, epoch)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        scheduler.step()

        model.eval()
        v_loss = 0.0
        v_correct = 0
        for inputs, labels in dataloader_validation:

            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])

            with torch.no_grad():
                if isinstance(criterion, AdaFaceLoss) or isinstance(criterion, ArcFaceLoss) or isinstance(criterion, CosFaceLoss):
                    embeddings = model.get_embedding(inputs)
                    loss = criterion(embeddings, labels)
                    preds = criterion.get_predictions(embeddings)
                else:
                    raise NotImplementedError(f"Criterion {criterion} not implemented")

            v_loss += loss.item() * inputs.size(0)
            v_correct += torch.sum(preds == labels.data)


        val_loss = v_loss / len(dataloader_validation.dataset)
        val_acc = v_correct.double() / len(dataloader_validation.dataset)


        current_lr = optimizer.param_groups[0]['lr']
        v_losses.append(val_loss)
        v_accs.append(val_acc)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        
        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f} LR: {current_lr}")
        print("-" * 10)
        print("-" * 10)

        # Save model last
        torch.save(model.state_dict(), os.path.join(config["output"], "model_last.pth"))
        print("Model saved as last")

        if val_loss == min(v_losses):
            torch.save(model.state_dict(), os.path.join(config["output"], "model_best.pth"))
            print("Model saved as best")

    print("Training finished")    
    writer.close()
    
def set_deterministic(config: dict):
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["seed"])
    os.environ['PYTHONHASHSEED'] = str(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)

def main(config: Path):
    with open(config) as f:
        config = json.load(f)

    if not os.path.exists(config["output"]):
        os.makedirs(config["output"])

    set_deterministic(config)
    train(config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model on the Handwritten Dataset")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    main(Path(args.config))