import torch
import json
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from models import get_model
from dataset import HandWrittenDataset
from loss.adaface import AdaFaceLoss


def evaluate_rank_k(model, criterion, data_loader, device, k=5):
    """
    Evaluate model's Top-K accuracy on the given data loader.
    Returns the average rank-k accuracy (float).
    """
    model.eval()
    total_samples = 0
    total_correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            embeddings = model.get_embedding(inputs)
            logits = criterion.get_logits(embeddings, labels)

            _, topk_indices = torch.topk(logits, k, dim=1)

            correct_mask = topk_indices.eq(labels.view(-1, 1))
            correct_per_sample = correct_mask.sum(dim=1).clamp(max=1)

            total_correct += correct_per_sample.sum().item()
            total_samples += labels.size(0)

    rank_k_acc = total_correct / total_samples
    return rank_k_acc


def get_dataloader(config, indices, is_eval=True):
    data_transforms = transforms.Compose([
        transforms.Resize([config["input_size"][1], config["input_size"][0]]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = HandWrittenDataset(config["data_dir"], indices, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=not is_eval)
    return dataloader


def main(config_path: Path, model_ckpt: Path, k=5):
    with open(config_path) as f:
        config = json.load(f)

    device = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(config).to(device)
    print(f"Loading model state from: {model_ckpt}")
    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict)

    if config["loss"]["name"].lower() == "adaface":
        criterion = AdaFaceLoss(
            class_num=config["architecture"]["num_classes"],
            embedding_size=config["embedding"]["dim"],
            device=device
        )
    else:
        raise NotImplementedError(f"Loss '{config['loss']['name']}' not supported here.")

    val_loader = get_dataloader(config, config["indicies"]["test"], is_eval=True)

    # Evaluate rank-k
    rank_k_acc = evaluate_rank_k(model, criterion, val_loader, device, k)
    print(f"Rank-{k} Accuracy on validation set = {rank_k_acc:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--k", type=int, default=5, help="Rank-K for evaluation")
    args = parser.parse_args()

    main(Path(args.config), Path(args.model_ckpt), args.k)
