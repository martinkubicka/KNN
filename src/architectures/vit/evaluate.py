import torch
from models import get_model
import json
import argparse
from dataset import HandWrittenDataset
from torchvision import transforms
from tqdm import tqdm
import numpy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

NUM_PAIRS = 3000


def get_data_loader(config):
    transform = transforms.Compose([
        transforms.Resize([config["input_size"][1], config["input_size"][0]]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = HandWrittenDataset(config["data_dir"], config["indicies"]["test"], transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=config["batch_size"]) 

    return dataloader

def get_embeddings(model, dataloader, opt):
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(opt.device)
            label = label.to(opt.device)

            embedding = model.get_embedding(data)
            embeddings.append(embedding)
            labels.append(label)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    return embeddings, labels

def create_pos_pairs(embeddings, labels):
    pos_pairs = []
   
    print("Creating positive pairs")
    while len(pos_pairs) < NUM_PAIRS:
        label_id = numpy.random.choice(labels, 1, replace=False)
        label_idx = numpy.where(labels == label_id)[0]
        
        if len(label_idx) < 2:
            continue

        idx1, idx2 = numpy.random.choice(label_idx, 2, replace=False)
        pair = (tuple(embeddings[idx1].tolist()), tuple(embeddings[idx2].tolist()))
        if pair not in pos_pairs:
            pos_pairs.append(pair)

    return pos_pairs

def create_neg_pairs(embeddings, labels):
    neg_pairs = []

    print("Creating negative pairs")
    while len(neg_pairs) < NUM_PAIRS:
        idx1, idx2 = numpy.random.choice(len(labels), 2, replace=False)
        if labels[idx1] != labels[idx2]:
            pair = (tuple(embeddings[idx1].tolist()), tuple(embeddings[idx2].tolist()))
            if pair not in neg_pairs:
                neg_pairs.append(pair)

    return neg_pairs   

def create_roc_curve(pos_pairs, neg_pairs):
    y_true = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    y_score = []

    for pair in pos_pairs + neg_pairs:
        tensor1 = torch.tensor(pair[0])  
        tensor2 = torch.tensor(pair[1])  
        similarity = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
        y_score.append(similarity.item())

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")

def create_geniune_impostor_score(pos_pairs, neg_pairs):
    genuine_scores = []
    imposter_scores = []

    for pair in pos_pairs:
        tensor1 = torch.tensor(pair[0])  
        tensor2 = torch.tensor(pair[1])  
        similarity = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
        genuine_scores.append(similarity.item())

    for pair in neg_pairs:
        tensor1 = torch.tensor(pair[0])  
        tensor2 = torch.tensor(pair[1])  
        similarity = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
        imposter_scores.append(similarity.item())

    sns.kdeplot(genuine_scores, color="blue", fill=True)
    sns.kdeplot(imposter_scores, color="red", fill=True)

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")

    plt.hist(genuine_scores, bins=80, alpha=0.5, label="Genuine", color="blue", density=True)
    plt.hist(imposter_scores, bins=80, alpha=0.5, label="Imposter", color="red", density=True)
    plt.legend(loc="upper right")
    plt.savefig("geniune_impostor_score.png")

def eval(config, opt):
    model = get_model(config)
    model.load_state_dict(torch.load(opt.weights))
    model.to(opt.device)
    model.eval()
    print("Model loaded")

    dataloader = get_data_loader(config)

    print("Loading embedings")
    embeddings, labels = get_embeddings(model, dataloader, opt)

    pos_pairs = create_pos_pairs(embeddings, labels)
    neg_pairs = create_neg_pairs(embeddings, labels)

    create_roc_curve(pos_pairs, neg_pairs)
    create_geniune_impostor_score(pos_pairs, neg_pairs)

def set_deterministic(config):
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(config["seed"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Handwritten Dataset")
    parser.add_argument("-c", "--config",default="config.json" ,type=str, help="Path to the config file")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use")
    parser.add_argument("-w", "--weights", default=None, type=str, help="Path to the weights file")

    args = parser.parse_args()
    config = args.config


    with open(config) as f:
        config = json.load(f)

    set_deterministic(config)

    eval(config, args)