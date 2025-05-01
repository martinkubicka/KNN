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
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

NUM_PAIRS = 3000
CMC_GALLERY_SIZES = [10, 25, 50, 100, 250, 529]
CMC_MAX_RANK = 5
RNG_SEED = 42


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
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(dataloader):
            data = data.to(opt.device)
            emb = model.get_embedding(data)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # Normalize
            embeddings.append(emb.cpu())
            labels.append(label.cpu())

    return torch.cat(embeddings), torch.cat(labels)


def _pairs_from_embeddings(embeddings, labels, positive=True, seed=RNG_SEED):
    """Creates NUM_PAIRS unique positive or negative pairs."""
    np.random.seed(seed)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = np.array(labels)

    pairs = set()
    while len(pairs) < NUM_PAIRS:
        idx1, idx2 = np.random.choice(len(labels), 2, replace=False)
        if positive and labels[idx1] != labels[idx2]:
            continue
        if not positive and labels[idx1] == labels[idx2]:
            continue

        pair = (tuple(embeddings[idx1].tolist()), tuple(embeddings[idx2].tolist()))
        pairs.add(pair)
    return list(pairs)


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
    plt.close()


def create_genuine_impostor_score(pos_pairs, neg_pairs):
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

    plt.figure()
    sns.kdeplot(genuine_scores, color="blue", fill=True)
    sns.kdeplot(imposter_scores, color="red", fill=True)

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")

    plt.hist(genuine_scores, bins=80, alpha=0.5, label="Genuine", color="blue", density=True)
    plt.hist(imposter_scores, bins=80, alpha=0.5, label="Imposter", color="red", density=True)
    plt.legend(loc="upper right")
    plt.savefig("geniune_impostor_score.png")
    plt.close()


def evaluate_closed_set_rank_k(embeddings,
                               labels,
                               gallery_sizes=CMC_GALLERY_SIZES,
                               max_rank=CMC_MAX_RANK):
    labels = np.array(labels)
    embeds = np.array(embeddings)  # shape [n, d]
    random.seed(RNG_SEED)

    identity_to_indices = defaultdict(list)
    for idx, lab in enumerate(labels):
        identity_to_indices[lab].append(idx)

    results = {}
    for N in gallery_sizes:
        correct_at_rank = np.zeros(max_rank, dtype=np.int32)

        for probe_idx in range(len(embeds)):
            probe_label = labels[probe_idx]
            probe_emb = embeds[probe_idx]

            impostor_ids = random.sample(
                [i for i in identity_to_indices.keys() if i != probe_label],
                k=N - 1)

            gallery_indices = []

            same_author_choices = [i for i in identity_to_indices[probe_label]
                                   if i != probe_idx]
            ref_idx = random.choice(same_author_choices)
            gallery_indices.append(ref_idx)

            for imp_id in impostor_ids:
                gallery_indices.append(random.choice(identity_to_indices[imp_id]))

            sims = cosine_similarity(probe_emb.reshape(1, -1),
                                     embeds[gallery_indices])[0]

            sorted_idx = np.argsort(sims)[::-1]
            sorted_labels = labels[gallery_indices][sorted_idx]

            for r in range(1, max_rank + 1):
                if probe_label in sorted_labels[:r]:
                    correct_at_rank[r - 1] += 1

        rank_acc = correct_at_rank / len(embeds)  # per‑rank accuracy
        results[N] = rank_acc.tolist()
    return results


def plot_cmc(results, out_path="cmc_curve.png"):
    """
    Plots accuracy vs gallery size for each rank-k.
    Each curve corresponds to a fixed rank-k across varying gallery sizes.
    """
    plt.figure()

    # Get all unique rank values (assume same length for all gallery sizes)
    rank_k = range(1, len(next(iter(results.values()))) + 1)
    gallery_sizes = sorted(results.keys())

    # Plot one curve per rank-k
    for rank in rank_k:
        accs_at_rank = [results[G][rank - 1] for G in gallery_sizes]
        plt.plot(gallery_sizes, accs_at_rank, marker="o", label=f"Rank-{rank}")

    plt.xlabel("Gallery size")
    plt.ylabel("Identification accuracy")
    plt.title("CMC Curve (Accuracy vs. Gallery Size)")
    plt.ylim(0.5, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] CMC plot saved to {out_path}")


def plot_embedding_clusters_3d(embeddings, labels):
    # No sample limit — use all embeddings
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Limit PCA to 50 components (or fewer if input dimension < 50)
    pca = PCA(n_components=min(50, embeddings_np.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings_np)

    # Take first 3 components for 3D visualization
    embeddings_3d = embeddings_pca[:, :3]

    # Ground truth visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c=labels_np, cmap='nipy_spectral', alpha=0.6
    )
    plt.colorbar(scatter, ax=ax, label='Ground truth')
    plt.title("3D PCA (Full Data) - Ground Truth Labels")
    plt.savefig("embedding_clusters_3d_ground_truth.png")
    plt.close()

    # KMeans clustering in 50-D PCA space
    kmeans = KMeans(n_clusters=10, random_state=42)
    predicted_clusters = kmeans.fit_predict(embeddings_pca)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c=predicted_clusters, cmap='nipy_spectral', alpha=0.6
    )
    plt.colorbar(scatter, ax=ax, label='Predicted Clusters (k=10)')
    plt.title("3D PCA (Full Data) - KMeans Clusters")
    plt.savefig("embedding_clusters_3d_predicted.png")
    plt.close()


def eval(config, opt):
    model = get_model(config)
    model.load_state_dict(torch.load(opt.weights, weights_only=False))
    model.to(opt.device)
    model.eval()
    print("Model loaded")

    dataloader = get_data_loader(config)

    print("Loading embedings")
    embeddings, labels = get_embeddings(model, dataloader, opt)

    pos_pairs = _pairs_from_embeddings(embeddings, labels, positive=True)
    neg_pairs = _pairs_from_embeddings(embeddings, labels, positive=False)

    create_roc_curve(pos_pairs, neg_pairs)
    create_genuine_impostor_score(pos_pairs, neg_pairs)
    plot_embedding_clusters_3d(embeddings, labels)

    rank_results = evaluate_closed_set_rank_k(
        embeddings,
        labels,
        gallery_sizes=CMC_GALLERY_SIZES,
        max_rank=5
    )

    with open("rank_results.txt", "w") as f:
        for N, accs in rank_results.items():
            print(f"Gallery size = {N}", file=f)
            for r, val in enumerate(accs, start=1):
                print(f"  Rank-{r} accuracy = {val:.4f}", file=f)

    plot_cmc(rank_results)


def set_deterministic(config):
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(config["seed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Handwritten Dataset")
    parser.add_argument("-c", "--config", default="config.json", type=str, help="Path to the config file")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str,
                        help="Device to use")
    parser.add_argument("-w", "--weights", default=None, type=str, help="Path to the weights file")

    args = parser.parse_args()
    config = args.config

    with open(config) as f:
        config = json.load(f)

    set_deterministic(config)

    eval(config, args)
