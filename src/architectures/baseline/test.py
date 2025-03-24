import torch
import torch.utils.data as data
from dataset import HWDataset
from globals import TRANSFORM, DEVICE, MODEL_PATH, IMG_SIZE, BATCH_SIZE, DATA_MDB_PATH, NB_CLASS_TEST, TEST_CSV_PATH, NB_CLASS
from model import create_model
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.spatial.distance import cdist
import random

test_dataset = HWDataset(DATA_MDB_PATH, TEST_CSV_PATH,transform=TRANSFORM)

# sample_size = int(0.5 * len(test_dataset))
# sampled_indices = random.sample(range(len(test_dataset)), sample_size)
# test_dataset = data.Subset(test_dataset, sampled_indices)

print(len(test_dataset))

test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = create_model(nb_cls=NB_CLASS, img_size=IMG_SIZE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
pretrained_dict = checkpoint["model_state_dict"]
model.load_state_dict(pretrained_dict, strict=False)
model.eval()
model.to(DEVICE)

embeddings = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        batch_embeddings = model(images).cpu().numpy().astype(np.float16) 
        embeddings.append(batch_embeddings)
        true_labels.append(labels.numpy())

embeddings = np.vstack(embeddings)
true_labels = np.concatenate(true_labels)

### K-means and PCA

kmeans = KMeans(n_clusters=NB_CLASS_TEST, random_state=42, n_init=10)
predicted_clusters = kmeans.fit_predict(embeddings)

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=true_labels, cmap='nipy_spectral', alpha=0.6)
plt.colorbar(scatter, ax=ax, label='Ground truth')
plt.title("K-means clusters - ground truth")
plt.savefig("embedding_clusters_3d_ground_truth.png")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=predicted_clusters, cmap='nipy_spectral', alpha=0.6)
plt.colorbar(scatter, ax=ax, label='Clusers')
plt.title("K-means clusters - predicted")
plt.savefig("embedding_clusters_3d_predicted.png")

#### ROC and Genuine-impostor

pairwise_distances = cdist(embeddings, embeddings, metric="cosine")
pairwise_similarities = 1 - pairwise_distances

genuine_scores = []
impostor_scores = []
genuine_pairs = []
impostor_pairs = []

for i in range(len(true_labels)):
    for j in range(i + 1, len(true_labels)):
        score = pairwise_similarities[i, j]
        if true_labels[i] == true_labels[j]:
            genuine_scores.append(score)
            genuine_pairs.append((i, j, score))
        else:
            impostor_scores.append(score)
            impostor_pairs.append((i, j, score))

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
y_scores = np.concatenate([genuine_scores, impostor_scores])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
predicted_labels = (y_scores >= optimal_threshold).astype(int)
accuracy = accuracy_score(y_true, predicted_labels)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(genuine_scores, bins=50, alpha=0.5, label="Genuine Pairs", color="blue", density=True)
plt.hist(impostor_scores, bins=50, alpha=0.5, label="Impostor Pairs", color="red", density=True)
plt.axvline(optimal_threshold, color="black", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.3f}")
plt.xlabel("Cosine Distance")
plt.ylabel("Density")
plt.title("Genuine vs. Impostor Pairs with optimal threshold")
plt.legend()
plt.grid()
plt.savefig("genuine_impostor_distribution.png")
plt.show()

genuine_x, genuine_y, genuine_scores = zip(*genuine_pairs)
impostor_x, impostor_y, impostor_scores = zip(*impostor_pairs)

print(f"Accuracy at Optimal Threshold: {(accuracy * 100):.3f}")
