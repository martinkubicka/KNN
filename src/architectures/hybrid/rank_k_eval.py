import numpy as np
import random
from collections import defaultdict
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

random.seed(42)
np.random.seed(42)

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def evaluate_closed_set_rank_k(
    embeddings,
    test_labels,
    gallery_sizes=[10, 50, 100],
    max_rank=5
):
    test_labels = np.array(test_labels)
    all_embeddings = np.array(embeddings) 
    
    print(test_labels.shape)
    print(all_embeddings.shape)

    identity_to_indices = defaultdict(list) # {lab1:[indicies], ..}
    for idx, label in enumerate(test_labels):
        identity_to_indices[label].append(idx)

    results = {}
    
    for N in gallery_sizes:
        correct_at_rank = np.zeros(max_rank, dtype=np.int32)
        
        for probe_idx in range(len(embeddings)):
            probe_label = test_labels[probe_idx] # actual ground truth
            probe_embedding = all_embeddings[probe_idx] # actual embedding

            # we dont want gt label
            possible_identities = list(identity_to_indices.keys())
            possible_identities.remove(probe_label)
            
            # pick ranom N identities (based on gallery size)
            chosen_identities = random.sample(possible_identities, k=N-1)
            
            # impostors for each picked author id
            impostor_indices = []
            for identity in chosen_identities:
                impostor_idx = random.choice(identity_to_indices[identity])
                impostor_indices.append(impostor_idx)
            
            # choose different sample from same author
            candidates = identity_to_indices[probe_label]
            ref_idx = random.choice([i for i in candidates if i != probe_idx])
            gallery_indices = [ref_idx] + impostor_indices
            
            similarities = []
            for g_idx in gallery_indices:
                score = cosine_similarity(probe_embedding, all_embeddings[g_idx])
                similarities.append(score)
            
            # sort -> we can get rank-N score
            sorted_gallery_indices = np.argsort(similarities)[::-1]
            sorted_labels = [test_labels[gallery_indices[i]] for i in sorted_gallery_indices]

            # get results
            for rank in range(1, max_rank+1):
                top_k_labels = sorted_labels[:rank]
                if probe_label in top_k_labels:
                    correct_at_rank[rank-1] += 1
        
        total_probes = len(embeddings)
        rank_accuracies = correct_at_rank / total_probes
        
        results[N] = rank_accuracies.tolist()

    return results

########

embeddings = []
true_labels = []

model = create_model(nb_cls=NB_CLASS, img_size=IMG_SIZE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
pretrained_dict = checkpoint["model_state_dict"]
model.load_state_dict(pretrained_dict, strict=False)
model.eval()
model.to(DEVICE)

test_dataset = HWDataset(DATA_MDB_PATH, TEST_CSV_PATH,transform=TRANSFORM)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        batch_embeddings = model(images).cpu().numpy().astype(np.float16) 
        for emb in batch_embeddings:
            embeddings.append(emb)
        for lab in labels:
            true_labels.append(lab.item())

gallery_sizes_to_test = [10, 25, 50, 100, 250, NB_CLASS_TEST]
rank_results = evaluate_closed_set_rank_k(
    embeddings,
    true_labels,
    gallery_sizes=gallery_sizes_to_test,
    max_rank=5
)

for N, accs in rank_results.items():
    print(f"Gallery size = {N}")
    for r, val in enumerate(accs, start=1):
        print(f"  Rank-{r} accuracy = {val:.4f}")