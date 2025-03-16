import torch
import torch.utils.data as data
from dataset import HWDataset
from globals import TRANSFORM, DEVICE, MODEL_PATH, IMG_SIZE, BATCH_SIZE, DATA_MDB_PATH, TEST_CSV_PATH
from model import create_model
from collections import defaultdict

def get_class(embedding, class_to_embedding):
    max_sim, best_class = -float("inf"), None
    for class_name, class_embedding in class_to_embedding.items():
        similarity = torch.nn.functional.cosine_similarity(embedding, class_embedding, dim=0)
        if similarity > max_sim:
            max_sim = similarity
            best_class = class_name
    return best_class

def get_representative_embeddings(model, test_loader, num_samples_per_class=10):
    class_to_embeddings = defaultdict(list)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            embeddings = model(images)

            for emb, label in zip(embeddings, labels):
                label_str = str(label.item())

                if len(class_to_embeddings[label_str]) < num_samples_per_class:
                    class_to_embeddings[label_str].append(emb)

            if all(len(v) >= num_samples_per_class for v in class_to_embeddings.values()):
                break

    class_to_embedding = {
        label: torch.stack(emb_list).mean(dim=0).to(DEVICE)
        for label, emb_list in class_to_embeddings.items()
    }

    return class_to_embedding

test_dataset = HWDataset(DATA_MDB_PATH, TEST_CSV_PATH,transform=TRANSFORM)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = create_model(num_classes=None, img_size=IMG_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

class_to_embedding = get_representative_embeddings(model, test_loader)

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        embeddings = model(images)

        for i in range(len(images)):
            pred_class = get_class(embeddings[i], class_to_embedding)
            true_class = labels[i].item()

            if pred_class == true_class:
                correct += 1
            total += 1

print(f"Test accuracy: {100 * correct / total:.2f}%")
