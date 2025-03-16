import argparse
import pandas as pd
import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_sample(image_db, id_csv):
    df = pd.read_csv(id_csv)
    # format check
    if "image_name" not in df.columns or "id" not in df.columns:
        raise ValueError("CSV does not contain 'image_name' or 'id' columns.")

    # sample first 25 authors
    sampled_df = df.drop_duplicates(subset="id")
    sampled_df = sampled_df[sampled_df["id"].between(0, 25)]
    sampled_df = sampled_df.sample(n=25, random_state=42)

    sampled_image_names = sampled_df["image_name"].tolist()
    sampled_authors = sampled_df["id"].tolist()

    env = lmdb.open(image_db, readonly=True, lock=False)
    txn = env.begin()
    images = []
    image_ids_loaded = []

    for image_id in sampled_image_names:
        img_data = txn.get(str(image_id).encode())
        if img_data:
            img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            images.append(img)
            image_ids_loaded.append(image_id)

    env.close()
    # plot 5x5
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Image author: {sampled_authors[i]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a sample image from the database.")
    parser.add_argument("--image_db", type=str, required=True, help="Path to the image database.")
    parser.add_argument("--id_csv", type=str, required=True, help="Path to the CSV file containing image IDs.")
    args = parser.parse_args()

    get_sample(args.image_db, args.id_csv)
