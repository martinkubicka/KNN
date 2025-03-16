import argparse
import pandas as pd
import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def split_dataset(old_ids_csv, test_ids_csv, train_ids_csv, image_db, plot_dir):
    df = pd.read_csv(old_ids_csv)
    if "image_name" not in df.columns or "id" not in df.columns:
        raise ValueError("CSV does not contain 'image_name' or 'id' columns.")

    # filter out authors with more than median pages
    author_counts = df["id"].value_counts()
    median_occurrences = author_counts.median()
    filtered_authors = author_counts[author_counts <= median_occurrences].index

    # sample for illustrative image
    sampled_df = df[df["id"].isin(filtered_authors)].drop_duplicates(subset="id")
    sampled_df = sampled_df.sample(n=1000, random_state=42)
    sampled_image_names = sampled_df["image_name"].tolist()

    sampled_authors = sampled_df["id"].tolist()

    # splitting into test and train
    test_df = df[df["id"].isin(sampled_authors)]
    train_df = df[~df["id"].isin(sampled_authors)]
    test_df.to_csv(test_ids_csv, index=False)
    train_df.to_csv(train_ids_csv, index=False)

    # open db to make the plot
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

    os.makedirs(plot_dir, exist_ok=True)
    num_images = len(images)
    cols = 25
    rows = num_images // cols + (num_images % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(50, rows * 3))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Author: {sampled_authors[i]}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sample_images.png"))
    plt.close()


def relabel_csv(id_csv):
    """ Relabeling so the ids are from zero"""
    df = pd.read_csv(id_csv)
    print(f"Number of unique authors: {df['id'].nunique()} in file {id_csv}")
    author_ids = df["id"].unique()
    author_mapping = {author_id: i for i, author_id in enumerate(author_ids)}
    df["id"] = df["id"].map(author_mapping)
    df.to_csv(id_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a sample image from the database.")
    parser.add_argument("--test_id_csv", type=str, required=True, help="Path to the image database.")
    parser.add_argument("--train_id_csv", type=str, required=True, help="Path to the image database.")
    parser.add_argument("--id_csv", type=str, required=True, help="Path to the CSV file containing image IDs.")
    parser.add_argument('--image_db', type=str, required=True, help='Path to the image database.')
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the plots.")
    args = parser.parse_args()
    split_dataset(args.id_csv, args.test_id_csv, args.train_id_csv, args.image_db, args.output_dir)
    relabel_csv(args.test_id_csv)
    relabel_csv(args.train_id_csv)
