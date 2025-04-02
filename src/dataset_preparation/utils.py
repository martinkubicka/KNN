import os
import pandas as pd
import numpy as np
import lmdb
import matplotlib.pyplot as plt
import cv2
import csv


def display_sample_images(image_db, sampled_image_names, num_images=100):
    """Display a sample of images from the LMDB database. For the puropse of debugging and documentation."""
    env = lmdb.open(image_db, readonly=True, lock=False)
    txn = env.begin()
    images = []
    print(f"Displaying {len(sampled_image_names)} images...")
    print(f"Sampled image names: {sampled_image_names[:num_images]}")

    for image_name in sampled_image_names[:num_images]:
        img_data = txn.get(image_name.encode())
        if img_data:
            img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            print(img.shape)
            images.append(img)

    env.close()

    cols = 10
    rows = num_images // cols + (num_images % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def check_db(db_path, csv1, csv2):
    """Check if all images in the CSV files exist in the LMDB database."""
    env = lmdb.open(db_path, readonly=True, lock=False)
    txn = env.begin()

    def check_csv(csv_file):
        missing = []
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row["image_name"].encode('utf-8')  # LMDB keys are bytes
                if not txn.get(key):
                    missing.append(row["image_name"])
        return missing

    missing1 = check_csv(csv1)
    missing2 = check_csv(csv2)

    print(f"\nCSV 1 ({csv1}) - Missing: {len(missing1)}")
    if missing1:
        print("Examples:", missing1[:5])

    print(f"\nCSV 2 ({csv2}) - Missing: {len(missing2)}")
    if missing2:
        print("Examples:", missing2[:5])


def relabel_csv(id_csv):
    """Relabel the author IDs in the CSV file to be continuous integers starting from zero."""
    df = pd.read_csv(id_csv)
    print(f"Number of unique authors: {df['id'].nunique()} in file {id_csv}")
    author_ids = df["id"].unique()
    author_mapping = {author_id: i for i, author_id in enumerate(author_ids)}
    df["id"] = df["id"].map(author_mapping)
    df.to_csv(id_csv, index=False)


def prepend_train_test_to_path(path):
    """Helper function to prepend 'train_' and 'test_' to the path."""
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    train_path = os.path.join(dirname, "train_" + basename)
    test_path = os.path.join(dirname, "test_" + basename)
    return train_path, test_path


def open_raw_ids(csv_file, data):
    with open(csv_file, "r", encoding="utf-8") as file:  # read the file in previous format
        for line in file:
            try:
                parts = line.strip().split(" ")
                image_name = parts[0]
                id_value = parts[1]
                data.append([image_name, id_value])
            except Exception as e:
                print(f"Error processing line: {line}")
                print(e)

    return data
