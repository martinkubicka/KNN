import argparse
import cv2
import lmdb
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import csv
import os
import polars as pl


def prepend_train_test_to_path(path):
    """Helper function to prepend 'train_' and 'test_' to the path."""
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    train_path = os.path.join(dirname, "train_" + basename)
    test_path = os.path.join(dirname, "test_" + basename)
    return train_path, test_path


class ImageProcessor:
    def __init__(self, source_db, target_db, id_csv, target_csv, target_width=1024, target_height=48, grayscale=True,
                 min_width=256, test_sample_max_img=20000):
        self.sampled_authors = None
        self.sampled_image_names = None
        self.source_db = source_db
        self.target_db = target_db
        self.id_csv = id_csv
        self.target_csv = target_csv
        self.df = None
        self.target_width = target_width
        self.target_height = target_height
        self.grayscale = grayscale
        self.txn_original = None
        self.min_width = min_width  # minimum width of the split image
        self.train_df = None
        self.test_df = None
        self.cnt = 0
        self.test_sample_max_img = test_sample_max_img

    def __call__(self):
        print("Processing data...")
        data = []
        with open(self.id_csv, "r", encoding="utf-8") as file: # read the file in previous format
            for line in file:
                try:
                    parts = line.strip().split(" ")
                    image_name = parts[0]
                    id_value = parts[1]
                    data.append([image_name, id_value])
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(e)
        print(f"Loaded {len(data)} lines from {self.id_csv}")
        self.df = pd.DataFrame(data, columns=["image_name", "id"])
        self.test_df = pd.DataFrame(columns=["image_name", "id"])
        print(f"Number of unique authors: {len(self.df['id'].unique())}")
        self.create_test_df()
        self.process_images()

    def create_test_df(self):
        """Create a test dataframe with authors having more than median pages and select a subset of images."""
        author_counts = self.df["id"].value_counts()
        median_occurrences = author_counts.median()
        filtered_authors = author_counts[author_counts >= median_occurrences].index

        # Shuffle authors before sampling for randomness
        sampled_authors = self.df[self.df["id"].isin(filtered_authors)]["id"].drop_duplicates().sample(frac=1,
                                                                                                       random_state=42).tolist()
        selected_authors = []
        selected_images = []
        total_images = 0
        max_images = self.test_sample_max_img

        for author in sampled_authors:
            author_images = self.df[self.df["id"] == author]["image_name"].tolist()
            author_image_count = len(author_images)

            # Check if adding this author's images would exceed the limit
            if total_images + author_image_count > max_images:
                break  # Stop before exceeding the limit

            # Add author's images and update count
            selected_authors.append(author)
            selected_images.extend(author_images)
            total_images += author_image_count

        self.sampled_authors = selected_authors
        self.sampled_image_names = selected_images
        print(f"Sampled {len(selected_authors)} authors with {total_images} images.")

    def pad_image_height(self, img):
        h, w = img.shape[:2]
        pad_height = (self.target_height - h) // 2
        if pad_height < 0:
            raise ValueError("Image height is greater than target height.")

        img = np.pad(img, ((pad_height, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        return img

    def gray_scale(self, img, author):
        if author in self.sampled_authors:
            return img  # Return early to avoid unnecessary computation

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None].repeat(3, axis=-1)

    def process_images(self):
        env_original = lmdb.open(self.source_db, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn_original = env_original.begin()
        map_size = 20 * 1024 * 1024 * 1024  # 20GB
        env_new = lmdb.open(self.target_db, map_size=map_size)

        def process_image(image_name):
            print(f"Processing image: {image_name}")
            key = self.txn_original.get(image_name.encode())
            if key is None:
                return None, None

            img = cv2.imdecode(np.frombuffer(key, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = self.pad_image_height(img)

            h, w = img.shape[:2]

            match = self.df.loc[self.df["image_name"] == image_name]
            if match.empty:
                return None, None
            author = match["id"].iloc[0]
            next_index = match.index[0] + 1

            if self.grayscale:
                img = self.gray_scale(img, author)
            concatenated_img = img
            next_image_name = ""
            while next_index < len(self.df):
                next_image_name = self.df.iloc[next_index]["image_name"]
                next_author = self.df.iloc[next_index]["id"]

                if author != next_author:
                    break

                next_key = self.txn_original.get(next_image_name.encode())
                if next_key is None:
                    break

                next_img = cv2.imdecode(np.frombuffer(next_key, dtype=np.uint8), cv2.IMREAD_COLOR)
                next_img = self.pad_image_height(next_img)
                if self.grayscale:
                    next_img = self.gray_scale(next_img, next_author)

                next_h, next_w = next_img.shape[:2]
                if h != next_h:
                    pad_height = int((self.target_height - h) / 2)
                    if pad_height < 0:
                        raise ValueError
                    next_img = np.pad(next_img, ((pad_height, pad_height), (0, 0), (0, 0)), mode='constant',
                                      constant_values=0)

                concatenated_img = np.concatenate((concatenated_img, next_img), axis=1)
                w += next_w

                if w >= self.target_width:
                    break

                next_index += 1

            if w > self.target_width:
                concatenated_img = concatenated_img[:, :self.target_width]

            pad_width = self.target_width - concatenated_img.shape[1]
            padded_img = np.pad(concatenated_img, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            _, img_encoded = cv2.imencode(".jpg", padded_img)
            img_bytes = img_encoded.tobytes()

            new_image_name = f"{image_name}_concat_{next_image_name}"
            with env_new.begin(write=True) as txn_new:
                txn_new.put(new_image_name.encode(), img_bytes)

            if author in self.sampled_authors:
                return None, {"image_name": new_image_name, "id": author}
            else:
                return {"image_name": new_image_name, "id": author}, None

        # Output paths
        train_csv_path, test_csv_path = prepend_train_test_to_path(self.target_csv)

        # Write headers once
        with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "id"])
        with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "id"])

            chunk_size = 3000
            start_index = len(self.df)

        for start in range(start_index, len(self.df), chunk_size):
            end = min(start + chunk_size, len(self.df))
            print(f"Batch {(start - start_index) // chunk_size + 1} of {(len(self.df) // chunk_size) + 1}")

            with ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(process_image, self.df["image_name"][start:end]),
                    total=end - start
                ))
            print("Unzipping results...")
            train_rows = []
            test_rows = []

            for train_row, test_row in results:
                if train_row is not None:
                    train_rows.append([train_row["image_name"], train_row["id"]])
                if test_row is not None:
                    test_rows.append([test_row["image_name"], test_row["id"]])

            if train_rows:
                with open(train_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(train_rows)

            if test_rows:
                with open(test_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(test_rows)

            print(f"Batch from {start} to {end} processed and saved.")

        print(f"Saved final CSVs to: \n  - {train_csv_path}\n  - {test_csv_path}")
        env_original.close()
        env_new.close()


def display_sample_images(image_db, sampled_image_names, num_images=100):
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
    # Open LMDB environment
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
    df = pd.read_csv(id_csv)
    print(f"Number of unique authors: {df['id'].nunique()} in file {id_csv}")
    author_ids = df["id"].unique()
    author_mapping = {author_id: i for i, author_id in enumerate(author_ids)}
    df["id"] = df["id"].map(author_mapping)
    df.to_csv(id_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Resizing and splitting images.")
    parser.add_argument("--source_db", type=str, default="lmdb.hwr_40-1.0", help="Path to the source image database.")
    parser.add_argument("--target_db", type=str, default="lmdb_processed", help="Path to the target image database.")
    parser.add_argument("--id_csv", type=str, default="lines.filtered_max_width.all",
                        help="Path to the CSV file containing ALL image IDs.")
    parser.add_argument("--target_csv", type=str, default='after_processing.csv', help="File for updated ids.")
    args = parser.parse_args()

    # Grayscale, resizing, and splitting images + test and train csvs
    processor = ImageProcessor(args.source_db, args.target_db, args.id_csv, args.target_csv)
    processor()

    # Relabeling the csv
    train_csv_path, test_csv_path = prepend_train_test_to_path(args.target_csv)
    relabel_csv(train_csv_path)
    relabel_csv(test_csv_path)

    # Displaying sample images
    df = pl.read_csv(train_csv_path)
    column_data = df["image_name"].to_list()
    display_sample_images(args.target_db, column_data)
    df = pl.read_csv(test_csv_path)
    column_data = df["image_name"].to_list()
    display_sample_images(args.target_db, column_data)

    # Check the db
    check_db(args.target_db, train_csv_path, test_csv_path)


if __name__ == "__main__":
    main()
