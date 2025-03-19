import argparse
import cv2
import lmdb
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class ImageProcessor:
    def __init__(self, source_db, target_db, id_csv, target_csv, target_width=512, target_height=48, grayscale=False, min_width=125):
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

    def __call__(self):
        print("Processing data...")
        self.process_images()

    def process_images(self):
        data = []
        with open(self.id_csv, "r", encoding="utf-8") as file:
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
        print(f"Number of unique authors: {len(self.df['id'].unique())}")

        env_original = lmdb.open(self.source_db, readonly=True, lock=False)
        self.txn_original = env_original.begin()
        map_size = 10 * 1024 * 1024 * 1024  # 10GB
        env_new = lmdb.open(self.target_db, map_size=map_size)

        def process_image(image_name):
            print(f"Processing {image_name}")
            key = self.txn_original.get(image_name.encode())
            if key is None:
                return None

            img = cv2.imdecode(np.frombuffer(key, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = img.shape[:2]
            pad_height = int(self.target_height - h / 2)
            if pad_height < 0:
                raise ValueError
            img = np.pad(img, ((pad_height, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

            if w > self.target_width:
                author = self.df[self.df["image_name"] == image_name]["id"].values[0]
                self.df.drop(self.df[self.df["image_name"] == image_name].index, inplace=True)
                num_splits = (w + self.target_width - 1) // self.target_width
                for i in range(num_splits):
                    start = i * self.target_width
                    end = min(start + self.target_width, w)
                    split_img = img[:, start:end]
                    if self.target_width > end - start > self.min_width:
                        pad_width = max(0, self.target_width - (end - start))
                        split_img = np.pad(split_img, ((0, 0), (0, pad_width), (0, 0)), mode='constant',
                                           constant_values=0)
                    _, img_encoded = cv2.imencode(".jpg", split_img)
                    img_bytes = img_encoded.tobytes()
                    new_key = f"{image_name.removesuffix('jpg')}_{i}.jpg".encode()
                    print(
                        f"Splitting {image_name} into {image_name.removesuffix('jpg')}_{i}.jpg of author: {author} with shape")
                    self.df = self.df._append({"image_name": f"{image_name.removesuffix('jpg')}_{i}.jpg", "id": author},
                                              ignore_index=True)
                    txn_new.put(new_key, img_bytes)
            else:
                author = self.df[self.df["image_name"] == image_name]["id"].values[0]
                pad_width = self.target_width - w
                padded_img = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
                _, img_encoded = cv2.imencode(".jpg", padded_img)
                img_bytes = img_encoded.tobytes()
                self.df = self.df._append({"image_name": image_name, "id": author}, ignore_index=True)
                txn_new.put(image_name.encode(), img_bytes)

                return self.df

        with env_new.begin(write=True) as txn_new:
            with ThreadPoolExecutor() as executor:
                results = list(
                    tqdm(executor.map(process_image, self.df["image_name"]), total=len(self.df["image_name"])))

        self.df = pd.concat(results, ignore_index=True)
        print(f"Final Number of unique authors: {len(self.df['id'].unique())}")
        self.df.to_csv(self.target_csv, index=False)
        env_original.close()


def main():
    parser = argparse.ArgumentParser(description="Resizing and splitting images.")
    parser.add_argument("--source_db", type=str, required=True, help="Path to the source image database.")
    parser.add_argument("--target_db", type=str, required=True, help="Path to the target image database.")
    parser.add_argument("--id_csv", type=str, required=True, help="Path to the CSV file containing ALL image IDs.")
    parser.add_argument("--target_csv", type=str, default='all_ids_split_img.csv', help="File for updated ids.")
    args = parser.parse_args()
    processor = ImageProcessor(args.source_db, args.target_db, args.id_csv, args.target_csv)
    processor()


if __name__ == "__main__":
    main()
