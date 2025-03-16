import argparse
import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def resize_and_pad(img, target_width=300):
    h, w = img.shape[:2]

    if w > target_width:
        img_resized = cv2.resize(img, (target_width, h), interpolation=cv2.INTER_AREA)
    else:  # padding
        pad_width = target_width - w
        if len(img.shape) == 3:  # Color image (BGR)
            img_resized = cv2.copyMakeBorder(img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            img_resized = cv2.copyMakeBorder(img, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

    return img_resized


class ImageProcessor:
    def __init__(self, source_db, target_db):
        self.source_db = source_db
        self.target_db = target_db

    def __call__(self):
        """Processes input data and stores results."""
        print("Processing data...")
        self.process_images()

    def process_images(self):
        env_original = lmdb.open(self.source_db, readonly=True, lock=False)
        txn_original = env_original.begin()

        with env_original.begin() as txn:
            total_keys = [key for key, _ in txn.cursor()]

        map_size = 10 * 1024 * 1024 * 1024  # 10GB
        env_new = lmdb.open(self.target_db, map_size=map_size)

        with env_new.begin(write=True) as txn_new:
            for key in tqdm(total_keys, desc="Processing Images"):
                img_data = txn_original.get(key)
                if img_data is None:
                    continue

                img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                adjusted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                adjusted_img = resize_and_pad(adjusted_img)
                _, img_encoded = cv2.imencode(".jpg", adjusted_img)
                img_bytes = img_encoded.tobytes()
                txn_new.put(key, img_bytes)

        env_original.close()
        env_new.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_db", type=str, required=True, help="Path to the source database")
    parser.add_argument("--target_db", type=str, required=True, help="Path to the target database")
    args = parser.parse_args()
    processor = ImageProcessor(args.source_db, args.target_db)
    processor()
