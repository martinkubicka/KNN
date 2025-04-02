import argparse
from image_processor import ImageProcessor
from utils import display_sample_images, check_db, relabel_csv, prepend_train_test_to_path
import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Scipt to process images and create new LMDB database with augmented images plus csv files describing the dataset.")
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
