import pandas as pd
import matplotlib.pyplot as plt
import argparse


class RebelAuthors:
    def __init__(self, infile, outfile):
        self.input_file = infile
        self.output_file = outfile
        self.linecounter = {}

    def __call__(self):
        print("Processing data...")
        self.relabel_authors()
        self.print_stats()

    def relabel_authors(self):
        """Relabels author IDs."""
        """Loads the file, reindexes IDs, and saves the processed data."""
        data = []
        with open(self.input_file, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    parts = line.strip().split(" ")
                    image_name = parts[0]
                    id_value = parts[1]  # text is useless for out task
                    data.append([image_name, id_value])
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(e)
        print(f"Loaded {len(data)} lines from {self.input_file}")
        df = pd.DataFrame(data, columns=["image_name", "id"])
        print(f"Number of unique authors: {len(df['id'].unique())}")
        unique_ids = {old_id: new_id for new_id, old_id in enumerate(sorted(df["id"].unique()))}
        df["id"] = df["id"].map(unique_ids)  # sequential mapping

        df.to_csv(self.output_file, index=False)
        print(f"Processed data saved to {self.output_file}")

    def print_stats(self):
        df = pd.read_csv(self.output_file, sep=",", names=["image_name", "id"], engine="python", quotechar='"')
        print(f"Number of unique authors: {len(df['id'].unique())}")
        print(f"Number of images: {len(df)}")
        id_counts = df["id"].value_counts().sort_index()

        # row stats
        plt.figure(figsize=(10, 5))
        plt.bar(id_counts.index, id_counts.values)
        plt.xlabel("ID")
        plt.ylabel("Occurrences")
        plt.title("Occurrences of Each ID")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating new author IDs csv from the format provided by school.")
    parser.add_argument("--infile", type=str, required=True, help="Input file with author IDs. In format provided by the school.")
    parser.add_argument("--outfile", type=str, required=True, help="Output file with relabeled author IDs.")
    args = parser.parse_args()
    processor = RebelAuthors(args.infile, args.outfile)
    processor()

