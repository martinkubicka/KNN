import pandas as pd
from sklearn.model_selection import train_test_split

input_csv = '/storage/brno2/home/xzarsk04/v4/train_after_processing.csv'
df = pd.read_csv(input_csv)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_output_csv = '/storage/brno2/home/xzarsk04/v4/train_ids.csv'
val_output_csv = '/storage/brno2/home/xzarsk04/v4/val_ids.csv'

train_df.to_csv(train_output_csv, index=False)
val_df.to_csv(val_output_csv, index=False)

print(f"Training set saved to {train_output_csv} with {len(train_df)} rows.")
print(f"Validation set saved to {val_output_csv} with {len(val_df)} rows.")
