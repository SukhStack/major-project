import pandas as pd
import os

def merge_csv_files(input_folder, output_file):
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    print(f"Found {len(all_files)} CSV files.")

    df_list = []

    for file in all_files:
        path = os.path.join(input_folder, file)
        print(f"Reading {file}...")
        df = pd.read_csv(path)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    print("Saving merged file...")
    merged_df.to_csv(output_file, index=False)

    print(f"Merged dataset saved to {output_file}")
    print(f"Final shape: {merged_df.shape}")

if __name__ == "__main__":
    input_folder = "../data/archive"
    output_file = "../data/CICIDS2017.csv"
    merge_csv_files(input_folder, output_file)