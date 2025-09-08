import os
import glob
import pandas as pd

def preprocess_data(data_folder="data", output_file="processed_data.csv"):
    # Initialize empty file lists
    file_pattern_csv = []
    file_pattern_xls = []

    # If CSVs exist, use them
    if glob.glob(os.path.join(data_folder, "*.csv")):
        file_pattern_csv = glob.glob(os.path.join(data_folder, "*.csv"))
    # Else if Excels exist, use them
    elif glob.glob(os.path.join(data_folder, "*.xls*")):
        file_pattern_xls = glob.glob(os.path.join(data_folder, "*.xls*"))

    # Merge whichever list is found
    files = file_pattern_csv + file_pattern_xls
    print("Files found:", files)   #  Debugging step

    if not files:
        print("No CSV/Excel files found in the data folder.")
        return

    all_data = []
    print("Starting preprocessing...")

    for file in files:
        try:
            print(f"Loading: {file}")
            if file.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # Clean column names
            df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

            # Drop duplicates (file-wise)
            df = df.drop_duplicates()

            # Add label from filename
            label = os.path.splitext(os.path.basename(file))[0]
            df["label"] = label

            all_data.append(df)

        except Exception as e:
            print(f" Error reading {file}: {e}")

    if all_data:
        # Merge all files
        combined_df = pd.concat(all_data, ignore_index=True, join="outer")

        # Fill missing values after merging
        combined_df = combined_df.fillna(0)

        print(f" Combined dataset shape: {combined_df.shape}")
        print(f" Columns: {combined_df.columns.tolist()}")
        combined_df.to_csv(output_file, index=False)
        print(f" Preprocessed data saved as {output_file}")
    else:
        print(" No data files were processed. Check your 'data/' folder.")


if __name__ == "__main__":
    preprocess_data(data_folder="data")
