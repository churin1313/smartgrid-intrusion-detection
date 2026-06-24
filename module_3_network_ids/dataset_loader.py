#dataset is not included due to size limitations, refer README for more details
#replace file path with loaded dataset

import pandas as pd

def load_dataset(file_path):
"""
Load a CICIDS2017 CSV dataset.

```
Parameters:
    file_path (str): Path to the dataset CSV file.

Returns:
    pandas.DataFrame: Loaded dataset.
"""

try:
    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df

except FileNotFoundError:
    print(f"Error: Dataset file not found at '{file_path}'")
    return None

except Exception as e:
    print(f"Error loading dataset: {e}")
    return None
```

if **name** == "**main**":
dataset_path = input("Enter dataset path: ")
data = load_dataset(dataset_path)

```
if data is not None:
    print(data.head())
```
