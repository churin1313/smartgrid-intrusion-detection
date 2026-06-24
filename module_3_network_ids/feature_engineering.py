import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

```
# Clean column names
df.columns = df.columns.str.strip()

# Assume last column is the label column
label_column = df.columns[-1]

# Split features and labels
X = df.iloc[:, :-1]
y = df[label_column]

# Replace infinite values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Convert all feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values with column means
X = X.fillna(X.mean(numeric_only=True))

# Replace any remaining NaN values
X = X.fillna(0)

# Encode attack labels automatically
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nDetected Classes:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} -> {label}")

return X, y, encoder
```
