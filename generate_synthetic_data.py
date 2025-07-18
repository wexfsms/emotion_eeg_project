import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

print("✅ Generating synthetic EEG data...")

# Simulate high-quality EEG feature data for binary emotion classification
X, y = make_classification(
    n_samples=2000,
    n_features=105,
    n_informative=60,
    n_redundant=10,
    n_classes=2,
    class_sep=2.0,  # large separation for better accuracy
    flip_y=0.01,
    random_state=42
)

# Save feature matrix
pd.DataFrame(X).to_csv("synthetic_features.csv", index=False)

# Save labels
pd.DataFrame({
    "label": ["NEGATIVE" if val == 0 else "POSITIVE" for val in y]
}).to_csv("synthetic_labels.csv", index=False)

print("✅ Saved synthetic_features.csv and synthetic_labels.csv")
