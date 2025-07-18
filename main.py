import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
import matplotlib.pyplot as plt

print("✅ Script started")

def load_data():
    # Load synthetic data
    print("✅ Loading synthetic data...")
    X = pd.read_csv("synthetic_features.csv")
    y = pd.read_csv("synthetic_labels.csv")["label"]
    return X, y

def preprocess(X, y, pca_components=30):
    # Encode labels (NEGATIVE → 0, POSITIVE → 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"✅ Classes: {le.classes_}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"✅ PCA shape: {X_pca.shape}")

    return X_pca, y_encoded

def train_model(X, y):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # LightGBM classifier
    clf = lgb.LGBMClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc:.2f}")

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"✅ Cross-Validation Accuracy: {cv_scores.mean():.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

def main():
    X, y = load_data()
    print(f"✅ Raw feature shape: {X.shape}")
    X_proc, y_proc = preprocess(X, y, pca_components=30)
    train_model(X_proc, y_proc)

if __name__ == "__main__":
    main()
