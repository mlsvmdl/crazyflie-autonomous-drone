import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "zigzag_training_data.csv"
MODEL_DIR = "./models"
MODEL_FILE = os.path.join(MODEL_DIR, "zigzag_drone_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "zigzag_drone_scaler.pkl")

ACTION_NAMES = {
    0: "hover",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
    5: "yaw_left",
    6: "yaw_right",
    7: "up",
    8: "down",
}


def main():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: data file not found: {DATA_FILE}")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    feature_cols = ["dist_front", "dist_back", "dist_left", "dist_right", "altitude"]
    X = df[feature_cols].values
    y = df["action"].values

    # Clean NaNs/Infs just in case
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Simple class distribution print
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print("\nClass distribution:")
    for a, c in zip(unique, counts):
        pct = 100.0 * c / total
        print(f"  {a} ({ACTION_NAMES[a]:10s}): {c:6d} ({pct:5.1f}%)")

    # Split with stratify when possible
    if len(unique) > 1 and min(counts) >= 2:
        stratify_arg = y
    else:
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Smallish MLP is enough; you can tune sizes later
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )

    print("\nTraining MLP...")
    model.fit(X_train_scaled, y_train)
    print("Training done.")

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model to  : {MODEL_FILE}")
    print(f"Saved scaler to : {SCALER_FILE}")


if __name__ == "__main__":
    main()
