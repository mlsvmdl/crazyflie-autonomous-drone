import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "zigzag_drone_training_data.csv"
MODEL_DIR = "./models/v1_zigzag"
MODEL_FILE = os.path.join(MODEL_DIR, "drone_mlp_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "drone_scaler.pkl")

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
        print(f"ERROR: {DATA_FILE} not found")
        return

    print("\n" + "=" * 70)
    print("  DRONE AI TRAINER - 6-FEATURE MODEL (F/B/L/R/Down/Up)")
    print("=" * 70 + "\n")

    df = pd.read_csv(DATA_FILE)

    expected_cols = [
        "dist_front",
        "dist_back",
        "dist_left",
        "dist_right",
        "dist_down",
        "dist_up",
        "action",
    ]
    if not all(col in df.columns for col in expected_cols):
        print(f"ERROR: Expected columns {expected_cols}")
        print(f"Found: {list(df.columns)}")
        return

    print(f"📊 Dataset loaded: {len(df):,} samples")
    print(
        f"   Features: dist_front, dist_back, dist_left, dist_right, dist_down, dist_up"
    )
    print(f"   Actions: {df['action'].nunique()} classes\n")

    print("Action distribution:")
    for action_id in sorted(df["action"].unique()):
        count = (df["action"] == action_id).sum()
        pct = count / len(df) * 100
        print(
            f"  {action_id} ({ACTION_NAMES[action_id]:10s}): {count:6d} ({pct:5.1f}%)"
        )

    X = df[
        ["dist_front", "dist_back", "dist_left", "dist_right", "dist_down", "dist_up"]
    ].values
    y = df["action"].values

    print(f"\n📐 Feature matrix shape: {X.shape}")
    print(f"   Target vector shape: {y.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Train/Test split:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing:  {len(X_test):,} samples\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("📈 Feature scaling applied (StandardScaler)\n")
    print("=" * 70)
    print("Training MLP Neural Network...")
    print("=" * 70 + "\n")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )
    model.fit(X_train_scaled, y_train)

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70 + "\n")

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Test Accuracy: {acc:.2%}\n")

    print("Per-action accuracy:")
    for action_id in sorted(np.unique(y_test)):
        mask = y_test == action_id
        if mask.sum() > 0:
            action_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {action_id} ({ACTION_NAMES[action_id]:10s}): {action_acc:.2%}")

    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70 + "\n")

    target_names = [ACTION_NAMES[i] for i in sorted(np.unique(y))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n" + "=" * 70)
    print("Confusion Matrix (sample - UP/DOWN actions):")
    print("=" * 70 + "\n")

    vertical_mask = np.isin(y_test, [7, 8])
    if vertical_mask.sum() > 0:
        cm_vertical = confusion_matrix(
            y_test[vertical_mask], y_pred[vertical_mask], labels=[7, 8]
        )
        print("UP/DOWN confusion:")
        print(f"           Pred UP  Pred DOWN")
        print(f"True UP    {cm_vertical[0, 0]:6d}   {cm_vertical[0, 1]:6d}")
        print(f"True DOWN  {cm_vertical[1, 0]:6d}   {cm_vertical[1, 1]:6d}")

    print("\n" + "=" * 70)
    print("Saving model...")
    print("=" * 70 + "\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Model saved: {MODEL_FILE}")
    print(f"✅ Scaler saved: {SCALER_FILE}")
    print(f"\n   Architecture: {model.hidden_layer_sizes}")
    print(f"   Input features: 6 (F/B/L/R/Down/Up)")
    print(f"   Output classes: 9 (hover + 8 actions)")
    print(f"   Iterations: {model.n_iter_}")
    print(
        f"   Best validation score: {model.best_validation_score_:.4f}"
        if hasattr(model, "best_validation_score_")
        else ""
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
