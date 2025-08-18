import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# PARAMETERS
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
SEED = 42

# 1) LOAD VALIDATION/TEST DATASET
# Point this to your "dataset/val" or "dataset/test" folder
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    labels="inferred",
    label_mode="binary",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode="rgb",
    seed=SEED,
    shuffle=False
)

# Prefetch
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# 2) LOAD MODEL
model = tf.keras.models.load_model("road_quality_model_transfer_512_mobilenet.keras")

# 3) GET PREDICTIONS
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

# 4) METRICS
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred, target_names=["Good", "Bad"]))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_true, y_pred))

roc_auc = roc_auc_score(y_true, y_pred_probs)
pr_auc = average_precision_score(y_true, y_pred_probs)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
