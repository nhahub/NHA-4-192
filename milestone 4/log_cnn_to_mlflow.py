"""
Log the already-completed CNN training run into MLflow.

This does NOT retrain anything. It replays the exact epoch-by-epoch
training log from your notebook (17 epochs, stopped early by
EarlyStopping out of a max of 80) into MLflow, plus your final
val/test accuracy and loss, your hyperparameters, and the saved
.keras model file as an artifact.

HOW TO USE:
1. Update MODEL_PATH below if your best_model.keras lives somewhere else.
2. Run this script once: python log_cnn_to_mlflow.py
3. Run `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`
   then open http://127.0.0.1:5000 to see the logged run.
"""

import os
import mlflow

# ─── CONFIG ──────────────────────────────────────────────────────
MODEL_PATH = r"./Model/best_model.keras"

EXPERIMENT_NAME = "traffic_sign_cnn_classification"
RUN_NAME = "cnn_gtsrb_128x128_run1"

# Hyperparameters pulled directly from the training notebook
HYPERPARAMS = {
    "architecture": "Sequential CNN (6 Conv2D blocks, 32->256 filters)",
    "input_size": "128x128x3",
    "num_classes": 43,
    "dataset": "GTSRB",
    "batch_size": 24,
    "max_epochs": 80,
    "epochs_actually_trained": 17,
    "optimizer": "Adam",
    "initial_learning_rate": 0.001,
    "loss_function": "sparse_categorical_crossentropy",
    "dropout_rate": 0.25,
    "data_augmentation": "zoom_range=0.2, rescale=1/255",
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.001,
    "reduce_lr_monitor": "val_loss",
    "reduce_lr_patience": 2,
    "reduce_lr_factor": 0.5,
    "reduce_lr_min_lr": 1e-6,
    "checkpoint_monitor": "val_accuracy",
    "checkpoint_save_best_only": True,
    "train_val_split": "80/20 stratified by ClassId",
}

# Full epoch-by-epoch history, transcribed from the notebook's training log.
# Each tuple: (epoch, train_acc, train_loss, val_acc, val_loss, learning_rate)
EPOCH_HISTORY = [
    (1,  0.5915, 1.5598, 0.9528, 0.1739, 0.0010),
    (2,  0.9572, 0.1559, 0.9809, 0.0599, 0.0010),
    (3,  0.9758, 0.0846, 0.9841, 0.0604, 0.0010),
    (4,  0.9797, 0.0682, 0.9477, 0.1777, 0.0010),
    (5,  0.9929, 0.0247, 0.9981, 0.0084, 0.0005),
    (6,  0.9949, 0.0175, 0.9955, 0.0138, 0.0005),
    (7,  0.9935, 0.0228, 0.9872, 0.0421, 0.0005),
    (8,  0.9971, 0.0100, 0.9981, 0.0045, 0.00025),
    (9,  0.9987, 0.0045, 0.9981, 0.0036, 0.00025),
    (10, 0.9978, 0.0064, 0.9955, 0.0184, 0.00025),
    (11, 0.9981, 0.0068, 0.9994, 0.0044, 0.00025),
    (12, 0.9991, 0.0040, 0.9994, 0.0020, 0.000125),
    (13, 0.9994, 0.0021, 0.9974, 0.0087, 0.000125),
    (14, 0.9987, 0.0043, 0.9987, 0.0039, 0.000125),
    (15, 0.9994, 0.0020, 0.9994, 0.0017, 0.0000625),
    (16, 0.9995, 0.0017, 1.0000, 0.0011, 0.0000625),
    (17, 0.9997, 0.0013, 0.9987, 0.0035, 0.0000625),
]

# Final evaluation results (separate evaluate() calls on val and test sets)
FINAL_METRICS = {
    "final_val_accuracy": 0.997449,   # 99.7448980808258 %
    "final_val_loss": 0.008801,
    "final_test_accuracy": 0.987807,  # 98.78067970275879 %
    "final_test_loss": 0.042820,
}
# ──────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Could not find model file at: {MODEL_PATH}")
        print("Logging will continue, but the model artifact will be skipped.")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # 1. Log hyperparameters
        mlflow.log_params(HYPERPARAMS)

        # 2. Replay epoch-by-epoch history
        for epoch, train_acc, train_loss, val_acc, val_loss, lr in EPOCH_HISTORY:
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", lr, step=epoch)

        print(f"Logged {len(EPOCH_HISTORY)} epochs of training history.")

        # 3. Log final headline metrics (from separate evaluate() calls)
        mlflow.log_metrics(FINAL_METRICS)

        # 4. Log the trained model file as an artifact
        if os.path.exists(MODEL_PATH):
            mlflow.log_artifact(MODEL_PATH, artifact_path="model")
            print(f"Logged model file: {MODEL_PATH}")

        print("\nDone! Run 'python -m mlflow ui --backend-store-uri sqlite:///mlflow.db'")
        print("then open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
