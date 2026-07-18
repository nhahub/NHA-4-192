"""
Log an already-completed YOLOv8 training run into MLflow.

This script does NOT retrain anything. It reads the results.csv file
that YOLOv8 automatically saved during your original training run,
and replays every epoch's metrics into MLflow so you get the full
training curve history without retraining.

HOW TO USE:
1. Update the paths in the CONFIG section below to match your machine.
2. Run this script once: python log_yolo_to_mlflow.py
3. Run `mlflow ui` in your terminal, then open http://127.0.0.1:5000
   to see the logged run.
"""

import os
import csv
import mlflow

# ─── CONFIG ──────────────────────────────────────────────────────
# Folder where YOLO saved this training run (contains results.csv and weights/best.pt)
RUN_FOLDER = r".\Model"

RESULTS_CSV = os.path.join(RUN_FOLDER, "results.csv")
BEST_MODEL_PATH = os.path.join(RUN_FOLDER, "best.pt")

# Hyperparameters from your training notebook (cell 2)
HYPERPARAMS = {
    "model": "yolov8n.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10,
    "translate": 0.1,
    "scale": 0.5,
    "lr0": 0.01,
    "lrf": 0.001,
    "warmup_epochs": 3,
    "dataset": "class_traffic_signs",
}

EXPERIMENT_NAME = "traffic_sign_yolo_detection"
RUN_NAME = "yolov8n_run1"
# ──────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"ERROR: Could not find results.csv at: {RESULTS_CSV}")
        print("Double check the RUN_FOLDER path at the top of this script.")
        return

    # Use the SQLite database in this folder instead of the default file-based mlruns
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # 1. Log hyperparameters
        mlflow.log_params(HYPERPARAMS)

        # 2. Read results.csv and log every epoch as a step
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            # YOLO sometimes adds extra spaces around column names — clean them
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            last_row = None
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                epoch = int(float(row.get("epoch", 0)))

                # Log whichever metrics exist in this row (column names can vary slightly by YOLO version)
                metric_map = {
                    "train/box_loss": "train_box_loss",
                    "train/cls_loss": "train_cls_loss",
                    "train/dfl_loss": "train_dfl_loss",
                    "metrics/precision(B)": "precision",
                    "metrics/recall(B)": "recall",
                    "metrics/mAP50(B)": "mAP50",
                    "metrics/mAP50-95(B)": "mAP50_95",
                    "val/box_loss": "val_box_loss",
                    "val/cls_loss": "val_cls_loss",
                    "val/dfl_loss": "val_dfl_loss",
                }

                for csv_col, mlflow_name in metric_map.items():
                    if csv_col in row and row[csv_col] not in ("", None):
                        try:
                            mlflow.log_metric(mlflow_name, float(row[csv_col]), step=epoch)
                        except ValueError:
                            pass

                last_row = row

            if last_row:
                print(f"Logged {epoch} epochs of training history.")
                # Log final epoch's metrics as top-level summary too
                if "metrics/mAP50(B)" in last_row:
                    mlflow.log_metric("final_mAP50", float(last_row["metrics/mAP50(B)"]))
                if "metrics/mAP50-95(B)" in last_row:
                    mlflow.log_metric("final_mAP50_95", float(last_row["metrics/mAP50-95(B)"]))

        # 3. Log the trained model file itself as an artifact (if it exists)
        if os.path.exists(BEST_MODEL_PATH):
            mlflow.log_artifact(BEST_MODEL_PATH, artifact_path="model")
            print(f"Logged model file: {BEST_MODEL_PATH}")
        else:
            print(f"WARNING: best.pt not found at {BEST_MODEL_PATH} — skipped model logging.")

        # 4. Log the results.csv itself too, for reference
        mlflow.log_artifact(RESULTS_CSV, artifact_path="training_logs")

        print("\nDone! Run 'mlflow ui' in your terminal, then open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
