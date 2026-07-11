from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sqlite3
import io
import os
import uuid
import pandas as pd
from datetime import datetime, timezone

app = FastAPI(title="Traffic Sign Detection & Classification API")

# Load both models ONCE when the app starts (not per-request — that would be slow)
yolo_model = YOLO(r"./Model/best.pt")
cnn_model = load_model(r"./Model/best_model.keras")

# ─── MONITORING SETUP ────────────────────────────────────────────
# A small SQLite database that logs every prediction the API makes,
# plus the actual uploaded image saved to disk, so usage and confidence
# trends — and what people actually uploaded — can be reviewed later.
DB_PATH = "predictions.db"
IMAGES_DIR = "uploaded_images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            class_id INTEGER,
            class_name TEXT,
            confidence REAL,
            image_filename TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_uploaded_image(image_bytes):
    """Saves the uploaded image to disk and returns its generated filename."""
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filename


def log_prediction(endpoint, class_id, class_name, confidence, image_filename):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, endpoint, class_id, class_name, confidence, image_filename) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), endpoint, class_id, class_name, confidence, image_filename)
    )
    conn.commit()
    conn.close()


init_db()
# ──────────────────────────────────────────────────────────────────

# CLASS NAMES - by actual ClassId (0-42), GTSRB standard signs
class_names = {
    0:  'Speed limit 20',       1:  'Speed limit 30',
    2:  'Speed limit 50',       3:  'Speed limit 60',
    4:  'Speed limit 70',       5:  'Speed limit 80',
    6:  'End speed limit 80',   7:  'Speed limit 100',
    8:  'Speed limit 120',      9:  'No passing',
    10: 'No passing >3.5t',     11: 'Right of way',
    12: 'Priority road',        13: 'Yield',
    14: 'Stop',                 15: 'No vehicles',
    16: 'No vehicles >3.5t',    17: 'No entry',
    18: 'General caution',      19: 'Curve left',
    20: 'Curve right',          21: 'Double curve',
    22: 'Bumpy road',           23: 'Slippery road',
    24: 'Road narrows right',   25: 'Road work',
    26: 'Traffic signals',      27: 'Pedestrians',
    28: 'Children crossing',    29: 'Bicycles crossing',
    30: 'Ice/Snow',             31: 'Wild animals',
    32: 'End restrictions',     33: 'Turn right ahead',
    34: 'Turn left ahead',      35: 'Go straight',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right',           39: 'Keep left',
    40: 'Roundabout',           41: 'End no passing',
    42: 'End no passing >3.5t'
}

# YOUR ACTUAL KERAS MAPPING (alphabetical sort from training) — fixes the
# class index bug where flow_from_dataframe sorted folder names as strings
keras_class_mapping = {
    '0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7,
    '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14,
    '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21,
    '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28,
    '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35,
    '41': 36, '42': 37, '5': 38, '6': 39, '7': 40, '8': 41, '9': 42
}
keras_index_to_classid = {v: int(k) for k, v in keras_class_mapping.items()}


@app.get("/")
def root():
    return FileResponse("static_index.html")


@app.get("/status")
def status():
    return {"message": "Traffic Sign Detection & Classification API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read the uploaded image bytes
    image_bytes = await file.read()

    # Save the uploaded image to disk so it can be reviewed later in /history
    image_filename = save_uploaded_image(image_bytes)

    # Convert bytes to an OpenCV image
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Run YOLO prediction
    results = yolo_model.predict(source=image, conf=0.25)

    # Build a clean JSON response
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })
            log_prediction("detect", class_id, class_name, confidence, image_filename)

    if not detections:
        # Still log that this image was tried, even with no detections found
        log_prediction("detect", None, "no detections", None, image_filename)

    return {"detections": detections}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Read the uploaded image bytes
    image_bytes = await file.read()

    # Save the uploaded image to disk so it can be reviewed later in /history
    image_filename = save_uploaded_image(image_bytes)

    # Convert bytes to an OpenCV image
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Preprocess: BGR->RGB, resize to 128x128, normalize 0-1
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))
    image_array = np.expand_dims(image_resized / 255.0, axis=0).astype('float32')

    # Run CNN prediction
    predictions = cnn_model.predict(image_array, verbose=0)
    keras_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    # Map back through the keras alphabetical-sort fix to the real class id
    actual_classid = keras_index_to_classid[keras_idx]
    label = class_names[actual_classid]

    log_prediction("classify", actual_classid, label, confidence, image_filename)

    return {
        "class_id": actual_classid,
        "class_name": label,
        "confidence": confidence
    }


@app.get("/history")
def history(limit: int = 100):
    """Returns the most recent logged predictions, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT id, timestamp, endpoint, class_id, class_name, confidence, image_filename FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    return {
        "count": len(rows),
        "predictions": [
            {
                "id": row[0],
                "timestamp": row[1],
                "endpoint": row[2],
                "class_id": row[3],
                "class_name": row[4],
                "confidence": row[5],
                "image_url": f"/history/image/{row[6]}" if row[6] else None
            }
            for row in rows
        ]
    }


@app.get("/history/image/{filename}")
def history_image(filename: str):
    """Serves a single previously uploaded image by its stored filename."""
    filepath = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(filepath):
        return {"error": "Image not found"}
    return FileResponse(filepath)


@app.get("/history/excel")
def history_excel():
    """Downloads the full prediction history as an Excel (.xlsx) file."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, endpoint, class_id, class_name, confidence, image_filename FROM predictions ORDER BY id DESC",
        conn
    )
    conn.close()

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Prediction History")
    buffer.seek(0)

    filename = f"prediction_history_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/history/db")
def history_db():
    """Downloads the raw SQLite database file containing all prediction history."""
    return FileResponse(
        DB_PATH,
        media_type="application/octet-stream",
        filename="predictions.db"
    )
