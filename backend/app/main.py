
import io
import base64
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "best.onnx"

IMG_SIZE = 960

# Default confidence threshold
CONF_THRESHOLD = 0.25

# Class-specific confidence thresholds
CLASS_THRESHOLDS = {
    "no_goggle": 0.15
}

IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "forklift",
    "no_vest"
]

# Convert class names to IDs
CLASS_THRESHOLDS_BY_ID = {
    CLASS_NAMES.index(class_name): threshold
    for class_name, threshold in CLASS_THRESHOLDS.items()
}


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Smart PPE Detection API",
    description="AI-powered Personal Protective Equipment Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD ONNX MODEL
# =========================================================

session = None
model_status = "not_loaded"


def load_model():
    global session, model_status

    if not MODEL_PATH.exists():
        model_status = "not_loaded"
        print(f"Model not found: {MODEL_PATH}")
        return

    try:
        session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        model_status = "loaded"
        print("ONNX model loaded successfully!")

    except Exception as error:
        session = None
        model_status = "error"
        print(f"Model loading error: {error}")


load_model()


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    original_width, original_height = image.size

    scale = min(
        IMG_SIZE / original_width,
        IMG_SIZE / original_height
    )

    new_width = round(original_width * scale)
    new_height = round(original_height * scale)

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (IMG_SIZE, IMG_SIZE),
        (114, 114, 114)
    )

    pad_x = (IMG_SIZE - new_width) / 2
    pad_y = (IMG_SIZE - new_height) / 2

    left = round(pad_x - 0.1)
    top = round(pad_y - 0.1)

    canvas.paste(resized, (left, top))

    image_array = np.asarray(canvas, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return (
        image_array.astype(np.float32),
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height
    )


# =========================================================
# NON-MAXIMUM SUPPRESSION
# =========================================================

def calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area1 = (
        max(0, box[2] - box[0]) *
        max(0, box[3] - box[1])
    )

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - intersection

    return intersection / np.maximum(union, 1e-7)


def non_max_suppression(boxes, scores, class_ids):
    selected = []

    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]

        while len(indices) > 0:
            current = indices[0]
            selected.append(current)

            if len(indices) == 1:
                break

            remaining = indices[1:]

            ious = calculate_iou(
                boxes[current],
                boxes[remaining]
            )

            indices = remaining[ious < IOU_THRESHOLD]

    selected.sort(
        key=lambda index: scores[index],
        reverse=True
    )

    return selected[:300]


# =========================================================
# POSTPROCESSING
# =========================================================

def postprocess_output(
    output,
    scale,
    pad_x,
    pad_y,
    original_width,
    original_height
):
    predictions = output[0].T

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Apply class-specific confidence thresholds
    thresholds = np.full(
        len(class_ids),
        CONF_THRESHOLD,
        dtype=np.float32
    )

    for class_id, threshold in CLASS_THRESHOLDS_BY_ID.items():
        thresholds[class_ids == class_id] = threshold

    mask = confidences >= thresholds

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    boxes = np.zeros_like(boxes_xywh)

    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    selected = non_max_suppression(
        boxes,
        confidences,
        class_ids
    )

    results = []

    for index in selected:
        x1, y1, x2, y2 = boxes[index]

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = float(np.clip(x1, 0, original_width))
        y1 = float(np.clip(y1, 0, original_height))
        x2 = float(np.clip(x2, 0, original_width))
        y2 = float(np.clip(y2, 0, original_height))

        class_id = int(class_ids[index])

        results.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": round(float(confidences[index]), 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    return results


# =========================================================
# PREDICTION PIPELINE
# =========================================================

def predict_image(image):
    if session is None:
        raise HTTPException(
            status_code=503,
            detail="ONNX model is not loaded"
        )

    input_tensor, scale, pad_x, pad_y, width, height = (
        preprocess_image(image)
    )

    start_time = time.perf_counter()

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: input_tensor}
    )[0]

    latency = (time.perf_counter() - start_time) * 1000

    detections = postprocess_output(
        output,
        scale,
        pad_x,
        pad_y,
        width,
        height
    )

    return {
        "success": True,
        "model": "YOLO11n-ONNX",
        "image_size": {
            "width": width,
            "height": height
        },
        "total_detections": len(detections),
        "inference_time_ms": round(latency, 2),
        "detections": detections
    }


# =========================================================
# IMAGE VALIDATION
# =========================================================

async def read_image(file: UploadFile):
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty image file"
        )

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart PPE Detection API",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_status": model_status
    }


@app.get("/model/metadata")
async def model_metadata():
    return {
        "model_name": "YOLO11n-ONNX",
        "model_status": model_status,
        "input_size": IMG_SIZE,
        "confidence_threshold": CONF_THRESHOLD,
        "class_thresholds": CLASS_THRESHOLDS,
        "iou_threshold": IOU_THRESHOLD,
        "classes": CLASS_NAMES,
        "number_of_classes": len(CLASS_NAMES),
        "runtime": "ONNX Runtime",
        "provider": "CPUExecutionProvider"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_image(file)
    return predict_image(image)


@app.post("/predict/visualize")
async def predict_visualize(file: UploadFile = File(...)):
    image = await read_image(file)

    prediction = predict_image(image)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for detection in prediction["detections"]:
        bbox = detection["bbox"]

        coordinates = (
            bbox["x1"],
            bbox["y1"],
            bbox["x2"],
            bbox["y2"]
        )

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        draw.rectangle(
            coordinates,
            outline="lime",
            width=3
        )

        draw.text(
            (bbox["x1"], max(0, bbox["y1"] - 18)),
            label,
            fill="lime"
        )

    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    prediction["annotated_image"] = image_base64

    return prediction
