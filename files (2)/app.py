"""
Cloud-Based Image Classification System
Backend: Flask + TensorFlow (MobileNetV2 - FREE pretrained on ImageNet)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import base64
import io
import os
import time
from PIL import Image
import logging

# ─── Setup ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ─── Load Model (once at startup) ────────────────────────────────────────────
logger.info("⏳ Loading MobileNetV2 model (first run downloads ~14MB)...")
model = MobileNetV2(weights='imagenet')
logger.info("✅ Model loaded successfully!")

# Warmup
dummy = np.zeros((1, 224, 224, 3))
model.predict(dummy, verbose=0)
logger.info("✅ Model warmed up!")


# ─── Helper Functions ────────────────────────────────────────────────────────
def preprocess_image(img_data: bytes) -> np.ndarray:
    """Convert raw image bytes → preprocessed NumPy array for MobileNetV2."""
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img = img.resize((224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def get_category(label: str) -> str:
    """Map ImageNet label to a simple category."""
    label_lower = label.lower()
    animals = ['dog','cat','bird','fish','horse','elephant','lion','tiger','bear',
               'wolf','fox','rabbit','deer','monkey','snake','frog','turtle','whale',
               'dolphin','shark','insect','butterfly','bee','spider','retriever',
               'labrador','poodle','beagle','husky','bulldog','terrier','spaniel']
    vehicles = ['car','truck','bus','train','plane','aircraft','boat','ship','bicycle',
                'motorcycle','jeep','van','taxi','ambulance','sports car']
    food    = ['pizza','burger','sandwich','cake','bread','fruit','vegetable','coffee',
               'sushi','noodle','rice','soup','salad','apple','banana','orange']
    plants  = ['flower','tree','rose','sunflower','daisy','tulip','grass','mushroom',
               'cactus','fern','coral','seaweed']
    tech    = ['phone','laptop','computer','keyboard','mouse','camera','tv','monitor',
               'headphone','speaker','remote','clock','watch']

    for w in animals:
        if w in label_lower: return "🐾 Animal"
    for w in vehicles:
        if w in label_lower: return "🚗 Vehicle"
    for w in food:
        if w in label_lower: return "🍕 Food"
    for w in plants:
        if w in label_lower: return "🌿 Plant"
    for w in tech:
        if w in label_lower: return "💻 Technology"
    return "📦 Object"


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return jsonify({
        "name": "CloudVision CNN API",
        "version": "1.0.0",
        "model": "MobileNetV2 (ImageNet)",
        "status": "running",
        "endpoints": {
            "POST /classify": "Classify an uploaded image file",
            "POST /classify_base64": "Classify a base64-encoded image",
            "GET /health": "Health check"
        }
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify an image uploaded as multipart/form-data.
    Field name: 'image'
    Returns top-5 predictions with confidence scores.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        return _run_inference(img_bytes, file.filename)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/classify_base64', methods=['POST'])
def classify_base64():
    """
    Classify an image sent as base64 JSON.
    Body: { "image": "data:image/jpeg;base64,..." or just the raw base64 string }
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No 'image' field in JSON body."}), 400

    try:
        b64 = data['image']
        # Strip data URI prefix if present
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        img_bytes = base64.b64decode(b64)
        filename = data.get('filename', 'image.jpg')
        return _run_inference(img_bytes, filename)
    except Exception as e:
        logger.error(f"Base64 classification error: {e}")
        return jsonify({"error": str(e)}), 500


def _run_inference(img_bytes: bytes, filename: str):
    """Core inference logic shared by both endpoints."""
    start = time.time()

    # Preprocess
    arr = preprocess_image(img_bytes)

    # Predict
    preds = model.predict(arr, verbose=0)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    # Decode top-5
    decoded = decode_predictions(preds, top=5)[0]
    # decoded: list of (imagenet_id, label, prob)

    results = []
    for i, (wn_id, label, prob) in enumerate(decoded):
        clean_label = label.replace('_', ' ').title()
        pct = round(float(prob) * 100, 2)
        results.append({
            "rank": i + 1,
            "label": clean_label,
            "imagenet_id": wn_id,
            "confidence": pct,
            "category": get_category(label)
        })

    top = results[0]
    logger.info(f"✅ {filename} → {top['label']} ({top['confidence']}%) in {elapsed_ms}ms")

    return jsonify({
        "success": True,
        "filename": filename,
        "top_prediction": top['label'],
        "confidence": top['confidence'],
        "category": top['category'],
        "inference_time_ms": elapsed_ms,
        "model": "MobileNetV2",
        "predictions": results
    })


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 CloudVision CNN API running on http://localhost:{port}")
    print("📖 Open index.html in your browser to use the frontend\n")
    app.run(debug=True, host='0.0.0.0', port=port)
