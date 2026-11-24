
import os
import gdown
import gradio as gr
import json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------
# BASE DIRECTORY
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cat_dog_cnn.keras"

FILE_ID = "1PkwTG4_D24EHrsldU35f4QvQxZ8CsYlr"

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------

if not MODEL_PATH.exists():

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        str(MODEL_PATH),
        quiet=False
    )

# -----------------------------
# LOAD MODEL
# -----------------------------

custom_objects = {
    "KerasLayer": hub.KerasLayer,
    "tensorflow_hub>KerasLayer": hub.KerasLayer
}

model = tf_keras.models.load_model(
    str(MODEL_PATH),
    custom_objects=custom_objects,
    compile=False
)

print("Model loaded successfully!")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_cat_dog(img):

    if img is None:
        return "Please upload an image."

    try:
        resized = cv2.resize(img, (224, 224))

        normalized = resized.astype("float32") / 255.0

        batch = np.expand_dims(normalized, axis=0)

        pred = model.predict(batch, verbose=0)

        label_idx = int(np.argmax(pred))

        if label_idx == 0:
            return "Prediction: Cat"
        else:
            return "Prediction: Dog"

    except Exception as e:

        print("Prediction error:", e)

        return "Unable to process this image."

# -----------------------------
# GRADIO INTERFACE
# -----------------------------

custom_css = """
/* Main app */
.gradio-container {
    width: 100% !important;
    max-width: 1100px !important;
    margin: auto !important;
    background: #f7f7fc !important;
    min-height: 100vh;
    padding: 12px !important;
    box-sizing: border-box !important;
}

/* App heading */
#app-header {
    text-align: center;
    padding: 18px 10px;
}

#app-header h1 {
    color: #5146c8;
    font-size: clamp(25px, 4vw, 32px);
    font-weight: 800;
    margin: 8px 0;
}

#app-header p {
    color: #77758c;
    font-size: 14px;
}

/* Hamburger menu */
#menu-area {
    position: relative;
    z-index: 200;
    padding: 8px 12px;
}

#menu-area details {
    position: relative;
}

#menu-area summary {
    list-style: none;
    cursor: pointer;
    background: #5146c8;
    color: white;
    width: 46px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 220;
}

#menu-area summary::-webkit-details-marker {
    display: none;
}

/* Left side drawer */
#side-menu {
    position: fixed;
    top: 0;
    left: 0;
    width: min(280px, 82vw);
    height: 100vh;
    height: 100dvh;
    box-sizing: border-box;
    background: #f7f7fc;
    padding: 75px 18px 20px;
    border-right: 1px solid #e6e3ff;
    box-shadow: 8px 0 25px rgba(0,0,0,0.10);
    z-index: 210;
}

/* Transparent menu buttons */
.menu-item {
    display: block;
    padding: 14px 16px;
    margin: 9px 0;
    border: 1px solid transparent;
    border-radius: 12px;
    color: #45405f !important;
    background: transparent !important;
    text-decoration: none !important;
    font-size: 15px;
    transition: 0.2s ease;
}

/* Glow only on selected/focused button */
.menu-item:focus,
.menu-item:focus-visible,
.menu-item:active {
    outline: none;
    border-color: #5146c8;
    color: #5146c8 !important;
    box-shadow: 0 0 10px rgba(81,70,200,0.35);
}

/* Main upload card */
#upload-card {
    width: calc(100% - 24px) !important;
    max-width: 1000px !important;
    box-sizing: border-box !important;
    background: white;
    border: 1px solid #e7e5f5;
    border-radius: 20px;
    padding: clamp(16px, 3vw, 28px);
    margin: 18px auto !important;
    box-shadow: 0 5px 18px rgba(50,40,100,0.06);
}

/* Photo upload and preview area */
#upload-card .image-container {
    min-height: clamp(280px, 42vw, 430px) !important;
}

/* Classify button */
#classify-btn {
    width: 100% !important;
    min-height: 60px !important;
    background: #5146c8 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Prediction result */
#result-box {
    border-radius: 12px;
    min-height: 110px !important;
}

/* Footer */
#app-footer {
    text-align: center;
    color: #88869b;
    font-size: 12px;
    padding: 18px;
}

/* Mobile responsive layout */
@media (max-width: 600px) {
    .gradio-container {
        padding: 8px !important;
    }

    #app-header {
        padding: 16px 8px;
    }

    #upload-card {
        width: 100% !important;
        margin: 12px auto !important;
        padding: 14px !important;
        border-radius: 16px;
    }

    #upload-card .image-container {
        min-height: 280px !important;
    }

    #classify-btn {
        min-height: 52px !important;
    }

    #result-box {
        min-height: 85px !important;
    }
}
"""


# -----------------------------
# SERVICE WORKER ROUTE
# -----------------------------

@demo.app.get("/service-worker.js")
async def serve_service_worker():

    worker_path = BASE_DIR / "service-worker.js"

    if worker_path.exists():

        return FileResponse(
            worker_path,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"}
        )

    return JSONResponse(
        status_code=404,
        content={"error": "Service Worker not found"}
    )

# -----------------------------
# MANIFEST ROUTE
# -----------------------------

@demo.app.get("/cat-dog-manifest.json")
async def serve_manifest():

    manifest_path = BASE_DIR / "manifest.json"

    if manifest_path.exists():

        data = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        return JSONResponse(
            content=data,
            media_type="application/manifest+json"
        )

    return JSONResponse(
        status_code=404,
        content={"error": "Manifest not found"}
    )

# -----------------------------
# APP ICON ROUTE
# -----------------------------

@demo.app.get("/{icon_name}.png")
async def serve_png(icon_name: str):

    icon_path = BASE_DIR / f"{icon_name}.png"

    if icon_path.exists():

        return FileResponse(
            icon_path,
            media_type="image/png"
        )

    return JSONResponse(
        status_code=404,
        content={"error": "Icon not found"}
    )

# -----------------------------
# LAUNCH APP
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        pwa=True,
        favicon_path=str(BASE_DIR / "launchericon-512x512.png"),
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet"
        ),
        css=custom_css
    )


