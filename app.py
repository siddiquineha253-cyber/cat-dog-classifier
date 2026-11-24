
import os
import gdown
import gradio as gr
import cv2
import json
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

demo = gr.Interface(

    fn=predict_cat_dog,

    inputs=gr.Image(
        type="numpy",
        label="Upload Cat or Dog Image"
    ),

    outputs=gr.Textbox(
        label="Result"
    ),

    title="Cat vs Dog Classifier",

    description="Upload an image to identify whether it is a Cat or a Dog."
)

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

    pwa_head = """
    <link rel="manifest" href="/cat-dog-manifest.json">
    <meta name="theme-color" content="#7C83FD">

    <script>
    if ("serviceWorker" in navigator) {
        window.addEventListener("load", () => {
            navigator.serviceWorker.register("/service-worker.js", {
                scope: "/"
            })
            .then(() => console.log("Service Worker registered"))
            .catch(error => console.error("Service Worker error:", error));
        });
    }
    </script>
    """

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        head=pwa_head
    )




