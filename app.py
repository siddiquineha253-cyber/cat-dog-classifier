
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


# -----------------------------
# PROFESSIONAL AI APP INTERFACE
# -----------------------------

custom_css = """
.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
    background: #f5f4ff !important;
}

#app-header {
    text-align: center;
    padding: 35px 15px 20px;
}

#app-header h1 {
    color: #4338ca;
    font-size: 34px;
    font-weight: 800;
}

#app-header p {
    color: #64748b;
    font-size: 16px;
}

#upload-card {
    background: white;
    border-radius: 22px;
    padding: 22px;
    border: 1px solid #e0e7ff;
}

#result-box {
    border-radius: 16px;
}

#classify-btn {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
}
"""

with gr.Blocks(
    title="Cat vs Dog AI Classifier",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet"
    ),
    css=custom_css
) as demo:

    gr.HTML("""
    <div id="app-header">
        <h1>🐾 PetVision AI</h1>
        <p>Smart Cat & Dog Image Classifier</p>
        <p>Upload a photo and let AI identify your pet.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_id="upload-card"):

                gr.Markdown("### 📸 Upload Your Image")

                image_input = gr.Image(
                    type="numpy",
                    label="Choose a Cat or Dog photo",
                    height=300
                )

                classify_btn = gr.Button(
                    "✨ Classify Image",
                    variant="primary",
                    elem_id="classify-btn",
                    size="lg"
                )

                result_output = gr.Textbox(
                    label="AI Prediction",
                    placeholder="Your prediction will appear here...",
                    interactive=False,
                    elem_id="result-box"
                )

                clear_btn = gr.ClearButton(
                    [image_input, result_output],
                    value="↻ Try Another Image"
                )

    gr.HTML("""
    <div class="footer">
        <p>Powered by Deep Learning & CNN Technology</p>
        <p>PetVision AI | Smart Image Classification</p>
    </div>
    """)

    classify_btn.click(
        fn=predict_cat_dog,
        inputs=image_input,
        outputs=result_output
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

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        pwa=True,
        favicon_path=str(BASE_DIR / "launchericon-512x512.png")
    )


