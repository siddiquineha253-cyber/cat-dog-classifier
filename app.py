
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
# PETVISION AI - MOBILE INTERFACE
# -----------------------------

custom_css = """
.gradio-container {
    max-width: 600px !important;
    margin: auto !important;
    background: #f7f7fc !important;
    min-height: 100vh;
}

#app-header {
    text-align: center;
    padding: 18px 10px;
}

#app-header h1 {
    color: #5146c8;
    font-size: 28px;
    font-weight: 800;
    margin: 8px 0;
}

#app-header p {
    color: #77758c;
    font-size: 14px;
}

#menu-area {
    position: relative;
    z-index: 100;
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
    width: 44px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
}

#menu-area summary::-webkit-details-marker {
    display: none;
}

#side-menu {
    position: absolute;
    top: 50px;
    left: 0;
    width: 230px;
    background: white;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    border: 1px solid #e6e3ff;
}

.menu-item {
    display: block;
    padding: 13px;
    margin: 5px 0;
    border-radius: 10px;
    color: #45405f !important;
    text-decoration: none !important;
    font-size: 15px;
}

.menu-item:hover {
    background: #f0efff;
}

#upload-card {
    background: white;
    border: 1px solid #e7e5f5;
    border-radius: 20px;
    padding: 16px;
    margin: 8px;
    box-shadow: 0 5px 18px rgba(50,40,100,0.06);
}

#classify-btn {
    background: #5146c8 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

#result-box {
    border-radius: 12px;
}

#app-footer {
    text-align: center;
    color: #88869b;
    font-size: 12px;
    padding: 18px;
}

@media (max-width: 480px) {
    #app-header h1 {
        font-size: 25px;
    }

    #upload-card {
        margin: 5px;
        padding: 12px;
    }
}
"""

with gr.Blocks(title="PetVision AI") as demo:

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div id="menu-area">
                <details>
                    <summary>☰</summary>
                    <div id="side-menu">
                        <a class="menu-item" href="#home">🏠 Home</a>
                        <a class="menu-item" href="#privacy">🔒 Privacy Policy</a>
                        <a class="menu-item" href="#share">📤 Share App</a>
                        <a class="menu-item" href="#rate">⭐ Rate App</a>
                    </div>
                </details>
            </div>
            """)

    gr.HTML("""
    <div id="app-header">
        <h1>🐾 PetVision AI</h1>
        <p>Smart Cat & Dog Image Classifier</p>
        <p>Discover your pet with Artificial Intelligence.</p>
    </div>
    """)

    with gr.Group(elem_id="upload-card"):

        gr.Markdown("### 📸 Upload Your Image")

        image_input = gr.Image(
            type="numpy",
            label="Choose a Cat or Dog photo",
            height=230
        )

        classify_btn = gr.Button(
            "✨  Classify Image",
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
    <div id="app-footer">
        <p>Powered by Deep Learning & CNN Technology</p>
        <p>PetVision AI | Version 1.0.0</p>
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
        favicon_path=str(BASE_DIR / "launchericon-512x512.png"),
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet"
        ),
        css=custom_css
    )


