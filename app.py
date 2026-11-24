import os
import gdown
import gradio as gr
import cv2
import json
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import tensorflow as tf

MODEL_PATH = "cat_dog_cnn.keras"
FILE_ID = "1PkwTG4_D24EHrsldU35f4QvQxZ8CsYlr"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

def predict_cat_dog(img):
    if img is None:
        return "Please upload an image."

    resized = cv2.resize(img, (224, 224))
    normalized = resized / 255.0
    batch = np.expand_dims(normalized, axis=0)

    pred = model.predict(batch, verbose=0)
    label_idx = int(np.argmax(pred))

    if label_idx == 0:
        return "Prediction: Cat"
    else:
        return "Prediction: Dog"

demo = gr.Interface(
    fn=predict_cat_dog,
    inputs=gr.Image(type="numpy", label="Upload Cat or Dog Image"),
    outputs=gr.Textbox(label="Result"),
    title="Cat vs Dog Classifier",
    head='<link rel="manifest" href="/manifest.json"><meta name="theme-color" content="#7C83FD">'
)
@demo.app.get("/manifest.json") 
async def serve_manifest():
    manifest_path = BASE_DIR / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return JSONResponse(content=data, media_type="application/manifest+json")
    return JSONResponse(status_code=404, content={"error": "manifest.json not found"})
@demo.app.get("/{icon_name}.png")
async def serve_png(icon_name: str):
    icon_path = BASE_DIR / f"{icon_name}.png"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Icon not found"})

if __name__ == "__main__":
     demo.launch(server_name="0.0.0.0", server_port=7860)
   