import os
import gdown
import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import keras


MODEL_PATH = "cat_dog_cnn.keras"
FILE_ID = "1PkwTG4_D24EHrsldU35f4QvQxZ8CsYlr"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.saving.load_model(MODEL_PATH,safe_mode=False)

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
    title="Cat vs Dog Classifier"
)

if _name_ == "_main_":
    demo.launch(server_name="0.0.0.0", server_port=7860)
