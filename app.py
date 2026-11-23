import os
import sys
import types
import cv2
import gdown
import gradio as gr
import keras
import numpy as np
import tensorflow as tf

load_context_mod = types.ModuleType(
    "tf_keras.src.saving.legacy.saved_model.load_context"
)
load_context_mod.register_load_context_function_in_load_context = (
    lambda *args, **kwargs: None
)
sys.modules["tf_keras.src.saving.legacy.saved_model.load_context"] = (
    load_context_mod
)

import tensorflow_hub as hub

MODEL_PATH = "cat_dog_cnn.keras"
FILE_ID = "1PkwTG4_D24EHrsldU35f4QvQxZ8CsYlr"

if not os.path.exists(MODEL_PATH):
  url = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"
  gdown.download(url, MODEL_PATH, quiet=False)

class SafeInputLayer(keras.layers.InputLayer):
  def _init_(self, **kwargs):
    kwargs.pop("optional", None)
    if "batch_input_shape" in kwargs and "batch_shape" not in kwargs:
      kwargs["batch_shape"] = kwargs.pop("batch_input_shape")
    kwargs.pop("input_shape", None)
    kwargs.pop("batch_size", None)
    super()._init_(**kwargs)


class WrappedKerasLayer(keras.layers.Layer):

  def _init_(self, handle=None, batch_input_shape=None, **kwargs):
    kwargs.pop("trainable", None)
    kwargs.pop("dtype", None)
    super()._init_(**kwargs)
    self.handle = handle
    self.batch_input_shape = batch_input_shape
    self.hub_layer = (
        hub.KerasLayer(handle, trainable=False) if handle else None
    )

  def call(self, inputs):
    return self.hub_layer(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        "handle": self.handle,
        "batch_input_shape": self.batch_input_shape,
    })
    return config

try:
  model = keras.saving.load_model(
      MODEL_PATH,
      custom_objects={
          "InputLayer": SafeInputLayer,
          "KerasLayer": WrappedKerasLayer,
      },
      safe_mode=False,
  )
except Exception as e:
  raise RuntimeError(f"Model load error: {e}")


def predict_cat_dog(img):
  if img is None:
    return "Please upload an image."

  resized = cv2.resize(img, (224, 224))
  normalized = resized / 255.0
  batch = np.expand_dims(normalized, axis=0)

  if hasattr(model, "predict"):
    pred = model.predict(batch, verbose=0)
  else:
    infer = model.signatures["serving_default"]
    out = infer(tf.constant(batch, dtype=tf.float32))
    pred = list(out.values())[0].numpy()

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
)
if _name_ == "_main_":
  port = int(os.environ.get("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=port)




