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

  def _init_(self, batch_input_shape=None, **kwargs):
    kwargs.pop("optional", None)
    kwargs.pop("input_shape", None)
    kwargs.pop("batch_size", None)
    if batch_input_shape is not None and "batch_shape" not in kwargs:
      kwargs["batch_shape"] = batch_input_shape
    super()._init_(**kwargs)


class WrappedKerasLayer(keras.layers.Layer):

  def _init_(self, *args, **kwargs):
    self.handle = kwargs.pop("handle", None)
    self.batch_input_shape = kwargs.pop("batch_input_shape", None)
    safe_kwargs = {}
    for key in ("name", "trainable", "dtype", "autocast"):
      if key in kwargs:
        safe_kwargs[key] = kwargs.pop(key)
    super()._init_(**safe_kwargs)
    self.layer = (
        hub.KerasLayer(self.handle, trainable=False) if self.handle else None
    )

  def call(self, inputs):
    return self.layer(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        "handle": self.handle,
        "batch_input_shape": self.batch_input_shape,
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


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
)

if _name_ == "_main_":
  port = int(os.environ.get("PORT", 7860))
  demo.launch(server_name="0.0.0.0", server_port=port)


