
import os
import cv2
import gdown
import gradio as gr

import keras
import numpy as np
import tensorflow as tf

# TensorFlow internal API compatibility
internal = tf.__internal__

if not hasattr(internal, "register_load_context_function"):
    internal.register_load_context_function = (
        internal.register_call_context_function
    )

import tensorflow_hub as hub

import tensorflow_hub as hub


# Model details
MODEL_PATH = "cat_dog_cnn.keras"
FILE_ID = "1PkwTG4_D24EHrsldU35f4QvQxZ8CsYlr"


# Download model if not available
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}&confirm=t"

    downloaded = gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

    if downloaded is None or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model download failed.")


# Custom Input Layer
class SafeInputLayer(keras.layers.InputLayer):

    def __init__(self, batch_input_shape=None, **kwargs):

        kwargs.pop("optional", None)
        kwargs.pop("input_shape", None)
        kwargs.pop("batch_size", None)

        if batch_input_shape is not None:
            kwargs["batch_shape"] = batch_input_shape

        super().__init__(**kwargs)


# Custom TensorFlow Hub Layer
class WrappedKerasLayer(keras.layers.Layer):

    def __init__(
        self,
        handle=None,
        batch_input_shape=None,
        **kwargs
    ):

        name = kwargs.pop("name", None)
        trainable = kwargs.pop("trainable", False)
        dtype = kwargs.pop("dtype", "float32")

        # Ignore unsupported legacy arguments
        kwargs.pop("optional", None)
        kwargs.pop("autocast", None)

        super().__init__(
            name=name,
            trainable=trainable,
            dtype=dtype
        )

        self.handle = handle
        self.batch_input_shape = batch_input_shape

        if not self.handle:
            raise ValueError("TensorFlow Hub handle is missing.")

        self.layer = hub.KerasLayer(
            self.handle,
            trainable=False
        )

    def call(self, inputs):
        return self.layer(inputs)

    def get_config(self):

        config = super().get_config()

        config.update({
            "handle": self.handle,
            "batch_input_shape": self.batch_input_shape
        })

        return config


# Load trained model
try:

    model = keras.saving.load_model(
        MODEL_PATH,

        custom_objects={
            "InputLayer": SafeInputLayer,
            "KerasLayer": WrappedKerasLayer,
            "WrappedLayer": WrappedKerasLayer
        },

        compile=False,
        safe_mode=False
    )

    print("CNN Model loaded successfully!")

except Exception as e:

    raise RuntimeError(
        f"Model load error: {e}"
    ) from e


# Prediction function
def predict_cat_dog(img):

    if img is None:
        return "Please upload an image."

    try:

        resized = cv2.resize(img, (224, 224))

        normalized = resized.astype("float32") / 255.0

        batch = np.expand_dims(normalized, axis=0)

        pred = model.predict(batch, verbose=0)

        label_idx = int(np.argmax(pred[0]))

        if label_idx == 0:
            return "Prediction: Cat"

        elif label_idx == 1:
            return "Prediction: Dog"

        else:
            return "Unknown prediction."

    except Exception as e:

        return f"Prediction error: {e}"


# Gradio application
demo = gr.Interface(

    fn=predict_cat_dog,

    inputs=gr.Image(
        type="numpy",
        label="Upload Cat or Dog Image"
    ),

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="Cat vs Dog Classifier",

    description="Upload an image to identify whether it is a Cat or a Dog."

)


# Launch application
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
