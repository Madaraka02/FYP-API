from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
# import cv2

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

MODEL = tf.keras.models.load_model("../saved_models/3")

CLASS_NAMES = ["COVID", "Normal"]

input_shape = MODEL.layers[0].input_shape
print(input_shape)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)):

    contents = await file.read()
    pil_image = Image.open(BytesIO(contents))
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))
    # add 3 channels to image
    pil_image = pil_image.convert('RGB')


    pil_image = tf.keras.preprocessing.image.img_to_array(pil_image)
    pil_image = pil_image / 255

    img_batch = np.expand_dims(pil_image, 0)
    
    predictions = MODEL.predict(img_batch)
    # score = tf.nn.softmax(predictions[0])
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)