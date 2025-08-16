from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model
from PIL import Image
import numpy as np
import json

app = FastAPI()

# Load the model and class names
model = load_model("foodvision_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

@app.get("/")
def read_root():
    return {"message": "FoodVision model is up!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input
    img_array = np.array(image) 
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return JSONResponse(content={
        "class": class_names[class_idx],
        "confidence": round(confidence * 100, 2)
    })
