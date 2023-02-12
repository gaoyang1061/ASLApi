from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import skimage
import keras.api._v2.keras as keras
import os
import uuid
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

#Change directory to /tmp folder
IMAGEDIR = "/tmp/"
IMAGE_SIZE = 64

def load_img(filePath, imgSize, model):
    dictLabels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                  'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                  'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
    X = np.empty((1, imgSize, imgSize, 3), dtype=np.float32)
    img_file = cv2.imread(filePath)
    if img_file is not None:
        img_file = skimage.transform.resize(img_file, (imgSize, imgSize, 3))
        img_arr = np.asarray(img_file).reshape((-1, imgSize, imgSize, 3))
        X[0] = img_arr

    prediction = model.predict(X)
    letterNum = prediction[0].tolist().index(max(prediction[0]))

    # convert num to letter
    letter = None
    keys = [k for k, v in dictLabels.items() if v == letterNum]
    if keys:
        letter = keys[0]

    confidencePercentage = max(prediction[0])
    return letter, confidencePercentage


@app.get('/')
def getSample():
    return JSONResponse({"result": "hello world"})

@app.post('/')
async def predict(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    filePath = f"{IMAGEDIR}{file.filename}"
    contents = await file.read()  # <-- Important!

    # example of how you can save the file
    with open(filePath, "wb") as f:
        f.write(contents)
    # filePath = "..\\F.jpg"

    # load model
    model_path = "model"
    model = keras.models.load_model(model_path)

    # sample code to load image and predict it with confidence
    predictLetter, confidence = load_img(filePath, IMAGE_SIZE, model)

    os.remove(filePath)
    return JSONResponse({"Letter": predictLetter, "Confidence": str(confidence * 100) + "%"})


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9000)
