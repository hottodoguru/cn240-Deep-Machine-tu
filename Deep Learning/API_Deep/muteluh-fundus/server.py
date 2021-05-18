from fastapi import FastAPI, UploadFile, Form, File

import cv2
import numpy as np
from keras.models import load_model
model = load_model('VGG16_epochs.h5')

def deep_model(img):
    IMG_SIZE = 224
    class_names = ["glaucoma", "normal", "other"]
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    
    conf = model.predict(img)
    conf = conf / np.sum(conf)
    
    argmax = np.argmax(conf)
    
    return class_names[argmax], conf[0][argmax]
    
app = FastAPI()

@app.get("/")
async def helloworld():
    return {"greeting": "Hello World"}


@app.post("/api/fundus")
async def upload_image(nonce: str=Form(None, title="Query Text"), 
                       image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    class_out, class_conf = deep_model(img)
    print(nonce)
    print(class_out)
    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
