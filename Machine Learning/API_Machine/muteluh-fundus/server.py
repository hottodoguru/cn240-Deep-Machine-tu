from fastapi import FastAPI, UploadFile, Form, File

import cv2
import imutils
import numpy as np
import pickle

glaucoma_model = pickle.load(open('glaucoma_model3.sav', 'rb'))
normal_model = pickle.load(open('normal_model5.sav', 'rb'))
other_model = pickle.load(open('other_model5.sav', 'rb'))
def machine_feature(img) :
    cup = 0
    disc = 1
    exudate = 0
    #split r,g,b_image
    b,g,r = cv2.split(img)

    #find cup
    ret,mask = cv2.threshold(g,198,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200,200))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    if len(cnts) > 0 :
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 0 :
            cup = radius*2

    #find disc
    ret,mask = cv2.threshold(r,254,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200,200))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    if len(cnts) > 0 :
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 0 :
            disc = radius*2

    #find exudate
    ret,mask = cv2.threshold(r+b,230,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200,200))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    if len(cnts) > 0 :
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cv2.drawContours(img, cnts, -1,(0, 55, 255), 2)
        exudate = cv2.contourArea(c)
  
    return (cup/disc) , exudate

def machine_model(img):
    class_names = ["glaucoma", "normal", "other"]
    conf = []
    
    cdr , exudate = machine_feature(img)
    conf.append(glaucoma_model.predict_proba([[cdr,exudate]])[0][1])
    conf.append(normal_model.predict_proba([[cdr,exudate]])[0][1])
    conf.append(other_model.predict_proba([[cdr,exudate]])[0][1])

    conf = conf / np.sum(conf)
    argmax = np.argmax(conf)

    
    return class_names[argmax], "{:.2f}".format(conf[argmax])
    
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
    
    class_out, class_conf = machine_model(img)

    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
