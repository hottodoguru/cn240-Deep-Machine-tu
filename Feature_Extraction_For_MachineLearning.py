import cv2
import imutils

### cup
def cup(img) :
    r = 0
    ret,mask = cv2.threshold(img,198,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200,200))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    if len(cnts) > 0 :
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 0 :
            r = radius*2
    return r
    
## disc
def disc(img) :
    r = 0
    ret,mask = cv2.threshold(img,254,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200,200))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    center = None
    if len(cnts) > 0 :
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 0 :
            r = radius*2

    return r

## exudate
def exudate(img) :
    contour = 0
    ret,mask = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200,200))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    center = None
    if len(cnts) > 0 :
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cv2.drawContours(img, cnts, -1,(0, 55, 255), 2)
        contour = cv2.contourArea(c)
        
    return contour

def main() :
    img  = 'C:\\Users\\sutic\\Desktop\\AI_augment_image_glaucoma\\0Glaucoma_jpg\\366.jpg'
    img = cv2.imread(img)
    b,g,r = cv2.split(img)
    img_cup =  cup(g)
    img_disc = disc(r)
    img_exudate = exudate(r+b)
    print(img_cup)
    print(img_disc)
    print(img_exudate)

if __name__ == "__main__" :
    main()
