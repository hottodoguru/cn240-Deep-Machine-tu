#3rotation image 10-20 dregree
from threading import Thread
import glob
import os
from PIL import Image
import cv2
import random
import math

for i in range(self.base,self.limit):
            image = Image.open(path[i])
            randforro = [10 + random.random()*10+10-(random.random())*10]
            angle = random.choice(randforro)
            image = image.rotate(angle)
            image.save("G://Deep_Learning_Dataset//023//Normal//" + "rotated" + os.path.basename(path[i]))
