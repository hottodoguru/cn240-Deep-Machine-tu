#6kernel filters sharpen
from threading import Thread
import glob
import os
import cv2
import numpy as np
import math



for i in range(self.base,self.limit):
            image = cv2.imread(path[i])
            kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            sharpened = cv2.filter2D(image,-1,kernel_sharpening)
            cv2.imwrite('G://Deep_Learning_Dataset//0562//Glaucoma//' + "sharpened" + os.path.basename(path[i]),sharpened)
