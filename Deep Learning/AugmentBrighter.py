#5color space Transformations brighter
from threading import Thread
import glob
import os
import cv2
import numpy as np
import math

for i in range(self.base,self.limit):
            image = cv2.imread(path[i])
            matrix = np.ones(image.shape,dtype = "uint8") * 35
            added = cv2.add(image,matrix)
            cv2.imwrite('G://Deep_Learning_Dataset//025//Normal//' + "brighter" + os.path.basename(path[i]),added)
            
