#4color space Transformations shader
from threading import Thread
import glob
import os
import cv2
import numpy as np
import math



for i in range(self.base,self.limit):
            image = cv2.imread(path[i])
            matrix = np.ones(image.shape,dtype = "uint8") * 35
            subtracted = cv2.subtract(image,matrix)
            cv2.imwrite('G://Deep_Learning_Dataset//024//Normal//' + "shader" + os.path.basename(path[i]),subtracted)
