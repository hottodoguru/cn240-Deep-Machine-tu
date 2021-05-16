from threading import Thread
import glob
import os
import cv2
import numpy as np
import math

for i in range(self.base,self.limit):
            image = cv2.imread(path[i])
            kernel_6x6 = np.ones((6,6),np.float32) / 36
            blurred = cv2.filter2D(image,-1,kernel_6x6)
            cv2.imwrite('G://Deep_Learning_Dataset//0572//Normal//' + "blurred" + os.path.basename(path[i]),blurred)
