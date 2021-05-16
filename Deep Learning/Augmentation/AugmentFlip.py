#1flip image Horizontally
from threading import Thread
import glob
import os
import cv2
import math

for i in range(self.base,self.limit):
            image = cv2.imread(path[i])
            flippedimage = cv2.flip(image, 1)
            cv2.imwrite('G://Deep_Learning_Dataset//012complete//Normal//' + 'flip' + os.path.basename(path[i]),flippedimage)
        print("complete flip :"+ self.base)
