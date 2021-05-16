#2salt and peper
from threading import Thread
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import glob
import math



def sp_noise(image,prob,img2):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img2[i][j]
    return output


path = glob.glob("G://Deep_Learning_Dataset//Healthy+traning400+Fundus_Train_Val_Data//*")
for i in path:
    img1 = img.imread(i,0)
    img2 = cv2.imread(i)
    cv2.resize(img1,(200,200))
    noise_img = sp_noise(img1,0.05,img2)
    cv2.imwrite('G://Deep_Learning_Dataset//02//Normal//' + os.path.basename(i) + '_noise.jpg',noise_img)
    print("complete make noise  " + os.path.basename(i))
